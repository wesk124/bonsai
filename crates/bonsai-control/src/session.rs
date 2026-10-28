//! Per-handoff state machine.
//!
//! ```text
//!            place()                SlotGranted              PrefillDone            Finished
//!  Queued ───────────▶ Leasing ─────────────────▶ Transferring ────────────▶ Decoding ─────────▶ Done
//!    ▲                   │ SlotDenied: try another decode worker
//!    └───────────────────┘ (P keeps its KV and waits)
//!
//!  any state ── HandoffFailed / timeout / worker lost ──▶ Failed ── fallback? ──▶ Queued (new HandoffId, BF16)
//! ```
//!
//! `Prefill` and `LeaseSlot` go out together on placement, so slot leasing
//! overlaps prefill compute and stays off the critical path.

use std::time::Instant;

use bonsai_core::{
    HandoffId, HandoffReport, RequestId, WorkerId,
    msg::{HandoffParams, InferenceRequest, SlotGrant},
};

use crate::coordinator::ConnId;

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum HandoffState {
    Queued,
    Leasing { prefill: WorkerId, decode: WorkerId, attempts: u32 },
    Transferring { prefill: WorkerId, decode: WorkerId },
    Decoding { prefill: WorkerId, decode: WorkerId },
    Done,
    Failed { reason: String },
}

impl HandoffState {
    pub fn name(&self) -> &'static str {
        match self {
            HandoffState::Queued => "queued",
            HandoffState::Leasing { .. } => "leasing",
            HandoffState::Transferring { .. } => "transferring",
            HandoffState::Decoding { .. } => "decoding",
            HandoffState::Done => "done",
            HandoffState::Failed { .. } => "failed",
        }
    }

    pub fn workers(&self) -> Option<(WorkerId, WorkerId)> {
        match *self {
            HandoffState::Leasing { prefill, decode, .. }
            | HandoffState::Transferring { prefill, decode }
            | HandoffState::Decoding { prefill, decode } => Some((prefill, decode)),
            _ => None,
        }
    }
}

#[derive(Debug, Clone)]
pub struct Session {
    pub request: InferenceRequest,
    pub params: HandoffParams,
    pub client: ConnId,
    pub state: HandoffState,
    pub grant: Option<SlotGrant>,
    pub submitted_at: Instant,
    pub state_since: Instant,
    pub first_token_at: Option<Instant>,
    /// Retries already used (BF16 fallback counts as one).
    pub retries: u32,
}

#[derive(Default)]
pub struct SessionManager {
    sessions: std::collections::HashMap<HandoffId, Session>,
    by_request: std::collections::HashMap<RequestId, HandoffId>,
    next_handoff: u64,
}

impl SessionManager {
    pub fn create(&mut self, request: InferenceRequest, params_without_id: HandoffParams, client: ConnId, now: Instant) -> HandoffId {
        let id = HandoffId(self.next_handoff);
        self.next_handoff += 1;
        let params = HandoffParams { handoff: id, ..params_without_id };
        self.by_request.insert(request.id, id);
        self.sessions.insert(
            id,
            Session {
                request,
                params,
                client,
                state: HandoffState::Queued,
                grant: None,
                submitted_at: now,
                state_since: now,
                first_token_at: None,
                retries: 0,
            },
        );
        id
    }

    pub fn get(&self, id: HandoffId) -> Option<&Session> {
        self.sessions.get(&id)
    }

    pub fn get_mut(&mut self, id: HandoffId) -> Option<&mut Session> {
        self.sessions.get_mut(&id)
    }

    /// Move to `next`, checking the transition is legal.
    pub fn transition(&mut self, id: HandoffId, next: HandoffState, now: Instant) -> bonsai_core::Result<&mut Session> {
        let s = self.sessions.get_mut(&id).ok_or(bonsai_core::BonsaiError::UnknownHandoff(id))?;
        use HandoffState::*;
        let ok = matches!(
            (&s.state, &next),
            (Queued, Leasing { .. })
                | (Leasing { .. }, Leasing { .. })
                | (Leasing { .. }, Queued)
                | (Leasing { .. }, Transferring { .. })
                | (Transferring { .. }, Decoding { .. })
                | (Transferring { .. }, Done) // Finished may overtake PrefillDone (different connections)
                | (Decoding { .. }, Done)
                | (_, Failed { .. })
        );
        if !ok {
            return Err(bonsai_core::BonsaiError::Protocol(format!(
                "illegal transition {} -> {} for {id:?}",
                s.state.name(),
                next.name()
            )));
        }
        s.state = next;
        s.state_since = now;
        Ok(s)
    }

    pub fn queued(&self) -> impl Iterator<Item = HandoffId> + '_ {
        let mut v: Vec<_> =
            self.sessions.iter().filter(|(_, s)| s.state == HandoffState::Queued).map(|(id, _)| *id).collect();
        v.sort(); // FIFO by id
        v.into_iter()
    }

    pub fn active_on(&self, worker: WorkerId) -> impl Iterator<Item = HandoffId> + '_ {
        self.sessions
            .iter()
            .filter(move |(_, s)| s.state.workers().is_some_and(|(p, d)| p == worker || d == worker))
            .map(|(id, _)| *id)
    }

    pub fn iter(&self) -> impl Iterator<Item = (&HandoffId, &Session)> {
        self.sessions.iter()
    }

    pub fn finish(&mut self, id: HandoffId) -> Option<Session> {
        let s = self.sessions.remove(&id)?;
        self.by_request.remove(&s.request.id);
        Some(s)
    }

    pub fn report_skeleton(s: &Session) -> HandoffReport {
        HandoffReport {
            request: Some(s.request.id),
            handoff: Some(s.params.handoff),
            tag: s.request.tag.clone(),
            codec: s.params.precision.name(&s.params.layout),
            tokens_per_chunk: s.params.chunk.tokens_per_chunk,
            layers_per_chunk: s.params.chunk.layers_per_chunk,
            seq_len: s.params.seq_len,
            ..Default::default()
        }
    }
}
