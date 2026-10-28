//! Sans-IO coordinator: `(state, message) -> (state', messages to send)`.

use std::time::{Duration, Instant};

use bonsai_core::{
    CodecSpec, HandoffId, HandoffReport, KvLayout, WorkerId,
    msg::{FromCoordinator, HandoffParams, InferenceRequest, ToCoordinator},
};
use tracing::warn;

use crate::{
    policy::{BandwidthEstimator, CodecPolicy, PlacementPolicy, PolicyContext},
    registry::{WorkerRegistry, WorkerState},
    session::{HandoffState, SessionManager},
    sink::MetricsSink,
};

/// One TCP connection (worker or client), assigned by the server.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct ConnId(pub u64);

#[derive(Debug)]
pub struct Outbound {
    pub to: ConnId,
    pub msg: FromCoordinator,
}

#[derive(Debug, Clone)]
pub struct CoordinatorConfig {
    /// All workers must serve this model/layout (checked at registration).
    pub layout: KvLayout,
    pub model: String,
    pub heartbeat_timeout: Duration,
    /// Max time in any non-terminal state except `Queued`/`Decoding`.
    pub handoff_timeout: Duration,
    pub decode_timeout: Duration,
    /// Decode workers to try per handoff before requeueing.
    pub max_lease_attempts: u32,
    /// On failure of a quantized handoff, retry once with BF16.
    pub fallback_to_bf16: bool,
    pub max_retries: u32,
    pub max_queue: usize,
}

pub struct Coordinator {
    cfg: CoordinatorConfig,
    registry: WorkerRegistry,
    sessions: SessionManager,
    placement: Box<dyn PlacementPolicy>,
    codec_policy: Box<dyn CodecPolicy>,
    sink: Box<dyn MetricsSink>,
    /// Decode workers that denied each leasing handoff.
    denied: std::collections::HashMap<HandoffId, Vec<WorkerId>>,
    /// Achieved goodput from recent handoffs; feeds adaptive precision.
    bandwidth: BandwidthEstimator,
}

impl Coordinator {
    pub fn new(
        cfg: CoordinatorConfig,
        placement: Box<dyn PlacementPolicy>,
        codec_policy: Box<dyn CodecPolicy>,
        sink: Box<dyn MetricsSink>,
    ) -> Self {
        Coordinator {
            cfg,
            registry: WorkerRegistry::default(),
            sessions: SessionManager::default(),
            placement,
            codec_policy,
            sink,
            denied: Default::default(),
            bandwidth: BandwidthEstimator::new(0.2),
        }
    }

    pub fn registry(&self) -> &WorkerRegistry {
        &self.registry
    }

    pub fn sessions(&self) -> &SessionManager {
        &self.sessions
    }

    pub fn on_message(&mut self, from: ConnId, msg: ToCoordinator, now: Instant) -> Vec<Outbound> {
        let mut out = Vec::new();
        match msg {
            ToCoordinator::Register(info) => {
                if info.layout != self.cfg.layout {
                    warn!(?info.layout, "worker layout mismatch; rejecting");
                    out.push(Outbound { to: from, msg: FromCoordinator::Shutdown });
                } else {
                    let id = self.registry.register(from, info, now);
                    out.push(Outbound { to: from, msg: FromCoordinator::Registered { worker: id } });
                }
            }
            ToCoordinator::Heartbeat { worker, load } => {
                self.registry.heartbeat(worker, load, now);
            }
            ToCoordinator::Deregister { worker } => self.worker_lost(worker, "deregistered", now, &mut out),
            ToCoordinator::Submit(req) => self.submit(from, req, now, &mut out),
            ToCoordinator::SlotGranted(grant) => {
                let id = grant.handoff;
                let Some((p, d)) = self.sessions.get(id).and_then(|s| s.state.workers()) else {
                    return out; // stale (already failed); decode worker will time out the lease
                };
                if let Ok(s) = self.sessions.transition(id, HandoffState::Transferring { prefill: p, decode: d }, now) {
                    s.grant = Some(grant.clone());
                    if let Some(pc) = self.registry.get(p).map(|w| w.conn) {
                        out.push(Outbound { to: pc, msg: FromCoordinator::SlotReady(grant) });
                    }
                }
            }
            ToCoordinator::SlotDenied { handoff, reason } => self.slot_denied(handoff, &reason, now, &mut out),
            ToCoordinator::PrefillDone { handoff, first_token: _, timings: _ } => {
                if let Some((p, d)) = self.sessions.get(handoff).and_then(|s| s.state.workers())
                    && let Ok(s) = self.sessions.transition(handoff, HandoffState::Decoding { prefill: p, decode: d }, now)
                {
                    s.first_token_at = Some(now);
                }
            }
            ToCoordinator::Finished { handoff, output_tokens, report } => {
                self.finished(handoff, output_tokens, report, now, &mut out)
            }
            ToCoordinator::HandoffFailed { handoff, reason } => self.fail(handoff, reason, now, &mut out),
        }
        self.schedule(now, &mut out);
        out
    }

    /// Call periodically (e.g. every 50 ms): timeouts and heartbeats.
    pub fn on_tick(&mut self, now: Instant) -> Vec<Outbound> {
        let mut out = Vec::new();
        let expired: Vec<_> = self
            .sessions
            .iter()
            .filter(|(_, s)| {
                let limit = match s.state {
                    HandoffState::Leasing { .. } | HandoffState::Transferring { .. } => self.cfg.handoff_timeout,
                    HandoffState::Decoding { .. } => self.cfg.decode_timeout,
                    _ => return false,
                };
                now.duration_since(s.state_since) > limit
            })
            .map(|(id, s)| (*id, s.state.name()))
            .collect();
        for (id, state) in expired {
            self.fail(id, format!("timeout in {state}"), now, &mut out);
        }
        let mut lost = Vec::new();
        for w in self.registry.iter_mut() {
            let idle = now.duration_since(w.last_seen);
            if idle > 3 * self.cfg.heartbeat_timeout {
                lost.push(w.id);
            } else if idle > self.cfg.heartbeat_timeout {
                w.state = WorkerState::Suspect;
            }
        }
        for id in lost {
            self.worker_lost(id, "heartbeat timeout", now, &mut out);
        }
        self.schedule(now, &mut out);
        out
    }

    pub fn on_disconnect(&mut self, conn: ConnId, now: Instant) -> Vec<Outbound> {
        let mut out = Vec::new();
        if let Some(id) = self.registry.by_conn(conn) {
            self.worker_lost(id, "disconnected", now, &mut out);
        }
        // A disconnected client's requests still run; their Completed is dropped by the server.
        self.schedule(now, &mut out);
        out
    }

    fn submit(&mut self, from: ConnId, req: InferenceRequest, now: Instant, out: &mut Vec<Outbound>) {
        let reject = |reason: String| Outbound { to: from, msg: FromCoordinator::Rejected { request: req.id, reason } };
        if req.prompt_tokens.is_empty() {
            out.push(reject("empty prompt".into()));
            return;
        }
        if self.sessions.queued().count() >= self.cfg.max_queue {
            out.push(reject("queue full".into()));
            return;
        }
        let ctx = PolicyContext {
            nominal_link_bps: self.registry.ready(bonsai_core::WorkerRole::Prefill).map(|w| w.info.link_bps).min().unwrap_or(0),
            measured_link_bps: self.bandwidth.estimate_bps(),
            inflight_handoffs: self.sessions.iter().filter(|(_, s)| s.state.workers().is_some()).count() as u32,
        };
        let (dc, dk) = self.codec_policy.choose(&req, &self.cfg.layout, &ctx);
        let params = HandoffParams {
            handoff: HandoffId(u64::MAX), // assigned by SessionManager::create
            request: req.id,
            layout: self.cfg.layout,
            precision: req.precision.clone().unwrap_or(dc),
            seq_len: req.prompt_tokens.len() as u32,
            chunk: req.chunk.unwrap_or(dk),
        };
        if let Err(e) = params.plan() {
            out.push(reject(e.to_string()));
            return;
        }
        let request = req.id;
        let id = self.sessions.create(req, params, from, now);
        out.push(Outbound { to: from, msg: FromCoordinator::Accepted { request, handoff: id } });
    }

    /// FIFO admission: place queued handoffs until one doesn't fit.
    fn schedule(&mut self, now: Instant, out: &mut Vec<Outbound>) {
        let queued: Vec<_> = self.sessions.queued().collect();
        for id in queued {
            let Some(s) = self.sessions.get(id) else { continue };
            let Ok(plan) = s.params.plan() else { continue };
            let exclude = self.denied.get(&id).cloned().unwrap_or_default();
            let Some((p, d)) = self.placement.place(&s.request, plan.slot_bytes, &self.registry, &exclude) else {
                break; // head-of-line: keep FIFO order under memory pressure
            };
            let (params, request) = (s.params.clone(), s.request.clone());
            if self.sessions.transition(id, HandoffState::Leasing { prefill: p, decode: d, attempts: 1 }, now).is_err() {
                continue;
            }
            if let Some(w) = self.registry.get_mut(d) {
                w.reserved_bytes += plan.slot_bytes;
            }
            let conn = |reg: &WorkerRegistry, w| reg.get(w).map(|e| e.conn);
            if let (Some(pc), Some(dc)) = (conn(&self.registry, p), conn(&self.registry, d)) {
                // Sent together: leasing overlaps prefill.
                out.push(Outbound { to: dc, msg: FromCoordinator::LeaseSlot { params: params.clone(), request: request.clone() } });
                out.push(Outbound { to: pc, msg: FromCoordinator::Prefill { params, request } });
            }
        }
    }

    fn slot_denied(&mut self, id: HandoffId, reason: &str, now: Instant, out: &mut Vec<Outbound>) {
        let Some(s) = self.sessions.get(id) else { return };
        let HandoffState::Leasing { prefill, decode, attempts } = s.state else { return };
        let Ok(plan) = s.params.plan() else { return };
        let (req, params) = (s.request.clone(), s.params.clone());
        self.release_reservation(decode, plan.slot_bytes);
        let denied = self.denied.entry(id).or_default();
        denied.push(decode);
        let exclude = denied.clone();
        // Try another decode worker while the prefill worker keeps its KV.
        if attempts < self.cfg.max_lease_attempts
            && let Some((_, d2)) = self.placement.place(&req, plan.slot_bytes, &self.registry, &exclude)
            && self.sessions.transition(id, HandoffState::Leasing { prefill, decode: d2, attempts: attempts + 1 }, now).is_ok()
        {
            if let Some(w) = self.registry.get_mut(d2) {
                w.reserved_bytes += plan.slot_bytes;
                out.push(Outbound { to: w.conn, msg: FromCoordinator::LeaseSlot { params, request: req } });
            }
            return;
        }
        warn!(?id, reason, "no decode worker could lease a slot; requeueing");
        if let Some(pc) = self.registry.get(prefill).map(|w| w.conn) {
            out.push(Outbound { to: pc, msg: FromCoordinator::Abort { handoff: id, reason: "requeued".into() } });
        }
        let _ = self.sessions.transition(id, HandoffState::Queued, now);
    }

    fn finished(&mut self, id: HandoffId, output: Vec<u32>, partial: HandoffReport, now: Instant, out: &mut Vec<Outbound>) {
        if self.sessions.transition(id, HandoffState::Done, now).is_err() {
            return;
        }
        let Some(s) = self.sessions.finish(id) else { return };
        self.denied.remove(&id);
        if let Some((_, d)) = s.state.workers().or(s.grant.as_ref().map(|g| (g.decode_worker, g.decode_worker)))
            && let Ok(plan) = s.params.plan()
        {
            self.release_reservation(d, plan.slot_bytes);
        }
        self.bandwidth.observe(&partial.sender);
        let mut report = SessionManager::report_skeleton(&s);
        report.sender = partial.sender;
        report.receiver = partial.receiver;
        report.concurrency = self.sessions.iter().count() as u32 + 1;
        report.ttft_ns = s.first_token_at.map(|t| (t - s.submitted_at).as_nanos() as u64).unwrap_or(0);
        report.ttfdt_ns = partial.ttfdt_ns; // measured by the client; see docs/HLD.md §7
        report.e2e_ns = (now - s.submitted_at).as_nanos() as u64;
        self.sink.record(&report);
        out.push(Outbound {
            to: s.client,
            msg: FromCoordinator::Completed { request: s.request.id, output_tokens: output, report },
        });
    }

    fn fail(&mut self, id: HandoffId, reason: String, now: Instant, out: &mut Vec<Outbound>) {
        let Some(s) = self.sessions.get(id) else { return };
        let workers = s.state.workers();
        let slot_bytes = s.params.plan().map(|p| p.slot_bytes).unwrap_or(0);
        if let Some((p, d)) = workers {
            for w in [p, d] {
                if let Some(c) = self.registry.get(w).map(|e| e.conn) {
                    out.push(Outbound { to: c, msg: FromCoordinator::Abort { handoff: id, reason: reason.clone() } });
                }
            }
            self.release_reservation(d, slot_bytes);
        }
        let _ = self.sessions.transition(id, HandoffState::Failed { reason: reason.clone() }, now);
        let Some(s) = self.sessions.finish(id) else { return };
        self.denied.remove(&id);

        let can_fallback =
            self.cfg.fallback_to_bf16 && !s.params.precision.is_uniform(CodecSpec::Bf16) && s.retries < self.cfg.max_retries;
        if can_fallback {
            // New HandoffId => new slot/epoch, so stale writes can't be miscounted.
            let mut req = s.request;
            req.precision = Some(CodecSpec::Bf16.into());
            let params = HandoffParams { precision: CodecSpec::Bf16.into(), ..s.params };
            let nid = self.sessions.create(req, params, s.client, now);
            if let Some(ns) = self.sessions.get_mut(nid) {
                ns.retries = s.retries + 1;
                ns.submitted_at = s.submitted_at;
            }
        } else {
            out.push(Outbound { to: s.client, msg: FromCoordinator::Rejected { request: s.request.id, reason } });
        }
    }

    fn worker_lost(&mut self, worker: WorkerId, why: &str, now: Instant, out: &mut Vec<Outbound>) {
        let affected: Vec<_> = self.sessions.active_on(worker).collect();
        for id in affected {
            self.fail(id, format!("worker {worker:?} {why}"), now, out);
        }
        self.registry.remove(worker);
    }

    fn release_reservation(&mut self, decode: WorkerId, bytes: u64) {
        if let Some(w) = self.registry.get_mut(decode) {
            w.reserved_bytes = w.reserved_bytes.saturating_sub(bytes);
        }
    }
}
