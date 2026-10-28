//! Pluggable decisions: where a request runs, and how its KV is encoded.

use bonsai_core::{
    ChunkPlan, ChunkSpec, CodecSpec, KvLayout, PrecisionMap, QuantBits, SenderTimings, WorkerId, WorkerRole,
    msg::InferenceRequest,
};

use crate::registry::WorkerRegistry;

pub trait PlacementPolicy: Send {
    /// Pick (prefill, decode). `slot_bytes` is what the decode worker must
    /// lease; `exclude` lists decode workers that already denied this handoff.
    fn place(
        &mut self,
        req: &InferenceRequest,
        slot_bytes: u64,
        reg: &WorkerRegistry,
        exclude: &[WorkerId],
    ) -> Option<(WorkerId, WorkerId)>;
}

/// Fewest active requests; decode worker must have room in its arena.
#[derive(Default)]
pub struct LeastLoaded;

impl PlacementPolicy for LeastLoaded {
    fn place(&mut self, _req: &InferenceRequest, slot_bytes: u64, reg: &WorkerRegistry, exclude: &[WorkerId]) -> Option<(WorkerId, WorkerId)> {
        let p = reg
            .ready(WorkerRole::Prefill)
            .filter(|w| w.load.active < w.info.max_concurrency)
            .min_by_key(|w| (w.load.active + w.load.queued, w.id))?;
        let d = reg
            .ready(WorkerRole::Decode)
            .filter(|w| !exclude.contains(&w.id))
            .filter(|w| w.load.active < w.info.max_concurrency)
            .filter(|w| w.load.arena_free_bytes.saturating_sub(w.reserved_bytes) >= slot_bytes)
            .min_by_key(|w| (w.load.active, w.id))?;
        Some((p.id, d.id))
    }
}

/// What the control plane knows when choosing a precision for a request.
#[derive(Debug, Clone, Copy, Default)]
pub struct PolicyContext {
    /// Line rate reported by the workers (`aggregated_link_speed`).
    pub nominal_link_bps: u64,
    /// EWMA of achieved goodput from recent handoffs (`None` until the first report).
    /// Lower than nominal under contention or throttling: the signal that should
    /// push precision down.
    pub measured_link_bps: Option<u64>,
    pub inflight_handoffs: u32,
}

impl PolicyContext {
    /// Bandwidth one new handoff can expect: measured if known, otherwise
    /// nominal shared across in-flight handoffs.
    pub fn effective_bps(&self) -> u64 {
        self.measured_link_bps.unwrap_or(self.nominal_link_bps / (self.inflight_handoffs as u64 + 1)).max(1)
    }
}

pub trait CodecPolicy: Send {
    fn choose(&self, req: &InferenceRequest, layout: &KvLayout, ctx: &PolicyContext) -> (PrecisionMap, ChunkSpec);
}

/// One fixed config for every request: what the experiment sweeps use.
pub struct StaticCodecPolicy {
    pub precision: PrecisionMap,
    pub chunk: ChunkSpec,
}

impl CodecPolicy for StaticCodecPolicy {
    fn choose(&self, _req: &InferenceRequest, _layout: &KvLayout, _ctx: &PolicyContext) -> (PrecisionMap, ChunkSpec) {
        (self.precision.clone(), self.chunk)
    }
}

/// Encode/decode throughput for one codec family, over raw BF16 bytes, bytes/s.
#[derive(Debug, Clone, Copy)]
pub struct CodecRates {
    pub encode_bps: f64,
    pub decode_bps: f64,
}

/// First-order model of handoff latency, calibrated from microbenchmarks
/// (`bonsai-bench codec` + `fabric-debug`). Used to predict where compression
/// should win before running the sweep, and to drive [`AdaptivePrecisionPolicy`].
#[derive(Debug, Clone, Copy)]
pub struct CostModel {
    pub bf16: CodecRates,
    pub int8: CodecRates,
    pub int4: CodecRates,
    pub per_write_ns: f64,
    /// Fixed cost per encode or decode kernel launch (one per chunk). Dominates tiny prompts.
    pub per_chunk_kernel_ns: f64,
    /// Achievable fraction of line rate for large writes.
    pub link_efficiency: f64,
}

impl CostModel {
    fn rates(&self, c: CodecSpec) -> CodecRates {
        match c {
            CodecSpec::Bf16 => self.bf16,
            CodecSpec::Quant { bits: QuantBits::Int8, .. } => self.int8,
            CodecSpec::Quant { bits: QuantBits::Int4Packed, .. } => self.int4,
        }
    }

    /// Pipelined estimate:
    /// `max(encode, transfer) + min(encode, transfer) / chunks + decode of the last group`.
    pub fn estimate_ns(&self, plan: &ChunkPlan, link_bps: u64) -> f64 {
        let raw_of = |c: &bonsai_core::ChunkDesc| {
            plan.layout.elems((c.layers.end - c.layers.start) as u64, (c.tokens.end - c.tokens.start) as u64) as f64
                * 2.0
                * plan.layout.dtype.size() as f64
        };
        let encode: f64 = plan
            .chunks
            .iter()
            .map(|c| raw_of(c) / self.rates(c.codec).encode_bps * 1e9 + self.per_chunk_kernel_ns)
            .sum();
        let n = plan.chunks.len().max(1) as f64;
        let transfer = plan.wire_bytes() as f64 * 8.0 / (link_bps as f64 * self.link_efficiency) * 1e9
            + (n + 1.0) * self.per_write_ns;
        let last = plan.num_groups.saturating_sub(1);
        let decode_last: f64 = plan
            .chunks
            .iter()
            .filter(|c| c.group == last)
            .map(|c| raw_of(c) / self.rates(c.codec).decode_bps * 1e9 + self.per_chunk_kernel_ns)
            .sum();
        encode.max(transfer) + encode.min(transfer) / n + decode_last
    }
}

/// A candidate configuration the adaptive policy may choose from.
#[derive(Debug, Clone)]
pub struct Candidate {
    pub precision: PrecisionMap,
    pub chunk: ChunkSpec,
    /// Offline quality loss vs BF16 on the dev split (F1 points), by context-length bucket.
    /// Candidates over the tolerance for a bucket are never chosen for it.
    pub quality_loss_by_ctx: Vec<(u32, f64)>,
}

impl Candidate {
    fn quality_loss(&self, seq_len: u32) -> f64 {
        self.quality_loss_by_ctx.iter().find(|(max_ctx, _)| seq_len <= *max_ctx).map(|(_, l)| *l).unwrap_or(f64::INFINITY)
    }
}

/// Adaptive precision.
///
/// Per request: drop candidates whose offline quality loss at this context length
/// exceeds `max_quality_loss`, then pick the lowest predicted handoff latency at
/// the bandwidth currently available. Effects:
/// * short prompts or idle fast links -> BF16/INT8 (compression can't pay off),
/// * long prompts, contention, or constrained links -> INT4 / mixed INT4+INT8.
///
/// `min_gain` adds hysteresis: a lower-precision candidate must beat the
/// highest-precision admissible one by this fraction, so quality isn't spent
/// on noise-level gains.
pub struct AdaptivePrecisionPolicy {
    pub candidates: Vec<Candidate>,
    pub model: CostModel,
    pub max_quality_loss: f64,
    pub min_gain: f64,
}

impl CodecPolicy for AdaptivePrecisionPolicy {
    fn choose(&self, req: &InferenceRequest, layout: &KvLayout, ctx: &PolicyContext) -> (PrecisionMap, ChunkSpec) {
        let seq = req.prompt_tokens.len() as u32;
        let bps = ctx.effective_bps();
        let mut scored: Vec<(&Candidate, f64, f64)> = self
            .candidates
            .iter()
            .filter(|c| c.quality_loss(seq) <= self.max_quality_loss)
            .filter_map(|c| {
                let plan = ChunkPlan::build(*layout, c.precision.clone(), seq, c.chunk).ok()?;
                Some((c, self.model.estimate_ns(&plan, bps), c.quality_loss(seq)))
            })
            .collect();
        // Baseline = least lossy admissible candidate.
        scored.sort_by(|a, b| a.2.total_cmp(&b.2));
        let Some(&(base, base_ns, _)) = scored.first() else {
            return (CodecSpec::Bf16.into(), ChunkSpec::PER_LAYER);
        };
        let best = scored
            .iter()
            .filter(|(_, ns, _)| *ns <= base_ns * (1.0 - self.min_gain))
            .min_by(|a, b| a.1.total_cmp(&b.1))
            .map(|(c, _, _)| *c)
            .unwrap_or(base);
        (best.precision.clone(), best.chunk)
    }
}

/// EWMA of achieved handoff goodput (wire bytes / sender handoff time).
pub struct BandwidthEstimator {
    alpha: f64,
    bps: Option<f64>,
}

impl BandwidthEstimator {
    pub fn new(alpha: f64) -> Self {
        BandwidthEstimator { alpha, bps: None }
    }

    pub fn observe(&mut self, t: &SenderTimings) {
        let send_ns = t.handoff_send_ns.saturating_sub(t.grant_wait_ns);
        if send_ns == 0 || t.wire_bytes == 0 {
            return;
        }
        let sample = t.wire_bytes as f64 * 8.0 * 1e9 / send_ns as f64;
        self.bps = Some(match self.bps {
            Some(b) => self.alpha * sample + (1.0 - self.alpha) * b,
            None => sample,
        });
    }

    pub fn estimate_bps(&self) -> Option<u64> {
        self.bps.map(|b| b as u64)
    }
}
