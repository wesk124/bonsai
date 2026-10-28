use std::time::{Duration, Instant};

use bonsai_control::{
    ConnId, Coordinator, CoordinatorConfig, Outbound,
    policy::{AdaptivePrecisionPolicy, Candidate, CodecPolicy, CodecRates, CostModel, LeastLoaded, PolicyContext, StaticCodecPolicy},
    sink::MemorySink,
};
use bonsai_core::{
    ChunkSpec, CodecSpec, HandoffReport, KvLayout, PrecisionMap, RequestId, SlotId, WorkerId, WorkerRole,
    msg::{FabricAddress, FromCoordinator, InferenceRequest, RemoteRegion, SamplingParams, SlotGrant, ToCoordinator, WorkerInfo},
};

const L: KvLayout = KvLayout::QWEN2_5_1_5B;
const P: ConnId = ConnId(1);
const D: ConnId = ConnId(2);
const C: ConnId = ConnId(3);

fn worker(role: WorkerRole) -> WorkerInfo {
    WorkerInfo {
        role,
        hostname: "node".into(),
        cuda_device: 0,
        fabric_addr: FabricAddress(vec![1, 2, 3].into()),
        nets_per_gpu: 1,
        link_bps: 400_000_000_000,
        model: "Qwen2.5-1.5B-Instruct".into(),
        layout: L,
        arena_bytes: 4 << 30,
        max_concurrency: 8,
    }
}

fn request(id: u64, len: usize) -> InferenceRequest {
    InferenceRequest {
        id: RequestId(id),
        prompt_tokens: vec![7; len],
        max_new_tokens: 32,
        stop_tokens: vec![],
        sampling: SamplingParams::default(),
        precision: None,
        chunk: None,
        tag: "test".into(),
    }
}

fn coordinator(fallback: bool) -> Coordinator {
    Coordinator::new(
        CoordinatorConfig {
            layout: L,
            model: "Qwen2.5-1.5B-Instruct".into(),
            heartbeat_timeout: Duration::from_secs(1),
            handoff_timeout: Duration::from_secs(5),
            decode_timeout: Duration::from_secs(60),
            max_lease_attempts: 2,
            fallback_to_bf16: fallback,
            max_retries: 1,
            max_queue: 128,
            bandwidth_ewma_alpha: 0.2,
        },
        Box::new(LeastLoaded),
        Box::new(StaticCodecPolicy { precision: CodecSpec::int8_default(&L).into(), chunk: ChunkSpec::PER_LAYER }),
        Box::new(MemorySink::default()),
    )
}

fn find<'a>(out: &'a [Outbound], to: ConnId, pred: impl Fn(&FromCoordinator) -> bool) -> &'a FromCoordinator {
    &out.iter().find(|o| o.to == to && pred(&o.msg)).unwrap_or_else(|| panic!("no matching message to {to:?}: {out:#?}")).msg
}

fn setup(c: &mut Coordinator, now: Instant) -> (WorkerId, WorkerId) {
    let ids: Vec<_> = [(P, WorkerRole::Prefill), (D, WorkerRole::Decode)]
        .into_iter()
        .map(|(conn, role)| match &c.on_message(conn, ToCoordinator::Register(worker(role)), now)[0].msg {
            FromCoordinator::Registered { worker } => *worker,
            m => panic!("{m:?}"),
        })
        .collect();
    (ids[0], ids[1])
}

#[test]
fn happy_path() {
    let now = Instant::now();
    let mut c = coordinator(false);
    let (_p, d) = setup(&mut c, now);

    let out = c.on_message(C, ToCoordinator::Submit(request(1, 4096)), now);
    find(&out, C, |m| matches!(m, FromCoordinator::Accepted { .. }));
    let FromCoordinator::LeaseSlot { params, .. } = find(&out, D, |m| matches!(m, FromCoordinator::LeaseSlot { .. })) else { unreachable!() };
    find(&out, P, |m| matches!(m, FromCoordinator::Prefill { .. }));
    let plan = params.plan().unwrap();

    let grant = SlotGrant {
        handoff: params.handoff,
        decode_worker: d,
        decode_addr: FabricAddress(vec![9].into()),
        slot: SlotId { index: 0, epoch: 1 },
        region: RemoteRegion(vec![0].into()),
        base_offset: 0,
        len: plan.slot_bytes,
    };
    let out = c.on_message(D, ToCoordinator::SlotGranted(grant), now);
    find(&out, P, |m| matches!(m, FromCoordinator::SlotReady(_)));

    c.on_message(P, ToCoordinator::PrefillDone { handoff: params.handoff, first_token: 42, timings: Default::default() }, now);
    let out = c.on_message(
        D,
        ToCoordinator::Finished { handoff: params.handoff, output_tokens: vec![1, 2, 3], report: HandoffReport::default() },
        now + Duration::from_millis(5),
    );
    let FromCoordinator::Completed { report, .. } = find(&out, C, |m| matches!(m, FromCoordinator::Completed { .. })) else { unreachable!() };
    assert!(report.codec.starts_with("int8"));
    assert_eq!(c.sessions().iter().count(), 0);
}

#[test]
fn failure_falls_back_to_bf16() {
    let now = Instant::now();
    let mut c = coordinator(true);
    setup(&mut c, now);
    let out = c.on_message(C, ToCoordinator::Submit(request(1, 1024)), now);
    let FromCoordinator::Prefill { params, .. } = find(&out, P, |m| matches!(m, FromCoordinator::Prefill { .. })) else { unreachable!() };

    let out = c.on_message(P, ToCoordinator::HandoffFailed { handoff: params.handoff, reason: "boom".into() }, now);
    find(&out, D, |m| matches!(m, FromCoordinator::Abort { .. }));
    // Rescheduled immediately as a new BF16 handoff.
    let FromCoordinator::Prefill { params: retry, .. } = find(&out, P, |m| matches!(m, FromCoordinator::Prefill { .. })) else { unreachable!() };
    assert_ne!(retry.handoff, params.handoff);
    assert!(retry.precision.is_uniform(CodecSpec::Bf16));
}

#[test]
fn adaptive_precision_tracks_context_and_bandwidth() {
    let int8 = CodecSpec::int8_default(&L);
    let int4 = CodecSpec::int4_default();
    let chunk = ChunkSpec { tokens_per_chunk: 0, layers_per_chunk: 4 };
    let ok = vec![(u32::MAX, 0.0)];
    let policy = AdaptivePrecisionPolicy {
        candidates: vec![
            Candidate { precision: CodecSpec::Bf16.into(), chunk, quality_loss_by_ctx: ok.clone() },
            Candidate { precision: int8.into(), chunk, quality_loss_by_ctx: vec![(u32::MAX, 0.2)] },
            Candidate {
                precision: PrecisionMap::protect_layers(int4, int8, &[0..4, 24..28]),
                chunk,
                quality_loss_by_ctx: vec![(u32::MAX, 0.8)],
            },
        ],
        model: CostModel {
            bf16: CodecRates { encode_bps: 400e9, decode_bps: 400e9 },
            int8: CodecRates { encode_bps: 200e9, decode_bps: 250e9 },
            int4: CodecRates { encode_bps: 150e9, decode_bps: 200e9 },
            per_write_ns: 5_000.0,
            per_chunk_kernel_ns: 5_000.0,
            link_efficiency: 0.9,
        },
        max_quality_loss: 1.0,
        min_gain: 0.1,
    };
    let fast = PolicyContext { nominal_link_bps: 400_000_000_000, measured_link_bps: None, inflight_handoffs: 0 };
    let slow = PolicyContext { measured_link_bps: Some(10_000_000_000), ..fast };

    // A handful of tokens: fixed per-write / per-launch costs dominate, so the
    // byte savings can't clear the 10% hysteresis.
    let (short_fast, _) = policy.choose(&request(1, 8), &L, &fast);
    let (long_slow, _) = policy.choose(&request(2, 32 * 1024), &L, &slow);
    assert!(short_fast.is_uniform(CodecSpec::Bf16), "short prompt on a fast link should stay BF16: {short_fast:?}");
    assert_eq!(long_slow.default, int4, "long prompt on a slow link should go to mixed INT4");
}
