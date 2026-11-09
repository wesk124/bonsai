use bonsai_control::{
    Coordinator, CoordinatorConfig,
    client::ControlClient,
    policy::{LeastLoaded, StaticCodecPolicy},
    server::serve_listener,
    sink::MemorySink,
};
use bonsai_core::{
    ChunkSpec, CodecSpec, HandoffReport, KvLayout, RequestId, SlotId, WorkerRole,
    msg::{
        FabricAddress, FromCoordinator, InferenceRequest, RemoteRegion, SamplingParams, SlotGrant,
        ToCoordinator, WorkerInfo, framing,
    },
};
use std::time::Duration;
use tokio::{
    io::{AsyncReadExt, AsyncWriteExt},
    net::{TcpListener, TcpStream},
    sync::oneshot,
    task::JoinHandle,
    time::timeout,
};

const L: KvLayout = KvLayout::QWEN3_4B_INSTRUCT_2507;
const MODEL: &str = "Qwen/Qwen3-4B-Instruct-2507";

fn coordinator() -> Coordinator {
    Coordinator::new(
        CoordinatorConfig {
            layout: L,
            model: MODEL.into(),
            heartbeat_timeout: Duration::from_secs(60),
            handoff_timeout: Duration::from_secs(60),
            decode_timeout: Duration::from_secs(60),
            max_lease_attempts: 2,
            fallback_to_bf16: false,
            max_retries: 0,
            max_queue: 8,
            bandwidth_ewma_alpha: 0.2,
        },
        Box::new(LeastLoaded),
        Box::new(StaticCodecPolicy {
            precision: CodecSpec::Bf16.into(),
            chunk: ChunkSpec::PER_LAYER,
        }),
        Box::new(MemorySink::default()),
    )
}

fn worker(role: WorkerRole) -> WorkerInfo {
    WorkerInfo {
        role,
        hostname: "test".into(),
        cuda_device: 0,
        fabric_addr: FabricAddress(vec![1].into()),
        nets_per_gpu: 1,
        link_bps: 100_000_000_000,
        model: MODEL.into(),
        layout: L,
        arena_bytes: 1 << 30,
        max_concurrency: 2,
    }
}
fn request(id: u64) -> ToCoordinator {
    ToCoordinator::Submit(InferenceRequest {
        id: RequestId(id),
        prompt_tokens: vec![1; 64],
        max_new_tokens: 2,
        stop_tokens: vec![],
        sampling: SamplingParams::default(),
        precision: None,
        chunk: None,
        tag: "tcp".into(),
    })
}
async fn recv(client: &mut ControlClient) -> FromCoordinator {
    timeout(Duration::from_secs(3), client.recv())
        .await
        .expect("message deadline")
        .expect("valid frame")
}
async fn start() -> (
    std::net::SocketAddr,
    oneshot::Sender<()>,
    JoinHandle<bonsai_core::Result<()>>,
) {
    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap();
    let (stop, stopped) = oneshot::channel();
    let task = tokio::spawn(serve_listener(
        listener,
        Duration::from_millis(10),
        coordinator(),
        async {
            let _ = stopped.await;
        },
    ));
    (addr, stop, task)
}
async fn stop(stop: oneshot::Sender<()>, task: JoinHandle<bonsai_core::Result<()>>) {
    stop.send(()).unwrap();
    timeout(Duration::from_secs(3), task)
        .await
        .unwrap()
        .unwrap()
        .unwrap();
}

#[tokio::test]
async fn actual_socket_handoff_and_split_reports() {
    let (addr, shutdown, server) = start().await;
    let mut prefill = ControlClient::connect(addr).await.unwrap();
    let mut decode = ControlClient::connect(addr).await.unwrap();
    let mut client = ControlClient::connect(addr).await.unwrap();
    prefill
        .send(&ToCoordinator::Register(worker(WorkerRole::Prefill)))
        .await
        .unwrap();
    assert!(matches!(
        recv(&mut prefill).await,
        FromCoordinator::Registered { .. }
    ));
    decode
        .send(&ToCoordinator::Register(worker(WorkerRole::Decode)))
        .await
        .unwrap();
    let FromCoordinator::Registered { worker: decode_id } = recv(&mut decode).await else {
        panic!("registration")
    };
    client.send(&request(1)).await.unwrap();
    let FromCoordinator::Accepted { handoff, .. } = recv(&mut client).await else {
        panic!("admission")
    };
    let FromCoordinator::LeaseSlot { params, .. } = recv(&mut decode).await else {
        panic!("lease")
    };
    assert_eq!(params.layout, L);
    assert!(matches!(
        recv(&mut prefill).await,
        FromCoordinator::Prefill { .. }
    ));
    decode
        .send(&ToCoordinator::SlotGranted(SlotGrant {
            handoff,
            decode_worker: decode_id,
            decode_addr: FabricAddress(vec![9].into()),
            slot: SlotId { index: 1, epoch: 2 },
            region: RemoteRegion(vec![0].into()),
            base_offset: 0,
            len: params.plan().unwrap().slot_bytes,
        }))
        .await
        .unwrap();
    assert!(matches!(
        recv(&mut prefill).await,
        FromCoordinator::SlotReady(_)
    ));
    let (mut sender, mut receiver) = prefill.split();
    sender
        .send(&ToCoordinator::PrefillDone {
            handoff,
            first_token: 7,
            timings: Default::default(),
        })
        .await
        .unwrap();
    decode
        .send(&ToCoordinator::Finished {
            handoff,
            output_tokens: vec![7, 8],
            report: HandoffReport {
                measured_handoff_ack_ns: Some(123),
                ..Default::default()
            },
        })
        .await
        .unwrap();
    // EOF must not overtake the report already queued by this connection.
    drop(decode);
    let FromCoordinator::Completed {
        request,
        output_tokens,
        report,
    } = recv(&mut client).await
    else {
        panic!("completion")
    };
    assert_eq!(request, RequestId(1));
    assert_eq!(output_tokens, [7, 8]);
    assert_eq!(report.tag, "tcp");
    assert_eq!(report.seq_len, 64);
    assert_eq!(report.handoff_latency_ns(), Some(123));
    stop(shutdown, server).await;
    assert!(
        timeout(Duration::from_secs(1), receiver.recv())
            .await
            .unwrap()
            .is_err()
    );
}

#[tokio::test]
async fn worker_disconnect_aborts_peer_and_rejects_request() {
    let (addr, shutdown, server) = start().await;
    let mut p = ControlClient::connect(addr).await.unwrap();
    let mut d = ControlClient::connect(addr).await.unwrap();
    let mut c = ControlClient::connect(addr).await.unwrap();
    for (client, role) in [(&mut p, WorkerRole::Prefill), (&mut d, WorkerRole::Decode)] {
        client
            .send(&ToCoordinator::Register(worker(role)))
            .await
            .unwrap();
        recv(client).await;
    }
    c.send(&request(1)).await.unwrap();
    recv(&mut c).await;
    recv(&mut p).await;
    recv(&mut d).await;
    drop(d);
    assert!(matches!(recv(&mut p).await, FromCoordinator::Abort { .. }));
    let FromCoordinator::Rejected { reason, .. } = recv(&mut c).await else {
        panic!("reject")
    };
    assert!(reason.contains("disconnected"));
    stop(shutdown, server).await;
}

#[tokio::test]
async fn client_disconnect_cancels_work_and_keeps_workers_usable() {
    let (addr, shutdown, server) = start().await;
    let mut p = ControlClient::connect(addr).await.unwrap();
    let mut d = ControlClient::connect(addr).await.unwrap();
    let mut c = ControlClient::connect(addr).await.unwrap();
    for (client, role) in [(&mut p, WorkerRole::Prefill), (&mut d, WorkerRole::Decode)] {
        client
            .send(&ToCoordinator::Register(worker(role)))
            .await
            .unwrap();
        recv(client).await;
    }
    c.send(&request(1)).await.unwrap();
    recv(&mut c).await;
    recv(&mut p).await;
    recv(&mut d).await;
    drop(c);
    assert!(matches!(recv(&mut p).await, FromCoordinator::Abort { .. }));
    assert!(matches!(recv(&mut d).await, FromCoordinator::Abort { .. }));
    let mut replacement = ControlClient::connect(addr).await.unwrap();
    replacement.send(&request(2)).await.unwrap();
    assert!(matches!(
        recv(&mut replacement).await,
        FromCoordinator::Accepted { .. }
    ));
    assert!(matches!(
        recv(&mut d).await,
        FromCoordinator::LeaseSlot { .. }
    ));
    stop(shutdown, server).await;
}

#[tokio::test]
async fn rejects_bad_frames_and_identity_without_stopping_server() {
    let (addr, shutdown, server) = start().await;
    for prefix in [0u32, framing::MAX_FRAME as u32 + 1, u32::MAX] {
        let mut bad = TcpStream::connect(addr).await.unwrap();
        bad.write_all(&prefix.to_le_bytes()).await.unwrap();
        assert_eq!(
            timeout(Duration::from_secs(1), bad.read(&mut [0; 1]))
                .await
                .unwrap()
                .unwrap(),
            0
        );
    }
    for change_layout in [false, true] {
        let mut bad = ControlClient::connect(addr).await.unwrap();
        let mut info = worker(WorkerRole::Decode);
        if change_layout {
            info.layout.num_layers -= 1;
        } else {
            info.model = "other model with identical layout".into();
        }
        bad.send(&ToCoordinator::Register(info)).await.unwrap();
        assert!(matches!(recv(&mut bad).await, FromCoordinator::Shutdown));
    }
    // TCP segmentation may split both the prefix and the postcard body.
    let mut fragmented = TcpStream::connect(addr).await.unwrap();
    for byte in framing::encode(&ToCoordinator::Register(worker(WorkerRole::Decode))).unwrap() {
        fragmented.write_all(&[byte]).await.unwrap();
    }
    let len = fragmented.read_u32_le().await.unwrap();
    let mut body = vec![0; len as usize];
    fragmented.read_exact(&mut body).await.unwrap();
    assert!(matches!(
        framing::decode::<FromCoordinator>(&body).unwrap(),
        FromCoordinator::Registered { .. }
    ));
    stop(shutdown, server).await;
    assert_eq!(fragmented.read(&mut [0; 1]).await.unwrap(), 0);
}
