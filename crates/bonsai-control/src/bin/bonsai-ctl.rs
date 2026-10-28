//! `bonsai-ctl` — the Bonsai coordinator.
//!
//! ```text
//! bonsai-ctl --listen 0.0.0.0:7000 --codec int8 --chunk-tokens 1024 \
//!            --chunk-layers 1 --metrics results/run.jsonl
//! ```

fn main() {
    // Parse args -> CoordinatorConfig + StaticCodecPolicy + JsonlSink,
    // build Coordinator, run server::serve on a multi-threaded tokio runtime.
    todo!("bonsai-ctl")
}
