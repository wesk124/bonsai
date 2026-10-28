//! Token-bucket egress limiter used to emulate bandwidth-constrained links.
//!
//! With ConnectX-7 / EFA at 100-400 Gb/s, compression rarely pays off; the
//! second hypothesis is about constrained links, so the sender can cap its own
//! rate. Combined with `write_shards = 1` this gives a controllable bottleneck.

use std::time::{Duration, Instant};

use parking_lot::Mutex;

pub struct Throttle {
    bps: u64,
    burst_bytes: u64,
    state: Mutex<(f64, Instant)>,
}

impl Throttle {
    pub fn new(bps: u64, burst_bytes: u64) -> Self {
        Throttle { bps, burst_bytes, state: Mutex::new((burst_bytes as f64, Instant::now())) }
    }

    /// Wait until `bytes` may be sent.
    pub async fn acquire(&self, bytes: u64) {
        let delay = {
            let mut s = self.state.lock();
            let now = Instant::now();
            let refill = now.duration_since(s.1).as_secs_f64() * self.bps as f64 / 8.0;
            s.0 = (s.0 + refill).min(self.burst_bytes as f64) - bytes as f64;
            s.1 = now;
            if s.0 >= 0.0 { Duration::ZERO } else { Duration::from_secs_f64(-s.0 * 8.0 / self.bps as f64) }
        };
        if !delay.is_zero() {
            tokio::time::sleep(delay).await;
        }
    }
}
