//! tokio TCP front end for the sans-IO [`Coordinator`].
//!
//! ```text
//! accept loop ──spawn──▶ conn reader task ──(ConnId, ToCoordinator)──▶ ┐
//!                        conn writer task ◀──FromCoordinator── mpsc ◀── coordinator task
//!                                                         tick (50 ms) ┘   (owns Coordinator)
//! ```
//! One task owns the coordinator, so there are no locks around protocol state.

use std::{net::SocketAddr, time::Duration};

use bonsai_core::Result;

use crate::coordinator::Coordinator;

#[derive(Debug, Clone)]
pub struct ServerConfig {
    pub listen: SocketAddr,
    pub tick: Duration,
}

pub async fn serve(_cfg: ServerConfig, _coordinator: Coordinator) -> Result<()> {
    todo!("accept loop, per-conn framed reader/writer tasks, single coordinator task")
}
