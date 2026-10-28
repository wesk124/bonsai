//! Bonsai control plane.
//!
//! The coordinator is the only stateful global component. It
//! * keeps the registry of prefill/decode workers and their load,
//! * admits requests, picks a (prefill, decode) pair and a codec/chunking,
//! * runs each handoff's state machine (lease -> transfer -> decode -> done),
//! * handles timeouts and failures (abort, quarantine, BF16 fallback),
//! * collects `HandoffReport`s into a metrics sink for the evaluation.
//!
//! It never sees KV bytes. Its messages are off the critical path except for
//! one: forwarding the `SlotGrant`, which overlaps prefill compute.
//!
//! [`coordinator::Coordinator`] is sans-IO: it consumes messages and returns
//! messages to send, so every protocol path can be unit-tested without
//! sockets. [`server`] wraps it with tokio TCP.

pub mod client;
pub mod config;
pub mod coordinator;
pub mod policy;
pub mod registry;
pub mod session;
pub mod sink;
pub mod server;

pub use coordinator::{ConnId, Coordinator, CoordinatorConfig, Outbound};
