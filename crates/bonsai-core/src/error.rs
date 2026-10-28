use thiserror::Error;

use crate::ids::{HandoffId, WorkerId};

pub type Result<T> = std::result::Result<T, BonsaiError>;

#[derive(Debug, Error)]
pub enum BonsaiError {
    #[error("invalid configuration: {0}")]
    InvalidConfig(String),
    #[error("receive arena exhausted: requested {requested} bytes, {available} available")]
    ArenaExhausted { requested: u64, available: u64 },
    #[error("staging ring exhausted")]
    StagingExhausted,
    #[error("unknown handoff {0:?}")]
    UnknownHandoff(HandoffId),
    #[error("unknown worker {0:?}")]
    UnknownWorker(WorkerId),
    #[error("handoff {0:?} timed out in state {1}")]
    Timeout(HandoffId, &'static str),
    #[error("handoff {0:?} aborted: {1}")]
    Aborted(HandoffId, String),
    #[error("transport error: {0}")]
    Transport(String),
    #[error("codec error: {0}")]
    Codec(String),
    #[error("cuda error: {0}")]
    Cuda(String),
    #[error("protocol error: {0}")]
    Protocol(String),
    #[error(transparent)]
    Io(#[from] std::io::Error),
    #[error(transparent)]
    Serde(#[from] postcard::Error),
}
