//! Control connection used by workers (via `bonsai-py`) and by the load generator.

use std::net::SocketAddr;

use bonsai_core::{
    Result,
    msg::{FromCoordinator, ToCoordinator},
};
use tokio::net::TcpStream;

pub struct ControlClient {
    _stream: TcpStream,
}

impl ControlClient {
    pub async fn connect(addr: SocketAddr) -> Result<Self> {
        Ok(ControlClient { _stream: TcpStream::connect(addr).await? })
    }

    pub async fn send(&mut self, _msg: &ToCoordinator) -> Result<()> {
        todo!("framing::encode + write_all")
    }

    pub async fn recv(&mut self) -> Result<FromCoordinator> {
        todo!("read u32 length, read body, framing::decode")
    }

    /// Split into halves so a worker can receive assignments while sending reports.
    pub fn split(self) -> (ControlSender, ControlReceiver) {
        todo!()
    }
}

pub struct ControlSender;
pub struct ControlReceiver;
