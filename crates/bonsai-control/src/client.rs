//! Framed control connections shared by workers and request clients.

use std::net::SocketAddr;

use bonsai_core::{
    BonsaiError, Result,
    msg::{FromCoordinator, ToCoordinator, framing},
};
use serde::{Serialize, de::DeserializeOwned};
use tokio::{
    io::{AsyncRead, AsyncReadExt, AsyncWrite, AsyncWriteExt},
    net::{
        TcpStream,
        tcp::{OwnedReadHalf, OwnedWriteHalf},
    },
};

pub(crate) async fn read_message<R: AsyncRead + Unpin, T: DeserializeOwned>(
    reader: &mut R,
) -> Result<T> {
    let len = reader.read_u32_le().await? as usize;
    // Check the untrusted prefix before allocating or waiting for a body.
    if len == 0 || len > framing::MAX_FRAME {
        return Err(BonsaiError::Protocol(format!(
            "invalid control frame length {len}"
        )));
    }
    let mut body = vec![0; len];
    reader.read_exact(&mut body).await?;
    framing::decode(&body)
}

pub(crate) async fn write_message<W: AsyncWrite + Unpin, T: Serialize>(
    writer: &mut W,
    msg: &T,
) -> Result<()> {
    writer.write_all(&framing::encode(msg)?).await?;
    Ok(())
}

pub struct ControlClient {
    stream: TcpStream,
}

impl ControlClient {
    pub async fn connect(addr: SocketAddr) -> Result<Self> {
        let stream = TcpStream::connect(addr).await?;
        stream.set_nodelay(true)?;
        Ok(Self { stream })
    }

    pub async fn send(&mut self, msg: &ToCoordinator) -> Result<()> {
        write_message(&mut self.stream, msg).await
    }

    pub async fn recv(&mut self) -> Result<FromCoordinator> {
        read_message(&mut self.stream).await
    }

    /// Receive assignments independently of outgoing worker reports.
    pub fn split(self) -> (ControlSender, ControlReceiver) {
        let (reader, writer) = self.stream.into_split();
        (ControlSender { writer }, ControlReceiver { reader })
    }
}

pub struct ControlSender {
    writer: OwnedWriteHalf,
}

impl ControlSender {
    pub async fn send(&mut self, msg: &ToCoordinator) -> Result<()> {
        write_message(&mut self.writer, msg).await
    }

    pub async fn shutdown(&mut self) -> Result<()> {
        Ok(self.writer.shutdown().await?)
    }
}

pub struct ControlReceiver {
    reader: OwnedReadHalf,
}

impl ControlReceiver {
    pub async fn recv(&mut self) -> Result<FromCoordinator> {
        read_message(&mut self.reader).await
    }
}
