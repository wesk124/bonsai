//! TCP front end. One task owns the coordinator; bounded connection queues
//! isolate stalled peers, and every connection failure triggers cleanup.

use std::{
    collections::{HashMap, VecDeque},
    future::Future,
    net::SocketAddr,
    time::{Duration, Instant},
};

use bonsai_core::{
    BonsaiError, Result,
    msg::{FromCoordinator, ToCoordinator},
};
use tokio::{
    net::{TcpListener, TcpStream},
    sync::{mpsc, watch},
    task::JoinSet,
};

use crate::{
    client::{read_message, write_message},
    coordinator::{ConnId, Coordinator, Outbound},
};

#[derive(Debug, Clone)]
pub struct ServerConfig {
    pub listen: SocketAddr,
    pub tick: Duration,
}

pub async fn serve(cfg: ServerConfig, coordinator: Coordinator) -> Result<()> {
    serve_until(cfg, coordinator, std::future::pending()).await
}

/// Run until cancellation, close all sockets, and join all connection tasks.
pub async fn serve_until(
    cfg: ServerConfig,
    coordinator: Coordinator,
    shutdown: impl Future<Output = ()>,
) -> Result<()> {
    let listener = TcpListener::bind(cfg.listen).await?;
    serve_listener(listener, cfg.tick, coordinator, shutdown).await
}

enum Event {
    Message(ConnId, Box<ToCoordinator>),
    Disconnected(ConnId),
}

struct Connection {
    outgoing: mpsc::Sender<FromCoordinator>,
    cancel: watch::Sender<bool>,
}

/// Accept an already bound listener (also permits race-free ephemeral-port tests).
pub async fn serve_listener(
    listener: TcpListener,
    tick: Duration,
    mut coordinator: Coordinator,
    shutdown: impl Future<Output = ()>,
) -> Result<()> {
    if tick.is_zero() {
        return Err(BonsaiError::InvalidConfig(
            "server tick must be positive".into(),
        ));
    }
    let (incoming, mut events) = mpsc::channel::<Event>(256);
    let mut connections = HashMap::new();
    let mut tasks = JoinSet::new();
    let mut interval = tokio::time::interval(tick);
    interval.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Skip);
    let mut next_id = 0u64;
    tokio::pin!(shutdown);
    let result = loop {
        let out = tokio::select! {
            _ = &mut shutdown => break Ok(()),
            accepted = listener.accept() => {
                let (stream, _) = match accepted { Ok(v) => v, Err(e) => break Err(e.into()) };
                let id = ConnId(next_id);
                next_id = match next_id.checked_add(1) { Some(v) => v, None => break Err(BonsaiError::Protocol("connection ids exhausted".into())) };
                let (outgoing, receiver) = mpsc::channel(64);
                let (cancel, cancelled) = watch::channel(false);
                connections.insert(id, Connection { outgoing, cancel });
                let incoming = incoming.clone();
                tasks.spawn(async move {
                    let result = connection(stream, id, incoming.clone(), receiver, cancelled).await;
                    // Share the reader's FIFO so EOF cannot overtake its last report.
                    let _ = incoming.send(Event::Disconnected(id)).await;
                    (id, result)
                });
                Vec::new()
            },
            Some(event) = events.recv() => match event {
                Event::Message(id, msg) if connections.contains_key(&id) => coordinator.on_message(id, *msg, Instant::now()),
                Event::Disconnected(id) if connections.remove(&id).is_some() => coordinator.on_disconnect(id, Instant::now()),
                _ => Vec::new(),
            },
            Some(joined) = tasks.join_next(), if !tasks.is_empty() => {
                match joined {
                    Ok((id, result)) => {
                        if let Err(error) = result { tracing::debug!(?id, %error, "control connection closed"); }
                        Vec::new()
                    },
                    Err(error) => break Err(BonsaiError::Transport(format!("connection task failed: {error}"))),
                }
            },
            _ = interval.tick() => coordinator.on_tick(Instant::now()),
        };
        dispatch(out, &mut connections, &mut coordinator);
    };
    // Release tasks waiting to report EOF on a full input queue during shutdown.
    events.close();
    // Cancellation interrupts blocked reads/writes as well as idle connections.
    for (_, connection) in connections.drain() {
        let _ = connection.cancel.send(true);
    }
    while tasks.join_next().await.is_some() {}
    result
}

fn dispatch(
    out: Vec<Outbound>,
    connections: &mut HashMap<ConnId, Connection>,
    coordinator: &mut Coordinator,
) {
    let mut pending = VecDeque::from(out);
    while let Some(out) = pending.pop_front() {
        let failed = connections
            .get(&out.to)
            .is_some_and(|c| c.outgoing.try_send(out.msg).is_err());
        if failed {
            if let Some(c) = connections.remove(&out.to) {
                let _ = c.cancel.send(true);
            }
            pending.extend(coordinator.on_disconnect(out.to, Instant::now()));
        }
    }
}

async fn connection(
    stream: TcpStream,
    id: ConnId,
    incoming: mpsc::Sender<Event>,
    mut outgoing: mpsc::Receiver<FromCoordinator>,
    mut cancelled: watch::Receiver<bool>,
) -> Result<()> {
    stream.set_nodelay(true)?;
    let (mut reader, mut writer) = stream.into_split();
    let read = async {
        loop {
            let message = read_message(&mut reader).await?;
            if incoming
                .send(Event::Message(id, Box::new(message)))
                .await
                .is_err()
            {
                return Ok(());
            }
        }
    };
    let write = async {
        while let Some(message) = outgoing.recv().await {
            let shutdown = matches!(message, FromCoordinator::Shutdown);
            write_message(&mut writer, &message).await?;
            if shutdown {
                return Ok(());
            }
        }
        Ok(())
    };
    tokio::select! {
        result = read => result,
        result = write => result,
        _ = cancelled.changed() => Ok(()),
    }
}
