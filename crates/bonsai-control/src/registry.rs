use std::{collections::HashMap, time::Instant};

use bonsai_core::{
    WorkerId, WorkerRole,
    msg::{WorkerInfo, WorkerLoad},
};

use crate::coordinator::ConnId;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum WorkerState {
    Ready,
    /// No heartbeat for `heartbeat_timeout`; not given new work.
    Suspect,
    Draining,
}

#[derive(Debug, Clone)]
pub struct WorkerEntry {
    pub id: WorkerId,
    pub conn: ConnId,
    pub info: WorkerInfo,
    pub load: WorkerLoad,
    pub state: WorkerState,
    pub last_seen: Instant,
    /// Arena bytes the coordinator has asked for but not yet seen released.
    /// Lets admission avoid obvious `SlotDenied`s between heartbeats.
    pub reserved_bytes: u64,
}

#[derive(Default)]
pub struct WorkerRegistry {
    workers: HashMap<WorkerId, WorkerEntry>,
    next_id: u32,
}

impl WorkerRegistry {
    pub fn register(&mut self, conn: ConnId, info: WorkerInfo, now: Instant) -> WorkerId {
        let id = WorkerId(self.next_id);
        self.next_id += 1;
        let load = WorkerLoad { arena_free_bytes: info.arena_bytes, ..Default::default() };
        self.workers.insert(
            id,
            WorkerEntry { id, conn, info, load, state: WorkerState::Ready, last_seen: now, reserved_bytes: 0 },
        );
        id
    }

    pub fn heartbeat(&mut self, id: WorkerId, load: WorkerLoad, now: Instant) -> bool {
        match self.workers.get_mut(&id) {
            Some(w) => {
                w.load = load;
                w.last_seen = now;
                if w.state == WorkerState::Suspect {
                    w.state = WorkerState::Ready;
                }
                true
            }
            None => false,
        }
    }

    pub fn remove(&mut self, id: WorkerId) -> Option<WorkerEntry> {
        self.workers.remove(&id)
    }

    pub fn by_conn(&self, conn: ConnId) -> Option<WorkerId> {
        self.workers.values().find(|w| w.conn == conn).map(|w| w.id)
    }

    pub fn get(&self, id: WorkerId) -> Option<&WorkerEntry> {
        self.workers.get(&id)
    }

    pub fn get_mut(&mut self, id: WorkerId) -> Option<&mut WorkerEntry> {
        self.workers.get_mut(&id)
    }

    /// Ready workers that can play `role` (Colocated counts as both).
    pub fn ready(&self, role: WorkerRole) -> impl Iterator<Item = &WorkerEntry> {
        self.workers.values().filter(move |w| {
            w.state == WorkerState::Ready && (w.info.role == role || w.info.role == WorkerRole::Colocated)
        })
    }

    pub fn iter(&self) -> impl Iterator<Item = &WorkerEntry> {
        self.workers.values()
    }

    pub fn iter_mut(&mut self) -> impl Iterator<Item = &mut WorkerEntry> {
        self.workers.values_mut()
    }
}
