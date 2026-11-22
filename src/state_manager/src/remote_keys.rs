//! Native button events from validated /joy, gated by /remote_status.
//! Native held/pressed edges; packet freshness comes from the remote receiver.
use crate::motion::Action;
use serde::Deserialize;

pub const START: u16 = 1 << 2;
pub const L2: u16 = 1 << 5;
pub const R2: u16 = 1 << 4;
pub const A: u16 = 1 << 8;
pub const B: u16 = 1 << 9;
pub const X: u16 = 1 << 10;
pub const Y: u16 = 1 << 11;
pub const UP: u16 = 1 << 12;
const JOY_TIMEOUT_MS: u64 = 250;
const HEALTH_TIMEOUT_MS: u64 = 350;

pub struct Edges {
    pub held: u16,
    pub pressed: u16,
    pub start_double: bool,
}

/// Original Static/Damping/ZeroTorque mappings; Stance is FSM 4, not Keep 5.
pub fn action_for(current: Option<Action>, edges: &Edges) -> Option<Action> {
    if current == Some(Action::LockedStanding) && edges.held & R2 != 0 {
        // A: manufacturer walking (811). X: our learned careful walking (812).
        if edges.pressed & A != 0 {
            return Some(Action::Walking);
        }
        if edges.pressed & X != 0 {
            return Some(Action::CarefulWalk);
        }
    }
    if edges.held & L2 == 0 {
        return None;
    }
    // Damping also interrupts interpolation, unlike the native MoveTo key gap.
    if edges.pressed & B != 0 {
        return Some(Action::Damping);
    }
    if current == Some(Action::Damping) && edges.pressed & Y != 0 {
        return Some(Action::ZeroMoment);
    }
    if matches!(
        current,
        Some(
            Action::ZeroMoment
                | Action::Damping
                | Action::Walking
                | Action::CarefulWalk
                | Action::SafeCrouch
        )
    ) && edges.pressed & UP != 0
    {
        return Some(Action::LockedStanding);
    }
    None
}

#[derive(Deserialize)]
struct Health {
    status_code: u16,
    connected: bool,
}

#[derive(Default)]
pub struct RemoteKeys {
    health: Option<(u64, bool)>,
    packet: Option<u64>,
    raw: u16,
    stable: u16,
    last_start: Option<u64>,
}

impl RemoteKeys {
    pub fn health(&mut self, text: &str, now: u64) {
        let valid = serde_json::from_str::<Health>(text)
            .is_ok_and(|health| health.status_code == 200 && health.connected);
        self.health = Some((now, valid));
    }

    pub fn receive(&mut self, buttons: &[i32], now: u64) {
        if buttons.len() != 16 || buttons.iter().any(|&b| b != 0 && b != 1) {
            self.packet = None;
            self.reset();
            return;
        }
        self.packet = Some(now);
        self.raw = buttons
            .iter()
            .enumerate()
            .fold(0u16, |bits, (i, &b)| bits | ((b as u16) << i));
    }

    pub fn connected(&self, now: u64) -> bool {
        self.health
            .is_some_and(|(time, valid)| valid && now.saturating_sub(time) <= HEALTH_TIMEOUT_MS)
            && self
                .packet
                .is_some_and(|time| now.saturating_sub(time) < JOY_TIMEOUT_MS)
    }

    pub fn poll(&mut self, now: u64) -> Option<Edges> {
        if !self.connected(now) {
            self.reset();
            return None;
        }
        if self.raw == self.stable {
            return None;
        }
        let previous = self.stable;
        self.stable = self.raw;
        let pressed = self.stable & !previous;
        let mut start_double = false;
        if pressed & START != 0 {
            start_double = self
                .last_start
                .is_some_and(|last| now.saturating_sub(last) <= 500);
            self.last_start = if start_double { None } else { Some(now) };
        }
        Some(Edges {
            held: previous & self.stable,
            pressed,
            start_double,
        })
    }

    fn reset(&mut self) {
        self.stable = 0;
        self.last_start = None;
    }
}
