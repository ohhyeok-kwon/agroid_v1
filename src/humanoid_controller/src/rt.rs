//! Real-time settings for the USB I/O thread (PREEMPT_RT R1, isolated CPU 7): SCHED_FIFO,
//! locked memory and an absolute-deadline period with per-window timing statistics.
use serde::Serialize;
use std::{io, time::Duration};

/// Above the default IRQ threads (FIFO 50); the MCU USB IRQ thread should sit above it (90).
pub const USB_PRIORITY: i32 = 80;
/// Linux `MCL_ONFAULT`: lock pages when first touched instead of populating every mapping
/// (DDS reserves large thread stacks that are mostly never used).
const MCL_ONFAULT: libc::c_int = 4;

pub fn set_fifo(priority: i32) -> io::Result<()> {
    schedule(libc::SCHED_FIFO, priority)
}

/// Threads created while SCHED_FIFO inherit it; libusb's own thread must stay normal.
pub fn set_normal() -> io::Result<()> {
    schedule(libc::SCHED_OTHER, 0)
}

fn schedule(policy: libc::c_int, priority: i32) -> io::Result<()> {
    let param = libc::sched_param {
        sched_priority: priority,
    };
    // SAFETY: pid 0 is the calling thread on Linux; the parameter is initialized.
    if unsafe { libc::sched_setscheduler(0, policy, &param) } != 0 {
        return Err(io::Error::last_os_error());
    }
    Ok(())
}

pub fn lock_memory() -> io::Result<()> {
    // SAFETY: plain syscall without pointers.
    if unsafe { libc::mlockall(libc::MCL_CURRENT | libc::MCL_FUTURE | MCL_ONFAULT) } != 0 {
        return Err(io::Error::last_os_error());
    }
    Ok(())
}

/// Touch the stack once so the periodic loop never takes a first-use page fault there.
pub fn prefault_stack() {
    let mut stack = [0u8; 128 * 1024];
    for i in (0..stack.len()).step_by(4096) {
        // SAFETY: in-bounds write to a local array; volatile keeps the touch.
        unsafe { std::ptr::write_volatile(&mut stack[i], 1) };
    }
    std::hint::black_box(&stack);
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Serialize)]
pub struct CycleTiming {
    pub period_us: u32,
    pub realtime: bool,
    pub memory_locked: bool,
    pub cycles: u64,
    /// Cycles that ended after their deadline; the next deadline restarts from that end.
    pub overruns: u64,
    pub window_cycles: u32,
    /// Maxima over the last window (about one second).
    pub max_wake_latency_us: u32,
    pub max_exec_us: u32,
}

fn now_ns() -> u64 {
    let mut now = libc::timespec {
        tv_sec: 0,
        tv_nsec: 0,
    };
    // SAFETY: valid output pointer; CLOCK_MONOTONIC always exists on Linux.
    unsafe { libc::clock_gettime(libc::CLOCK_MONOTONIC, &mut now) };
    now.tv_sec as u64 * 1_000_000_000 + now.tv_nsec as u64
}

fn sleep_until(deadline_ns: u64) {
    let deadline = libc::timespec {
        tv_sec: (deadline_ns / 1_000_000_000) as libc::time_t,
        tv_nsec: (deadline_ns % 1_000_000_000) as libc::c_long,
    };
    loop {
        // SAFETY: valid absolute deadline; the remainder pointer is unused with TIMER_ABSTIME.
        let result = unsafe {
            libc::clock_nanosleep(
                libc::CLOCK_MONOTONIC,
                libc::TIMER_ABSTIME,
                &deadline,
                std::ptr::null_mut(),
            )
        };
        if result != libc::EINTR {
            break;
        }
    }
}

fn micros(ns: u64) -> u32 {
    (ns / 1000).min(u64::from(u32::MAX)) as u32
}

/// Absolute-deadline periodic loop: `begin()` at the top of a cycle, `end()` after the work,
/// then `sleep()`. Deadlines advance by the period (no drift from the work's own duration).
pub struct Periodic {
    period_ns: u64,
    next_ns: u64,
    started_ns: u64,
    window_target: u32,
    window: CycleTiming,
    cycles: u64,
    overruns: u64,
}

impl Periodic {
    pub fn new(period: Duration, window: Duration) -> Self {
        let period_ns = (period.as_nanos() as u64).max(1);
        let now = now_ns();
        Self {
            period_ns,
            next_ns: now,
            started_ns: now,
            window_target: (window.as_nanos() as u64 / period_ns).clamp(1, u64::from(u32::MAX))
                as u32,
            window: CycleTiming {
                period_us: micros(period_ns),
                ..CycleTiming::default()
            },
            cycles: 0,
            overruns: 0,
        }
    }

    pub fn begin(&mut self) {
        let now = now_ns();
        let latency = micros(now.saturating_sub(self.next_ns));
        self.window.max_wake_latency_us = self.window.max_wake_latency_us.max(latency);
        self.started_ns = now;
    }

    /// Records the cycle and sets the next deadline. A cycle that overran restarts the
    /// schedule from its end instead of running missed cycles back to back. Returns the
    /// finished window about once per `window`.
    pub fn end(&mut self, realtime: bool, memory_locked: bool) -> Option<CycleTiming> {
        let end = now_ns();
        let exec = micros(end.saturating_sub(self.started_ns));
        self.window.max_exec_us = self.window.max_exec_us.max(exec);
        self.window.window_cycles += 1;
        self.cycles += 1;
        self.next_ns += self.period_ns;
        if self.next_ns <= end {
            self.overruns += 1;
            self.next_ns = end + self.period_ns;
        }
        if self.window.window_cycles < self.window_target {
            return None;
        }
        let mut done = self.window;
        done.cycles = self.cycles;
        done.overruns = self.overruns;
        done.realtime = realtime;
        done.memory_locked = memory_locked;
        self.window = CycleTiming {
            period_us: done.period_us,
            ..CycleTiming::default()
        };
        Some(done)
    }

    pub fn sleep(&self) {
        sleep_until(self.next_ns);
    }
}
