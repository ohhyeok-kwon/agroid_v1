//! CLOCK_MONOTONIC in milliseconds: one clock for every process on this host, never stepped.
//! Use it for ages, leases and ordering; wall-clock stamps are only for display.
//! `state_manager` (with the action managers) and `remote_controller` include this file by path.

pub fn mono_ms() -> u64 {
    mono_us() / 1000
}

pub fn mono_us() -> u64 {
    let mut now = libc::timespec {
        tv_sec: 0,
        tv_nsec: 0,
    };
    // SAFETY: valid output pointer; CLOCK_MONOTONIC always exists on Linux.
    unsafe { libc::clock_gettime(libc::CLOCK_MONOTONIC, &mut now) };
    now.tv_sec as u64 * 1_000_000 + now.tv_nsec as u64 / 1000
}
