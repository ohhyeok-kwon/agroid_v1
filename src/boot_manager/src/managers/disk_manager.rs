use std::io;

const MIN_FREE_PERCENT: f64 = 20.0;

pub fn check() -> io::Result<bool> {
    // SAFETY: statvfs receives a NUL-terminated path and a writable output structure.
    let mut stats = unsafe { std::mem::zeroed::<libc::statvfs>() };
    if unsafe { libc::statvfs(c"/".as_ptr(), &mut stats) } != 0 {
        return Err(io::Error::last_os_error());
    }
    healthy(
        stats.f_blocks,
        stats.f_bavail,
        stats.f_flag & libc::ST_RDONLY != 0,
    )
}

fn healthy(total: u64, available: u64, read_only: bool) -> io::Result<bool> {
    if total == 0 || available > total {
        return Err(super::invalid_data("invalid filesystem counters"));
    }
    if read_only {
        return Err(io::Error::new(
            io::ErrorKind::PermissionDenied,
            "root filesystem is read-only",
        ));
    }
    Ok(100.0 * available as f64 / (total as f64) >= MIN_FREE_PERCENT)
}
