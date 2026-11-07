use std::{io, net::Ipv4Addr, ptr};

pub fn check() -> io::Result<bool> {
    let mut head = ptr::null_mut();
    // SAFETY: getifaddrs owns the returned linked list until freeifaddrs below.
    if unsafe { libc::getifaddrs(&mut head) } != 0 {
        return Err(io::Error::last_os_error());
    }
    let mut current = head;
    let mut healthy = false;
    while !current.is_null() {
        // SAFETY: current/ifa_addr belong to the live getifaddrs list. The address
        // is cast to sockaddr_in only after checking its AF_INET family.
        unsafe {
            let interface = &*current;
            if !interface.ifa_addr.is_null()
                && (*interface.ifa_addr).sa_family as i32 == libc::AF_INET
            {
                let address = &*interface.ifa_addr.cast::<libc::sockaddr_in>();
                let ip = Ipv4Addr::from(address.sin_addr.s_addr.to_ne_bytes());
                if usable(interface.ifa_flags, ip) {
                    healthy = true;
                    break;
                }
            }
            current = interface.ifa_next;
        }
    }
    // SAFETY: head is the unmodified pointer returned successfully by getifaddrs.
    unsafe { libc::freeifaddrs(head) };
    Ok(healthy)
}

fn usable(flags: u32, ip: Ipv4Addr) -> bool {
    let required = (libc::IFF_UP | libc::IFF_RUNNING) as u32;
    flags & required == required
        && flags & libc::IFF_LOOPBACK as u32 == 0
        && !ip.is_unspecified()
        && !ip.is_loopback()
}
