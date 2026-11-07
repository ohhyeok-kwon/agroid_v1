//! Owns the USB handle and byte transfers; no MCU/BMS interpretation.
use std::{
    ffi::{c_int, c_void},
    io, ptr,
    time::Duration,
};

const VENDOR: u16 = 0x3c7b;
const PRODUCT: u16 = 0x4d8e;
const INTERFACE: c_int = 0;
const IN_ENDPOINT: u8 = 0x81;
const OUT_ENDPOINT: u8 = 0x01;
const TIMEOUT_MS: u32 = 100;
// R1 reserves CPU 7 (isolcpus). Native sendRev is explicitly bound there.
const USB_CPU: usize = 7;

#[link(name = "libusb-1.0.so.0", kind = "dylib", modifiers = "+verbatim")]
unsafe extern "C" {
    fn libusb_init(context: *mut *mut c_void) -> c_int;
    fn libusb_exit(context: *mut c_void);
    fn libusb_open_device_with_vid_pid(
        context: *mut c_void,
        vendor: u16,
        product: u16,
    ) -> *mut c_void;
    fn libusb_close(handle: *mut c_void);
    fn libusb_claim_interface(handle: *mut c_void, interface: c_int) -> c_int;
    fn libusb_release_interface(handle: *mut c_void, interface: c_int) -> c_int;
    fn libusb_bulk_transfer(
        handle: *mut c_void,
        endpoint: u8,
        data: *mut u8,
        length: c_int,
        transferred: *mut c_int,
        timeout: u32,
    ) -> c_int;
}

pub struct UsbTransport {
    context: *mut c_void,
    handle: *mut c_void,
    periodic: bool,
}

pub trait UsbWrite {
    fn write_packet(&mut self, buffer: &[u8]) -> io::Result<()>;
}

pub trait UsbExchange {
    fn exchange(
        &mut self,
        request: &[u8],
        reply: &mut [u8],
        timeout: Duration,
    ) -> io::Result<usize>;
}

impl UsbExchange for UsbTransport {
    fn exchange(
        &mut self,
        request: &[u8],
        reply: &mut [u8],
        timeout: Duration,
    ) -> io::Result<usize> {
        let millis = u32::try_from(timeout.as_millis())
            .map_err(|_| io::Error::new(io::ErrorKind::InvalidInput, "USB timeout is too large"))?;
        if millis == 0 {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                "USB timeout must be bounded",
            ));
        }
        self.write_with_timeout(request, millis)?;
        self.read_with_timeout(reply, millis)
    }
}

impl UsbWrite for UsbTransport {
    fn write_packet(&mut self, buffer: &[u8]) -> io::Result<()> {
        self.write(buffer)
    }
}

fn error(operation: &str, code: c_int) -> io::Error {
    io::Error::other(format!("USB {operation} failed (libusb {code})"))
}

fn pin_thread_to_cpu(cpu: usize) -> io::Result<()> {
    if cpu >= libc::CPU_SETSIZE as usize {
        return Err(io::Error::new(
            io::ErrorKind::InvalidInput,
            "invalid USB CPU",
        ));
    }
    // SAFETY: the set is initialized, the CPU index is bounded, and pid 0
    // changes only this calling thread. ROS threads keep their existing mask.
    let result = unsafe {
        let mut cpus = std::mem::zeroed::<libc::cpu_set_t>();
        libc::CPU_ZERO(&mut cpus);
        libc::CPU_SET(cpu, &mut cpus);
        libc::sched_setaffinity(0, std::mem::size_of_val(&cpus), &cpus)
    };
    if result != 0 {
        let error = io::Error::last_os_error();
        return Err(io::Error::new(
            error.kind(),
            format!("USB CPU {cpu} affinity failed: {error}"),
        ));
    }
    Ok(())
}

impl UsbTransport {
    pub fn open() -> io::Result<Self> {
        // Do this before libusb_init so its event thread inherits CPU 7 too.
        // Without the native affinity, scheduler delays can skip RF fragments. Like the
        // native sendRev thread, a failed pin is not fatal: motor/BMS I/O must still open.
        if let Err(error) = pin_thread_to_cpu(USB_CPU) {
            eprintln!("humanoid_controller: {error}; USB continues without CPU {USB_CPU}");
        }
        let mut context = ptr::null_mut();
        // SAFETY: pointers are outputs owned by libusb; every success has a matching close.
        unsafe {
            let result = libusb_init(&mut context);
            if result != 0 {
                return Err(error("initialize", result));
            }
            let handle = libusb_open_device_with_vid_pid(context, VENDOR, PRODUCT);
            if handle.is_null() {
                libusb_exit(context);
                return Err(io::Error::new(
                    io::ErrorKind::NotFound,
                    "USB 3c7b:4d8e is unavailable",
                ));
            }
            let result = libusb_claim_interface(handle, INTERFACE);
            if result != 0 {
                libusb_close(handle);
                libusb_exit(context);
                return Err(error("claim interface 0", result));
            }
            Ok(Self {
                context,
                handle,
                periodic: false,
            })
        }
    }

    /// Send one complete transfer. A partial write must not be followed by a read.
    pub fn write(&mut self, buffer: &[u8]) -> io::Result<()> {
        self.write_with_timeout(buffer, if self.periodic { 1 } else { TIMEOUT_MS })
    }

    /// Bound ordinary transfers; startup still uses its explicit one-second exchange timeout.
    pub fn set_periodic(&mut self) {
        self.periodic = true;
    }

    fn write_with_timeout(&mut self, buffer: &[u8], timeout_ms: u32) -> io::Result<()> {
        let length = c_int::try_from(buffer.len())
            .map_err(|_| io::Error::new(io::ErrorKind::InvalidInput, "USB buffer is too large"))?;
        let mut transferred = 0;
        // SAFETY: the handle is live; libusb reads, but does not modify, an OUT buffer.
        let result = unsafe {
            libusb_bulk_transfer(
                self.handle,
                OUT_ENDPOINT,
                buffer.as_ptr().cast_mut(),
                length,
                &mut transferred,
                timeout_ms,
            )
        };
        if result != 0 {
            return Err(error("write endpoint 0x01", result));
        }
        if transferred != length {
            return Err(io::Error::new(
                io::ErrorKind::WriteZero,
                format!("USB partial write: {transferred}/{length} bytes"),
            ));
        }
        Ok(())
    }

    /// A timeout is not a new sample. Partial data returned at timeout is retained.
    pub fn read(&mut self, buffer: &mut [u8]) -> io::Result<usize> {
        self.read_with_timeout(buffer, if self.periodic { 1 } else { TIMEOUT_MS })
    }

    fn read_with_timeout(&mut self, buffer: &mut [u8], timeout_ms: u32) -> io::Result<usize> {
        let length = c_int::try_from(buffer.len())
            .map_err(|_| io::Error::new(io::ErrorKind::InvalidInput, "USB buffer is too large"))?;
        let mut transferred = 0;
        // SAFETY: handle is live; libusb can write at most buffer.len() bytes.
        let result = unsafe {
            libusb_bulk_transfer(
                self.handle,
                IN_ENDPOINT,
                buffer.as_mut_ptr(),
                length,
                &mut transferred,
                timeout_ms,
            )
        };
        if result == 0 || result == -7 {
            if !(0..=length).contains(&transferred) {
                return Err(error("invalid transfer length", result));
            }
            Ok(transferred as usize)
        } else {
            Err(error("read endpoint 0x81", result))
        }
    }
}

impl Drop for UsbTransport {
    fn drop(&mut self) {
        // SAFETY: this object uniquely owns a claimed handle and context.
        unsafe {
            libusb_release_interface(self.handle, INTERFACE);
            libusb_close(self.handle);
            libusb_exit(self.context);
        }
    }
}
