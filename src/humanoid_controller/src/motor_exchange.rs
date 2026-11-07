//! Native motor command/response channel (basic_service 0x10171628).
//! Separate from motion targets. TX[3] and RX[21] exchange nibbles every six calls.
use std::{
    ffi::{c_int, c_void},
    io, ptr,
};

// Factory protocol constant from the verified R1 image at 0x1054d010.
// It is not a user credential. Keep the binary representation out of logs.
const KEY: &[u8; 16] = include_bytes!("../config/r1_motor_exchange.bin");

#[link(name = "crypto")]
unsafe extern "C" {
    fn EVP_CIPHER_CTX_new() -> *mut c_void;
    fn EVP_CIPHER_CTX_free(ctx: *mut c_void);
    fn EVP_aes_128_ecb() -> *const c_void;
    fn EVP_EncryptInit_ex(
        ctx: *mut c_void,
        cipher: *const c_void,
        engine: *mut c_void,
        key: *const u8,
        iv: *const u8,
    ) -> c_int;
    fn EVP_CIPHER_CTX_set_padding(ctx: *mut c_void, padding: c_int) -> c_int;
    fn EVP_EncryptUpdate(
        ctx: *mut c_void,
        out: *mut u8,
        len: *mut c_int,
        input: *const u8,
        input_len: c_int,
    ) -> c_int;
}

struct Cipher(*mut c_void);
impl Cipher {
    fn new() -> Self {
        // SAFETY: OpenSSL returns a thread-owned opaque context.
        Self(unsafe { EVP_CIPHER_CTX_new() })
    }
}
impl Drop for Cipher {
    fn drop(&mut self) {
        unsafe {
            if !self.0.is_null() {
                EVP_CIPHER_CTX_free(self.0);
            }
        }
    }
}
thread_local! { static CIPHER: Cipher = Cipher::new(); }

fn encrypt_first_block(bytes: &mut [u8; 32]) -> io::Result<()> {
    CIPHER.with(|cached| {
        let ctx = cached.0;
        if ctx.is_null() {
            return Err(io::Error::other("motor exchange cipher allocation failed"));
        }
        let mut output = [0u8; 32];
        let mut len = 0;
        // SAFETY: this context belongs exclusively to this thread, buffers are initialized.
        let ok = unsafe {
            EVP_EncryptInit_ex(
                ctx,
                EVP_aes_128_ecb(),
                ptr::null_mut(),
                KEY.as_ptr(),
                ptr::null(),
            ) == 1
                && EVP_CIPHER_CTX_set_padding(ctx, 0) == 1
                && EVP_EncryptUpdate(ctx, output.as_mut_ptr(), &mut len, bytes.as_ptr(), 16) == 1
                && len == 16
        };
        if !ok {
            return Err(io::Error::other("motor exchange cipher failed"));
        }
        bytes[..16].copy_from_slice(&output[..16]);
        Ok(())
    })
}

#[derive(Clone, Default)]
pub struct MotorExchange {
    tx: u8,
    prescaler: u8,
    index: u32,
    last_toggle: u8,
    challenge: [u8; 32],
    response: [u8; 32],
}

impl MotorExchange {
    pub fn next(&mut self, rx: u8, random_bit: u8) -> io::Result<u8> {
        if self.prescaler < 5 {
            self.prescaler += 1;
            return Ok(self.tx);
        }
        self.prescaler = 0;
        self.tx = (self.tx & 0x6f) | ((!rx) & 0x80) | ((random_bit & 1) << 4);
        if self.last_toggle != rx >> 7 {
            self.tx = (self.tx & 0x9f) | (rx & 0x20);
            if rx & 0x20 != 0 {
                self.index = 0;
                if rx & 0x40 != 0 {
                    self.challenge.fill(0);
                } else {
                    self.response = self.challenge;
                    encrypt_first_block(&mut self.response)?;
                }
            }
            if rx & 0x40 != 0 {
                if self.index < 64 {
                    let i = (self.index / 2) as usize;
                    self.challenge[i] |= (rx & 15) << (4 * (self.index & 1));
                    if self.index & 1 != 0 {
                        self.challenge[i] ^= if i == 0 { 0x77 } else { self.challenge[0] };
                    }
                } else {
                    self.tx |= 0x40;
                }
                self.tx = (self.tx & 0xf0) | (self.index as u8 & 15);
            } else {
                if self.index < 64 {
                    self.tx = (self.tx & 0xf0)
                        | ((self.response[(self.index / 2) as usize] >> (4 * (self.index & 1)))
                            & 15);
                } else {
                    self.tx |= 0x40;
                }
                if self.index & 15 != u32::from(rx & 15) {
                    self.tx |= 0x40;
                }
            }
            self.index = self.index.wrapping_add(1);
        }
        self.last_toggle = rx >> 7;
        Ok(self.tx)
    }

    pub fn self_check() -> io::Result<()> {
        let mut block = [0; 32];
        encrypt_first_block(&mut block)
    }
}
