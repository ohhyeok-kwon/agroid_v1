//! Transmission requests. Payload bytes are opaque; no motion format is defined here.

#[derive(Clone, Copy, Debug)]
pub enum ControllerCommand<'a> {
    ReadStatus,
    Motion(&'a [u8]),
    Peripheral(&'a [u8]),
}
