//! MCU response framing, independent of USB and individual state decoders.
pub const FRAME_SIZE: usize = 0x12a0;
const HEADER: [u8; 2] = [0xfc, 0xfd];
// basic_service: BMS fragment at object+0xb3a3, RX buffer at object+0xb1f8.
const BMS_OFFSET: usize = 0xb3a3 - 0xb1f8;
pub const BMS_FRAGMENT_SIZE: usize = 11;

#[derive(Debug, PartialEq)]
pub struct McuFrame {
    pub bytes: [u8; FRAME_SIZE],
}

impl McuFrame {
    pub fn bms_fragment(&self) -> &[u8; BMS_FRAGMENT_SIZE] {
        self.bytes[BMS_OFFSET..BMS_OFFSET + BMS_FRAGMENT_SIZE]
            .try_into()
            .unwrap()
    }
}

pub struct McuReceiver {
    pending: [u8; FRAME_SIZE],
    length: usize,
}

impl Default for McuReceiver {
    fn default() -> Self {
        Self {
            pending: [0; FRAME_SIZE],
            length: 0,
        }
    }
}

impl McuReceiver {
    pub fn reset(&mut self) {
        self.length = 0;
    }

    /// Fixed storage and bounded parsing for the USB thread. The callback borrows one frame.
    pub fn receive_each(&mut self, mut bytes: &[u8], mut emit: impl FnMut(&[u8; FRAME_SIZE])) {
        while !bytes.is_empty() {
            if self.length < 2 {
                let b = bytes[0];
                bytes = &bytes[1..];
                if self.length == 0 {
                    if b == HEADER[0] {
                        self.pending[0] = b;
                        self.length = 1;
                    }
                } else if b == HEADER[1] {
                    self.pending[1] = b;
                    self.length = 2;
                } else {
                    self.length = usize::from(b == HEADER[0]);
                }
                continue;
            }
            let n = bytes.len().min(FRAME_SIZE - self.length);
            self.pending[self.length..self.length + n].copy_from_slice(&bytes[..n]);
            self.length += n;
            bytes = &bytes[n..];
            if self.length == FRAME_SIZE {
                emit(&self.pending);
                self.length = 0;
            }
        }
    }

    /// Handles partial USB transfers and several MCU frames in one transfer.
    pub fn receive(&mut self, bytes: &[u8]) -> Vec<McuFrame> {
        let mut output = Vec::new();
        self.receive_each(bytes, |frame| output.push(McuFrame { bytes: *frame }));
        output
    }
}
