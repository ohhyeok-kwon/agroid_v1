//! Factory profile selected by basic_service for R1 (sect=10, type=1, peripheral=0).
//! Verified against the native 0x1019b438 selector and 0x1056ea4c model table.
use serde::Serialize;
use std::{fs, io, path::Path};

pub const CONFIG_OFFSET: usize = 0x20;
pub const CONFIG_END: usize = 0x134;
pub const TX_SLOT_OFFSET: usize = 0x138;
pub const RX_SLOT_OFFSET: usize = 0x1d8;
pub const SLOT_SIZE: usize = 0x38;
pub const BUS_COUNT: usize = 8;
pub const SLOTS_PER_BUS: usize = 9;

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize)]
pub struct MotorAddress {
    pub bus: u8,
    pub id: u8,
}

pub const HEAD_YAW: MotorAddress = MotorAddress { bus: 2, id: 0 };
pub const HEAD_PITCH: MotorAddress = MotorAddress { bus: 4, id: 0 };
pub const AUXILIARY_IMU: MotorAddress = MotorAddress { bus: 6, id: 0 };

/// Original 35-slot LowCmd/LowState indices, in our 26-joint URDF order.
pub const NATIVE_INDICES: [usize; 26] = [
    0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 15, 16, 17, 18, 19, 22, 23, 24, 25, 26, 29, 30,
];

/// URDF order: address, direction, effective transmission ratio. Native R1 table.
pub const JOINT_MOTORS: [(MotorAddress, f32, f32); 26] = {
    const R: f32 = 12.666667;
    const fn m(bus: u8, id: u8) -> MotorAddress {
        MotorAddress { bus, id }
    }
    [
        (m(5, 1), 1., 32.),
        (m(5, 2), 1., 32.),
        (m(5, 3), -1., 32.),
        (m(5, 4), 1., 32.),
        (m(5, 5), 1., R),
        (m(5, 6), 1., R),
        (m(6, 1), -1., 32.),
        (m(6, 2), 1., 32.),
        (m(6, 3), -1., 32.),
        (m(6, 4), -1., 32.),
        (m(6, 5), 1., R),
        (m(6, 6), 1., R),
        (m(5, 0), 1., 32.),
        (m(1, 0), 1., 32.),
        (m(0, 0), 1., 32.),
        (m(0, 1), 1., 32.),
        (m(0, 2), -1., R),
        (m(0, 3), 1., R),
        (m(0, 4), 1., R),
        (m(3, 0), -1., 32.),
        (m(3, 1), 1., 32.),
        (m(3, 2), -1., R),
        (m(3, 3), -1., R),
        (m(3, 4), 1., R),
        (m(4, 0), 1., R),
        (m(2, 0), -1., 214.34805),
    ]
};

// Bus/id are physical UART addresses, not ROS or manufacturer joint indices.
pub const MOTORS: [MotorAddress; 26] = [
    MotorAddress { bus: 0, id: 0 },
    MotorAddress { bus: 0, id: 1 },
    MotorAddress { bus: 0, id: 2 },
    MotorAddress { bus: 0, id: 3 },
    MotorAddress { bus: 0, id: 4 },
    MotorAddress { bus: 1, id: 0 },
    HEAD_YAW,
    MotorAddress { bus: 3, id: 0 },
    MotorAddress { bus: 3, id: 1 },
    MotorAddress { bus: 3, id: 2 },
    MotorAddress { bus: 3, id: 3 },
    MotorAddress { bus: 3, id: 4 },
    HEAD_PITCH,
    MotorAddress { bus: 5, id: 0 },
    MotorAddress { bus: 5, id: 1 },
    MotorAddress { bus: 5, id: 2 },
    MotorAddress { bus: 5, id: 3 },
    MotorAddress { bus: 5, id: 4 },
    MotorAddress { bus: 5, id: 5 },
    MotorAddress { bus: 5, id: 6 },
    MotorAddress { bus: 6, id: 1 },
    MotorAddress { bus: 6, id: 2 },
    MotorAddress { bus: 6, id: 3 },
    MotorAddress { bus: 6, id: 4 },
    MotorAddress { bus: 6, id: 5 },
    MotorAddress { bus: 6, id: 6 },
];

pub fn slot_offset(base: usize, address: MotorAddress) -> usize {
    base + (usize::from(address.bus) * SLOTS_PER_BUS + usize::from(address.id)) * SLOT_SIZE
}

/// A checked model selection; no guessed profile is applied to another robot.
pub struct R1Profile(());

impl R1Profile {
    pub fn load() -> io::Result<Self> {
        let primary = "/unitree/robot/basic/basic_service.json";
        let fallback = "/unitree/module/basic_service/basic_service.json";
        let settings = match fs::read_to_string(primary) {
            Ok(settings) => settings,
            Err(error) if error.kind() == io::ErrorKind::NotFound => fs::read_to_string(fallback)?,
            Err(error) => return Err(error),
        };
        let profile = Self::from_settings(
            &settings,
            optional_number(Path::new("/unitree/robot/basic/machinetype"))?,
            optional_number(Path::new("/unitree/robot/info/g1_peripheral_version"))?.unwrap_or(0),
        )?;
        for primary in [
            "/unitree/robot/basic/basic_motor.json",
            "/unitree/robot/basic/basic_protect.json",
        ] {
            let fallback = primary.replace("/robot/basic/", "/module/basic_service/");
            let text = match fs::read_to_string(primary) {
                Ok(text) => Some(text),
                Err(e) if e.kind() == io::ErrorKind::NotFound => {
                    match fs::read_to_string(&fallback) {
                        Ok(text) => Some(text),
                        Err(e)
                            if e.kind() == io::ErrorKind::NotFound
                                && primary.ends_with("basic_protect.json") =>
                        {
                            None
                        }
                        Err(e) => return Err(e),
                    }
                }
                Err(e) => return Err(e),
            };
            if let Some(text) = text {
                let value = serde_json::from_str(&text).map_err(io::Error::other)?;
                reject_overrides(&value, "", primary.ends_with("basic_protect.json"))?;
            }
        }
        Ok(profile)
    }

    pub fn from_settings(
        settings: &str,
        machine_type: Option<u64>,
        peripheral: u64,
    ) -> io::Result<Self> {
        let json: serde_json::Value = serde_json::from_str(settings)
            .map_err(|error| io::Error::new(io::ErrorKind::InvalidData, error))?;
        let options = &json["basicSetupOptions"];
        reject_overrides(&json["developerMode"], "developerMode", false)?;
        let machine_type = machine_type.or_else(|| options["machinetype"].as_u64());
        if options["machinesect"].as_u64() != Some(10)
            || machine_type != Some(1)
            || peripheral != 0
            || options["jointmotor"].as_bool() != Some(false)
            || options["externaljointcalibration"].as_bool() != Some(false)
            || options["changeconfig"].as_bool() != Some(false)
        {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "unsupported robot profile; expected R1 sect=10/type=1/peripheral=0 with factory configuration",
            ));
        }
        Ok(Self(()))
    }

    /// Native config: 9 baud values, 8x10 enable bytes, 8x10 raw u16 timeouts.
    pub(crate) fn write_config(&self, frame: &mut [u8]) {
        frame[CONFIG_OFFSET..CONFIG_END].fill(0);
        for i in 0..9 {
            let baud: u32 = if i == 8 { 921_600 } else { 6_000_000 };
            frame[0x20 + i * 4..0x24 + i * 4].copy_from_slice(&baud.to_le_bytes());
        }
        for address in MOTORS.iter().chain(std::iter::once(&AUXILIARY_IMU)) {
            frame[0x44 + usize::from(address.bus) * 10 + usize::from(address.id)] = 1;
        }
        // Preserve the table's disabled-slot timeout values as well as active ones.
        for (bus, count) in [9, 1, 1, 9, 1, 7, 7, 0].into_iter().enumerate() {
            for id in 0..count {
                let offset = 0x94 + (bus * 10 + id) * 2;
                frame[offset..offset + 2].copy_from_slice(&100_u16.to_le_bytes());
            }
        }
    }
}

fn reject_overrides(value: &serde_json::Value, path: &str, protect: bool) -> io::Result<()> {
    if let Some(object) = value.as_object() {
        for (key, child) in object {
            let name = format!("{path}/{key}");
            // LED color does not change mechanics; all motor/protection overrides
            // must be implemented explicitly before applying this factory profile.
            let led = protect && name.starts_with("/basicStart/led/");
            if !led
                && (key.to_ascii_lowercase().ends_with("enable"))
                && child != &serde_json::Value::Bool(false)
            {
                return Err(io::Error::new(
                    io::ErrorKind::InvalidData,
                    format!("unsupported enabled or malformed factory override: {name}"),
                ));
            }
            reject_overrides(child, &name, protect)?;
        }
    }
    Ok(())
}

fn optional_number(path: &Path) -> io::Result<Option<u64>> {
    match fs::read_to_string(path) {
        Ok(value) if value.trim().is_empty() => Ok(None),
        Ok(value) => value.trim().parse().map(Some).map_err(|_| {
            io::Error::new(
                io::ErrorKind::InvalidData,
                format!("invalid robot setting: {}", path.display()),
            )
        }),
        Err(error) if error.kind() == io::ErrorKind::NotFound => Ok(None),
        Err(error) => Err(error),
    }
}
