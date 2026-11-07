use std::{fs, io};

const MAX_TEMPERATURE_C: f64 = 90.0;

pub fn check() -> io::Result<bool> {
    let mut found = false;
    let mut healthy = true;
    for entry in fs::read_dir("/sys/class/thermal")? {
        let entry = entry?;
        if !entry
            .file_name()
            .to_string_lossy()
            .starts_with("thermal_zone")
        {
            continue;
        }
        let value = fs::read_to_string(entry.path().join("temp"))?;
        found = true;
        healthy &= below_threshold(&value)?;
    }
    if !found {
        return Err(super::invalid_data("no temperature sensors found"));
    }
    Ok(healthy)
}

fn below_threshold(value: &str) -> io::Result<bool> {
    let temperature = value
        .trim()
        .parse::<f64>()
        .map_err(|_| super::invalid_data("invalid temperature"))?
        / 1000.0;
    if !temperature.is_finite() || temperature < -273.15 {
        return Err(super::invalid_data("invalid temperature"));
    }
    Ok(temperature < MAX_TEMPERATURE_C)
}
