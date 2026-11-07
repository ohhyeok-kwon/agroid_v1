fn main() -> std::process::ExitCode {
    let result = locked_standing_action_manager::LockedStanding::load()
        .map_err(|error| -> Box<dyn std::error::Error> { error.into() })
        .and_then(locked_standing_action_manager::motion::runtime::run);
    match result {
        Ok(()) => std::process::ExitCode::SUCCESS,
        Err(error) => {
            eprintln!("locked_standing_action_manager: {error}");
            std::process::ExitCode::FAILURE
        }
    }
}
