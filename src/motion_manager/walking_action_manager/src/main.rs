fn main() -> std::process::ExitCode {
    match walking_action_manager::runtime::run() {
        Ok(()) => std::process::ExitCode::SUCCESS,
        Err(error) => {
            eprintln!("walking_action_manager: {error}");
            std::process::ExitCode::FAILURE
        }
    }
}
