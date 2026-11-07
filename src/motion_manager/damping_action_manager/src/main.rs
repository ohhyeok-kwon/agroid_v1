fn main() -> std::process::ExitCode {
    match damping_action_manager::motion::runtime::run(damping_action_manager::Damping) {
        Ok(()) => std::process::ExitCode::SUCCESS,
        Err(error) => {
            eprintln!("damping_action_manager: {error}");
            std::process::ExitCode::FAILURE
        }
    }
}
