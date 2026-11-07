fn main() -> std::process::ExitCode {
    match zero_moment_action_manager::motion::runtime::run(zero_moment_action_manager::ZeroMoment) {
        Ok(()) => std::process::ExitCode::SUCCESS,
        Err(error) => {
            eprintln!("zero_moment_action_manager: {error}");
            std::process::ExitCode::FAILURE
        }
    }
}
