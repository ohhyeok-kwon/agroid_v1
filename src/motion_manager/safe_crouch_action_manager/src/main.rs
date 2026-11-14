fn main() {
    if let Err(error) =
        walking_action_manager::policy_runtime::run(safe_crouch_action_manager::SafeCrouch::load())
    {
        eprintln!("safe_crouch_action_manager: {error}");
        std::process::exit(1);
    }
}
