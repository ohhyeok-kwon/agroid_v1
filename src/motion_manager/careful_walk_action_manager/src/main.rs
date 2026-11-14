fn main() {
    if let Err(error) = walking_action_manager::policy_runtime::run(
        careful_walk_action_manager::CarefulWalk::load(),
    ) {
        eprintln!("careful_walk_action_manager: {error}");
        std::process::exit(1);
    }
}
