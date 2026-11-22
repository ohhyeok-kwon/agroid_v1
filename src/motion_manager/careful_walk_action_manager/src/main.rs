fn main() {
    // Before any other thread exists: every thread of the node inherits the CPU.
    let bound = careful_walk_action_manager::bind_cpu();
    let cpu = bound.clone().unwrap_or_else(|error| error);
    if std::env::args().nth(1).as_deref() == Some("--self-test") {
        // 250 ticks = 5 s at the control cadence. A binding that was due and failed fails the test.
        match careful_walk_action_manager::self_test(250) {
            Ok(report) if bound.is_ok() => println!("{report}; {cpu}"),
            Ok(report) => {
                eprintln!("careful_walk_action_manager: FAILED {cpu}; {report}");
                std::process::exit(1);
            }
            Err(error) => {
                eprintln!("careful_walk_action_manager: {error}; {cpu}");
                std::process::exit(1);
            }
        }
        return;
    }
    eprintln!("careful_walk_action_manager: {cpu}");
    // Without its CPU the node stays up and reports not ready, so the walk cannot be entered.
    let policy = bound.and_then(|_| careful_walk_action_manager::CarefulWalk::load());
    if let Err(error) = walking_action_manager::policy_runtime::run(policy) {
        eprintln!("careful_walk_action_manager: {error}");
        std::process::exit(1);
    }
}
