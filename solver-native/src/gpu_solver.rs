use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;

fn main() {
    let shutdown = Arc::new(AtomicBool::new(false));

    // Use nix to install SIGINT handler without SA_RESTART, so blocked syscalls
    // (Metal IOKit, thread::sleep) get interrupted and return EINTR.
    {
        use nix::sys::signal;
        static SIGINT_RECEIVED: AtomicBool = AtomicBool::new(false);
        extern "C" fn sigint_handler(_: nix::libc::c_int) {
            SIGINT_RECEIVED.store(true, Ordering::SeqCst);
        }
        let action = signal::SigAction::new(
            signal::SigHandler::Handler(sigint_handler),
            signal::SaFlags::empty(), // no SA_RESTART
            signal::SigSet::empty(),
        );
        unsafe { signal::sigaction(signal::Signal::SIGINT, &action).ok(); }

        // Bridge thread: poll the static flag and set the Arc
        let s = Arc::clone(&shutdown);
        std::thread::spawn(move || {
            // Wait for first ctrl+c
            loop {
                if SIGINT_RECEIVED.load(Ordering::SeqCst) { break; }
                std::thread::sleep(std::time::Duration::from_millis(50));
            }
            eprintln!("\nShutting down...");
            s.store(true, Ordering::SeqCst);
            // Reset and wait for second ctrl+c to force exit
            SIGINT_RECEIVED.store(false, Ordering::SeqCst);
            loop {
                if SIGINT_RECEIVED.load(Ordering::SeqCst) {
                    eprintln!("\nForce exit");
                    std::process::exit(1);
                }
                std::thread::sleep(std::time::Duration::from_millis(50));
            }
        });
    }

    let args: Vec<String> = std::env::args().collect();

    let league = args.iter()
        .position(|a| a == "--league")
        .and_then(|i| args.get(i + 1))
        .map(|s| s.as_str())
        .unwrap_or("winter");

    match league {
        "winter" => solver_native::winter_main::run(shutdown, &args),
        "summer" => solver_native::summer_main::run(shutdown, &args),
        "summer-fixed" => solver_native::summer_fixed_main::run(shutdown, &args),
        "winter-fixed" => solver_native::winter_fixed_main::run(shutdown, &args),
        other => {
            eprintln!("Unknown league: {}. Use --league winter, summer, summer-fixed, or winter-fixed", other);
            std::process::exit(1);
        }
    }
}
