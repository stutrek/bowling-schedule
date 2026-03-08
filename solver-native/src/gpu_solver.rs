use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;

fn main() {
    let shutdown = Arc::new(AtomicBool::new(false));
    {
        let s = Arc::clone(&shutdown);
        ctrlc::set_handler(move || {
            eprintln!("\nShutting down...");
            s.store(true, Ordering::Relaxed);
        })
        .expect("Error setting Ctrl-C handler");
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
        other => {
            eprintln!("Unknown league: {}. Use --league winter, summer, or summer-fixed", other);
            std::process::exit(1);
        }
    }
}
