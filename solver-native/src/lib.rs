pub use solver_core::*;

use chrono::Local;
use std::fs;

pub fn now_iso() -> String {
    Local::now().format("%Y-%m-%dT%H:%M:%S%:z").to_string()
}

pub fn new_generation_dir(base: &str) -> String {
    let ts = Local::now().format("%Y-%m-%dT%H%M%S%.3f");
    let dir = format!("{}/gen-{}", base, ts);
    fs::create_dir_all(&dir).expect("Failed to create generation directory");
    dir
}

pub fn complete_dir(base: &str) -> String {
    let dir = format!("{}/complete", base);
    fs::create_dir_all(&dir).expect("Failed to create complete directory");
    dir
}

pub fn save_assignment(
    results_dir: &str,
    prefix: &str,
    key: u32,
    a: &Assignment,
    c: &CostBreakdown,
    last_saved: &mut Option<Assignment>,
) -> bool {
    if last_saved.as_ref() == Some(a) {
        return false;
    }
    let ts = Local::now().format("%Y%m%d-%H%M%S%z");
    let filename = format!("{}/{:04}-{}-{}.tsv", results_dir, key, prefix, ts);
    let tsv = assignment_to_tsv(a);
    let _ = fs::write(&filename, &tsv);
    eprintln!("[{}] Saved {} ({})", now_iso(), filename, cost_label(c));
    *last_saved = Some(*a);
    true
}
