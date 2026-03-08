use crate::cpu_sa_summer_fixed::FixedWorkerReport;
use crate::output::{GlobalBestMeta, fmt_elapsed};
use solver_core::summer_fixed::{evaluate_fixed, fixed_cost_label, FixedWeights};
use std::time::Instant;

pub struct FixedWorkerMeta {
    pub reseeded_at: Instant,
    pub cost_at_reseed: u32,
    pub last_report: Option<FixedWorkerReport>,
    pub prev_iterations: u64,
    pub prev_iter_time: Instant,
    pub iters_per_sec: u64,
}

fn fmt_ips(ips: u64) -> String {
    if ips >= 1_000_000_000 {
        format!("{:.2}B", ips as f64 / 1_000_000_000.0)
    } else if ips >= 1_000_000 {
        format!("{:.1}M", ips as f64 / 1_000_000.0)
    } else if ips >= 1_000 {
        format!("{:.0}K", ips as f64 / 1_000.0)
    } else {
        format!("{}", ips)
    }
}

pub fn print_fixed_table_banner(
    global_best_cost: u32,
    w8: &FixedWeights,
    best_sched: &solver_core::summer_fixed::FixedSchedule,
    meta: &GlobalBestMeta,
    start_time: Instant,
    gpu_ips: u64,
    cpu_ips: u64,
) {
    let now = chrono::Local::now().format("%H:%M:%S");
    let age = meta.found_at.elapsed().as_secs();
    let age_str = if age < 60 { format!("{}s ago", age) }
                  else if age < 3600 { format!("{}m{}s ago", age / 60, age % 60) }
                  else { format!("{}h{}m ago", age / 3600, (age % 3600) / 60) };
    eprintln!(
        "── {} {} best={} from {} ({}) gpu:{} cpu:{} it/s ──\x1b[K",
        now, fmt_elapsed(start_time.elapsed()),
        global_best_cost, meta.source, age_str,
        fmt_ips(gpu_ips), fmt_ips(cpu_ips),
    );
    let bd = evaluate_fixed(best_sched, w8);
    eprintln!("   {}\x1b[K", fixed_cost_label(&bd));
}

pub fn print_fixed_table_header() {
    eprintln!(
        "{:>4} {:>5}  {:>5} {:>5}  {:>9} {:>9} {:>9} {:>9} {:>9} {:>9}  {:>6}  {}\x1b[K",
        "src", "temp", "cur", "best",
        "matchup", "slot", "lane", "g5lane", "same", "comm",
        "stag", "state",
    );
}

pub fn print_fixed_cpu_row(
    core_id: usize,
    report: &FixedWorkerReport,
    w8: &FixedWeights,
    meta: &FixedWorkerMeta,
    temp: f64,
    _stagnation: u64,
) {
    let cur_bd = evaluate_fixed(&report.current_sched, w8);
    let best_bd = evaluate_fixed(&report.best_sched, w8);
    let since = meta.reseeded_at.elapsed().as_secs();
    let state = if since < 30 { format!("shook+{}s", since) } else { "normal".to_string() };
    let bold = if since < 30 { "\x1b[1m" } else { "" };
    let reset = if since < 30 { "\x1b[0m" } else { "" };
    let stag_k = report.iterations_since_improve / 1000;
    eprintln!(
        "{}cpu{:<1} {:>5.1}  {:>5} {:>5}  {:>4}/{:<4} {:>4}/{:<4} {:>4}/{:<4} {:>4}/{:<4} {:>4}/{:<4} {:>4}/{:<4}  {:>5}k  {}{}\x1b[K",
        bold,
        core_id, temp,
        cur_bd.total, best_bd.total,
        cur_bd.matchup_balance, best_bd.matchup_balance,
        cur_bd.slot_balance, best_bd.slot_balance,
        cur_bd.lane_balance, best_bd.lane_balance,
        cur_bd.game5_lane_balance, best_bd.game5_lane_balance,
        cur_bd.same_lane_balance, best_bd.same_lane_balance,
        cur_bd.commissioner_overlap, best_bd.commissioner_overlap,
        stag_k,
        state, reset,
    );
}
