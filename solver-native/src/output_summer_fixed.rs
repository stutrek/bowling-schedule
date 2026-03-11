use crate::cpu_sa_summer_fixed::FixedWorkerReport;
use crate::output::{GlobalBestMeta, fmt_elapsed};
use solver_core::summer_fixed::{evaluate_fixed, fixed_cost_label, FixedWeights};
use std::time::Instant;

pub struct FixedWorkerMeta {
    pub last_report: Option<FixedWorkerReport>,
    pub prev_iterations: u64,
    pub prev_iter_time: Instant,
    pub iters_per_sec: u64,
    pub best_found_at: Instant,
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
        "{:>4} {:>9}  {:>5} {:>5}  {:>9} {:>9} {:>9} {:>9} {:>9} {:>9} {:>9}  {}\x1b[K",
        "src", "temp", "cur", "best",
        "matchup", "slot", "lane", "g5lane", "comm", "spacing", "break",
        "state",
    );
}

pub fn print_fixed_cpu_row(
    core_id: usize,
    report: &FixedWorkerReport,
    w8: &FixedWeights,
    meta: &FixedWorkerMeta,
    initial_temp: f64,
) {
    let cur_bd = evaluate_fixed(&report.current_sched, w8);
    let best_bd = evaluate_fixed(&report.best_sched, w8);
    let state = if let Some((cur, total)) = report.sweep_round {
            format!("SWEEP {}/{}", cur, total)
        } else { "normal".to_string() };
    let best_age = meta.best_found_at.elapsed().as_secs();
    let bold = if best_age < 10 { "\x1b[1m" } else { "" };
    let reset = if best_age < 10 { "\x1b[0m" } else { "" };
    let temp_str = format!("{:.0}/{:.1}", initial_temp, report.current_temp);
    eprintln!(
        "{}cpu{:<1} {:>9}  {:>5} {:>5}  {:>4}/{:<4} {:>4}/{:<4} {:>4}/{:<4} {:>4}/{:<4} {:>4}/{:<4} {:>4}/{:<4} {:>4}/{:<4}  {}{}\x1b[K",
        bold,
        core_id, temp_str,
        cur_bd.total, best_bd.total,
        cur_bd.matchup_balance, best_bd.matchup_balance,
        cur_bd.slot_balance, best_bd.slot_balance,
        cur_bd.lane_balance, best_bd.lane_balance,
        cur_bd.game5_lane_balance, best_bd.game5_lane_balance,
        cur_bd.commissioner_overlap, best_bd.commissioner_overlap,
        cur_bd.matchup_spacing, best_bd.matchup_spacing,
        cur_bd.break_balance, best_bd.break_balance,
        state, reset,
    );
}


