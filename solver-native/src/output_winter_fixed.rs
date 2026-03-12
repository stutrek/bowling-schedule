use crate::cpu_sa_winter_fixed::WinterFixedWorkerReport;
use crate::output::{GlobalBestMeta, fmt_elapsed};
use solver_core::winter_fixed::{evaluate_fixed, fixed_cost_label, WinterFixedCostBreakdown, WinterFixedWeights};
use std::time::Instant;

pub struct WinterFixedWorkerMeta {
    pub last_report: Option<WinterFixedWorkerReport>,
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
    global_best_bd: &WinterFixedCostBreakdown,
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
    eprintln!("   {}\x1b[K", fixed_cost_label(global_best_bd));
}

pub fn print_fixed_table_header() {
    eprintln!(
        "{:>4} {:>9}  {:>5} {:>5}  {:>9} {:>9} {:>9} {:>9} {:>9} {:>9} {:>9} {:>9} {:>9}  {}\x1b[K",
        "src", "temp", "cur", "best",
        "match", "consec", "el_bal", "el_alt", "lane", "switch", "ll_bal", "comm", "hs_rpt",
        "state",
    );
}

pub fn print_fixed_cpu_row(
    core_id: usize,
    report: &WinterFixedWorkerReport,
    w8: &WinterFixedWeights,
    meta: &WinterFixedWorkerMeta,
    initial_temp: f64,
) {
    let cur_bd = evaluate_fixed(&report.current_schedule, w8);
    let best_bd = evaluate_fixed(&report.best_schedule, w8);
    let best_age = meta.best_found_at.elapsed().as_secs();
    let bold = if best_age < 10 { "\x1b[1m" } else { "" };
    let reset = if best_age < 10 { "\x1b[0m" } else { "" };
    let temp_str = format!("{:.0}/{:.1}", initial_temp, report.current_temp);
    let state = if report.sweep_round > 0 {
        format!("sweep-{}", report.sweep_round)
    } else {
        "normal".to_string()
    };
    eprintln!(
        "{}cpu{:<1} {:>9}  {:>5} {:>5}  {:>4}/{:<4} {:>4}/{:<4} {:>4}/{:<4} {:>4}/{:<4} {:>4}/{:<4} {:>4}/{:<4} {:>4}/{:<4} {:>4}/{:<4} {:>4}/{:<4}  {}{}\x1b[K",
        bold,
        core_id, temp_str,
        cur_bd.total, best_bd.total,
        cur_bd.matchup_balance, best_bd.matchup_balance,
        cur_bd.consecutive_opponents, best_bd.consecutive_opponents,
        cur_bd.early_late_balance, best_bd.early_late_balance,
        cur_bd.early_late_alternation, best_bd.early_late_alternation,
        cur_bd.lane_balance, best_bd.lane_balance,
        cur_bd.lane_switch_balance, best_bd.lane_switch_balance,
        cur_bd.late_lane_balance, best_bd.late_lane_balance,
        cur_bd.commissioner_overlap, best_bd.commissioner_overlap,
        cur_bd.half_season_repeat, best_bd.half_season_repeat,
        state, reset,
    );
}

pub fn print_fixed_gpu_row(
    gpu_best_cost: u32,
    gpu_median: u32,
    best_bd: &WinterFixedCostBreakdown,
    gpu_ips: u64,
) {
    let ips = if gpu_ips > 0 {
        format!("{}M", gpu_ips / 1_000_000)
    } else {
        "-".to_string()
    };
    eprintln!(
        "gpu   {:>5}  ~{:<4} {:>5}  {:>4}/{:<4} {:>4}/{:<4} {:>4}/{:<4} {:>4}/{:<4} {:>4}/{:<4} {:>4}/{:<4} {:>4}/{:<4} {:>4}/{:<4} {:>4}/{:<4}  {:>6}\x1b[K",
        "-", gpu_median, gpu_best_cost,
        "-", best_bd.matchup_balance,
        "-", best_bd.consecutive_opponents,
        "-", best_bd.early_late_balance,
        "-", best_bd.early_late_alternation,
        "-", best_bd.lane_balance,
        "-", best_bd.lane_switch_balance,
        "-", best_bd.late_lane_balance,
        "-", best_bd.commissioner_overlap,
        "-", best_bd.half_season_repeat,
        ips,
    );
}
