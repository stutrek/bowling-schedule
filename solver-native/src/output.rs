use crate::cpu_sa::WorkerReport;
use solver_core::{cost_label, evaluate, CostBreakdown, Weights};
use std::time::{Duration, Instant};

pub const FRESH_TABLE_INTERVAL_SECS: u64 = 600;

pub struct WorkerMeta {
    pub reseeded_at: Instant,
    pub cost_at_reseed: u32,
    pub last_report: Option<WorkerReport>,
    pub prev_iterations: u64,
    pub prev_iter_time: Instant,
    pub iters_per_sec: u64,
}

pub struct GlobalBestMeta {
    pub source: String,
    pub found_at: Instant,
}

pub fn fmt_elapsed(d: Duration) -> String {
    let secs = d.as_secs();
    format!("+{:02}:{:02}:{:02}", secs / 3600, (secs % 3600) / 60, secs % 60)
}

pub fn print_table_banner(
    global_best_cost: u32,
    global_best_bd: &CostBreakdown,
    meta: &GlobalBestMeta,
    start_time: Instant,
) {
    let now = chrono::Local::now().format("%H:%M:%S");
    let age = meta.found_at.elapsed().as_secs();
    let age_str = if age < 60 { format!("{}s ago", age) }
                  else if age < 3600 { format!("{}m{}s ago", age / 60, age % 60) }
                  else { format!("{}h{}m ago", age / 3600, (age % 3600) / 60) };
    eprintln!(
        "── {} {} best={} from {} ({}) ──\x1b[K",
        now, fmt_elapsed(start_time.elapsed()),
        global_best_cost, meta.source, age_str,
    );
    eprintln!("   {}\x1b[K", cost_label(global_best_bd));
}

pub fn print_table_header() {
    eprintln!(
        "{:>4} {:>5}  {:>5} {:>5}  {:>9} {:>9} {:>9} {:>9} {:>9} {:>9} {:>9} {:>9} {:>9}  {:>6}  {}\x1b[K",
        "src", "temp", "cur", "best",
        "match", "consec", "el_bal", "el_alt", "lane", "switch", "ll_bal", "comm", "hs_rpt",
        "it/s", "state",
    );
}

pub fn print_cpu_row(
    core_id: usize,
    report: &WorkerReport,
    w8: &Weights,
    meta: &WorkerMeta,
    temp: f64,
) {
    let cur_bd = evaluate(&report.current_assignment, w8);
    let best_bd = evaluate(&report.best_assignment, w8);
    let since = meta.reseeded_at.elapsed().as_secs();
    let state = if since < 30 { format!("shook+{}s", since) } else { "normal".to_string() };
    let ips = if meta.iters_per_sec > 0 {
        format!("{}k", meta.iters_per_sec / 1000)
    } else {
        "-".to_string()
    };
    eprintln!(
        "cpu{:<1} {:>5.1}  {:>5} {:>5}  {:>4}/{:<4} {:>4}/{:<4} {:>4}/{:<4} {:>4}/{:<4} {:>4}/{:<4} {:>4}/{:<4} {:>4}/{:<4} {:>4}/{:<4} {:>4}/{:<4}  {:>6}  {}\x1b[K",
        core_id, temp,
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
        ips,
        state,
    );
}

pub fn print_gpu_row(
    gpu_best_cost: u32,
    gpu_median: u32,
    best_bd: &CostBreakdown,
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

pub fn format_event(elapsed: Duration, msg: &str) -> String {
    format!("{:>9}  >>>  {}", fmt_elapsed(elapsed), msg)
}
