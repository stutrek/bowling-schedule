use crate::cpu_sa_summer::SummerWorkerReport;
use crate::output::{GlobalBestMeta, fmt_elapsed};
use solver_core::summer::{evaluate_summer, summer_cost_label, SummerCostBreakdown, SummerWeights};
use std::time::Instant;

pub struct SummerWorkerMeta {
    pub reseeded_at: Instant,
    pub cost_at_reseed: u32,
    pub last_report: Option<SummerWorkerReport>,
    pub prev_iterations: u64,
    pub prev_iter_time: Instant,
    pub iters_per_sec: u64,
}

pub fn print_summer_table_banner(
    global_best_cost: u32,
    global_best_bd: &SummerCostBreakdown,
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
    eprintln!("   {}\x1b[K", summer_cost_label(global_best_bd));
}

pub fn print_summer_table_header() {
    eprintln!(
        "{:>4} {:>5}  {:>5} {:>5}  {:>9} {:>9} {:>9} {:>9} {:>9} {:>9} {:>9}  {:>6}  {}\x1b[K",
        "src", "temp", "cur", "best",
        "match", "lane_sw", "gaps", "lane", "comm", "repeat", "slot",
        "stag", "state",
    );
}

pub fn print_summer_cpu_row(
    core_id: usize,
    report: &SummerWorkerReport,
    w8: &SummerWeights,
    meta: &SummerWorkerMeta,
    temp: f64,
    stagnation: u64,
) {
    let cur_bd = evaluate_summer(&report.current_assignment, w8);
    let best_bd = evaluate_summer(&report.best_assignment, w8);
    let since = meta.reseeded_at.elapsed().as_secs();
    let state = if since < 30 { format!("shook+{}s", since) } else { "normal".to_string() };
    let bold = if since < 30 { "\x1b[1m" } else { "" };
    let reset = if since < 30 { "\x1b[0m" } else { "" };
    eprintln!(
        "{}cpu{:<1} {:>5.1}  {:>5} {:>5}  {:>4}/{:<4} {:>4}/{:<4} {:>4}/{:<4} {:>4}/{:<4} {:>4}/{:<4} {:>4}/{:<4} {:>4}/{:<4}  {:>6}  {}{}\x1b[K",
        bold,
        core_id, temp,
        cur_bd.total, best_bd.total,
        cur_bd.matchup_balance, best_bd.matchup_balance,
        cur_bd.lane_switches, best_bd.lane_switches,
        cur_bd.time_gaps, best_bd.time_gaps,
        cur_bd.lane_balance, best_bd.lane_balance,
        cur_bd.commissioner_overlap, best_bd.commissioner_overlap,
        cur_bd.repeat_matchup_same_night, best_bd.repeat_matchup_same_night,
        cur_bd.slot_balance, best_bd.slot_balance,
        stagnation,
        state, reset,
    );
}
