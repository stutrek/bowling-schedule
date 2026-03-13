use crate::cpu_sa_winter_fixed::WinterFixedWorkerReport;
use crate::output::{GlobalBestMeta, fmt_elapsed, fmt_ips};
pub use crate::output::{HistogramOutput, build_histogram};
use solver_core::winter_fixed::{evaluate_fixed, fixed_cost_label, WinterFixedCostBreakdown, WinterFixedWeights};
use std::time::Instant;

pub struct WinterFixedWorkerMeta {
    pub last_report: Option<WinterFixedWorkerReport>,
    pub prev_iterations: u64,
    pub prev_iter_time: Instant,
    pub iters_per_sec: u64,
    pub best_found_at: Instant,
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
        "{:>4} {:>9}  {:>5} {:>5}  {:>5} {:>5} {:>5} {:>5} {:>5} {:>5} {:>5} {:>5} {:>5} {:>5}  {}\x1b[K",
        "src", "temp", "best", "worst",
        "match", "consec", "el_bal", "el_alt", "el_con", "lane", "switch", "ll_bal", "comm", "hs_rpt",
        "state",
    );
}

pub fn print_fixed_cpu_row(
    core_id: usize,
    report: &WinterFixedWorkerReport,
    w8: &WinterFixedWeights,
    meta: &WinterFixedWorkerMeta,
    info_col: &str,
    best_cost: u32,
    worst_cost: u32,
) {
    let best_bd = evaluate_fixed(&report.best_schedule, w8);
    let best_age = meta.best_found_at.elapsed().as_secs();
    let bold = if best_age < 10 { "\x1b[1m" } else { "" };
    let reset = if best_age < 10 { "\x1b[0m" } else { "" };
    let temp_str = info_col;
    let state = if report.sweep_round > 0 {
        format!("sweep-{}", report.sweep_round)
    } else {
        "normal".to_string()
    };
    eprintln!(
        "{}cpu{:<1} {:>9}  {:>5} {:>5}  {:>5} {:>5} {:>5} {:>5} {:>5} {:>5} {:>5} {:>5} {:>5} {:>5}  {}{}\x1b[K",
        bold,
        core_id, temp_str,
        best_cost, worst_cost,
        best_bd.matchup_balance,
        best_bd.consecutive_opponents,
        best_bd.early_late_balance,
        best_bd.early_late_alternation,
        best_bd.early_late_consecutive,
        best_bd.lane_balance,
        best_bd.lane_switch_balance,
        best_bd.late_lane_balance,
        best_bd.commissioner_overlap,
        best_bd.half_season_repeat,
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
        "gpu   {:>5}  ~{:<4} {:>5}  {:>5} {:>5} {:>5} {:>5} {:>5} {:>5} {:>5} {:>5} {:>5} {:>5}  {:>6}\x1b[K",
        "-", gpu_median, gpu_best_cost,
        best_bd.matchup_balance,
        best_bd.consecutive_opponents,
        best_bd.early_late_balance,
        best_bd.early_late_alternation,
        best_bd.early_late_consecutive,
        best_bd.lane_balance,
        best_bd.lane_switch_balance,
        best_bd.late_lane_balance,
        best_bd.commissioner_overlap,
        best_bd.half_season_repeat,
        ips,
    );
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn histogram_rows_are_fixed_width() {
        let costs: Vec<u32> = (1000..2000).collect();
        let hist = build_histogram(&costs, 140);
        for row in &hist.rows {
            assert_eq!(row.chars().count(), 140, "row char count should be 140");
        }
        assert_eq!(hist.legend.len(), 140, "legend byte len should be 140 (ascii)");
    }

    #[test]
    fn histogram_stable_within_boundary() {
        // Two frames where min/max stay within the same 500-boundaries
        let costs_a: Vec<u32> = (1000..1400).collect();
        let mut costs_b: Vec<u32> = (1000..1400).collect();
        costs_b.push(1450); // still within same range_hi (1500)
        let hist_a = build_histogram(&costs_a, 140);
        let hist_b = build_histogram(&costs_b, 140);
        assert_eq!(hist_a.legend, hist_b.legend, "legend should be stable within same 500-boundary");
    }

    #[test]
    fn histogram_shifts_on_boundary_crossing() {
        // Crossing a 500-boundary should change the range
        let costs_a: Vec<u32> = (1000..1499).collect();
        let costs_b: Vec<u32> = (1000..1501).collect();
        let hist_a = build_histogram(&costs_a, 140);
        let hist_b = build_histogram(&costs_b, 140);
        assert_ne!(hist_a.legend, hist_b.legend, "crossing 500-boundary should change legend");
    }

    #[test]
    fn histogram_skips_sentinels() {
        let costs = vec![0, 1000, 1500, u32::MAX, 1_000_000, 2000];
        let hist = build_histogram(&costs, 20);
        // n label (row 1) should show 3 (only 1000, 1500, 2000 counted)
        assert_eq!(hist.labels[1].trim(), "3");
    }

    #[test]
    fn histogram_all_sentinels() {
        let costs = vec![0, u32::MAX, 1_000_000];
        let hist = build_histogram(&costs, 20);
        // Should not panic, n=0
        assert_eq!(hist.labels[1].trim(), "0");
    }

    #[test]
    fn histogram_single_value() {
        let costs = vec![1500; 100];
        let hist = build_histogram(&costs, 40);
        assert_eq!(hist.rows[0].chars().count(), 40);
        // All data in one bucket — bottom row should have exactly one non-space column
        let non_space: usize = hist.rows[2].chars().filter(|&c| c != ' ').count();
        assert!(non_space >= 1, "should have at least one bar column");
    }

    #[test]
    fn histogram_max_bucket_is_full_height() {
        let costs = vec![1500; 100];
        let hist = build_histogram(&costs, 40);
        // The peak bucket should be full height (█ in all 3 rows)
        assert!(hist.rows[0].contains('█'), "top row should have full block at peak");
        assert!(hist.rows[1].contains('█'), "mid row should have full block at peak");
        assert!(hist.rows[2].contains('█'), "bottom row should have full block at peak");
    }

    #[test]
    fn histogram_legend_starts_with_range_lo() {
        let costs: Vec<u32> = (2000..3000).collect();
        let hist = build_histogram(&costs, 140);
        assert!(hist.legend.starts_with("2000"), "legend should start with range_lo=2000, got: {}", &hist.legend[..20]);
    }
}
