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

pub struct HistogramOutput {
    pub rows: [String; 3],      // top, mid, bottom bar rows
    pub labels: [String; 3],    // right-side labels per row
    pub legend: String,          // x-axis legend
}

pub fn build_histogram(costs: &[u32], hist_width: usize) -> HistogramOutput {
    // Pass 1: find min/max (skip sentinel values)
    let mut min_cost = u32::MAX;
    let mut max_cost = 0u32;
    let mut n = 0usize;
    for &c in costs {
        if c == 0 || c >= 1_000_000 { continue; }
        if c < min_cost { min_cost = c; }
        if c > max_cost { max_cost = c; }
        n += 1;
    }
    if min_cost > max_cost { min_cost = 0; max_cost = 1; }
    let range_lo = (min_cost / 500) * 500;
    let range_hi = ((max_cost / 500) + 1) * 500;
    let range = (range_hi - range_lo).max(1);
    // Integer bucket width: every bucket covers exactly `bucket_width` cost units
    let bucket_width = (range + hist_width as u32 - 1) / hist_width as u32; // ceiling division
    let bucket_width = bucket_width.max(1);

    // Pass 2: bucket using integer division (no fractional aliasing)
    let mut buckets = vec![0u32; hist_width];
    for &c in costs {
        if c == 0 || c >= 1_000_000 { continue; }
        let b = ((c - range_lo) / bucket_width) as usize;
        buckets[b.min(hist_width - 1)] += 1;
    }

    // Use max of self and neighbors so single-gap stripes fill in,
    // but real spikes are preserved (max never reduces a peak)
    let mut display_counts = vec![0u32; hist_width];
    for i in 0..hist_width {
        let left = if i > 0 { buckets[i - 1] } else { 0 };
        let right = if i + 1 < hist_width { buckets[i + 1] } else { 0 };
        display_counts[i] = buckets[i].max(left.min(right));
    }
    let max_count = *display_counts.iter().max().unwrap_or(&1).max(&1);
    let bars = [" ", "▁", "▂", "▃", "▄", "▅", "▆", "▇", "█"];
    let total_levels: u32 = 24;
    let heights: Vec<u32> = display_counts.iter().map(|&c| {
        if c == 0 { 0 }
        else { ((c as f64 / max_count as f64) * total_levels as f64).ceil() as u32 }
    }).collect();

    let mut rows = [String::new(), String::new(), String::new()];
    let mut labels = [String::new(), String::new(), String::new()];
    for row in 0..3u32 {
        let row_base = (2 - row) * 8;
        rows[row as usize] = heights.iter().map(|&h| {
            if h <= row_base { bars[0] }
            else if h >= row_base + 8 { bars[8] }
            else { bars[(h - row_base) as usize] }
        }).collect();
        labels[row as usize] = if row == 0 {
            format!("{:>5}", max_count)
        } else if row == 1 {
            format!("{:>5}", n)
        } else {
            "     ".to_string()
        };
    }

    let mut legend = String::with_capacity(hist_width + 20);
    let mut pos = 0usize;
    // Pick label step to get ~5 labels: try 100, 250, 500, 1000, 2500, 5000...
    let total_range = hist_width as u32 * bucket_width;
    let label_step = [100u32, 250, 500, 1000, 2500, 5000, 10000]
        .iter()
        .copied()
        .find(|&s| total_range / s <= 6)
        .unwrap_or(10000);
    // Start with range_lo label
    let fmt_cost = |c: u32| -> String {
        if c >= 10000 { format!("{}k", c / 1000) } else { format!("{}", c) }
    };
    let first_label = fmt_cost(range_lo);
    legend.push_str(&first_label);
    pos += first_label.len();
    // Place labels at multiples of label_step
    let mut cost = ((range_lo / label_step) + 1) * label_step;
    while cost < range_lo + total_range {
        let bi = ((cost - range_lo) / bucket_width) as usize;
        if bi < hist_width && pos + 2 <= bi {
            while pos < bi { legend.push('-'); pos += 1; }
            let label = fmt_cost(cost);
            if pos + label.len() <= hist_width {
                legend.push_str(&label);
                pos += label.len();
            }
        }
        cost += label_step;
    }
    while pos < hist_width { legend.push('-'); pos += 1; }
    legend.truncate(hist_width);

    HistogramOutput { rows, labels, legend }
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
