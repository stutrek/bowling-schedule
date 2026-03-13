use std::time::{Duration, Instant};

pub const FRESH_TABLE_INTERVAL_SECS: u64 = 600;

pub struct GlobalBestMeta {
    pub source: String,
    pub found_at: Instant,
}

pub fn fmt_elapsed(d: Duration) -> String {
    let secs = d.as_secs();
    format!("+{:02}:{:02}:{:02}", secs / 3600, (secs % 3600) / 60, secs % 60)
}

pub fn format_event(elapsed: Duration, msg: &str) -> String {
    format!("{:>9}  >>>  {}", fmt_elapsed(elapsed), msg)
}

pub fn fmt_ips(ips: u64) -> String {
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

pub struct HistogramOutput {
    pub rows: [String; 3],
    pub labels: [String; 3],
    pub legend: String,
}

pub fn build_histogram(costs: &[u32], hist_width: usize) -> HistogramOutput {
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
    let bucket_width = (range + hist_width as u32 - 1) / hist_width as u32;
    let bucket_width = bucket_width.max(1);

    let mut buckets = vec![0u32; hist_width];
    for &c in costs {
        if c == 0 || c >= 1_000_000 { continue; }
        let b = ((c - range_lo) / bucket_width) as usize;
        buckets[b.min(hist_width - 1)] += 1;
    }

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
    let total_range = hist_width as u32 * bucket_width;
    let label_step = [100u32, 250, 500, 1000, 2500, 5000, 10000]
        .iter()
        .copied()
        .find(|&s| total_range / s <= 6)
        .unwrap_or(10000);
    let fmt_cost = |c: u32| -> String {
        if c >= 10000 { format!("{}k", c / 1000) } else { format!("{}", c) }
    };
    let first_label = fmt_cost(range_lo);
    legend.push_str(&first_label);
    pos += first_label.len();
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
