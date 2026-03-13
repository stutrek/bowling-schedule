use crate::cpu_sa_winter_fixed::WinterFixedWorkerReport;
use crate::island_pool::{IslandPoolStats, NUM_ISLANDS};
use crate::output::{GlobalBestMeta, fmt_elapsed, fmt_ips};
use solver_core::winter_fixed::{fixed_cost_label, WinterFixedCostBreakdown, NUM_MOVES, MOVE_NAMES};
use std::time::Instant;

pub struct EliteWorkerMeta {
    pub last_report: Option<WinterFixedWorkerReport>,
    pub prev_iterations: u64,
    pub prev_iter_time: Instant,
    pub iters_per_sec: u64,
    pub best_found_at: Instant,
}

pub enum EliteWorkerState {
    Idle,
    Refining {
        island_idx: usize,
        start_iters: u64,
        start_cost: u32,
    },
}

impl EliteWorkerState {
    pub fn island_idx(&self) -> Option<usize> {
        match self {
            EliteWorkerState::Refining { island_idx, .. } => Some(*island_idx),
            EliteWorkerState::Idle => None,
        }
    }
}

pub fn print_elite_banner(
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

pub fn print_island_summary(stats: &IslandPoolStats, avg_d: f64) {
    eprintln!(
        "islands: {} total, {} active, {} stagnant, {} reset(dedup), {} reset(stag) | min_d={:.1} avg_d={:.1}\x1b[K",
        NUM_ISLANDS, stats.active, stats.stagnant,
        stats.dedup_resets, stats.stagnation_resets,
        stats.min_distance, avg_d,
    );
}

pub fn print_elite_table_header() {
    eprintln!(
        "{:>4} {:>6}  {:>5} {:>5} {:>6}  {:>6}  {}\x1b[K",
        "src", "island", "best", "start", "delta", "iters", "state",
    );
}

pub fn print_elite_worker_row(
    core_id: usize,
    state: &EliteWorkerState,
    meta: &EliteWorkerMeta,
    refinement_iters: u64,
) {
    match state {
        EliteWorkerState::Idle => {
            eprintln!(
                "cpu{:<1} {:>6}  {:>5} {:>5} {:>6}  {:>6}  {}\x1b[K",
                core_id, "--", "--", "--", "--", "--", "idle",
            );
        }
        EliteWorkerState::Refining { island_idx, start_iters, start_cost } => {
            let (best_cost, current_iters) = meta.last_report.as_ref()
                .map(|r| (r.best_cost, r.iterations_total))
                .unwrap_or((*start_cost, *start_iters));

            let delta = best_cost as i64 - *start_cost as i64;
            let delta_str = if delta < 0 { format!("{}", delta) } else { format!("+{}", delta) };
            let cycle_iters = current_iters.saturating_sub(*start_iters);
            let pct = if refinement_iters > 0 {
                (cycle_iters as f64 / refinement_iters as f64 * 100.0).min(100.0)
            } else {
                0.0
            };
            let iters_str = if cycle_iters >= 1_000_000 {
                format!("{:.1}M", cycle_iters as f64 / 1_000_000.0)
            } else if cycle_iters >= 1_000 {
                format!("{:.0}K", cycle_iters as f64 / 1_000.0)
            } else {
                format!("{}", cycle_iters)
            };

            let best_age = meta.best_found_at.elapsed().as_secs();
            let bold = if best_age < 10 { "\x1b[1m" } else { "" };
            let reset = if best_age < 10 { "\x1b[0m" } else { "" };

            eprintln!(
                "{}cpu{:<1} {:>6}  {:>5} {:>5} {:>6}  {:>6}  refining ({:.0}%){}\x1b[K",
                bold, core_id,
                format!("#{}", island_idx),
                best_cost, *start_cost, delta_str,
                iters_str, pct, reset,
            );
        }
    }
}

pub fn print_elite_move_stats(worker_metas: &[EliteWorkerMeta]) {
    let mut avg_rates = [0.0f64; NUM_MOVES];
    let mut avg_shares = [0.0f64; NUM_MOVES];
    let mut n = 0usize;
    for meta in worker_metas {
        if let Some(ref r) = meta.last_report {
            for m in 0..NUM_MOVES {
                avg_rates[m] += r.move_rates[m];
                avg_shares[m] += r.move_shares[m];
            }
            n += 1;
        }
    }
    if n > 0 {
        let nf = n as f64;
        let header: Vec<String> = (0..NUM_MOVES).map(|m| {
            format!("{:>7}", MOVE_NAMES[m])
        }).collect();
        let values: Vec<String> = (0..NUM_MOVES).map(|m| {
            format!("{:>4.1}/{:<2.0}", avg_rates[m] / nf * 100.0, avg_shares[m] / nf * 100.0)
        }).collect();
        eprintln!("  move:     {}\x1b[K", header.join(" "));
        eprintln!("  acc%/sel%: {}\x1b[K", values.join(" "));
    }
}
