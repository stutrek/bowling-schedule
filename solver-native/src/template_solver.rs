//! SA solver to find the optimal night template for summer_fixed.
//! Maximizes 3-in-a-row consecutive games while maintaining lane consistency.
//!
//! The template defines 18 matchup entries across 5 slots:
//!   Slots 0-3: 4 lanes each (8 teams per slot)
//!   Slot 4: 2 lanes (lanes 2-3 only, 4 teams)
//! Each of 12 positions appears exactly 3 times (3 games per night).

use rand::rngs::SmallRng;
use rand::{Rng, SeedableRng};
use serde::{Deserialize, Serialize};
use std::fs;
use std::sync::atomic::{AtomicBool, AtomicU32, Ordering};
use std::sync::{mpsc, Arc};
use std::time::Instant;

const POSITIONS: usize = 12;
const SLOTS: usize = 5;
const LANES: usize = 4;
const ENTRIES: usize = 18; // 4*4 + 1*2

#[derive(Clone, Copy, Debug, Serialize, Deserialize)]
pub struct TemplateEntry {
    pub slot: u8,
    pub lane: u8,
    pub pos_a: u8,
    pub pos_b: u8,
}

/// Fixed structure: which (slot, lane) positions exist.
const ENTRY_SLOTS_LANES: [(u8, u8); ENTRIES] = [
    (0, 0), (0, 1), (0, 2), (0, 3),
    (1, 0), (1, 1), (1, 2), (1, 3),
    (2, 0), (2, 1), (2, 2), (2, 3),
    (3, 0), (3, 1), (3, 2), (3, 3),
    (4, 2), (4, 3),
];

#[derive(Clone, Deserialize)]
struct TemplateWeights {
    not_consecutive: u32,
    lane_switch_consecutive: u32,
    lane_switch_same_pair: u32,
    lane_pair_break: u32,
    time_gap_large: u32,
    repeat_matchup: u32,
}

impl Default for TemplateWeights {
    fn default() -> Self {
        TemplateWeights {
            not_consecutive: 50,
            lane_switch_consecutive: 40,
            lane_switch_same_pair: 20,
            lane_pair_break: 15,
            time_gap_large: 60,
            repeat_matchup: 200,
        }
    }
}

#[derive(Clone)]
struct Template {
    pos_a: [u8; ENTRIES],
    pos_b: [u8; ENTRIES],
}

/// Cost breakdown for display.
struct CostBreakdown {
    total: u32,
    not_consecutive: u32,
    lane_switch: u32,
    lane_pair_break: u32,
    time_gap: u32,
    repeat_matchup: u32,
    consec_count: u32,
    consec_same_lane: u32,
    consec_lane_penalized: u32,
    non_consec_penalized: u32,
}

/// For each position, collect its (slot, lane) appearances sorted by slot.
fn position_games(t: &Template) -> Vec<Vec<(u8, u8)>> {
    let mut games: Vec<Vec<(u8, u8)>> = vec![Vec::new(); POSITIONS];
    for i in 0..ENTRIES {
        let (slot, lane) = ENTRY_SLOTS_LANES[i];
        games[t.pos_a[i] as usize].push((slot, lane));
        games[t.pos_b[i] as usize].push((slot, lane));
    }
    for g in &mut games {
        g.sort_by_key(|&(s, _)| s);
    }
    games
}

fn evaluate_breakdown(t: &Template, w8: &TemplateWeights) -> CostBreakdown {
    let games = position_games(t);
    let mut bd = CostBreakdown {
        total: 0, not_consecutive: 0, lane_switch: 0,
        lane_pair_break: 0, time_gap: 0, repeat_matchup: 0, consec_count: 0,
        consec_same_lane: 0, consec_lane_penalized: 0, non_consec_penalized: 0,
    };

    // Repeat matchup penalty
    let mut pair_counts = [0u8; POSITIONS * (POSITIONS - 1) / 2];
    for i in 0..ENTRIES {
        let a = t.pos_a[i].min(t.pos_b[i]) as usize;
        let b = t.pos_a[i].max(t.pos_b[i]) as usize;
        let idx = a * (2 * POSITIONS - a - 1) / 2 + (b - a - 1);
        pair_counts[idx] += 1;
    }
    for &c in &pair_counts {
        if c > 1 {
            bd.repeat_matchup += w8.repeat_matchup * (c as u32 - 1);
        }
    }

    for p in 0..POSITIONS {
        let g = &games[p];
        if g.len() != 3 {
            bd.total = u32::MAX;
            return bd;
        }

        let slots = [g[0].0, g[1].0, g[2].0];
        let lanes = [g[0].1, g[1].1, g[2].1];

        let is_consecutive = slots[1] == slots[0] + 1 && slots[2] == slots[1] + 1;

        // Positions playing slot 4 from lanes 0-1 are forced to switch lane pairs
        // (slot 4 only has lanes 2-3). Skip lane penalties for them — only
        // penalize not_consecutive, repeat_matchup, and time_gap.
        let plays_slot4 = slots.iter().any(|&s| s == 4);
        let non_slot4_lanes: Vec<u8> = (0..3).filter(|&i| slots[i] != 4).map(|i| lanes[i]).collect();
        let forced_lane_switch = plays_slot4 && non_slot4_lanes.iter().all(|&l| l <= 1);

        if is_consecutive {
            bd.consec_count += 1;
            if lanes[0] == lanes[1] && lanes[1] == lanes[2] {
                bd.consec_same_lane += 1;
            }
        } else {
            bd.not_consecutive += w8.not_consecutive;
        }

        for i in 0..2 {
            let gap = slots[i + 1] - slots[i] - 1;
            if gap >= 2 {
                bd.time_gap += w8.time_gap_large;
            }
        }

        if !forced_lane_switch {
            if is_consecutive {
                let mut lane_pen = false;
                for i in 0..2 {
                    if lanes[i] != lanes[i + 1] {
                        lane_pen = true;
                        if lanes[i] / 2 == lanes[i + 1] / 2 {
                            bd.lane_switch += w8.lane_switch_same_pair;
                        } else {
                            bd.lane_switch += w8.lane_switch_consecutive;
                        }
                    }
                }
                if lane_pen {
                    bd.consec_lane_penalized += 1;
                }
            } else {
                // Two consecutive + one break game
                let mut pen = false;
                if slots[1] == slots[0] + 1 {
                    if lanes[0] != lanes[1] {
                        pen = true;
                        if lanes[0] / 2 == lanes[1] / 2 {
                            bd.lane_switch += w8.lane_switch_same_pair;
                        } else {
                            bd.lane_switch += w8.lane_switch_consecutive;
                        }
                    }
                    if lanes[0] / 2 != lanes[2] / 2 {
                        pen = true;
                        bd.lane_pair_break += w8.lane_pair_break;
                    }
                } else if slots[2] == slots[1] + 1 {
                    if lanes[1] != lanes[2] {
                        pen = true;
                        if lanes[1] / 2 == lanes[2] / 2 {
                            bd.lane_switch += w8.lane_switch_same_pair;
                        } else {
                            bd.lane_switch += w8.lane_switch_consecutive;
                        }
                    }
                    if lanes[1] / 2 != lanes[0] / 2 {
                        pen = true;
                        bd.lane_pair_break += w8.lane_pair_break;
                    }
                } else {
                    pen = true;
                    bd.lane_switch += w8.lane_switch_consecutive * 2;
                }
                if pen {
                    bd.non_consec_penalized += 1;
                }
            }
        }
    }

    bd.total = bd.not_consecutive + bd.lane_switch + bd.lane_pair_break + bd.time_gap + bd.repeat_matchup;
    bd
}

fn evaluate(t: &Template, w8: &TemplateWeights) -> u32 {
    evaluate_breakdown(t, w8).total
}

fn is_valid(t: &Template) -> bool {
    let mut counts = [0u8; POSITIONS];
    let mut slot_sets: [u16; SLOTS] = [0; SLOTS];

    for i in 0..ENTRIES {
        let (slot, _) = ENTRY_SLOTS_LANES[i];
        let s = slot as usize;
        let a = t.pos_a[i] as usize;
        let b = t.pos_b[i] as usize;

        if a >= POSITIONS || b >= POSITIONS || a == b { return false; }

        counts[a] += 1;
        counts[b] += 1;

        let mask_a = 1u16 << a;
        let mask_b = 1u16 << b;
        if slot_sets[s] & mask_a != 0 || slot_sets[s] & mask_b != 0 { return false; }
        slot_sets[s] |= mask_a | mask_b;
    }

    if !counts.iter().all(|&c| c == 3) {
        return false;
    }

    // Hard constraints: at least 2 consecutive games, and span <= 4
    let games = position_games(t);
    for p in 0..POSITIONS {
        let g = &games[p];
        let slots: Vec<u8> = g.iter().map(|&(s, _)| s).collect();
        let span = slots[2] - slots[0];
        if span > 4 { return false; }
        let has_consec = (slots[1] == slots[0] + 1) || (slots[2] == slots[1] + 1);
        if !has_consec { return false; }
    }

    true
}

fn random_template(rng: &mut SmallRng) -> Template {
    loop {
        let mut t = Template {
            pos_a: [0; ENTRIES],
            pos_b: [0; ENTRIES],
        };

        let mut remaining = [3u8; POSITIONS];
        let mut slot_used: [u16; SLOTS] = [0; SLOTS];

        let mut order: Vec<usize> = (0..ENTRIES).collect();
        for i in (1..order.len()).rev() {
            let j = rng.random_range(0..=i);
            order.swap(i, j);
        }

        let mut ok = true;
        for &idx in &order {
            let (slot, _) = ENTRY_SLOTS_LANES[idx];
            let s = slot as usize;

            let mut cands: Vec<u8> = (0..POSITIONS as u8)
                .filter(|&p| remaining[p as usize] > 0 && (slot_used[s] & (1 << p)) == 0)
                .collect();

            if cands.len() < 2 { ok = false; break; }

            for i in (1..cands.len()).rev() {
                let j = rng.random_range(0..=i);
                cands.swap(i, j);
            }

            t.pos_a[idx] = cands[0];
            t.pos_b[idx] = cands[1];
            remaining[cands[0] as usize] -= 1;
            remaining[cands[1] as usize] -= 1;
            slot_used[s] |= (1 << cands[0]) | (1 << cands[1]);
        }

        if ok && remaining.iter().all(|&r| r == 0) {
            debug_assert!(is_valid(&t));
            return t;
        }
    }
}

fn perturb(t: &Template, rng: &mut SmallRng) -> Option<Template> {
    let mut t2 = t.clone();
    let move_type = rng.random_range(0..3u8);

    match move_type {
        0 => {
            // Position swap: pick two entries in different slots, swap one position each
            let e1 = rng.random_range(0..ENTRIES);
            let e2 = loop {
                let e = rng.random_range(0..ENTRIES);
                if ENTRY_SLOTS_LANES[e].0 != ENTRY_SLOTS_LANES[e1].0 { break e; }
            };

            let side1 = rng.random_range(0..2u8);
            let side2 = rng.random_range(0..2u8);

            let p1 = if side1 == 0 { &mut t2.pos_a[e1] } else { &mut t2.pos_b[e1] };
            let v1 = *p1;
            let p2 = if side2 == 0 { &mut t2.pos_a[e2] } else { &mut t2.pos_b[e2] };
            let v2 = *p2;

            *p2 = v1;
            if side1 == 0 { t2.pos_a[e1] = v2; } else { t2.pos_b[e1] = v2; }
        }
        1 => {
            // Intra-slot swap: swap two positions within the same slot
            let slot = rng.random_range(0..SLOTS as u8);
            let entries_in_slot: Vec<usize> = (0..ENTRIES)
                .filter(|&i| ENTRY_SLOTS_LANES[i].0 == slot)
                .collect();
            if entries_in_slot.len() < 2 { return None; }

            let i1 = entries_in_slot[rng.random_range(0..entries_in_slot.len())];
            let i2 = loop {
                let i = entries_in_slot[rng.random_range(0..entries_in_slot.len())];
                if i != i1 { break i; }
            };

            let side1 = rng.random_range(0..2u8);
            let side2 = rng.random_range(0..2u8);

            let v1 = if side1 == 0 { t2.pos_a[i1] } else { t2.pos_b[i1] };
            let v2 = if side2 == 0 { t2.pos_a[i2] } else { t2.pos_b[i2] };

            if side1 == 0 { t2.pos_a[i1] = v2; } else { t2.pos_b[i1] = v2; }
            if side2 == 0 { t2.pos_a[i2] = v1; } else { t2.pos_b[i2] = v1; }
        }
        2 => {
            // Lane reassign: swap two entries' matchups between lanes in same slot
            let slot = rng.random_range(0..SLOTS as u8);
            let entries_in_slot: Vec<usize> = (0..ENTRIES)
                .filter(|&i| ENTRY_SLOTS_LANES[i].0 == slot)
                .collect();
            if entries_in_slot.len() < 2 { return None; }

            let i1 = entries_in_slot[rng.random_range(0..entries_in_slot.len())];
            let i2 = loop {
                let i = entries_in_slot[rng.random_range(0..entries_in_slot.len())];
                if i != i1 { break i; }
            };

            let (a1, b1) = (t2.pos_a[i1], t2.pos_b[i1]);
            let (a2, b2) = (t2.pos_a[i2], t2.pos_b[i2]);
            t2.pos_a[i1] = a2;
            t2.pos_b[i1] = b2;
            t2.pos_a[i2] = a1;
            t2.pos_b[i2] = b1;
        }
        _ => unreachable!(),
    }

    if is_valid(&t2) { Some(t2) } else { None }
}

fn template_to_entries(t: &Template) -> Vec<TemplateEntry> {
    (0..ENTRIES).map(|i| {
        let (slot, lane) = ENTRY_SLOTS_LANES[i];
        TemplateEntry { slot, lane, pos_a: t.pos_a[i], pos_b: t.pos_b[i] }
    }).collect()
}

fn template_to_tsv(t: &Template) -> String {
    let mut lines = vec!["Slot\tLane 1\tLane 2\tLane 3\tLane 4".to_string()];
    for slot in 0..SLOTS {
        let mut lane_strs = vec!["-".to_string(); LANES];
        for i in 0..ENTRIES {
            if ENTRY_SLOTS_LANES[i].0 == slot as u8 {
                let lane = ENTRY_SLOTS_LANES[i].1 as usize;
                lane_strs[lane] = format!("{} v {}", t.pos_a[i] + 1, t.pos_b[i] + 1);
            }
        }
        lines.push(format!("{}\t{}", slot + 1, lane_strs.join("\t")));
    }
    lines.join("\n") + "\n"
}

fn cost_label(bd: &CostBreakdown) -> String {
    let non_consec = 12 - bd.consec_count;
    let consec_same = bd.consec_count - bd.consec_lane_penalized;
    format!(
        "{} consec ({} same lane, {} diff lane) | {} non-consec ({} diff lane) | gap={} repeat={}",
        bd.consec_count, consec_same, bd.consec_lane_penalized,
        non_consec, bd.non_consec_penalized,
        bd.time_gap, bd.repeat_matchup,
    )
}

// ─── Worker thread ──────────────────────────────────────────────────────────

const BATCH_SIZE: u64 = 50_000;

struct WorkerReport {
    thread_id: usize,
    best_template: Template,
    best_cost: u32,
    current_cost: u32,
    current_temp: f64,
    iterations_total: u64,
    iterations_since_improve: u64,
}

fn worker_loop(
    thread_id: usize,
    initial_temp: f64,
    w8: TemplateWeights,
    shutdown: Arc<AtomicBool>,
    global_best_cost: Arc<AtomicU32>,
    global_best_template: Arc<std::sync::Mutex<Option<Template>>>,
    report_tx: mpsc::Sender<WorkerReport>,
) {
    let mut rng = SmallRng::from_os_rng();
    let mut current = random_template(&mut rng);
    let mut current_cost = evaluate(&current, &w8);
    let mut best = current.clone();
    let mut best_cost = current_cost;

    let mut temp: f64 = initial_temp;
    let cooling_rate: f64 = 0.9999995;
    let min_temp: f64 = 0.1;
    let mut iterations_total: u64 = 0;
    let mut iterations_since_improve: u64 = 0;

    while !shutdown.load(Ordering::Relaxed) {
        // When cooled, start from a completely new random template
        if temp <= min_temp + 0.01 {
            current = random_template(&mut rng);
            current_cost = evaluate(&current, &w8);
            best = current.clone();
            best_cost = current_cost;
            temp = initial_temp;
            iterations_since_improve = 0;
        }

        for _ in 0..BATCH_SIZE {
            if let Some(candidate) = perturb(&current, &mut rng) {
                let new_cost = evaluate(&candidate, &w8);
                let delta = new_cost as i64 - current_cost as i64;

                let accept = if delta < 0 {
                    true
                } else if delta == 0 {
                    rng.random_range(0..5u8) == 0
                } else {
                    let p = (-delta as f64 / temp).exp();
                    rng.random::<f64>() < p
                };

                if accept {
                    current = candidate;
                    current_cost = new_cost;

                    if current_cost < best_cost {
                        best_cost = current_cost;
                        best = current.clone();
                        iterations_since_improve = 0;

                        let gb = global_best_cost.load(Ordering::Relaxed);
                        if best_cost < gb {
                            global_best_cost.fetch_min(best_cost, Ordering::Relaxed);
                            let mut guard = global_best_template.lock().unwrap();
                            *guard = Some(best.clone());
                        }
                    }
                }
            }

            temp = (temp * cooling_rate).max(min_temp);
            iterations_total += 1;
            iterations_since_improve += 1;
        }

        let _ = report_tx.send(WorkerReport {
            thread_id,
            best_template: best.clone(),
            best_cost,
            current_cost,
            current_temp: temp,
            iterations_total,
            iterations_since_improve,
        });
    }
}

// ─── Output ─────────────────────────────────────────────────────────────────

fn fmt_elapsed(d: std::time::Duration) -> String {
    let secs = d.as_secs();
    format!("+{:02}:{:02}:{:02}", secs / 3600, (secs % 3600) / 60, secs % 60)
}

fn fmt_ips(ips: u64) -> String {
    if ips >= 1_000_000 {
        format!("{:.1}M", ips as f64 / 1_000_000.0)
    } else if ips >= 1_000 {
        format!("{:.0}K", ips as f64 / 1_000.0)
    } else {
        format!("{}", ips)
    }
}

fn print_banner(
    global_best_cost: u32,
    best_template: &Template,
    best_source: &str,
    best_age: u64,
    start_time: Instant,
    total_ips: u64,
    w8: &TemplateWeights,
) {
    let now = chrono::Local::now().format("%H:%M:%S");
    let age_str = if best_age < 60 { format!("{}s ago", best_age) }
                  else if best_age < 3600 { format!("{}m{}s ago", best_age / 60, best_age % 60) }
                  else { format!("{}h{}m ago", best_age / 3600, (best_age % 3600) / 60) };
    eprintln!(
        "── {} {} best={} from {} ({}) {} it/s ──\x1b[K",
        now, fmt_elapsed(start_time.elapsed()),
        global_best_cost, best_source, age_str, fmt_ips(total_ips),
    );
    let bd = evaluate_breakdown(best_template, w8);
    eprintln!("   {}\x1b[K", cost_label(&bd));
}

fn print_table_header() {
    eprintln!(
        "{:>4} {:>8}  {:>5} {:>5}  {:>7} {:>7} {:>7} {:>7} {:>7}  {:>6}  {}\x1b[K",
        "src", "temp", "cur", "best",
        "consec", "lane_sw", "pair", "gap", "repeat",
        "stag", "state",
    );
}

fn print_thread_row(report: &WorkerReport, initial_temp: f64, w8: &TemplateWeights) {
    let bd_cur = evaluate_breakdown(&report.best_template, w8);
    let stag_k = report.iterations_since_improve / 1000;
    let state = if report.current_temp <= 0.02 { "cooled" }
        else { "normal" };
    let bold = if report.iterations_since_improve < 50_000 { "\x1b[1m" } else { "" };
    let reset = if !bold.is_empty() { "\x1b[0m" } else { "" };
    let temp_str = format!("{:.0}/{:.2}", initial_temp, report.current_temp);
    eprintln!(
        "{}t{:<2} {:>8}  {:>5} {:>5}  {:>7} {:>7} {:>7} {:>7} {:>7}  {:>5}k  {}{}\x1b[K",
        bold,
        report.thread_id, temp_str,
        report.current_cost, report.best_cost,
        bd_cur.consec_count, bd_cur.lane_switch, bd_cur.lane_pair_break,
        bd_cur.time_gap, bd_cur.repeat_matchup,
        stag_k, state, reset,
    );
}

fn print_template_layout(t: &Template) {
    let games = position_games(t);
    for slot in 0..SLOTS {
        let mut lane_strs = vec!["-\t".to_string(); LANES];
        for i in 0..ENTRIES {
            if ENTRY_SLOTS_LANES[i].0 == slot as u8 {
                let lane = ENTRY_SLOTS_LANES[i].1 as usize;
                lane_strs[lane] = format!("{:>2}v{:<2}", t.pos_a[i] + 1, t.pos_b[i] + 1);
            }
        }
        eprintln!("   Slot {}: {}\x1b[K", slot + 1, lane_strs.join("  "));
    }
    let mut consec = 0u32;
    let mut consec_diff_lane = 0u32;
    let mut non_consec_diff_lane = 0u32;
    for p in 0..POSITIONS {
        let g = &games[p];
        let slots: Vec<u8> = g.iter().map(|&(s, _)| s).collect();
        let lanes: Vec<u8> = g.iter().map(|&(_, l)| l).collect();
        let is_consec = slots.len() == 3 && slots[1] == slots[0] + 1 && slots[2] == slots[1] + 1;
        if is_consec {
            consec += 1;
            if lanes[0] != lanes[1] || lanes[1] != lanes[2] {
                consec_diff_lane += 1;
            }
        } else if slots.len() == 3 {
            // check if the consecutive pair has a lane change
            let has_pen = if slots[1] == slots[0] + 1 { lanes[0] != lanes[1] || lanes[0] / 2 != lanes[2] / 2 }
                else if slots[2] == slots[1] + 1 { lanes[1] != lanes[2] || lanes[1] / 2 != lanes[0] / 2 }
                else { true };
            if has_pen { non_consec_diff_lane += 1; }
        }
    }
    let non_consec = 12 - consec;
    eprintln!("   {} consec ({} same lane, {} diff lane) | {} non-consec ({} diff lane)\x1b[K",
        consec, consec - consec_diff_lane, consec_diff_lane,
        non_consec, non_consec_diff_lane);
}

// ─── Main ───────────────────────────────────────────────────────────────────

fn main() {
    let shutdown = Arc::new(AtomicBool::new(false));
    {
        let s = Arc::clone(&shutdown);
        ctrlc::set_handler(move || { s.store(true, Ordering::SeqCst); }).ok();
    }

    let num_threads = std::thread::available_parallelism()
        .map(|n| n.get())
        .unwrap_or(4);

    // Load weights
    let w8: TemplateWeights = fs::read_to_string("template_weights.json")
        .or_else(|_| fs::read_to_string("../template_weights.json"))
        .ok()
        .and_then(|s| serde_json::from_str(&s).ok())
        .unwrap_or_else(|| {
            eprintln!("No template_weights.json found, using defaults");
            TemplateWeights::default()
        });
    eprintln!("Weights: not_consec={} lane_sw={} pair={} gap={} repeat={}",
        w8.not_consecutive, w8.lane_switch_consecutive, w8.lane_pair_break,
        w8.time_gap_large, w8.repeat_matchup);

    eprintln!("Template SA solver — {} threads", num_threads);
    eprintln!("Searching for template maximizing 3-in-a-row...");

    // Temperature spread across threads
    let temp_min = 5.0f64;
    let temp_max = 20.0f64;
    let temps: Vec<f64> = if num_threads <= 1 {
        vec![temp_min]
    } else {
        (0..num_threads)
            .map(|i| temp_min + (temp_max - temp_min) * i as f64 / (num_threads - 1) as f64)
            .collect()
    };

    let global_best_cost = Arc::new(AtomicU32::new(u32::MAX));
    let global_best_template: Arc<std::sync::Mutex<Option<Template>>> = Arc::new(std::sync::Mutex::new(None));

    let (report_tx, report_rx) = mpsc::channel();
    let mut handles = Vec::new();

    for i in 0..num_threads {
        let shutdown = Arc::clone(&shutdown);
        let gbc = Arc::clone(&global_best_cost);
        let gbt = Arc::clone(&global_best_template);
        let tx = report_tx.clone();
        let temp = temps[i];
        let w8 = w8.clone();
        handles.push(std::thread::spawn(move || {
            worker_loop(i, temp, w8, shutdown, gbc, gbt, tx);
        }));
    }
    drop(report_tx);

    let start_time = Instant::now();
    let mut last_reports: Vec<Option<WorkerReport>> = (0..num_threads).map(|_| None).collect();
    let mut best_cost = u32::MAX;
    let mut best_template: Option<Template> = None;
    let mut best_source = "none".to_string();
    let mut best_found_at = Instant::now();
    let mut last_print = Instant::now();
    let mut last_fresh = Instant::now();
    let mut prev_lines: u32 = 0;
    let mut pending_events: Vec<String> = Vec::new();

    loop {
        // Drain reports
        let deadline = Instant::now() + std::time::Duration::from_millis(200);
        loop {
            match report_rx.recv_timeout(deadline.saturating_duration_since(Instant::now())) {
                Ok(report) => {
                    let tid = report.thread_id;
                    if report.best_cost < best_cost {
                        best_cost = report.best_cost;
                        best_template = Some(report.best_template.clone());
                        best_source = format!("t{}", tid);
                        best_found_at = Instant::now();
                        let bd = evaluate_breakdown(&report.best_template, &w8);
                        let tsv = template_to_tsv(&report.best_template);
                        pending_events.push(format!(
                            "{:>9}  >>>  New best {} from thread {} ({})\n{}",
                            fmt_elapsed(start_time.elapsed()), best_cost, tid,
                            cost_label(&bd), tsv,
                        ));
                        // Save immediately
                        save_results(&report.best_template);
                    }
                    last_reports[tid] = Some(report);
                }
                Err(mpsc::RecvTimeoutError::Timeout) => break,
                Err(mpsc::RecvTimeoutError::Disconnected) => break,
            }
        }

        if shutdown.load(Ordering::Relaxed) { break; }

        // Print table
        if last_print.elapsed().as_millis() >= 1000 || !pending_events.is_empty() {
            let fresh = last_fresh.elapsed().as_secs() >= 600;
            if !fresh && prev_lines > 0 {
                eprint!("\x1b[{}A\r\x1b[J", prev_lines);
            } else if fresh {
                last_fresh = Instant::now();
            }
            for msg in pending_events.drain(..) {
                eprintln!("{}", msg);
            }
            let mut lines: u32 = 0;

            if let Some(ref bt) = best_template {
                let total_ips: u64 = last_reports.iter().filter_map(|r| {
                    r.as_ref().map(|r| {
                        let elapsed = start_time.elapsed().as_secs_f64().max(0.001);
                        (r.iterations_total as f64 / elapsed) as u64
                    })
                }).sum();
                let age = best_found_at.elapsed().as_secs();
                print_banner(best_cost, bt, &best_source, age, start_time, total_ips, &w8);
                lines += 2;
                print_template_layout(bt);
                lines += SLOTS as u32 + 1;
            }

            print_table_header();
            lines += 1;
            for (i, report_opt) in last_reports.iter().enumerate() {
                if let Some(ref report) = report_opt {
                    print_thread_row(report, temps[i], &w8);
                    lines += 1;
                }
            }
            eprint!("\x1b[J");
            prev_lines = lines;
            last_print = Instant::now();
        }

        // Check if all threads disconnected
        if last_reports.iter().all(|r| r.is_some()) && report_rx.try_recv().is_err() {
            if shutdown.load(Ordering::Relaxed) { break; }
        }
    }

    // Shutdown
    shutdown.store(true, Ordering::SeqCst);
    for h in handles {
        let _ = h.join();
    }

    if let Some(ref bt) = best_template {
        eprintln!("\n=== Final best (cost: {}) ===", best_cost);
        let bd = evaluate_breakdown(bt, &w8);
        eprintln!("   {}", cost_label(&bd));
        save_results(bt);
    } else {
        eprintln!("No valid template found!");
    }
}

fn save_results(t: &Template) {
    let entries = template_to_entries(t);
    let json = serde_json::to_string_pretty(&entries).unwrap();
    let _ = fs::write("summer_fixed_template.json", &json);
    let tsv = template_to_tsv(t);
    let _ = fs::write("summer_fixed_template.tsv", &tsv);
}
