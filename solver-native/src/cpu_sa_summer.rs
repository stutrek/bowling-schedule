use std::sync::atomic::{AtomicBool, AtomicU32, Ordering};
use std::sync::{mpsc, Arc};
use std::thread;

use rand::rngs::SmallRng;
use rand::{Rng, SeedableRng};
use solver_core::summer::*;

pub const BATCH_SIZE: u64 = 10_000;
const STATS_RECOMPUTE: u64 = 10_000;

pub const NUM_MOVES: usize = 15;
const BASE_WEIGHTS: [f64; NUM_MOVES] = [
    0.12, 0.08, 0.08, 0.04, 0.04, 0.08, 0.08, 0.06, 0.08, 0.06, 0.05, 0.05,
    0.08, 0.06, 0.04,
];

pub const MOVE_NAMES: [&str; NUM_MOVES] = [
    "tm_swap", "mtch_sw", "opp_sw", "ln_week",
    "slot_sw", "g_match", "g_lane", "g_slot", "g_lnsw",
    "pr_swap", "g_ln_xs", "ln_chas", "g_brkfx", "g_lnsaf", "g_lnpr",
];

pub enum SummerWorkerCommand {
    SetState(SummerAssignment),
    SetTemp(f64),
    Shutdown,
}

pub struct SummerWorkerReport {
    pub core_id: usize,
    pub best_assignment: SummerAssignment,
    pub best_cost: u32,
    pub current_assignment: SummerAssignment,
    pub current_cost: u32,
    pub iterations_total: u64,
    pub move_rates: [f64; NUM_MOVES],
    pub move_shares: [f64; NUM_MOVES],
}

pub struct SummerCpuWorkers {
    pub handles: Vec<thread::JoinHandle<()>>,
    pub commands: Vec<mpsc::Sender<SummerWorkerCommand>>,
    pub reports: mpsc::Receiver<SummerWorkerReport>,
    pub live_best_costs: Vec<Arc<AtomicU32>>,
}

pub fn run_summer_cpu_workers(
    num_cores: usize,
    w8: SummerWeights,
    temps: Vec<f64>,
    shutdown: Arc<AtomicBool>,
) -> SummerCpuWorkers {
    let (report_tx, report_rx) = mpsc::channel();
    let mut cmd_txs = Vec::with_capacity(num_cores);
    let mut handles = Vec::with_capacity(num_cores);
    let mut live_best_costs = Vec::with_capacity(num_cores);

    for core_id in 0..num_cores {
        let (cmd_tx, cmd_rx) = mpsc::channel();
        cmd_txs.push(cmd_tx);

        let live_best = Arc::new(AtomicU32::new(u32::MAX));
        live_best_costs.push(Arc::clone(&live_best));

        let w8 = w8.clone();
        let temp = temps[core_id];
        let shutdown = Arc::clone(&shutdown);
        let report_tx = report_tx.clone();

        handles.push(thread::spawn(move || {
            worker_loop(core_id, w8, temp, shutdown, cmd_rx, report_tx, live_best);
        }));
    }

    SummerCpuWorkers {
        handles,
        commands: cmd_txs,
        reports: report_rx,
        live_best_costs,
    }
}

fn worker_loop(
    core_id: usize,
    initial_w8: SummerWeights,
    initial_temp: f64,
    shutdown: Arc<AtomicBool>,
    cmd_rx: mpsc::Receiver<SummerWorkerCommand>,
    report_tx: mpsc::Sender<SummerWorkerReport>,
    live_best_cost: Arc<AtomicU32>,
) {
    let mut rng = SmallRng::from_os_rng();
    let mut a = random_summer_assignment(&mut rng);
    let active_w8 = initial_w8;
    let mut active_temp = initial_temp;
    let mut cost = evaluate_summer(&a, &active_w8);
    let mut best_a = a;
    let mut best_cost = cost.total;
    let mut stats = MoveStats::new();
    let mut iterations_total: u64 = 0;

    loop {
        while let Ok(cmd) = cmd_rx.try_recv() {
            match cmd {
                SummerWorkerCommand::SetState(new_a) => {
                    a = new_a;
                    cost = evaluate_summer(&a, &active_w8);
                    if cost.total < best_cost {
                        best_a = a;
                        best_cost = cost.total;
                        live_best_cost.store(best_cost, Ordering::Relaxed);
                    }
                }
                SummerWorkerCommand::SetTemp(t) => {
                    active_temp = t;
                }
                SummerWorkerCommand::Shutdown => return,
            }
        }

        if shutdown.load(Ordering::Relaxed) {
            return;
        }

        let batch_end = iterations_total + BATCH_SIZE;
        for i in iterations_total..batch_end {
            if i > 0 && i % STATS_RECOMPUTE == 0 {
                stats.recompute();
            }

            // Exhaustive single-position search (periodic)
            if i > 0 && i % 100_000 == 0 && cost.total > 0 {
                exhaustive_local_search(&mut a, &mut cost, &mut best_a, &mut best_cost, &active_w8, &mut rng);
            }

            // Compound move at low cost
            let compound_prob = ((1000.0 - cost.total as f64) / 800.0).clamp(0.0, 0.5);
            if rng.random::<f64>() < compound_prob {
                let saved = a;
                let max_swaps = if cost.total < 200 { 10 } else if cost.total < 400 { 6 } else { 4 };
                let num_swaps = rng.random_range(2..=max_swaps);
                for _ in 0..num_swaps {
                    try_team_swap(&mut a, &mut rng);
                }
                let new_cost = evaluate_summer(&a, &active_w8);
                let delta = new_cost.total as i64 - cost.total as i64;
                if sa_accept(delta, active_temp, &mut rng) {
                    cost = new_cost;
                    if cost.total < best_cost { best_cost = cost.total; best_a = a; }
                } else {
                    a = saved;
                }
            } else {
                let move_id = stats.select(rng.random::<f64>());
                stats.attempts[move_id] += 1;

                let accepted = match move_id {
                    0 => do_team_swap(&mut a, &mut cost, &mut best_a, &mut best_cost, &active_w8, active_temp, &mut rng),
                    1 => do_matchup_swap(&mut a, &mut cost, &mut best_a, &mut best_cost, &active_w8, active_temp, &mut rng),
                    2 => do_opponent_swap(&mut a, &mut cost, &mut best_a, &mut best_cost, &active_w8, active_temp, &mut rng),
                    3 => do_lane_swap_week(&mut a, &mut cost, &mut best_a, &mut best_cost, &active_w8, active_temp, &mut rng),
                    4 => do_slot_swap(&mut a, &mut cost, &mut best_a, &mut best_cost, &active_w8, active_temp, &mut rng),
                    5 => do_guided_matchup(&mut a, &mut cost, &mut best_a, &mut best_cost, &active_w8, active_temp, &mut rng),
                    6 => do_guided_lane(&mut a, &mut cost, &mut best_a, &mut best_cost, &active_w8, active_temp, &mut rng),
                    7 => do_guided_slot(&mut a, &mut cost, &mut best_a, &mut best_cost, &active_w8, active_temp, &mut rng),
                    8 => do_guided_lane_switch(&mut a, &mut cost, &mut best_a, &mut best_cost, &active_w8, active_temp, &mut rng),
                    9 => do_pair_swap_in_slot(&mut a, &mut cost, &mut best_a, &mut best_cost, &active_w8, active_temp, &mut rng),
                    10 => do_guided_lane_cross_slot(&mut a, &mut cost, &mut best_a, &mut best_cost, &active_w8, active_temp, &mut rng),
                    11 => do_lane_chase(&mut a, &mut cost, &mut best_a, &mut best_cost, &active_w8, active_temp, &mut rng),
                    12 => do_guided_break_fix(&mut a, &mut cost, &mut best_a, &mut best_cost, &active_w8, active_temp, &mut rng),
                    13 => do_guided_lane_safe(&mut a, &mut cost, &mut best_a, &mut best_cost, &active_w8, active_temp, &mut rng),
                    _ => do_guided_lane_pair(&mut a, &mut cost, &mut best_a, &mut best_cost, &active_w8, active_temp, &mut rng),
                };

                if accepted { stats.accepts[move_id] += 1; }
            }
        }
        iterations_total = batch_end;
        live_best_cost.store(best_cost, Ordering::Relaxed);

        let _ = report_tx.send(SummerWorkerReport {
            core_id,
            best_assignment: best_a,
            best_cost,
            current_assignment: a,
            current_cost: cost.total,
            iterations_total,
            move_rates: stats.last_rates,
            move_shares: stats.last_shares,
        });
    }
}

// ─── SA acceptance ─────────────────────────────────────────────────────────

fn sa_accept(delta: i64, temp: f64, rng: &mut SmallRng) -> bool {
    if delta < 0 {
        true
    } else if delta == 0 {
        rng.random::<f64>() < 0.2
    } else {
        rng.random::<f64>() < (-delta as f64 / temp).exp()
    }
}

// ─── Helper: collect filled/empty positions ────────────────────────────────

fn filled_positions(a: &SummerAssignment, w: usize) -> Vec<(usize, usize)> {
    let mut v = Vec::new();
    for s in 0..S_SLOTS {
        for p in 0..S_PAIRS {
            if a[w][s][p].0 != EMPTY {
                v.push((s, p));
            }
        }
    }
    v
}

/// Check if team is already in the given slot (at any pair other than exclude_pair).
fn team_in_slot(a: &SummerAssignment, w: usize, slot: usize, team: u8, exclude_pair: Option<usize>) -> bool {
    for p in 0..S_PAIRS {
        if Some(p) == exclude_pair { continue; }
        let (t1, t2) = a[w][slot][p];
        if t1 == team || t2 == team { return true; }
    }
    false
}

// ─── Move 0: team_swap ─────────────────────────────────────────────────────

/// Pick two filled matchups in different slots of the same week.
/// Swap one team from each (the swapped teams exchange slots).
fn try_team_swap(a: &mut SummerAssignment, rng: &mut SmallRng) -> bool {
    let w = rng.random_range(0..S_WEEKS);
    let filled = filled_positions(a, w);
    if filled.len() < 2 { return false; }

    let i1 = rng.random_range(0..filled.len());
    let mut i2 = rng.random_range(0..(filled.len() - 1));
    if i2 >= i1 { i2 += 1; }

    let (s1, p1) = filled[i1];
    let (s2, p2) = filled[i2];
    if s1 == s2 { return false; }

    let side1 = rng.random_range(0..2usize);
    let side2 = rng.random_range(0..2usize);

    let t1 = if side1 == 0 { a[w][s1][p1].0 } else { a[w][s1][p1].1 };
    let t2 = if side2 == 0 { a[w][s2][p2].0 } else { a[w][s2][p2].1 };

    // Check t1 not already in s2 (excluding the position we're swapping into)
    if team_in_slot(a, w, s2, t1, Some(p2)) { return false; }
    if team_in_slot(a, w, s1, t2, Some(p1)) { return false; }

    // Check we don't create same-team matchups
    let opp1 = if side1 == 0 { a[w][s1][p1].1 } else { a[w][s1][p1].0 };
    let opp2 = if side2 == 0 { a[w][s2][p2].1 } else { a[w][s2][p2].0 };
    if opp1 == t2 || opp2 == t1 { return false; }

    if side1 == 0 { a[w][s1][p1].0 = t2; } else { a[w][s1][p1].1 = t2; }
    if side2 == 0 { a[w][s2][p2].0 = t1; } else { a[w][s2][p2].1 = t1; }
    true
}

fn do_team_swap(
    a: &mut SummerAssignment, cost: &mut SummerCostBreakdown,
    best_a: &mut SummerAssignment, best_cost: &mut u32,
    w8: &SummerWeights, temp: f64, rng: &mut SmallRng,
) -> bool {
    let saved = *a;
    if !try_team_swap(a, rng) { return false; }
    let new_cost = evaluate_summer(a, w8);
    let delta = new_cost.total as i64 - cost.total as i64;
    if sa_accept(delta, temp, rng) {
        *cost = new_cost;
        if cost.total < *best_cost { *best_cost = cost.total; *best_a = *a; }
        true
    } else {
        *a = saved;
        false
    }
}

// ─── Move 1: matchup_swap ─────────────────────────────────────────────────

fn do_matchup_swap(
    a: &mut SummerAssignment, cost: &mut SummerCostBreakdown,
    best_a: &mut SummerAssignment, best_cost: &mut u32,
    w8: &SummerWeights, temp: f64, rng: &mut SmallRng,
) -> bool {
    let w = rng.random_range(0..S_WEEKS);
    let filled = filled_positions(a, w);
    if filled.len() < 2 { return false; }

    let i1 = rng.random_range(0..filled.len());
    let mut i2 = rng.random_range(0..(filled.len() - 1));
    if i2 >= i1 { i2 += 1; }

    let (s1, p1) = filled[i1];
    let (s2, p2) = filled[i2];
    if s1 == s2 { return false; } // same slot swap doesn't change slot/lane assignments meaningfully

    let (a1, b1) = a[w][s1][p1];
    let (a2, b2) = a[w][s2][p2];

    // Check: no team from matchup 1 already in slot 2 (excluding the swap partner)
    if team_in_slot(a, w, s2, a1, Some(p2)) { return false; }
    if team_in_slot(a, w, s2, b1, Some(p2)) { return false; }
    if team_in_slot(a, w, s1, a2, Some(p1)) { return false; }
    if team_in_slot(a, w, s1, b2, Some(p1)) { return false; }

    // Swap entire matchups
    a[w][s1][p1] = (a2, b2);
    a[w][s2][p2] = (a1, b1);

    let new_cost = evaluate_summer(a, w8);
    let delta = new_cost.total as i64 - cost.total as i64;
    if sa_accept(delta, temp, rng) {
        *cost = new_cost;
        if cost.total < *best_cost { *best_cost = cost.total; *best_a = *a; }
        true
    } else {
        a[w][s1][p1] = (a1, b1);
        a[w][s2][p2] = (a2, b2);
        false
    }
}

// ─── Move 2: opponent_swap ─────────────────────────────────────────────────

fn do_opponent_swap(
    a: &mut SummerAssignment, cost: &mut SummerCostBreakdown,
    best_a: &mut SummerAssignment, best_cost: &mut u32,
    w8: &SummerWeights, temp: f64, rng: &mut SmallRng,
) -> bool {
    let w = rng.random_range(0..S_WEEKS);
    let s = rng.random_range(0..S_SLOTS);

    // Find filled pairs in this slot
    let mut slot_filled: Vec<usize> = Vec::new();
    for p in 0..S_PAIRS {
        if a[w][s][p].0 != EMPTY {
            slot_filled.push(p);
        }
    }
    if slot_filled.len() < 2 { return false; }

    let i1 = rng.random_range(0..slot_filled.len());
    let mut i2 = rng.random_range(0..(slot_filled.len() - 1));
    if i2 >= i1 { i2 += 1; }

    let p1 = slot_filled[i1];
    let p2 = slot_filled[i2];

    // Swap one team between them: A vs B and C vs D → A vs D and C vs B
    let side1 = rng.random_range(0..2usize);
    let side2 = rng.random_range(0..2usize);

    let t1 = if side1 == 0 { a[w][s][p1].0 } else { a[w][s][p1].1 };
    let t2 = if side2 == 0 { a[w][s][p2].0 } else { a[w][s][p2].1 };

    // Check we don't create same-team matchups
    let opp1 = if side1 == 0 { a[w][s][p1].1 } else { a[w][s][p1].0 };
    let opp2 = if side2 == 0 { a[w][s][p2].1 } else { a[w][s][p2].0 };
    if opp1 == t2 || opp2 == t1 { return false; }

    if side1 == 0 { a[w][s][p1].0 = t2; } else { a[w][s][p1].1 = t2; }
    if side2 == 0 { a[w][s][p2].0 = t1; } else { a[w][s][p2].1 = t1; }

    let new_cost = evaluate_summer(a, w8);
    let delta = new_cost.total as i64 - cost.total as i64;
    if sa_accept(delta, temp, rng) {
        *cost = new_cost;
        if cost.total < *best_cost { *best_cost = cost.total; *best_a = *a; }
        true
    } else {
        // Undo
        if side1 == 0 { a[w][s][p1].0 = t1; } else { a[w][s][p1].1 = t1; }
        if side2 == 0 { a[w][s][p2].0 = t2; } else { a[w][s][p2].1 = t2; }
        false
    }
}

// ─── Move 3: lane_swap_week — swap two entire lanes across all slots in a week

fn do_lane_swap_week(
    a: &mut SummerAssignment, cost: &mut SummerCostBreakdown,
    best_a: &mut SummerAssignment, best_cost: &mut u32,
    w8: &SummerWeights, temp: f64, rng: &mut SmallRng,
) -> bool {
    let w = rng.random_range(0..S_WEEKS);
    let p1 = rng.random_range(0..S_PAIRS);
    let mut p2 = rng.random_range(0..(S_PAIRS - 1));
    if p2 >= p1 { p2 += 1; }

    let saved = *a;
    for s in 0..S_SLOTS {
        if is_valid_position(s, p1) && is_valid_position(s, p2) {
            let tmp = a[w][s][p1];
            a[w][s][p1] = a[w][s][p2];
            a[w][s][p2] = tmp;
        }
    }

    let new_cost = evaluate_summer(a, w8);
    let delta = new_cost.total as i64 - cost.total as i64;
    if sa_accept(delta, temp, rng) {
        *cost = new_cost;
        if cost.total < *best_cost { *best_cost = cost.total; *best_a = *a; }
        true
    } else {
        *a = saved;
        false
    }
}

// ─── Move 4: slot_swap ─────────────────────────────────────────────────────

fn do_slot_swap(
    a: &mut SummerAssignment, cost: &mut SummerCostBreakdown,
    best_a: &mut SummerAssignment, best_cost: &mut u32,
    w8: &SummerWeights, temp: f64, rng: &mut SmallRng,
) -> bool {
    let w = rng.random_range(0..S_WEEKS);

    // Pick two full slots to swap (0-3). Slot 4 is partial (2 pairs), excluded.
    let s1 = rng.random_range(0..4usize);
    let mut s2 = rng.random_range(0..3usize);
    if s2 >= s1 { s2 += 1; }

    let tmp = a[w][s1];
    a[w][s1] = a[w][s2];
    a[w][s2] = tmp;

    let new_cost = evaluate_summer(a, w8);
    let delta = new_cost.total as i64 - cost.total as i64;
    if sa_accept(delta, temp, rng) {
        *cost = new_cost;
        if cost.total < *best_cost { *best_cost = cost.total; *best_a = *a; }
        true
    } else {
        a[w][s2] = a[w][s1];
        a[w][s1] = tmp;
        false
    }
}

// ─── Move 5: guided_matchup ────────────────────────────────────────────────

fn do_guided_matchup(
    a: &mut SummerAssignment, cost: &mut SummerCostBreakdown,
    best_a: &mut SummerAssignment, best_cost: &mut u32,
    w8: &SummerWeights, temp: f64, rng: &mut SmallRng,
) -> bool {
    // Count matchups
    let mut matchup_counts = [0i32; S_TEAMS * S_TEAMS];
    for w in 0..S_WEEKS {
        for s in 0..S_SLOTS {
            for p in 0..S_PAIRS {
                let (t1, t2) = a[w][s][p];
                if t1 == EMPTY { continue; }
                let lo = (t1 as usize).min(t2 as usize);
                let hi = (t1 as usize).max(t2 as usize);
                matchup_counts[lo * S_TEAMS + hi] += 1;
            }
        }
    }

    // Find a pair outside target [2, 3]
    let start = rng.random_range(0..S_TEAMS);
    let mut ta = 0u8;
    let mut tb = 0u8;
    let mut found = false;
    let mut need_more = false;
    'outer: for off_i in 0..S_TEAMS {
        let i = (start + off_i) % S_TEAMS;
        for j in (i + 1)..S_TEAMS {
            let c = matchup_counts[i * S_TEAMS + j];
            if c < 2 || c > 3 {
                ta = i as u8;
                tb = j as u8;
                need_more = c < 2;
                found = true;
                break 'outer;
            }
        }
    }
    if !found { return false; }

    let saved = *a;

    if need_more {
        // Need to create a matchup between ta and tb
        // Find a week where both are in different slots and swap to bring them together
        let week_start = rng.random_range(0..S_WEEKS);
        for off in 0..S_WEEKS {
            let w = (week_start + off) % S_WEEKS;
            // Find ta and tb positions
            let pos_a = find_team_pos(a, w, ta);
            let pos_b = find_team_pos(a, w, tb);
            if pos_a.is_empty() || pos_b.is_empty() { continue; }

            // Pick a game of ta and try to swap tb into the opponent slot
            let (sa, pa, side_a) = pos_a[rng.random_range(0..pos_a.len())];
            // tb needs to get into the same matchup
            // Find which team is ta's opponent in this matchup
            let opp = if side_a == 0 { a[w][sa][pa].1 } else { a[w][sa][pa].0 };

            // Find a game of tb that we can swap the opponent into
            for &(sb, pb, side_b) in &pos_b {
                if sa == sb { continue; } // same slot
                // tb's opponent in its current matchup
                let _tb_opp = if side_b == 0 { a[w][sb][pb].1 } else { a[w][sb][pb].0 };

                // Swap: put tb where opp is, put opp where tb is
                // Check opp not in slot sb, tb not in slot sa (except at current positions)
                if team_in_slot(a, w, sa, tb, Some(pa)) { continue; }
                if team_in_slot(a, w, sb, opp, Some(pb)) { continue; }
                // Check we don't create a self-matchup: opp replacing tb at (sb,pb)
                // means opp paired with _tb_opp — reject if they're the same
                let tb_opp = if side_b == 0 { a[w][sb][pb].1 } else { a[w][sb][pb].0 };
                if tb_opp == opp { continue; }

                // Perform: swap opp and tb
                if side_a == 0 { a[w][sa][pa].1 = tb; } else { a[w][sa][pa].0 = tb; }
                if side_b == 0 { a[w][sb][pb].0 = opp; } else { a[w][sb][pb].1 = opp; }

                let new_cost = evaluate_summer(a, w8);
                let delta = new_cost.total as i64 - cost.total as i64;
                if sa_accept(delta, temp, rng) {
                    *cost = new_cost;
                    if cost.total < *best_cost { *best_cost = cost.total; *best_a = *a; }
                    return true;
                } else {
                    *a = saved;
                    return false;
                }
            }
        }
    } else {
        // Need to break a matchup between ta and tb (count > 3)
        // Find a week where they play each other and swap one out
        let week_start = rng.random_range(0..S_WEEKS);
        for off in 0..S_WEEKS {
            let w = (week_start + off) % S_WEEKS;
            // Find if ta and tb are matched this week
            for s in 0..S_SLOTS {
                for p in 0..S_PAIRS {
                    let (t1, t2) = a[w][s][p];
                    if (t1 == ta && t2 == tb) || (t1 == tb && t2 == ta) {
                        // Found it. Try a team_swap to break it up
                        if try_team_swap(a, rng) {
                            let new_cost = evaluate_summer(a, w8);
                            let delta = new_cost.total as i64 - cost.total as i64;
                            if sa_accept(delta, temp, rng) {
                                *cost = new_cost;
                                if cost.total < *best_cost { *best_cost = cost.total; *best_a = *a; }
                                return true;
                            } else {
                                *a = saved;
                                return false;
                            }
                        }
                    }
                }
            }
        }
    }

    false
}

// ─── Move 6: guided_lane ──────────────────────────────────────────────────

fn do_guided_lane(
    a: &mut SummerAssignment, cost: &mut SummerCostBreakdown,
    best_a: &mut SummerAssignment, best_cost: &mut u32,
    w8: &SummerWeights, temp: f64, rng: &mut SmallRng,
) -> bool {
    // Compute lane counts (4 lanes = pair indices)
    let mut lane_counts = [0i32; S_TEAMS * S_LANES];
    for w in 0..S_WEEKS {
        for s in 0..S_SLOTS {
            for p in 0..S_PAIRS {
                let (t1, t2) = a[w][s][p];
                if t1 == EMPTY { continue; }
                lane_counts[t1 as usize * S_LANES + p] += 1;
                lane_counts[t2 as usize * S_LANES + p] += 1;
            }
        }
    }

    let target = (S_WEEKS as f64 * 3.0) / S_LANES as f64; // 7.5
    let mut worst_team = 0usize;
    let mut worst_lane = 0usize;
    let mut worst_dev = 0.0f64;
    let mut worst_over = false;
    for t in 0..S_TEAMS {
        for l in 0..S_LANES {
            let dev = lane_counts[t * S_LANES + l] as f64 - target;
            if dev.abs() > worst_dev {
                worst_dev = dev.abs();
                worst_team = t;
                worst_lane = l;
                worst_over = dev > 0.0;
            }
        }
    }
    if worst_dev < 1.0 { return false; }

    // Find worst_team on the over-represented lane and opponent-swap to a different lane
    let saved = *a;
    let w_start = rng.random_range(0..S_WEEKS);
    for off in 0..S_WEEKS {
        let w = (w_start + off) % S_WEEKS;
        let positions = find_team_pos(a, w, worst_team as u8);
        if positions.is_empty() { continue; }

        // Find a position on the over-represented lane (or under-represented for swapping into)
        let source_lane = worst_lane;
        let source_pos: Vec<_> = if worst_over {
            positions.iter().filter(|&&(_, p, _)| p == source_lane).collect()
        } else {
            positions.iter().filter(|&&(_, p, _)| p != source_lane).collect()
        };
        if source_pos.is_empty() { continue; }
        let &(s, p_src, side) = source_pos[rng.random_range(0..source_pos.len())];

        // Find another matchup in the same slot on a different lane to swap into
        let mut targets: Vec<usize> = Vec::new();
        for p in 0..S_PAIRS {
            if p == p_src || a[w][s][p].0 == EMPTY { continue; }
            targets.push(p);
        }
        if targets.is_empty() { continue; }
        let p_tgt = targets[rng.random_range(0..targets.len())];
        let swap_side = rng.random_range(0..2usize);

        let my_team = worst_team as u8;
        let their_team = if swap_side == 0 { a[w][s][p_tgt].0 } else { a[w][s][p_tgt].1 };

        // Check constraints: no self-matchup
        let my_opp = if side == 0 { a[w][s][p_src].1 } else { a[w][s][p_src].0 };
        let their_opp = if swap_side == 0 { a[w][s][p_tgt].1 } else { a[w][s][p_tgt].0 };
        if my_opp == their_team || their_opp == my_team { continue; }

        // Perform swap
        if side == 0 { a[w][s][p_src].0 = their_team; } else { a[w][s][p_src].1 = their_team; }
        if swap_side == 0 { a[w][s][p_tgt].0 = my_team; } else { a[w][s][p_tgt].1 = my_team; }

        let new_cost = evaluate_summer(a, w8);
        let delta = new_cost.total as i64 - cost.total as i64;
        if sa_accept(delta, temp, rng) {
            *cost = new_cost;
            if cost.total < *best_cost { *best_cost = cost.total; *best_a = *a; }
            return true;
        } else {
            *a = saved;
            return false;
        }
    }
    false
}

// ─── Move 7: guided_slot ──────────────────────────────────────────────────

fn do_guided_slot(
    a: &mut SummerAssignment, cost: &mut SummerCostBreakdown,
    best_a: &mut SummerAssignment, best_cost: &mut u32,
    w8: &SummerWeights, temp: f64, rng: &mut SmallRng,
) -> bool {
    // Compute slot counts
    let mut slot_counts = [0i32; S_TEAMS * S_SLOTS];
    for w in 0..S_WEEKS {
        for s in 0..S_SLOTS {
            for p in 0..S_PAIRS {
                let (t1, t2) = a[w][s][p];
                if t1 == EMPTY { continue; }
                slot_counts[t1 as usize * S_SLOTS + s] += 1;
                slot_counts[t2 as usize * S_SLOTS + s] += 1;
            }
        }
    }

    let slot4_target: f64 = (4.0 * S_WEEKS as f64) / S_TEAMS as f64;
    let slot03_target: f64 = (S_WEEKS as f64 * 3.0 - slot4_target) / 4.0;

    let mut worst_team = 0usize;
    let mut worst_slot = 0usize;
    let mut worst_dev = 0.0f64;
    for t in 0..S_TEAMS {
        for s in 0..S_SLOTS {
            let target = if s < 4 { slot03_target } else { slot4_target };
            let dev = slot_counts[t * S_SLOTS + s] as f64 - target;
            if -dev > worst_dev {
                worst_dev = -dev;
                worst_team = t;
                worst_slot = s;
            }
        }
    }
    if worst_dev < 1.0 { return false; }

    // Find worst_team in a different slot and team_swap into worst_slot
    let saved = *a;
    let w_start = rng.random_range(0..S_WEEKS);
    for off in 0..S_WEEKS {
        let w = (w_start + off) % S_WEEKS;
        let positions = find_team_pos(a, w, worst_team as u8);
        if positions.is_empty() { continue; }

        // Find worst_team NOT in worst_slot
        let from: Vec<_> = positions.iter().filter(|&&(s, _, _)| s != worst_slot).collect();
        if from.is_empty() { continue; }
        let &(sf, pf, side_f) = from[rng.random_range(0..from.len())];

        // Find a matchup in worst_slot to team_swap with
        let mut targets: Vec<(usize, usize)> = Vec::new();
        for p in 0..S_PAIRS {
            if !is_valid_position(worst_slot, p) { continue; }
            if a[w][worst_slot][p].0 == EMPTY { continue; }
            targets.push((worst_slot, p));
        }
        if targets.is_empty() { continue; }
        let (st, pt) = targets[rng.random_range(0..targets.len())];
        let side_t = rng.random_range(0..2usize);

        let my_team = worst_team as u8;
        let their_team = if side_t == 0 { a[w][st][pt].0 } else { a[w][st][pt].1 };

        // Check constraints
        if team_in_slot(a, w, st, my_team, Some(pt)) { continue; }
        if team_in_slot(a, w, sf, their_team, Some(pf)) { continue; }
        let my_opp = if side_f == 0 { a[w][sf][pf].1 } else { a[w][sf][pf].0 };
        let their_opp = if side_t == 0 { a[w][st][pt].1 } else { a[w][st][pt].0 };
        if my_opp == their_team || their_opp == my_team { continue; }

        // Perform team swap
        if side_f == 0 { a[w][sf][pf].0 = their_team; } else { a[w][sf][pf].1 = their_team; }
        if side_t == 0 { a[w][st][pt].0 = my_team; } else { a[w][st][pt].1 = my_team; }

        let new_cost = evaluate_summer(a, w8);
        let delta = new_cost.total as i64 - cost.total as i64;
        if sa_accept(delta, temp, rng) {
            *cost = new_cost;
            if cost.total < *best_cost { *best_cost = cost.total; *best_a = *a; }
            return true;
        } else {
            *a = saved;
            return false;
        }
    }
    false
}

// ─── Move 8: guided_lane_switch ───────────────────────────────────────────

fn do_guided_lane_switch(
    a: &mut SummerAssignment, cost: &mut SummerCostBreakdown,
    best_a: &mut SummerAssignment, best_cost: &mut u32,
    w8: &SummerWeights, temp: f64, rng: &mut SmallRng,
) -> bool {
    let w = rng.random_range(0..S_WEEKS);

    // Find team with worst lane-switch penalty in this week
    let mut worst_team = 0u8;
    let mut worst_penalty = 0u32;
    let t_start = rng.random_range(0..S_TEAMS);

    for t_off in 0..S_TEAMS {
        let t = (t_start + t_off) % S_TEAMS;
        let positions = find_team_pos(a, w, t as u8);
        if positions.len() < 2 { continue; }

        let mut pen = 0u32;
        for i in 0..(positions.len() - 1) {
            let (s1, p1, _) = positions[i];
            let (s2, p2, _) = positions[i + 1];
            let gap = s2 - s1 - 1;
            if p1 != p2 {
                if gap == 0 {
                    pen += w8.lane_switch_consecutive;
                } else {
                    pen += w8.lane_switch_post_break;
                }
            }
        }

        if pen > worst_penalty {
            worst_penalty = pen;
            worst_team = t as u8;
        }
    }

    if worst_penalty == 0 { return false; }

    let saved = *a;
    let positions = find_team_pos(a, w, worst_team);

    // Find worst transition
    let mut worst_idx = 0usize;
    let mut worst_trans_pen = 0u32;
    for i in 0..(positions.len().saturating_sub(1)) {
        let (s1, p1, _) = positions[i];
        let (s2, p2, _) = positions[i + 1];
        if p1 == p2 { continue; }
        let gap = s2 - s1 - 1;
        let pen = if gap == 0 { w8.lane_switch_consecutive } else { w8.lane_switch_post_break };
        if pen > worst_trans_pen {
            worst_trans_pen = pen;
            worst_idx = i;
        }
    }

    let (s1, p1, side1) = positions[worst_idx];
    let (s2, p2, side2) = positions[worst_idx + 1];

    // Try to move the team from one lane to match the other via opponent swap
    // Pick which game to fix (randomly)
    let (fix_s, fix_p, fix_side, target_p) = if rng.random_range(0..2usize) == 0 {
        (s1, p1, side1, p2) // move game 1's team to pair p2
    } else {
        (s2, p2, side2, p1) // move game 2's team to pair p1
    };

    // Find a matchup at target_p in the same slot to opponent-swap with
    if a[w][fix_s][target_p].0 != EMPTY {
        let swap_side = rng.random_range(0..2usize);
        let their_team = if swap_side == 0 { a[w][fix_s][target_p].0 } else { a[w][fix_s][target_p].1 };

        // Check constraints
        let my_opp = if fix_side == 0 { a[w][fix_s][fix_p].1 } else { a[w][fix_s][fix_p].0 };
        let their_opp = if swap_side == 0 { a[w][fix_s][target_p].1 } else { a[w][fix_s][target_p].0 };
        if my_opp == their_team || their_opp == worst_team {
            *a = saved;
            return false;
        }

        // Perform swap
        if fix_side == 0 { a[w][fix_s][fix_p].0 = their_team; } else { a[w][fix_s][fix_p].1 = their_team; }
        if swap_side == 0 { a[w][fix_s][target_p].0 = worst_team; } else { a[w][fix_s][target_p].1 = worst_team; }
    } else {
        return false;
    }

    let new_cost = evaluate_summer(a, w8);
    let delta = new_cost.total as i64 - cost.total as i64;
    if sa_accept(delta, temp, rng) {
        *cost = new_cost;
        if cost.total < *best_cost { *best_cost = cost.total; *best_a = *a; }
        true
    } else {
        *a = saved;
        false
    }
}

// ─── Move 9: pair_swap_in_slot ─────────────────────────────────────────────

fn do_pair_swap_in_slot(
    a: &mut SummerAssignment, cost: &mut SummerCostBreakdown,
    best_a: &mut SummerAssignment, best_cost: &mut u32,
    w8: &SummerWeights, temp: f64, rng: &mut SmallRng,
) -> bool {
    let w = rng.random_range(0..S_WEEKS);
    let s = rng.random_range(0..S_SLOTS);
    let p1 = rng.random_range(0..S_PAIRS);
    if !is_valid_position(s, p1) || a[w][s][p1].0 == EMPTY { return false; }
    let mut p2 = rng.random_range(0..(S_PAIRS - 1));
    if p2 >= p1 { p2 += 1; }
    if !is_valid_position(s, p2) || a[w][s][p2].0 == EMPTY { return false; }

    let saved = *a;
    let tmp = a[w][s][p1];
    a[w][s][p1] = a[w][s][p2];
    a[w][s][p2] = tmp;

    let new_cost = evaluate_summer(a, w8);
    let delta = new_cost.total as i64 - cost.total as i64;
    if sa_accept(delta, temp, rng) {
        *cost = new_cost;
        if cost.total < *best_cost { *best_cost = cost.total; *best_a = *a; }
        true
    } else {
        *a = saved;
        false
    }
}

// ─── Move 10: guided_lane_cross_slot ──────────────────────────────────────

fn do_guided_lane_cross_slot(
    a: &mut SummerAssignment, cost: &mut SummerCostBreakdown,
    best_a: &mut SummerAssignment, best_cost: &mut u32,
    w8: &SummerWeights, temp: f64, rng: &mut SmallRng,
) -> bool {
    // Compute lane counts
    let mut lane_counts = [0i32; S_TEAMS * S_LANES];
    for w in 0..S_WEEKS {
        for s in 0..S_SLOTS {
            for p in 0..S_PAIRS {
                let (t1, t2) = a[w][s][p];
                if t1 == EMPTY { continue; }
                lane_counts[t1 as usize * S_LANES + p] += 1;
                lane_counts[t2 as usize * S_LANES + p] += 1;
            }
        }
    }

    let target = (S_WEEKS as f64 * 3.0) / S_LANES as f64;
    let mut worst_team = 0usize;
    let mut worst_lane = 0usize;
    let mut worst_dev = 0.0f64;
    let mut worst_over = false;
    for t in 0..S_TEAMS {
        for l in 0..S_LANES {
            let dev = lane_counts[t * S_LANES + l] as f64 - target;
            if dev.abs() > worst_dev {
                worst_dev = dev.abs();
                worst_team = t;
                worst_lane = l;
                worst_over = dev > 0.0;
            }
        }
    }
    if worst_dev < 1.0 { return false; }

    // Team_swap worst_team from over-represented lane to a different lane across slots
    let saved = *a;
    let w = rng.random_range(0..S_WEEKS);
    let positions = find_team_pos(a, w, worst_team as u8);
    if positions.is_empty() { return false; }

    // Find a position on the source lane
    let source_pos: Vec<_> = if worst_over {
        positions.iter().filter(|&&(_, p, _)| p == worst_lane).collect()
    } else {
        positions.iter().filter(|&&(_, p, _)| p != worst_lane).collect()
    };
    if source_pos.is_empty() { return false; }
    let &(sf, pf, side_f) = source_pos[rng.random_range(0..source_pos.len())];

    // Find a target in a different slot on a different lane pair
    let mut targets: Vec<(usize, usize, usize)> = Vec::new();
    for s in 0..S_SLOTS {
        if s == sf { continue; }
        for p in 0..S_PAIRS {
            if !is_valid_position(s, p) || a[w][s][p].0 == EMPTY { continue; }
            let on_target = if worst_over { p != worst_lane } else { p == worst_lane };
            if !on_target { continue; }
            targets.push((s, p, 0));
            targets.push((s, p, 1));
        }
    }
    if targets.is_empty() { return false; }
    let (st, pt, side_t) = targets[rng.random_range(0..targets.len())];

    let my_team = worst_team as u8;
    let their_team = if side_t == 0 { a[w][st][pt].0 } else { a[w][st][pt].1 };

    // Check constraints
    if team_in_slot(a, w, st, my_team, Some(pt)) { return false; }
    if team_in_slot(a, w, sf, their_team, Some(pf)) { return false; }
    let my_opp = if side_f == 0 { a[w][sf][pf].1 } else { a[w][sf][pf].0 };
    let their_opp = if side_t == 0 { a[w][st][pt].1 } else { a[w][st][pt].0 };
    if my_opp == their_team || their_opp == my_team { return false; }

    // Perform team swap
    if side_f == 0 { a[w][sf][pf].0 = their_team; } else { a[w][sf][pf].1 = their_team; }
    if side_t == 0 { a[w][st][pt].0 = my_team; } else { a[w][st][pt].1 = my_team; }

    let new_cost = evaluate_summer(a, w8);
    let delta = new_cost.total as i64 - cost.total as i64;
    if sa_accept(delta, temp, rng) {
        *cost = new_cost;
        if cost.total < *best_cost { *best_cost = cost.total; *best_a = *a; }
        true
    } else {
        *a = saved;
        false
    }
}

// ─── Move 11: lane_chase — multi-week compound move to fix worst lane imbalance ──

fn do_lane_chase(
    a: &mut SummerAssignment, cost: &mut SummerCostBreakdown,
    best_a: &mut SummerAssignment, best_cost: &mut u32,
    w8: &SummerWeights, temp: f64, rng: &mut SmallRng,
) -> bool {
    // Compute lane counts
    let mut lane_counts = [0i32; S_TEAMS * S_LANES];
    for w in 0..S_WEEKS {
        for s in 0..S_SLOTS {
            for p in 0..S_PAIRS {
                let (t1, t2) = a[w][s][p];
                if t1 == EMPTY { continue; }
                lane_counts[t1 as usize * S_LANES + p] += 1;
                lane_counts[t2 as usize * S_LANES + p] += 1;
            }
        }
    }

    let target = (S_WEEKS as f64 * 3.0) / S_LANES as f64;
    let mut worst_team = 0usize;
    let mut worst_lane = 0usize;
    let mut worst_dev = 0.0f64;
    let mut worst_over = false;
    for t in 0..S_TEAMS {
        for l in 0..S_LANES {
            let dev = lane_counts[t * S_LANES + l] as f64 - target;
            if dev.abs() > worst_dev {
                worst_dev = dev.abs();
                worst_team = t;
                worst_lane = l;
                worst_over = dev > 0.0;
            }
        }
    }
    if worst_dev < 1.0 { return false; }

    let saved = *a;
    let mut changed = false;

    // Iterate all weeks, trying to fix the team's lane in each
    let w_start = rng.random_range(0..S_WEEKS);
    for w_off in 0..S_WEEKS {
        let w = (w_start + w_off) % S_WEEKS;
        let positions = find_team_pos(a, w, worst_team as u8);
        if positions.is_empty() { continue; }

        // Find team on the source lane (over: on worst_lane, under: not on worst_lane)
        let source: Vec<_> = if worst_over {
            positions.iter().filter(|&&(_, p, _)| p == worst_lane).collect()
        } else {
            positions.iter().filter(|&&(_, p, _)| p != worst_lane).collect()
        };
        if source.is_empty() { continue; }
        let &(sf, pf, side_f) = source[rng.random_range(0..source.len())];

        // Strategy 1: pair-swap within the same slot
        let target_lanes: Vec<usize> = if worst_over {
            (0..S_PAIRS).filter(|&p| p != worst_lane && is_valid_position(sf, p) && a[w][sf][p].0 != EMPTY).collect()
        } else {
            vec![worst_lane].into_iter().filter(|&p| is_valid_position(sf, p) && a[w][sf][p].0 != EMPTY).collect()
        };
        let mut did_swap = false;
        for &tp in &target_lanes {
            // Pair-swap: swap entire matchups between pf and tp in slot sf
            let tmp = a[w][sf][pf];
            a[w][sf][pf] = a[w][sf][tp];
            a[w][sf][tp] = tmp;
            changed = true;
            did_swap = true;
            break;
        }
        if did_swap { continue; }

        // Strategy 2: team-swap across slots
        let mut did_cross_slot = false;
        for s in 0..S_SLOTS {
            if s == sf { continue; }
            if did_cross_slot { break; }
            for p in 0..S_PAIRS {
                if !is_valid_position(s, p) || a[w][s][p].0 == EMPTY { continue; }
                let on_target = if worst_over { p != worst_lane } else { p == worst_lane };
                if !on_target { continue; }

                let side_t = rng.random_range(0..2usize);
                let their_team = if side_t == 0 { a[w][s][p].0 } else { a[w][s][p].1 };

                if team_in_slot(a, w, s, worst_team as u8, Some(p)) { continue; }
                if team_in_slot(a, w, sf, their_team, Some(pf)) { continue; }
                let my_opp = if side_f == 0 { a[w][sf][pf].1 } else { a[w][sf][pf].0 };
                let their_opp = if side_t == 0 { a[w][s][p].1 } else { a[w][s][p].0 };
                if my_opp == their_team || their_opp == worst_team as u8 { continue; }

                if side_f == 0 { a[w][sf][pf].0 = their_team; } else { a[w][sf][pf].1 = their_team; }
                if side_t == 0 { a[w][s][p].0 = worst_team as u8; } else { a[w][s][p].1 = worst_team as u8; }
                changed = true;
                did_cross_slot = true;
                break;
            }
        }
    }

    if !changed {
        return false;
    }

    let new_cost = evaluate_summer(a, w8);
    let delta = new_cost.total as i64 - cost.total as i64;
    if sa_accept(delta, temp, rng) {
        *cost = new_cost;
        if cost.total < *best_cost { *best_cost = cost.total; *best_a = *a; }
        true
    } else {
        *a = saved;
        false
    }
}

// ─── Move 12: guided_break_fix — fix post-break lane switches ─────────────

fn do_guided_break_fix(
    a: &mut SummerAssignment, cost: &mut SummerCostBreakdown,
    best_a: &mut SummerAssignment, best_cost: &mut u32,
    w8: &SummerWeights, temp: f64, rng: &mut SmallRng,
) -> bool {
    let w = rng.random_range(0..S_WEEKS);
    let t_start = rng.random_range(0..S_TEAMS);

    // Find a team with a post-break lane change in this week
    for t_off in 0..S_TEAMS {
        let t = (t_start + t_off) % S_TEAMS;
        let positions = find_team_pos(a, w, t as u8);
        if positions.len() < 2 { continue; }

        // Find a post-break lane change: gap >= 1 between consecutive games, different pairs
        for i in 0..(positions.len() - 1) {
            let (s1, p1, _) = positions[i];
            let (s2, p2, _) = positions[i + 1];
            let gap = s2 - s1 - 1;
            if gap == 0 || p1 == p2 { continue; }

            // Found a post-break lane switch. Determine which game is isolated:
            // If i==0 and i+1 has a consecutive neighbor after it, fix game i (the isolated one)
            // If i+1==last and i has a consecutive neighbor before it, fix game i+1
            // Otherwise pick randomly
            let (fix_idx, target_pair) = if i == 0 && positions.len() > 2 && positions[2].0 == s2 + 1 {
                // Games i+1 and i+2 are consecutive, game i is isolated → fix game i to match p2
                (0, p2)
            } else if i + 1 == positions.len() - 1 && i > 0 && positions[i - 1].0 == s1 - 1 {
                // Games i-1 and i are consecutive, game i+1 is isolated → fix game i+1 to match p1
                (i + 1, p1)
            } else if rng.random_range(0..2usize) == 0 {
                (i, p2)
            } else {
                (i + 1, p1)
            };

            let (fix_s, fix_p, fix_side) = positions[fix_idx];
            if fix_p == target_pair { continue; } // already on the right lane

            // Opponent-swap within the slot: find a matchup on target_pair to swap with
            if !is_valid_position(fix_s, target_pair) || a[w][fix_s][target_pair].0 == EMPTY {
                continue;
            }

            let swap_side = rng.random_range(0..2usize);
            let their_team = if swap_side == 0 { a[w][fix_s][target_pair].0 } else { a[w][fix_s][target_pair].1 };
            let my_opp = if fix_side == 0 { a[w][fix_s][fix_p].1 } else { a[w][fix_s][fix_p].0 };
            let their_opp = if swap_side == 0 { a[w][fix_s][target_pair].1 } else { a[w][fix_s][target_pair].0 };
            if my_opp == their_team || their_opp == t as u8 { continue; }

            let saved = *a;
            if fix_side == 0 { a[w][fix_s][fix_p].0 = their_team; } else { a[w][fix_s][fix_p].1 = their_team; }
            if swap_side == 0 { a[w][fix_s][target_pair].0 = t as u8; } else { a[w][fix_s][target_pair].1 = t as u8; }

            let new_cost = evaluate_summer(a, w8);
            let delta = new_cost.total as i64 - cost.total as i64;
            if sa_accept(delta, temp, rng) {
                *cost = new_cost;
                if cost.total < *best_cost { *best_cost = cost.total; *best_a = *a; }
                return true;
            } else {
                *a = saved;
                return false;
            }
        }
    }
    false
}

// ─── Move 13: guided_lane_safe — lane balance fix that avoids creating lane switches

fn do_guided_lane_safe(
    a: &mut SummerAssignment, cost: &mut SummerCostBreakdown,
    best_a: &mut SummerAssignment, best_cost: &mut u32,
    w8: &SummerWeights, temp: f64, rng: &mut SmallRng,
) -> bool {
    // Compute lane counts
    let mut lane_counts = [0i32; S_TEAMS * S_LANES];
    for w in 0..S_WEEKS {
        for s in 0..S_SLOTS {
            for p in 0..S_PAIRS {
                let (t1, t2) = a[w][s][p];
                if t1 == EMPTY { continue; }
                lane_counts[t1 as usize * S_LANES + p] += 1;
                lane_counts[t2 as usize * S_LANES + p] += 1;
            }
        }
    }

    let target = (S_WEEKS as f64 * 3.0) / S_LANES as f64;
    let mut worst_team = 0usize;
    let mut worst_lane = 0usize;
    let mut worst_dev = 0.0f64;
    let mut worst_over = false;
    for t in 0..S_TEAMS {
        for l in 0..S_LANES {
            let dev = lane_counts[t * S_LANES + l] as f64 - target;
            if dev.abs() > worst_dev {
                worst_dev = dev.abs();
                worst_team = t;
                worst_lane = l;
                worst_over = dev > 0.0;
            }
        }
    }
    if worst_dev < 1.0 { return false; }

    let saved = *a;
    let w_start = rng.random_range(0..S_WEEKS);
    for off in 0..S_WEEKS {
        let w = (w_start + off) % S_WEEKS;
        let positions = find_team_pos(a, w, worst_team as u8);
        if positions.is_empty() { continue; }

        // Find a position on the source lane (over: on worst_lane, under: not on worst_lane)
        let source_pos: Vec<_> = if worst_over {
            positions.iter().filter(|&&(_, p, _)| p == worst_lane).collect()
        } else {
            positions.iter().filter(|&&(_, p, _)| p != worst_lane).collect()
        };
        if source_pos.is_empty() { continue; }
        let &(s, p_src, side) = source_pos[rng.random_range(0..source_pos.len())];

        // Find which lane pair an adjacent game uses (so swapping to it won't create a lane switch)
        let src_idx = positions.iter().position(|&(ps, pp, _)| ps == s && pp == p_src).unwrap();
        let mut adjacent_lane = None;
        if src_idx > 0 {
            let (prev_s, prev_p, _) = positions[src_idx - 1];
            if s - prev_s <= 1 { adjacent_lane = Some(prev_p); }
        }
        if adjacent_lane.is_none() && src_idx + 1 < positions.len() {
            let (next_s, next_p, _) = positions[src_idx + 1];
            if next_s - s <= 1 { adjacent_lane = Some(next_p); }
        }

        let target_p = match adjacent_lane {
            Some(p) if p != p_src => p,
            _ => continue, // no adjacent game on a different lane, or already on same lane
        };

        // Only swap if it helps balance (moving toward target_p should reduce the imbalance)
        let helps = if worst_over { target_p != worst_lane } else { target_p == worst_lane };
        if !helps { continue; }

        // Opponent-swap within the slot
        if !is_valid_position(s, target_p) || a[w][s][target_p].0 == EMPTY { continue; }

        let swap_side = rng.random_range(0..2usize);
        let their_team = if swap_side == 0 { a[w][s][target_p].0 } else { a[w][s][target_p].1 };
        let my_opp = if side == 0 { a[w][s][p_src].1 } else { a[w][s][p_src].0 };
        let their_opp = if swap_side == 0 { a[w][s][target_p].1 } else { a[w][s][target_p].0 };
        if my_opp == their_team || their_opp == worst_team as u8 { continue; }

        if side == 0 { a[w][s][p_src].0 = their_team; } else { a[w][s][p_src].1 = their_team; }
        if swap_side == 0 { a[w][s][target_p].0 = worst_team as u8; } else { a[w][s][target_p].1 = worst_team as u8; }

        let new_cost = evaluate_summer(a, w8);
        let delta = new_cost.total as i64 - cost.total as i64;
        if sa_accept(delta, temp, rng) {
            *cost = new_cost;
            if cost.total < *best_cost { *best_cost = cost.total; *best_a = *a; }
            return true;
        } else {
            *a = saved;
            return false;
        }
    }
    false
}

// ─── Move 14: guided_lane_pair — swap two teams with complementary lane imbalances

fn do_guided_lane_pair(
    a: &mut SummerAssignment, cost: &mut SummerCostBreakdown,
    best_a: &mut SummerAssignment, best_cost: &mut u32,
    w8: &SummerWeights, temp: f64, rng: &mut SmallRng,
) -> bool {
    // Compute lane counts
    let mut lane_counts = [0i32; S_TEAMS * S_LANES];
    for w in 0..S_WEEKS {
        for s in 0..S_SLOTS {
            for p in 0..S_PAIRS {
                let (t1, t2) = a[w][s][p];
                if t1 == EMPTY { continue; }
                lane_counts[t1 as usize * S_LANES + p] += 1;
                lane_counts[t2 as usize * S_LANES + p] += 1;
            }
        }
    }

    let target = (S_WEEKS as f64 * 3.0) / S_LANES as f64;

    // Find two teams with complementary imbalances: team_a over on lane_x, team_b over on lane_y
    let t_start = rng.random_range(0..S_TEAMS);
    for t_off in 0..S_TEAMS {
        let ta = (t_start + t_off) % S_TEAMS;
        // Find ta's most overrepresented lane
        let mut best_lane_a = 0usize;
        let mut best_dev_a = 0.0f64;
        for l in 0..S_LANES {
            let dev = lane_counts[ta * S_LANES + l] as f64 - target;
            if dev > best_dev_a { best_dev_a = dev; best_lane_a = l; }
        }
        if best_dev_a < 1.0 { continue; }

        // Find team_b that is overrepresented on a different lane and underrepresented on lane_a's lane
        for tb in 0..S_TEAMS {
            if tb == ta { continue; }
            let mut best_lane_b = 0usize;
            let mut best_dev_b = 0.0f64;
            for l in 0..S_LANES {
                let dev = lane_counts[tb * S_LANES + l] as f64 - target;
                if dev > best_dev_b { best_dev_b = dev; best_lane_b = l; }
            }
            if best_dev_b < 1.0 || best_lane_b == best_lane_a { continue; }

            // Check complementary: ta should be under on lane_b, tb under on lane_a
            let ta_on_lb = lane_counts[ta * S_LANES + best_lane_b] as f64 - target;
            let tb_on_la = lane_counts[tb * S_LANES + best_lane_a] as f64 - target;
            if ta_on_lb >= 0.0 || tb_on_la >= 0.0 { continue; }

            // Find a week where ta is on lane_a and tb is on lane_b in the same slot
            let saved = *a;
            let w_start = rng.random_range(0..S_WEEKS);
            for w_off in 0..S_WEEKS {
                let w = (w_start + w_off) % S_WEEKS;
                // Find ta on best_lane_a in some slot
                let mut ta_pos: Option<(usize, usize)> = None;
                let mut tb_pos: Option<(usize, usize)> = None;
                for s in 0..S_SLOTS {
                    let (t1, t2) = a[w][s][best_lane_a];
                    if t1 as usize == ta { ta_pos = Some((s, 0)); }
                    else if t2 as usize == ta { ta_pos = Some((s, 1)); }
                }
                if ta_pos.is_none() { continue; }
                let (sa, side_a) = ta_pos.unwrap();

                // Find tb on best_lane_b in the same slot
                if !is_valid_position(sa, best_lane_b) { continue; }
                let (t1, t2) = a[w][sa][best_lane_b];
                if t1 as usize == tb { tb_pos = Some((sa, 0)); }
                else if t2 as usize == tb { tb_pos = Some((sa, 1)); }
                if tb_pos.is_none() { continue; }
                let (_sb, side_b) = tb_pos.unwrap();

                // Opponent-swap: ta and tb exchange lane pairs within the same slot
                let opp_a = if side_a == 0 { a[w][sa][best_lane_a].1 } else { a[w][sa][best_lane_a].0 };
                let opp_b = if side_b == 0 { a[w][sa][best_lane_b].1 } else { a[w][sa][best_lane_b].0 };
                if opp_a == tb as u8 || opp_b == ta as u8 { continue; }

                if side_a == 0 { a[w][sa][best_lane_a].0 = tb as u8; } else { a[w][sa][best_lane_a].1 = tb as u8; }
                if side_b == 0 { a[w][sa][best_lane_b].0 = ta as u8; } else { a[w][sa][best_lane_b].1 = ta as u8; }

                let new_cost = evaluate_summer(a, w8);
                let delta = new_cost.total as i64 - cost.total as i64;
                if sa_accept(delta, temp, rng) {
                    *cost = new_cost;
                    if cost.total < *best_cost { *best_cost = cost.total; *best_a = *a; }
                    return true;
                } else {
                    *a = saved;
                    return false;
                }
            }
        }
    }
    false
}

// ─── Exhaustive local search ───────────────────────────────────────────────

fn exhaustive_local_search(
    a: &mut SummerAssignment, cost: &mut SummerCostBreakdown,
    best_a: &mut SummerAssignment, best_cost: &mut u32,
    w8: &SummerWeights, rng: &mut SmallRng,
) {
    let w = rng.random_range(0..S_WEEKS);
    let filled = filled_positions(a, w);
    if filled.len() < 2 { return; }

    let mut best_delta = 0i64;
    let mut best_move: Option<((usize, usize, usize), (usize, usize, usize))> = None;

    // Try team swaps between all pairs of filled positions in different slots
    for i in 0..filled.len() {
        for j in (i + 1)..filled.len() {
            let (s1, p1) = filled[i];
            let (s2, p2) = filled[j];
            if s1 == s2 { continue; }

            for side1 in 0..2usize {
                for side2 in 0..2usize {
                    let t1 = if side1 == 0 { a[w][s1][p1].0 } else { a[w][s1][p1].1 };
                    let t2 = if side2 == 0 { a[w][s2][p2].0 } else { a[w][s2][p2].1 };

                    if team_in_slot(a, w, s2, t1, Some(p2)) { continue; }
                    if team_in_slot(a, w, s1, t2, Some(p1)) { continue; }

                    // Check we don't create same-team matchups
                    let opp1 = if side1 == 0 { a[w][s1][p1].1 } else { a[w][s1][p1].0 };
                    let opp2 = if side2 == 0 { a[w][s2][p2].1 } else { a[w][s2][p2].0 };
                    if opp1 == t2 || opp2 == t1 { continue; }

                    if side1 == 0 { a[w][s1][p1].0 = t2; } else { a[w][s1][p1].1 = t2; }
                    if side2 == 0 { a[w][s2][p2].0 = t1; } else { a[w][s2][p2].1 = t1; }

                    let nc = evaluate_summer(a, w8);
                    let d = nc.total as i64 - cost.total as i64;
                    if d < best_delta {
                        best_delta = d;
                        best_move = Some(((s1, p1, side1), (s2, p2, side2)));
                    }

                    // Undo
                    if side1 == 0 { a[w][s1][p1].0 = t1; } else { a[w][s1][p1].1 = t1; }
                    if side2 == 0 { a[w][s2][p2].0 = t2; } else { a[w][s2][p2].1 = t2; }
                }
            }
        }
    }

    if let Some(((s1, p1, side1), (s2, p2, side2))) = best_move {
        {
            let t1 = if side1 == 0 { a[w][s1][p1].0 } else { a[w][s1][p1].1 };
            let t2 = if side2 == 0 { a[w][s2][p2].0 } else { a[w][s2][p2].1 };
            if side1 == 0 { a[w][s1][p1].0 = t2; } else { a[w][s1][p1].1 = t2; }
            if side2 == 0 { a[w][s2][p2].0 = t1; } else { a[w][s2][p2].1 = t1; }
        }
        *cost = evaluate_summer(a, w8);
        if cost.total < *best_cost { *best_cost = cost.total; *best_a = *a; }
    }
}

// ─── Helpers ───────────────────────────────────────────────────────────────

/// Find all positions of a team in a given week: (slot, pair, side)
fn find_team_pos(a: &SummerAssignment, w: usize, team: u8) -> Vec<(usize, usize, usize)> {
    let mut v = Vec::new();
    for s in 0..S_SLOTS {
        for p in 0..S_PAIRS {
            let (t1, t2) = a[w][s][p];
            if t1 == team { v.push((s, p, 0)); }
            if t2 == team { v.push((s, p, 1)); }
        }
    }
    v
}

// ─── Move stats (adaptive selection) ───────────────────────────────────────

struct MoveStats {
    attempts: [u64; NUM_MOVES],
    accepts: [u64; NUM_MOVES],
    cumulative: [f64; NUM_MOVES],
    last_rates: [f64; NUM_MOVES],
    last_shares: [f64; NUM_MOVES],
}

impl MoveStats {
    fn new() -> Self {
        let mut s = MoveStats {
            attempts: [0; NUM_MOVES],
            accepts: [0; NUM_MOVES],
            cumulative: [0.0; NUM_MOVES],
            last_rates: [0.0; NUM_MOVES],
            last_shares: [0.0; NUM_MOVES],
        };
        s.recompute();
        s
    }

    fn recompute(&mut self) {
        let total_attempts: u64 = self.attempts.iter().sum();
        let mut weights = [0.0f64; NUM_MOVES];
        for m in 0..NUM_MOVES {
            let rate = if self.attempts[m] > 0 {
                self.accepts[m] as f64 / self.attempts[m] as f64
            } else {
                0.5
            };
            self.last_rates[m] = rate;
            self.last_shares[m] = if total_attempts > 0 {
                self.attempts[m] as f64 / total_attempts as f64
            } else {
                0.0
            };
            weights[m] = BASE_WEIGHTS[m] * (0.1 + rate);
        }
        let sum: f64 = weights.iter().sum();
        let mut cum = 0.0;
        for m in 0..NUM_MOVES {
            cum += weights[m] / sum;
            self.cumulative[m] = cum;
        }
        self.cumulative[NUM_MOVES - 1] = 1.0;
        self.attempts = [0; NUM_MOVES];
        self.accepts = [0; NUM_MOVES];
    }

    fn select(&self, rand_val: f64) -> usize {
        for m in 0..NUM_MOVES {
            if rand_val < self.cumulative[m] { return m; }
        }
        NUM_MOVES - 1
    }
}
