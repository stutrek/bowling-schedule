use std::fs;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};
use std::thread;

use rand::rngs::SmallRng;
use rand::{Rng, SeedableRng};
use solver_core::*;

use crate::now_iso;

pub struct SharedBest {
    pub cost: u32,
    pub assignment: Assignment,
}

impl SharedBest {
    pub fn new() -> Self {
        SharedBest {
            cost: u32::MAX,
            assignment: [[[0u8; POS]; QUADS]; WEEKS],
        }
    }
}

const MAX_ITERATIONS: u64 = 6_000_000_000;
const SYNC_INTERVAL: u64 = 100_000;
const STATS_RECOMPUTE: u64 = 10_000;
const PROGRESS_INTERVAL: u64 = 50_000_000;
const STAGNATION_ITERS: u64 = 500_000_000;
const FOCUS_ITERS: u64 = 1_000_000_000;

const QUAD_PAIRS: [(usize, usize); 6] = [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)];

const NUM_MOVES: usize = 11;
const BASE_WEIGHTS: [f64; NUM_MOVES] = [
    0.25, 0.15, 0.10, 0.08, 0.06, 0.06, 0.05, 0.06, 0.06, 0.06, 0.07,
];

struct MoveStats {
    attempts: [u64; NUM_MOVES],
    accepts: [u64; NUM_MOVES],
    cumulative: [f64; NUM_MOVES],
}

impl MoveStats {
    fn new() -> Self {
        let mut s = MoveStats {
            attempts: [0; NUM_MOVES],
            accepts: [0; NUM_MOVES],
            cumulative: [0.0; NUM_MOVES],
        };
        s.recompute();
        s
    }

    fn recompute(&mut self) {
        let mut weights = [0.0f64; NUM_MOVES];
        for m in 0..NUM_MOVES {
            let rate = if self.attempts[m] > 0 {
                self.accepts[m] as f64 / self.attempts[m] as f64
            } else {
                0.5
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
            if rand_val < self.cumulative[m] {
                return m;
            }
        }
        NUM_MOVES - 1
    }
}

fn find_team_in_week(a: &Assignment, w: usize, team: u8) -> Option<(usize, usize)> {
    for q in 0..QUADS {
        for p in 0..POS {
            if a[w][q][p] == team {
                return Some((q, p));
            }
        }
    }
    None
}

fn same_half(q1: usize, q2: usize) -> bool {
    (q1 < 2 && q2 < 2) || (q1 >= 2 && q2 >= 2)
}

fn guided_matchup(a: &mut Assignment, rng: &mut SmallRng) -> bool {
    let mut matchups = [false; TEAMS * TEAMS];
    for w in 0..WEEKS {
        for q in 0..QUADS {
            let [pa, pb, pc, pd] = a[w][q];
            for &(t1, t2) in &[(pa, pb), (pc, pd), (pa, pd), (pc, pb)] {
                matchups[t1.min(t2) as usize * TEAMS + t1.max(t2) as usize] = true;
            }
        }
    }

    let start = rng.random_range(0..TEAMS);
    let mut ta = 0u8;
    let mut tb = 0u8;
    let mut found = false;
    'outer: for off_i in 0..TEAMS {
        let i = (start + off_i) % TEAMS;
        for j in (i + 1)..TEAMS {
            if !matchups[i * TEAMS + j] {
                ta = i as u8;
                tb = j as u8;
                found = true;
                break 'outer;
            }
        }
    }
    if !found {
        return false;
    }

    let week_start = rng.random_range(0..WEEKS);
    for off in 0..WEEKS {
        let w = (week_start + off) % WEEKS;
        let (qa, _) = match find_team_in_week(a, w, ta) {
            Some(x) => x,
            None => continue,
        };
        let (qb, pb) = match find_team_in_week(a, w, tb) {
            Some(x) => x,
            None => continue,
        };
        if !same_half(qa, qb) || qa == qb {
            continue;
        }

        let candidates: Vec<usize> = (0..POS).filter(|&p| a[w][qa][p] != ta).collect();
        if candidates.is_empty() {
            continue;
        }
        let swap_pos = candidates[rng.random_range(0..candidates.len())];
        let tmp = a[w][qa][swap_pos];
        a[w][qa][swap_pos] = a[w][qb][pb];
        a[w][qb][pb] = tmp;
        return true;
    }
    false
}

fn guided_lane(a: &mut Assignment, rng: &mut SmallRng) -> bool {
    let mut lane_counts = [0i32; TEAMS * LANES];
    for w in 0..WEEKS {
        for q in 0..QUADS {
            let [pa, pb, pc, pd] = a[w][q];
            let lo = (q % 2) * 2;
            lane_counts[pa as usize * LANES + lo] += 2;
            lane_counts[pb as usize * LANES + lo] += 1;
            lane_counts[pb as usize * LANES + lo + 1] += 1;
            lane_counts[pc as usize * LANES + lo + 1] += 2;
            lane_counts[pd as usize * LANES + lo + 1] += 1;
            lane_counts[pd as usize * LANES + lo] += 1;
        }
    }

    let target_l = (WEEKS as f64 * 2.0) / LANES as f64;
    let mut worst_team = 0usize;
    let mut worst_dev = 0.0f64;
    for t in 0..TEAMS {
        for l in 0..LANES {
            let dev = (lane_counts[t * LANES + l] as f64 - target_l).abs();
            if dev > worst_dev {
                worst_dev = dev;
                worst_team = t;
            }
        }
    }
    if worst_dev < 1.0 {
        return false;
    }

    let start = rng.random_range(0..WEEKS * QUADS);
    for off in 0..WEEKS * QUADS {
        let idx = (start + off) % (WEEKS * QUADS);
        let w = idx / QUADS;
        let q = idx % QUADS;
        let mut team_pos = None;
        for p in 0..POS {
            if a[w][q][p] == worst_team as u8 {
                team_pos = Some(p);
                break;
            }
        }
        let tp = match team_pos {
            Some(p) => p,
            None => continue,
        };
        let mut swap_pos = rng.random_range(0..(POS - 1));
        if swap_pos >= tp {
            swap_pos += 1;
        }
        a[w][q].swap(tp, swap_pos);
        return true;
    }
    false
}

fn guided_early_late(a: &mut Assignment, rng: &mut SmallRng) -> bool {
    let mut early_count = [0i32; TEAMS];
    for w in 0..WEEKS {
        for q in 0..2 {
            for p in 0..POS {
                early_count[a[w][q][p] as usize] += 1;
            }
        }
    }

    let target_e = WEEKS as f64 / 2.0;
    let mut worst_team = 0usize;
    let mut worst_dev = 0.0f64;
    let mut too_many_early = false;
    for t in 0..TEAMS {
        let dev = early_count[t] as f64 - target_e;
        if dev.abs() > worst_dev {
            worst_dev = dev.abs();
            worst_team = t;
            too_many_early = dev > 0.0;
        }
    }
    if worst_dev < 1.0 {
        return false;
    }

    let start = rng.random_range(0..WEEKS);
    for off in 0..WEEKS {
        let w = (start + off) % WEEKS;
        let team = worst_team as u8;
        let in_early = (0..2).any(|q| (0..POS).any(|p| a[w][q][p] == team));
        if (too_many_early && in_early) || (!too_many_early && !in_early) {
            let tmp0 = a[w][0];
            let tmp1 = a[w][1];
            a[w][0] = a[w][2];
            a[w][2] = tmp0;
            a[w][1] = a[w][3];
            a[w][3] = tmp1;
            return true;
        }
    }
    false
}

fn pick_focus_constraint(cost: &CostBreakdown, original: &Weights) -> (&'static str, Weights) {
    let candidates = [
        (cost.matchup_balance, "matchup"),
        (cost.consecutive_opponents, "consec"),
        (cost.early_late_balance, "el_bal"),
        (cost.early_late_alternation, "el_alt"),
        (cost.lane_balance, "lane"),
        (cost.lane_switch_balance, "switch"),
        (cost.late_lane_balance, "ll_bal"),
        (cost.commissioner_overlap, "comm"),
    ];

    let &(_, name) = candidates
        .iter()
        .filter(|(v, _)| *v > 0)
        .max_by_key(|(v, _)| *v)
        .unwrap_or(&(0, "matchup"));

    let mut w = Weights {
        matchup_zero: 0,
        matchup_triple: 0,
        consecutive_opponents: 0,
        early_late_balance: 0.0,
        early_late_alternation: 0,
        lane_balance: 0.0,
        lane_switch: 0.0,
        late_lane_balance: 0.0,
        commissioner_overlap: 0,
    };

    match name {
        "matchup" => {
            w.matchup_zero = original.matchup_zero;
            w.matchup_triple = original.matchup_triple;
        }
        "consec" => w.consecutive_opponents = original.consecutive_opponents,
        "el_bal" => w.early_late_balance = original.early_late_balance,
        "el_alt" => w.early_late_alternation = original.early_late_alternation,
        "lane" => w.lane_balance = original.lane_balance,
        "switch" => w.lane_switch = original.lane_switch,
        "ll_bal" => w.late_lane_balance = original.late_lane_balance,
        "comm" => w.commissioner_overlap = original.commissioner_overlap,
        _ => unreachable!(),
    }

    (name, w)
}

fn sa_accept(delta: i64, temp: f64, rng: &mut SmallRng) -> bool {
    if delta < 0 {
        true
    } else if delta == 0 {
        rng.random::<f64>() < 0.2
    } else {
        rng.random::<f64>() < (-delta as f64 / temp).exp()
    }
}

pub fn run_cpu_workers(
    num_cores: usize,
    w8: Weights,
    shared_best: Arc<Mutex<SharedBest>>,
    shutdown: Arc<AtomicBool>,
    results_dir: String,
) -> Vec<thread::JoinHandle<()>> {
    if num_cores == 0 {
        return vec![];
    }

    let temps: Vec<f64> = if num_cores == 1 {
        vec![1.0]
    } else {
        (0..num_cores)
            .map(|i| 0.1 * (100.0f64).powf(i as f64 / (num_cores - 1) as f64))
            .collect()
    };

    eprintln!(
        "[{}] CPU SA: {} cores, temps: {:.1}..{:.1}",
        now_iso(),
        num_cores,
        temps[0],
        temps[num_cores - 1],
    );

    let w8 = Arc::new(w8);
    let temps = Arc::new(temps);
    let results_dir = Arc::new(results_dir);

    (0..num_cores)
        .map(|core_id| {
            let shutdown = Arc::clone(&shutdown);
            let shared_best = Arc::clone(&shared_best);
            let w8 = Arc::clone(&w8);
            let temps = Arc::clone(&temps);
            let results_dir = Arc::clone(&results_dir);

            thread::spawn(move || {
                let mut rng = SmallRng::from_os_rng();
                let temp = temps[core_id];
                let cold_temp = temps[0];
                let mut last_saved: Option<Assignment> = None;

                loop {
                    if shutdown.load(Ordering::Relaxed) {
                        return;
                    }

                    let mut a = {
                        let sb = shared_best.lock().unwrap();
                        if sb.cost < u32::MAX {
                            let mut a = sb.assignment;
                            perturb(&mut a, &mut rng, 5 + core_id * 2);
                            a
                        } else {
                            random_assignment(&mut rng)
                        }
                    };

                    let mut active_w8 = (*w8).clone();
                    let mut cost = evaluate(&a, &active_w8);
                    let mut best_a = a;
                    let mut best_cost = cost.total;
                    let mut stats = MoveStats::new();
                    let mut last_improvement: u64 = 0;
                    let mut focus_name: Option<String> = None;
                    let mut focus_end: u64 = 0;
                    let mut active_temp = temp;

                    for i in 0..MAX_ITERATIONS {
                        if shutdown.load(Ordering::Relaxed) {
                            break;
                        }
                        if best_cost == 0 {
                            break;
                        }

                        if i > 0 && i % STATS_RECOMPUTE == 0 {
                            stats.recompute();

                            if focus_name.is_some() && (i >= focus_end || cost.total == 0) {
                                let name = focus_name.take().unwrap();
                                active_w8 = (*w8).clone();
                                active_temp = temp;
                                cost = evaluate(&a, &active_w8);
                                if cost.total < best_cost {
                                    best_a = a;
                                    best_cost = cost.total;
                                }
                                eprintln!(
                                    "[{}] cpu {} exiting focus (\x1b[1m{}\x1b[0m) | cost: {}",
                                    now_iso(),
                                    core_id,
                                    name,
                                    cost.total,
                                );
                                last_improvement = i;
                            }

                            if focus_name.is_none()
                                && i > last_improvement + STAGNATION_ITERS
                            {
                                let base = {
                                    let sb = shared_best.lock().unwrap();
                                    if sb.cost < u32::MAX {
                                        sb.assignment
                                    } else {
                                        best_a
                                    }
                                };
                                let full_cost = evaluate(&base, &w8);
                                if full_cost.total > 0 {
                                    let (name, fw) = pick_focus_constraint(&full_cost, &w8);
                                    eprintln!(
                                        "[{}] cpu {} focusing on \x1b[1m{}\x1b[0m (stagnant {}M iters)",
                                        now_iso(),
                                        core_id,
                                        name,
                                        (i - last_improvement) / 1_000_000,
                                    );
                                    active_w8 = fw;
                                    focus_name = Some(name.to_string());
                                    focus_end = i + FOCUS_ITERS;
                                    active_temp = cold_temp;
                                    a = base;
                                    perturb(&mut a, &mut rng, 10);
                                    cost = evaluate(&a, &active_w8);
                                } else {
                                    last_improvement = i;
                                }
                            }
                        }

                        // Exhaustive single-quad-pair search (periodic)
                        if i > 0 && i % 100_000 == 0 && cost.total > 0 {
                            let ew = rng.random_range(0..WEEKS);
                            let (eq1, eq2) =
                                QUAD_PAIRS[rng.random_range(0..QUAD_PAIRS.len())];
                            let mut best_delta = 0i64;
                            let mut best_swap: Option<(usize, usize)> = None;
                            for ep1 in 0..POS {
                                for ep2 in 0..POS {
                                    let tmp = a[ew][eq1][ep1];
                                    a[ew][eq1][ep1] = a[ew][eq2][ep2];
                                    a[ew][eq2][ep2] = tmp;
                                    let nc = evaluate(&a, &active_w8);
                                    let d = nc.total as i64 - cost.total as i64;
                                    if d < best_delta {
                                        best_delta = d;
                                        best_swap = Some((ep1, ep2));
                                    }
                                    a[ew][eq2][ep2] = a[ew][eq1][ep1];
                                    a[ew][eq1][ep1] = tmp;
                                }
                            }
                            if let Some((ep1, ep2)) = best_swap {
                                let tmp = a[ew][eq1][ep1];
                                a[ew][eq1][ep1] = a[ew][eq2][ep2];
                                a[ew][eq2][ep2] = tmp;
                                cost = evaluate(&a, &active_w8);
                                if focus_name.is_none() && cost.total < best_cost {
                                    best_cost = cost.total;
                                    best_a = a;
                                    last_improvement = i;
                                }
                            }
                        }

                        // Compound move
                        let compound_prob =
                            ((1000.0 - cost.total as f64) / 800.0).clamp(0.0, 0.5);
                        if rng.random::<f64>() < compound_prob {
                            let saved = a;
                            let max_swaps = if cost.total < 200 {
                                12
                            } else if cost.total < 400 {
                                6
                            } else {
                                4
                            };
                            let num_swaps = rng.random_range(2..=max_swaps);
                            for _ in 0..num_swaps {
                                let w = rng.random_range(0..WEEKS);
                                let (q1, q2) =
                                    QUAD_PAIRS[rng.random_range(0..QUAD_PAIRS.len())];
                                let p1 = rng.random_range(0..POS);
                                let p2 = rng.random_range(0..POS);
                                let tmp = a[w][q1][p1];
                                a[w][q1][p1] = a[w][q2][p2];
                                a[w][q2][p2] = tmp;
                            }
                            let new_cost = evaluate(&a, &active_w8);
                            let delta = new_cost.total as i64 - cost.total as i64;
                            if sa_accept(delta, active_temp, &mut rng) {
                                cost = new_cost;
                                if focus_name.is_none() && cost.total < best_cost {
                                    best_cost = cost.total;
                                    best_a = a;
                                    last_improvement = i;
                                }
                            } else {
                                a = saved;
                            }
                        } else {
                            let move_id = stats.select(rng.random::<f64>());
                            stats.attempts[move_id] += 1;

                            let accepted = match move_id {
                                0 => {
                                    // Inter-quad player swap
                                    let w = rng.random_range(0..WEEKS);
                                    let (q1, q2) =
                                        QUAD_PAIRS[rng.random_range(0..QUAD_PAIRS.len())];
                                    let p1 = rng.random_range(0..POS);
                                    let p2 = rng.random_range(0..POS);
                                    let tmp = a[w][q1][p1];
                                    a[w][q1][p1] = a[w][q2][p2];
                                    a[w][q2][p2] = tmp;
                                    let new_cost = evaluate(&a, &active_w8);
                                    let delta = new_cost.total as i64 - cost.total as i64;
                                    if sa_accept(delta, active_temp, &mut rng) {
                                        cost = new_cost;
                                        if focus_name.is_none() && cost.total < best_cost {
                                            best_cost = cost.total;
                                            best_a = a;
                                            last_improvement = i;
                                        }
                                        true
                                    } else {
                                        a[w][q2][p2] = a[w][q1][p1];
                                        a[w][q1][p1] = tmp;
                                        false
                                    }
                                }
                                1 => {
                                    // Intra-quad player swap
                                    let w = rng.random_range(0..WEEKS);
                                    let q = rng.random_range(0..QUADS);
                                    let p1 = rng.random_range(0..POS);
                                    let mut p2 = rng.random_range(0..(POS - 1));
                                    if p2 >= p1 {
                                        p2 += 1;
                                    }
                                    let tmp = a[w][q][p1];
                                    a[w][q][p1] = a[w][q][p2];
                                    a[w][q][p2] = tmp;
                                    let new_cost = evaluate(&a, &active_w8);
                                    let delta = new_cost.total as i64 - cost.total as i64;
                                    if sa_accept(delta, active_temp, &mut rng) {
                                        cost = new_cost;
                                        if focus_name.is_none() && cost.total < best_cost {
                                            best_cost = cost.total;
                                            best_a = a;
                                            last_improvement = i;
                                        }
                                        true
                                    } else {
                                        a[w][q][p2] = a[w][q][p1];
                                        a[w][q][p1] = tmp;
                                        false
                                    }
                                }
                                2 => {
                                    // Team cross-week swap
                                    let team = rng.random_range(0..TEAMS) as u8;
                                    let w1 = rng.random_range(0..WEEKS);
                                    let mut w2 = rng.random_range(0..(WEEKS - 1));
                                    if w2 >= w1 {
                                        w2 += 1;
                                    }
                                    let pos1 = find_team_in_week(&a, w1, team);
                                    let pos2 = find_team_in_week(&a, w2, team);
                                    if let (Some((q1, p1)), Some((q2, p2))) = (pos1, pos2) {
                                        let save = (
                                            a[w1][q1][p1],
                                            a[w1][q2][p2],
                                            a[w2][q1][p1],
                                            a[w2][q2][p2],
                                        );
                                        let other1 = a[w2][q1][p1];
                                        let other2 = a[w1][q2][p2];
                                        a[w1][q1][p1] = other2;
                                        a[w1][q2][p2] = team;
                                        a[w2][q2][p2] = other1;
                                        a[w2][q1][p1] = team;
                                        let new_cost = evaluate(&a, &active_w8);
                                        let delta =
                                            new_cost.total as i64 - cost.total as i64;
                                        if sa_accept(delta, active_temp, &mut rng) {
                                            cost = new_cost;
                                            if focus_name.is_none()
                                                && cost.total < best_cost
                                            {
                                                best_cost = cost.total;
                                                best_a = a;
                                                last_improvement = i;
                                            }
                                            true
                                        } else {
                                            a[w1][q1][p1] = save.0;
                                            a[w1][q2][p2] = save.1;
                                            a[w2][q1][p1] = save.2;
                                            a[w2][q2][p2] = save.3;
                                            false
                                        }
                                    } else {
                                        false
                                    }
                                }
                                3 => {
                                    // Quad swap
                                    let w = rng.random_range(0..WEEKS);
                                    let (q1, q2) =
                                        QUAD_PAIRS[rng.random_range(0..QUAD_PAIRS.len())];
                                    let tmp = a[w][q1];
                                    a[w][q1] = a[w][q2];
                                    a[w][q2] = tmp;
                                    let new_cost = evaluate(&a, &active_w8);
                                    let delta = new_cost.total as i64 - cost.total as i64;
                                    if sa_accept(delta, active_temp, &mut rng) {
                                        cost = new_cost;
                                        if focus_name.is_none() && cost.total < best_cost {
                                            best_cost = cost.total;
                                            best_a = a;
                                            last_improvement = i;
                                        }
                                        true
                                    } else {
                                        a[w][q2] = a[w][q1];
                                        a[w][q1] = tmp;
                                        false
                                    }
                                }
                                4 => {
                                    // Week swap
                                    let w1 = rng.random_range(0..WEEKS);
                                    let mut w2 = rng.random_range(0..(WEEKS - 1));
                                    if w2 >= w1 {
                                        w2 += 1;
                                    }
                                    let tmp = a[w1];
                                    a[w1] = a[w2];
                                    a[w2] = tmp;
                                    let new_cost = evaluate(&a, &active_w8);
                                    let delta = new_cost.total as i64 - cost.total as i64;
                                    if sa_accept(delta, active_temp, &mut rng) {
                                        cost = new_cost;
                                        if focus_name.is_none() && cost.total < best_cost {
                                            best_cost = cost.total;
                                            best_a = a;
                                            last_improvement = i;
                                        }
                                        true
                                    } else {
                                        a[w2] = a[w1];
                                        a[w1] = tmp;
                                        false
                                    }
                                }
                                5 => {
                                    // Early/late flip
                                    let w = rng.random_range(0..WEEKS);
                                    let tmp0 = a[w][0];
                                    let tmp1 = a[w][1];
                                    a[w][0] = a[w][2];
                                    a[w][2] = tmp0;
                                    a[w][1] = a[w][3];
                                    a[w][3] = tmp1;
                                    let new_cost = evaluate(&a, &active_w8);
                                    let delta = new_cost.total as i64 - cost.total as i64;
                                    if sa_accept(delta, active_temp, &mut rng) {
                                        cost = new_cost;
                                        if focus_name.is_none() && cost.total < best_cost {
                                            best_cost = cost.total;
                                            best_a = a;
                                            last_improvement = i;
                                        }
                                        true
                                    } else {
                                        let t0 = a[w][0];
                                        let t1 = a[w][1];
                                        a[w][0] = a[w][2];
                                        a[w][2] = t0;
                                        a[w][1] = a[w][3];
                                        a[w][3] = t1;
                                        false
                                    }
                                }
                                6 => {
                                    // Lane pair swap
                                    let w = rng.random_range(0..WEEKS);
                                    let tmp0 = a[w][0];
                                    let tmp2 = a[w][2];
                                    a[w][0] = a[w][1];
                                    a[w][1] = tmp0;
                                    a[w][2] = a[w][3];
                                    a[w][3] = tmp2;
                                    let new_cost = evaluate(&a, &active_w8);
                                    let delta = new_cost.total as i64 - cost.total as i64;
                                    if sa_accept(delta, active_temp, &mut rng) {
                                        cost = new_cost;
                                        if focus_name.is_none() && cost.total < best_cost {
                                            best_cost = cost.total;
                                            best_a = a;
                                            last_improvement = i;
                                        }
                                        true
                                    } else {
                                        let t1 = a[w][1];
                                        let t3 = a[w][3];
                                        a[w][1] = a[w][0];
                                        a[w][0] = t1;
                                        a[w][3] = a[w][2];
                                        a[w][2] = t3;
                                        false
                                    }
                                }
                                7 => {
                                    // Stay/switch rotation
                                    let w = rng.random_range(0..WEEKS);
                                    let q = rng.random_range(0..QUADS);
                                    a[w][q].swap(0, 1);
                                    a[w][q].swap(2, 3);
                                    let new_cost = evaluate(&a, &active_w8);
                                    let delta = new_cost.total as i64 - cost.total as i64;
                                    if sa_accept(delta, active_temp, &mut rng) {
                                        cost = new_cost;
                                        if focus_name.is_none() && cost.total < best_cost {
                                            best_cost = cost.total;
                                            best_a = a;
                                            last_improvement = i;
                                        }
                                        true
                                    } else {
                                        a[w][q].swap(0, 1);
                                        a[w][q].swap(2, 3);
                                        false
                                    }
                                }
                                8 => {
                                    // Guided matchup
                                    let saved = a;
                                    if guided_matchup(&mut a, &mut rng) {
                                        let new_cost = evaluate(&a, &active_w8);
                                        let delta =
                                            new_cost.total as i64 - cost.total as i64;
                                        if sa_accept(delta, active_temp, &mut rng) {
                                            cost = new_cost;
                                            if focus_name.is_none()
                                                && cost.total < best_cost
                                            {
                                                best_cost = cost.total;
                                                best_a = a;
                                                last_improvement = i;
                                            }
                                            true
                                        } else {
                                            a = saved;
                                            false
                                        }
                                    } else {
                                        false
                                    }
                                }
                                9 => {
                                    // Guided lane
                                    let saved = a;
                                    if guided_lane(&mut a, &mut rng) {
                                        let new_cost = evaluate(&a, &active_w8);
                                        let delta =
                                            new_cost.total as i64 - cost.total as i64;
                                        if sa_accept(delta, active_temp, &mut rng) {
                                            cost = new_cost;
                                            if focus_name.is_none()
                                                && cost.total < best_cost
                                            {
                                                best_cost = cost.total;
                                                best_a = a;
                                                last_improvement = i;
                                            }
                                            true
                                        } else {
                                            a = saved;
                                            false
                                        }
                                    } else {
                                        false
                                    }
                                }
                                _ => {
                                    // Guided early/late (move 10)
                                    let saved = a;
                                    if guided_early_late(&mut a, &mut rng) {
                                        let new_cost = evaluate(&a, &active_w8);
                                        let delta =
                                            new_cost.total as i64 - cost.total as i64;
                                        if sa_accept(delta, active_temp, &mut rng) {
                                            cost = new_cost;
                                            if focus_name.is_none()
                                                && cost.total < best_cost
                                            {
                                                best_cost = cost.total;
                                                best_a = a;
                                                last_improvement = i;
                                            }
                                            true
                                        } else {
                                            a = saved;
                                            false
                                        }
                                    } else {
                                        false
                                    }
                                }
                            };

                            if accepted {
                                stats.accepts[move_id] += 1;
                            }
                        }

                        // Sync with shared best
                        if i > 0 && i % SYNC_INTERVAL == 0 && focus_name.is_none() {
                            let mut sb = shared_best.lock().unwrap();
                            if best_cost < sb.cost {
                                sb.cost = best_cost;
                                sb.assignment = best_a;
                            } else if sb.cost < best_cost {
                                best_cost = sb.cost;
                                best_a = sb.assignment;
                                a = sb.assignment;
                                drop(sb);
                                perturb(&mut a, &mut rng, 3);
                                cost = evaluate(&a, &active_w8);
                                last_improvement = i;
                            }
                        }

                        if i > 0 && i % PROGRESS_INTERVAL == 0 {
                            let label = if i >= 1_000_000_000 {
                                format!("{:.2}B", i as f64 / 1_000_000_000.0)
                            } else {
                                format!("{}M", i / 1_000_000)
                            };
                            eprintln!(
                                "[{}] cpu {} @ {} | best: {} | temp: {:.2}{}",
                                now_iso(),
                                core_id,
                                label,
                                best_cost,
                                active_temp,
                                if let Some(ref name) = focus_name {
                                    format!(" | \x1b[1mfocus: {}\x1b[0m", name)
                                } else {
                                    String::new()
                                },
                            );
                        }
                    }

                    // End of run: sync and save
                    if best_cost < u32::MAX {
                        let mut sb = shared_best.lock().unwrap();
                        if best_cost < sb.cost {
                            sb.cost = best_cost;
                            sb.assignment = best_a;
                        }
                        drop(sb);

                        if best_cost <= 160 && last_saved.as_ref() != Some(&best_a) {
                            let ts = chrono::Local::now().format("%Y%m%d-%H%M%S%z");
                            let filename = format!(
                                "{}/{:04}-cpu{}-{}.tsv",
                                results_dir, best_cost, core_id, ts
                            );
                            let mut out = best_a;
                            reassign_commissioners(&mut out);
                            let _ = fs::write(&filename, assignment_to_tsv(&out));
                            eprintln!("[{}] Saved {}", now_iso(), filename);
                            last_saved = Some(best_a);
                        }
                    }

                    eprintln!(
                        "[{}] cpu {} run done | best: {}",
                        now_iso(),
                        core_id,
                        best_cost,
                    );
                }
            })
        })
        .collect()
}
