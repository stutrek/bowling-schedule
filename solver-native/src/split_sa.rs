use rand::rngs::SmallRng;
use rand::{Rng, SeedableRng};
use serde::Deserialize;
use std::fs;
use std::sync::atomic::{AtomicBool, AtomicU32, Ordering};
use std::sync::{Arc, Condvar, Mutex};
use std::thread;

use solver_native::*;

#[derive(Deserialize)]
struct Config {
    solver: SolverParams,
}

#[derive(Deserialize)]
struct SolverParams {
    t0: f64,
    #[serde(default = "default_progress_interval")]
    progress_interval: u64,
    #[serde(default)]
    temp_floor: f64,
    #[serde(default = "default_weights_path")]
    weights_path: String,
    #[serde(default)]
    cores: f64,
}

fn default_weights_path() -> String { "../weights.json".to_string() }
fn default_progress_interval() -> u64 { 10_000_000 }

const MAX_ITERATIONS: u64 = 6_000_000_000;
const SPLIT_WEEK: usize = 7;
const SWAP_INTERVAL: u64 = 100_000;
const STATS_RECOMPUTE: u64 = 10_000;

type LockMask = [[bool; QUADS]; WEEKS];

fn phase1_lock() -> LockMask {
    let mut m = [[false; QUADS]; WEEKS];
    m[SPLIT_WEEK][1] = true;
    m[SPLIT_WEEK][3] = true;
    for w in (SPLIT_WEEK + 1)..WEEKS {
        for q in 0..QUADS { m[w][q] = true; }
    }
    m
}

fn phase2_lock() -> LockMask {
    let mut m = [[false; QUADS]; WEEKS];
    for w in 0..SPLIT_WEEK {
        for q in 0..QUADS { m[w][q] = true; }
    }
    m[SPLIT_WEEK][0] = true;
    m[SPLIT_WEEK][2] = true;
    m
}

// ═══════════════════════════════════════════════════════════════════════════
// Precomputed unlocked indices
// ═══════════════════════════════════════════════════════════════════════════

struct ActiveIndices {
    active_weeks: Vec<usize>,
    full_weeks: Vec<usize>,
    active_quads: Vec<(usize, usize)>,
    active_quad_pairs: Vec<(usize, usize, usize)>,
}

fn build_active_indices(locked: &LockMask) -> ActiveIndices {
    let mut active_weeks = Vec::new();
    let mut full_weeks = Vec::new();
    let mut active_quads = Vec::new();
    let mut active_quad_pairs = Vec::new();

    for w in 0..WEEKS {
        let unlocked: Vec<usize> = (0..QUADS).filter(|&q| !locked[w][q]).collect();
        if !unlocked.is_empty() { active_weeks.push(w); }
        if unlocked.len() == QUADS { full_weeks.push(w); }
        for &q in &unlocked { active_quads.push((w, q)); }
        for i in 0..unlocked.len() {
            for j in (i + 1)..unlocked.len() {
                active_quad_pairs.push((w, unlocked[i], unlocked[j]));
            }
        }
    }

    ActiveIndices { active_weeks, full_weeks, active_quads, active_quad_pairs }
}

// ═══════════════════════════════════════════════════════════════════════════
// Masked evaluation
// ═══════════════════════════════════════════════════════════════════════════

struct MaskedCost {
    matchup_balance: u32,
    consecutive_opponents: u32,
    early_late_balance: u32,
    early_late_alternation: u32,
    lane_balance: u32,
    lane_switch_balance: u32,
    late_lane_balance: u32,
    total: u32,
}

fn evaluate_masked(
    a: &Assignment,
    w8: &Weights,
    locked: &LockMask,
    baseline_matchups: &[i32; TEAMS * TEAMS],
    full_schedule: bool,
) -> MaskedCost {
    let mut matchups = *baseline_matchups;
    let mut lane_counts = [0i32; TEAMS * LANES];
    let mut late_lane_counts = [0i32; TEAMS * LANES];
    let mut stay_count = [0i32; TEAMS];
    let mut early_count = [0i32; TEAMS];
    let mut early_late = [255u8; TEAMS * WEEKS];
    let mut week_matchup = [0u8; WEEKS * TEAMS * TEAMS];

    for w in 0..WEEKS {
        for q in 0..QUADS {
            let count_this = if full_schedule { true } else { !locked[w][q] };
            if !count_this { continue; }

            let [pa, pb, pc, pd] = a[w][q];
            let early: u8 = if q < 2 { 1 } else { 0 };
            let lane_off = (q % 2) * 2;

            let pairs: [(u8, u8); 4] = [(pa, pb), (pc, pd), (pa, pd), (pc, pb)];
            for &(t1, t2) in &pairs {
                let lo = t1.min(t2) as usize;
                let hi = t1.max(t2) as usize;
                matchups[lo * TEAMS + hi] += 1;
                week_matchup[w * TEAMS * TEAMS + lo * TEAMS + hi] = 1;
            }

            lane_counts[pa as usize * LANES + lane_off] += 2;
            lane_counts[pb as usize * LANES + lane_off] += 1;
            lane_counts[pb as usize * LANES + lane_off + 1] += 1;
            lane_counts[pc as usize * LANES + lane_off + 1] += 2;
            lane_counts[pd as usize * LANES + lane_off + 1] += 1;
            lane_counts[pd as usize * LANES + lane_off] += 1;

            if q >= 2 {
                late_lane_counts[pa as usize * LANES + lane_off] += 2;
                late_lane_counts[pb as usize * LANES + lane_off] += 1;
                late_lane_counts[pb as usize * LANES + lane_off + 1] += 1;
                late_lane_counts[pc as usize * LANES + lane_off + 1] += 2;
                late_lane_counts[pd as usize * LANES + lane_off + 1] += 1;
                late_lane_counts[pd as usize * LANES + lane_off] += 1;
            }

            stay_count[pa as usize] += 1;
            stay_count[pc as usize] += 1;

            for &t in &[pa, pb, pc, pd] {
                early_late[t as usize * WEEKS + w] = early;
                if early == 1 { early_count[t as usize] += 1; }
            }
        }
    }

    let mut matchup_balance: u32 = 0;
    for i in 0..TEAMS {
        for j in (i + 1)..TEAMS {
            let c = matchups[i * TEAMS + j];
            if c == 0 { matchup_balance += w8.matchup_zero; }
            else if c >= 3 { matchup_balance += (c - 2) as u32 * w8.matchup_triple; }
        }
    }

    let mut consecutive_opponents: u32 = 0;
    let active_week_set: [bool; WEEKS] = std::array::from_fn(|w| {
        if full_schedule { true } else { (0..QUADS).any(|q| !locked[w][q]) }
    });
    for w in 0..(WEEKS - 1) {
        if !active_week_set[w] || !active_week_set[w + 1] { continue; }
        let b1 = w * TEAMS * TEAMS;
        let b2 = (w + 1) * TEAMS * TEAMS;
        for i in 0..TEAMS {
            for j in (i + 1)..TEAMS {
                let idx = i * TEAMS + j;
                if week_matchup[b1 + idx] != 0 && week_matchup[b2 + idx] != 0 {
                    consecutive_opponents += w8.consecutive_opponents;
                }
            }
        }
    }

    let active_quad_count: f64 = (0..WEEKS).map(|w| {
        (0..QUADS).filter(|&q| if full_schedule { true } else { !locked[w][q] }).count() as f64
    }).sum::<f64>();
    let scale = active_quad_count / (WEEKS * QUADS) as f64;

    let target_e: f64 = WEEKS as f64 / 2.0 * scale;
    let mut early_late_balance: u32 = 0;
    for t in 0..TEAMS {
        let dev = (early_count[t] as f64 - target_e).abs();
        early_late_balance += (dev * dev * w8.early_late_balance) as u32;
    }

    let mut early_late_alternation: u32 = 0;
    for t in 0..TEAMS {
        let base = t * WEEKS;
        let mut run = 0u8;
        let mut prev = 255u8;
        for w in 0..WEEKS {
            let v = early_late[base + w];
            if v == 255 { run = 0; prev = 255; continue; }
            if v == prev { run += 1; } else { run = 1; prev = v; }
            if run >= 3 { early_late_alternation += w8.early_late_alternation; }
        }
    }

    let target_l: f64 = (WEEKS as f64 * 2.0) / LANES as f64 * scale;
    let mut lane_balance: u32 = 0;
    for t in 0..TEAMS {
        for l in 0..LANES {
            lane_balance +=
                ((lane_counts[t * LANES + l] as f64 - target_l).abs() * w8.lane_balance) as u32;
        }
    }

    let target_stay: f64 = WEEKS as f64 / 2.0 * scale;
    let mut lane_switch_balance: u32 = 0;
    for t in 0..TEAMS {
        let dev = (stay_count[t] as f64 - target_stay).abs();
        lane_switch_balance += (dev * w8.lane_switch) as u32;
    }

    let active_late_quad_count: f64 = (0..WEEKS).map(|w| {
        (2..QUADS).filter(|&q| if full_schedule { true } else { !locked[w][q] }).count() as f64
    }).sum::<f64>();
    let late_scale = active_late_quad_count / (WEEKS * 2) as f64;
    let late_target_l: f64 = WEEKS as f64 / LANES as f64 * late_scale;
    let mut late_lane_balance: u32 = 0;
    for t in 0..TEAMS {
        for l in 0..LANES {
            late_lane_balance +=
                ((late_lane_counts[t * LANES + l] as f64 - late_target_l).abs() * w8.late_lane_balance) as u32;
        }
    }

    let total = matchup_balance + consecutive_opponents + early_late_balance
        + early_late_alternation + lane_balance + lane_switch_balance + late_lane_balance;

    MaskedCost {
        matchup_balance, consecutive_opponents, early_late_balance,
        early_late_alternation, lane_balance, lane_switch_balance,
        late_lane_balance, total,
    }
}

fn cost_label_masked(c: &MaskedCost) -> String {
    format!(
        "total: {:>4} matchup: {:>3} consec: {:>3} el_bal: {:>3} el_alt: {:>3} lane: {:>3} switch: {:>3} ll_bal: {:>3}",
        c.total, c.matchup_balance, c.consecutive_opponents,
        c.early_late_balance, c.early_late_alternation, c.lane_balance,
        c.lane_switch_balance, c.late_lane_balance,
    )
}

// ═══════════════════════════════════════════════════════════════════════════
// Adaptive move selection
// ═══════════════════════════════════════════════════════════════════════════

const NUM_MOVES: usize = 11;
const BASE_WEIGHTS: [f64; NUM_MOVES] = [
    0.25,  // 0: inter-quad swap
    0.15,  // 1: intra-quad swap
    0.10,  // 2: team cross-week swap
    0.08,  // 3: quad swap
    0.06,  // 4: week swap
    0.06,  // 5: early/late flip
    0.05,  // 6: lane pair swap
    0.06,  // 7: stay/switch rotation
    0.06,  // 8: guided matchup
    0.06,  // 9: guided lane
    0.07,  // 10: guided early/late
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
            if rand_val < self.cumulative[m] { return m; }
        }
        NUM_MOVES - 1
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// Guided move helpers
// ═══════════════════════════════════════════════════════════════════════════

fn find_team_in_week(a: &Assignment, w: usize, team: u8) -> Option<(usize, usize)> {
    for q in 0..QUADS {
        for p in 0..POS {
            if a[w][q][p] == team { return Some((q, p)); }
        }
    }
    None
}

fn same_half(q1: usize, q2: usize) -> bool {
    (q1 < 2 && q2 < 2) || (q1 >= 2 && q2 >= 2)
}

fn guided_matchup(
    a: &mut Assignment,
    locked: &LockMask,
    active_weeks: &[usize],
    full_schedule: bool,
    rng: &mut SmallRng,
) -> bool {
    let mut matchups = [false; TEAMS * TEAMS];
    for w in 0..WEEKS {
        for q in 0..QUADS {
            if !full_schedule && locked[w][q] { continue; }
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
    if !found { return false; }

    let week_start = rng.random_range(0..active_weeks.len());
    for off in 0..active_weeks.len() {
        let w = active_weeks[(week_start + off) % active_weeks.len()];
        let pos_a = find_team_in_week(a, w, ta);
        let pos_b = find_team_in_week(a, w, tb);
        let (qa, _pa) = match pos_a { Some(x) => x, None => continue };
        let (qb, pb) = match pos_b { Some(x) => x, None => continue };

        if !same_half(qa, qb) || qa == qb { continue; }
        if locked[w][qa] || locked[w][qb] { continue; }

        // Swap B with a random non-A player in A's quad
        let candidates: Vec<usize> = (0..POS)
            .filter(|&p| a[w][qa][p] != ta)
            .collect();
        if candidates.is_empty() { continue; }
        let ci = rng.random_range(0..candidates.len());
        let swap_pos = candidates[ci];

        let tmp = a[w][qa][swap_pos];
        a[w][qa][swap_pos] = a[w][qb][pb];
        a[w][qb][pb] = tmp;
        return true;
    }
    false
}

fn guided_lane(
    a: &mut Assignment,
    locked: &LockMask,
    active_quads: &[(usize, usize)],
    full_schedule: bool,
    scale: f64,
    rng: &mut SmallRng,
) -> bool {
    let mut lane_counts = [0i32; TEAMS * LANES];
    for w in 0..WEEKS {
        for q in 0..QUADS {
            if !full_schedule && locked[w][q] { continue; }
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

    let target_l = (WEEKS as f64 * 2.0) / LANES as f64 * scale;
    let mut worst_team = 0usize;
    let mut worst_dev = 0.0f64;
    for t in 0..TEAMS {
        for l in 0..LANES {
            let dev = (lane_counts[t * LANES + l] as f64 - target_l).abs();
            if dev > worst_dev { worst_dev = dev; worst_team = t; }
        }
    }
    if worst_dev < 1.0 { return false; }

    let start = rng.random_range(0..active_quads.len());
    for off in 0..active_quads.len() {
        let (w, q) = active_quads[(start + off) % active_quads.len()];
        let mut team_pos = None;
        for p in 0..POS {
            if a[w][q][p] == worst_team as u8 { team_pos = Some(p); break; }
        }
        let tp = match team_pos { Some(p) => p, None => continue };

        // Find a quad-mate to swap with (different position)
        let mut swap_pos = rng.random_range(0..(POS - 1));
        if swap_pos >= tp { swap_pos += 1; }
        a[w][q].swap(tp, swap_pos);
        return true;
    }
    false
}

fn guided_early_late(
    a: &mut Assignment,
    locked: &LockMask,
    full_weeks: &[usize],
    full_schedule: bool,
    scale: f64,
    rng: &mut SmallRng,
) -> bool {
    if full_weeks.is_empty() { return false; }

    let mut early_count = [0i32; TEAMS];
    for w in 0..WEEKS {
        for q in 0..QUADS {
            if !full_schedule && locked[w][q] { continue; }
            if q < 2 {
                for p in 0..POS { early_count[a[w][q][p] as usize] += 1; }
            }
        }
    }

    let target_e = WEEKS as f64 / 2.0 * scale;
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
    if worst_dev < 1.0 { return false; }

    let start = rng.random_range(0..full_weeks.len());
    for off in 0..full_weeks.len() {
        let w = full_weeks[(start + off) % full_weeks.len()];
        let team = worst_team as u8;
        let in_early = (0..2).any(|q| (0..POS).any(|p| a[w][q][p] == team));

        if (too_many_early && in_early) || (!too_many_early && !in_early) {
            // Flip this week's early/late
            let tmp0 = a[w][0]; let tmp1 = a[w][1];
            a[w][0] = a[w][2]; a[w][2] = tmp0;
            a[w][1] = a[w][3]; a[w][3] = tmp1;
            return true;
        }
    }
    false
}

// ═══════════════════════════════════════════════════════════════════════════
// Random assignment for unlocked quads only
// ═══════════════════════════════════════════════════════════════════════════

fn randomize_unlocked(a: &mut Assignment, locked: &LockMask, rng: &mut SmallRng) {
    for w in 0..WEEKS {
        let unlocked: Vec<usize> = (0..QUADS).filter(|&q| !locked[w][q]).collect();
        if unlocked.is_empty() { continue; }

        if unlocked.len() == QUADS {
            let mut teams: [u8; TEAMS] = std::array::from_fn(|i| i as u8);
            for i in (1..TEAMS).rev() {
                let j = rng.random_range(0..=i);
                teams.swap(i, j);
            }
            for q in 0..QUADS {
                for p in 0..POS { a[w][q][p] = teams[q * POS + p]; }
            }
        } else {
            let mut used = [false; TEAMS];
            for q in 0..QUADS {
                if locked[w][q] {
                    for p in 0..POS { used[a[w][q][p] as usize] = true; }
                }
            }
            let mut avail: Vec<u8> = (0..TEAMS as u8).filter(|&t| !used[t as usize]).collect();
            for i in (1..avail.len()).rev() {
                let j = rng.random_range(0..=i);
                avail.swap(i, j);
            }
            let mut ai = 0;
            for &q in &unlocked {
                for p in 0..POS { a[w][q][p] = avail[ai]; ai += 1; }
            }
        }
    }
}

fn load_random_solved(dir: &str, rng: &mut SmallRng) -> Option<Assignment> {
    let entries: Vec<_> = fs::read_dir(dir).ok()?
        .filter_map(|e| e.ok())
        .filter(|e| e.path().extension().and_then(|s| s.to_str()) == Some("tsv"))
        .collect();
    if entries.is_empty() { return None; }
    let idx = rng.random_range(0..entries.len());
    let content = fs::read_to_string(entries[idx].path()).ok()?;
    parse_tsv(&content)
}

// ═══════════════════════════════════════════════════════════════════════════
// Parallel tempering
// ═══════════════════════════════════════════════════════════════════════════

struct ReplicaSlot {
    assignment: Assignment,
    cost: u32,
    best_assignment: Assignment,
    best_cost: u32,
}

struct SwapBuffer {
    slots: Vec<ReplicaSlot>,
    checked_in: usize,
    epoch: u64,
    global_best_cost: u32,
    global_best_assignment: Assignment,
    stagnation: u32,
    prev_best_cost: u32,
}

// ═══════════════════════════════════════════════════════════════════════════
// SA phase runner
// ═══════════════════════════════════════════════════════════════════════════

fn run_phase(
    phase_name: &str,
    locked: &LockMask,
    w8: &Weights,
    t0: f64,
    temp_floor: f64,
    progress_interval: u64,
    num_cores: usize,
    shutdown: &Arc<AtomicBool>,
    results_dir: &str,
    seed_files: &[Assignment],
    baseline_matchups: &[i32; TEAMS * TEAMS],
    full_schedule_eval: bool,
) -> Option<Assignment> {
    let idx = build_active_indices(locked);
    if idx.active_quads.is_empty() {
        eprintln!("[{}] {} -- no active quads, skipping", now_iso(), phase_name);
        return None;
    }

    fs::create_dir_all(results_dir).expect("Failed to create results directory");

    // Parallel tempering: geometric temperature ladder
    let temps: Vec<f64> = if num_cores == 1 {
        vec![temp_floor.max(1.0)]
    } else {
        (0..num_cores).map(|i| {
            let tf = temp_floor.max(0.1);
            tf * (t0 / tf).powf(i as f64 / (num_cores - 1) as f64)
        }).collect()
    };

    let active_quad_count: f64 = (0..WEEKS).map(|w| {
        (0..QUADS).filter(|&q| if full_schedule_eval { true } else { !locked[w][q] }).count() as f64
    }).sum::<f64>();
    let scale = active_quad_count / (WEEKS * QUADS) as f64;

    let global_best = Arc::new(AtomicU32::new(u32::MAX));
    let global_best_assignment: Arc<Mutex<Option<Assignment>>> =
        Arc::new(Mutex::new(None));

    let dummy: Assignment = [[[0u8; POS]; QUADS]; WEEKS];
    let swap_pair = Arc::new((
        Mutex::new(SwapBuffer {
            slots: (0..num_cores).map(|_| ReplicaSlot { assignment: dummy, cost: u32::MAX, best_assignment: dummy, best_cost: u32::MAX }).collect(),
            checked_in: 0,
            epoch: 0,
            global_best_cost: u32::MAX,
            global_best_assignment: dummy,
            stagnation: 0,
            prev_best_cost: u32::MAX,
        }),
        Condvar::new(),
    ));

    let locked = Arc::new(*locked);
    let idx = Arc::new(idx);
    let w8 = Arc::new(w8.clone());
    let baseline = Arc::new(*baseline_matchups);
    let results_dir = Arc::new(results_dir.to_string());
    let seed_files = Arc::new(seed_files.to_vec());
    let temps = Arc::new(temps);

    eprintln!(
        "[{}] {} -- {} cores, {} active quads, {} full weeks, {}B iters/run, temps: {:.1}..{:.1}",
        now_iso(), phase_name, num_cores,
        idx.active_quads.len(), idx.full_weeks.len(),
        MAX_ITERATIONS as f64 / 1e9,
        temps[0], temps[num_cores - 1],
    );

    let handles: Vec<_> = (0..num_cores)
        .map(|core_id| {
            let shutdown = Arc::clone(shutdown);
            let global_best = Arc::clone(&global_best);
            let global_best_assignment = Arc::clone(&global_best_assignment);
            let locked = Arc::clone(&locked);
            let idx = Arc::clone(&idx);
            let w8 = Arc::clone(&w8);
            let baseline = Arc::clone(&baseline);
            let results_dir = Arc::clone(&results_dir);
            let seed_files = Arc::clone(&seed_files);
            let temps = Arc::clone(&temps);
            let swap_pair = Arc::clone(&swap_pair);

            thread::spawn(move || {
                let mut rng = SmallRng::from_os_rng();
                let mut first_run = true;
                let mut last_saved: Option<Assignment> = None;
                let temp = temps[core_id];

                loop {
                    if shutdown.load(Ordering::Relaxed) { return; }

                    let use_seed = first_run && core_id < seed_files.len();
                    let mut a: Assignment = if use_seed {
                        first_run = false;
                        seed_files[core_id]
                    } else {
                        first_run = false;
                        if rng.random::<f64>() < 0.5 {
                            load_random_solved(&results_dir, &mut rng).unwrap_or_else(|| {
                                let mut a = [[[0u8; POS]; QUADS]; WEEKS];
                                randomize_unlocked(&mut a, &locked, &mut rng);
                                a
                            })
                        } else {
                            let mut a = [[[0u8; POS]; QUADS]; WEEKS];
                            randomize_unlocked(&mut a, &locked, &mut rng);
                            a
                        }
                    };

                    if !use_seed {
                        randomize_unlocked(&mut a, &locked, &mut rng);
                    }

                    let mut cost = evaluate_masked(&a, &w8, &locked, &baseline, full_schedule_eval);
                    let mut best_a = a;
                    let mut best_cost = cost.total;
                    let mut stats = MoveStats::new();

                    for i in 0..MAX_ITERATIONS {
                        if shutdown.load(Ordering::Relaxed) { break; }
                        if best_cost == 0 { break; }

                        // Recompute adaptive weights periodically
                        if i > 0 && i % STATS_RECOMPUTE == 0 {
                            stats.recompute();
                        }

                        // Exhaustive single-week inter-quad swap search (periodic)
                        if i > 0 && i % 100_000 == 0 && cost.total > 0 && !idx.active_quad_pairs.is_empty() {
                            let pi = rng.random_range(0..idx.active_quad_pairs.len());
                            let (ew, eq1, eq2) = idx.active_quad_pairs[pi];
                            let mut best_delta = 0i64;
                            let mut best_swap: Option<(usize, usize)> = None;
                            for ep1 in 0..POS {
                                for ep2 in 0..POS {
                                    let tmp = a[ew][eq1][ep1];
                                    a[ew][eq1][ep1] = a[ew][eq2][ep2];
                                    a[ew][eq2][ep2] = tmp;
                                    let nc = evaluate_masked(&a, &w8, &locked, &baseline, full_schedule_eval);
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
                                cost = evaluate_masked(&a, &w8, &locked, &baseline, full_schedule_eval);
                                if cost.total < best_cost { best_cost = cost.total; best_a = a; }
                            }
                        }

                        // Compound move (kept separate -- probability based on cost)
                        let compound_prob = ((1000.0 - cost.total as f64) / 800.0).clamp(0.0, 0.5);
                        if rng.random::<f64>() < compound_prob && !idx.active_quad_pairs.is_empty() {
                            let saved = a;
                            let max_swaps = if cost.total < 200 { 12 } else if cost.total < 400 { 6 } else { 4 };
                            let num_swaps = rng.random_range(2..=max_swaps);
                            for _ in 0..num_swaps {
                                let pi = rng.random_range(0..idx.active_quad_pairs.len());
                                let (w, q1, q2) = idx.active_quad_pairs[pi];
                                let p1 = rng.random_range(0..POS);
                                let p2 = rng.random_range(0..POS);
                                let tmp = a[w][q1][p1];
                                a[w][q1][p1] = a[w][q2][p2];
                                a[w][q2][p2] = tmp;
                            }
                            let new_cost = evaluate_masked(&a, &w8, &locked, &baseline, full_schedule_eval);
                            let delta = new_cost.total as i64 - cost.total as i64;
                            if delta <= 0 || rng.random::<f64>() < (-delta as f64 / temp).exp() {
                                cost = new_cost;
                                if cost.total < best_cost { best_cost = cost.total; best_a = a; }
                            } else {
                                a = saved;
                            }
                        } else {
                            // Adaptive move selection
                            let move_id = stats.select(rng.random::<f64>());
                            stats.attempts[move_id] += 1;
                            let accepted;

                            match move_id {
                                0 => {
                                    // Inter-quad player swap
                                    if idx.active_quad_pairs.is_empty() { accepted = false; } else {
                                        let pi = rng.random_range(0..idx.active_quad_pairs.len());
                                        let (w, q1, q2) = idx.active_quad_pairs[pi];
                                        let p1 = rng.random_range(0..POS);
                                        let p2 = rng.random_range(0..POS);
                                        let tmp = a[w][q1][p1];
                                        a[w][q1][p1] = a[w][q2][p2];
                                        a[w][q2][p2] = tmp;
                                        let new_cost = evaluate_masked(&a, &w8, &locked, &baseline, full_schedule_eval);
                                        let delta = new_cost.total as i64 - cost.total as i64;
                                        if delta <= 0 || rng.random::<f64>() < (-delta as f64 / temp).exp() {
                                            cost = new_cost;
                                            if cost.total < best_cost { best_cost = cost.total; best_a = a; }
                                            accepted = true;
                                        } else {
                                            a[w][q2][p2] = a[w][q1][p1];
                                            a[w][q1][p1] = tmp;
                                            accepted = false;
                                        }
                                    }
                                }
                                1 => {
                                    // Intra-quad player swap
                                    if idx.active_quads.is_empty() { accepted = false; } else {
                                        let qi = rng.random_range(0..idx.active_quads.len());
                                        let (w, q) = idx.active_quads[qi];
                                        let p1 = rng.random_range(0..POS);
                                        let mut p2 = rng.random_range(0..(POS - 1));
                                        if p2 >= p1 { p2 += 1; }
                                        let tmp = a[w][q][p1];
                                        a[w][q][p1] = a[w][q][p2];
                                        a[w][q][p2] = tmp;
                                        let new_cost = evaluate_masked(&a, &w8, &locked, &baseline, full_schedule_eval);
                                        let delta = new_cost.total as i64 - cost.total as i64;
                                        if delta <= 0 || rng.random::<f64>() < (-delta as f64 / temp).exp() {
                                            cost = new_cost;
                                            if cost.total < best_cost { best_cost = cost.total; best_a = a; }
                                            accepted = true;
                                        } else {
                                            a[w][q][p2] = a[w][q][p1];
                                            a[w][q][p1] = tmp;
                                            accepted = false;
                                        }
                                    }
                                }
                                2 => {
                                    // Team cross-week swap
                                    accepted = if idx.active_weeks.len() >= 2 {
                                        let team = rng.random_range(0..TEAMS) as u8;
                                        let wi1 = rng.random_range(0..idx.active_weeks.len());
                                        let mut wi2 = rng.random_range(0..(idx.active_weeks.len() - 1));
                                        if wi2 >= wi1 { wi2 += 1; }
                                        let w1 = idx.active_weeks[wi1];
                                        let w2 = idx.active_weeks[wi2];

                                        let mut qi1 = None; let mut pi1 = None;
                                        let mut qi2 = None; let mut pi2 = None;
                                        for q in 0..QUADS {
                                            if locked[w1][q] { continue; }
                                            for p in 0..POS {
                                                if a[w1][q][p] == team && qi1.is_none() { qi1 = Some(q); pi1 = Some(p); }
                                            }
                                        }
                                        for q in 0..QUADS {
                                            if locked[w2][q] { continue; }
                                            for p in 0..POS {
                                                if a[w2][q][p] == team && qi2.is_none() { qi2 = Some(q); pi2 = Some(p); }
                                            }
                                        }

                                        if let (Some(qi1), Some(pi1), Some(qi2), Some(pi2)) = (qi1, pi1, qi2, pi2) {
                                            if !locked[w2][qi1] && !locked[w1][qi2] {
                                                let save = (a[w1][qi1][pi1], a[w1][qi2][pi2], a[w2][qi1][pi1], a[w2][qi2][pi2]);
                                                let other1 = a[w2][qi1][pi1];
                                                let other2 = a[w1][qi2][pi2];
                                                a[w1][qi1][pi1] = other2;
                                                a[w1][qi2][pi2] = team;
                                                a[w2][qi2][pi2] = other1;
                                                a[w2][qi1][pi1] = team;
                                                let new_cost = evaluate_masked(&a, &w8, &locked, &baseline, full_schedule_eval);
                                                let delta = new_cost.total as i64 - cost.total as i64;
                                                if delta <= 0 || rng.random::<f64>() < (-delta as f64 / temp).exp() {
                                                    cost = new_cost;
                                                    if cost.total < best_cost { best_cost = cost.total; best_a = a; }
                                                    true
                                                } else {
                                                    a[w1][qi1][pi1] = save.0;
                                                    a[w1][qi2][pi2] = save.1;
                                                    a[w2][qi1][pi1] = save.2;
                                                    a[w2][qi2][pi2] = save.3;
                                                    false
                                                }
                                            } else { false }
                                        } else { false }
                                    } else { false };
                                }
                                3 => {
                                    // Quad swap
                                    if idx.active_quad_pairs.is_empty() { accepted = false; } else {
                                        let pi = rng.random_range(0..idx.active_quad_pairs.len());
                                        let (w, q1, q2) = idx.active_quad_pairs[pi];
                                        let tmp = a[w][q1];
                                        a[w][q1] = a[w][q2];
                                        a[w][q2] = tmp;
                                        let new_cost = evaluate_masked(&a, &w8, &locked, &baseline, full_schedule_eval);
                                        let delta = new_cost.total as i64 - cost.total as i64;
                                        if delta <= 0 || rng.random::<f64>() < (-delta as f64 / temp).exp() {
                                            cost = new_cost;
                                            if cost.total < best_cost { best_cost = cost.total; best_a = a; }
                                            accepted = true;
                                        } else {
                                            a[w][q2] = a[w][q1];
                                            a[w][q1] = tmp;
                                            accepted = false;
                                        }
                                    }
                                }
                                4 => {
                                    // Week swap
                                    if idx.full_weeks.len() < 2 { accepted = false; } else {
                                        let wi1 = rng.random_range(0..idx.full_weeks.len());
                                        let mut wi2 = rng.random_range(0..(idx.full_weeks.len() - 1));
                                        if wi2 >= wi1 { wi2 += 1; }
                                        let w1 = idx.full_weeks[wi1];
                                        let w2 = idx.full_weeks[wi2];
                                        let tmp = a[w1];
                                        a[w1] = a[w2];
                                        a[w2] = tmp;
                                        let new_cost = evaluate_masked(&a, &w8, &locked, &baseline, full_schedule_eval);
                                        let delta = new_cost.total as i64 - cost.total as i64;
                                        if delta <= 0 || rng.random::<f64>() < (-delta as f64 / temp).exp() {
                                            cost = new_cost;
                                            if cost.total < best_cost { best_cost = cost.total; best_a = a; }
                                            accepted = true;
                                        } else {
                                            a[w2] = a[w1];
                                            a[w1] = tmp;
                                            accepted = false;
                                        }
                                    }
                                }
                                5 => {
                                    // Early/late flip
                                    if idx.full_weeks.is_empty() { accepted = false; } else {
                                        let wi = rng.random_range(0..idx.full_weeks.len());
                                        let w = idx.full_weeks[wi];
                                        let tmp0 = a[w][0]; let tmp1 = a[w][1];
                                        a[w][0] = a[w][2]; a[w][2] = tmp0;
                                        a[w][1] = a[w][3]; a[w][3] = tmp1;
                                        let new_cost = evaluate_masked(&a, &w8, &locked, &baseline, full_schedule_eval);
                                        let delta = new_cost.total as i64 - cost.total as i64;
                                        if delta <= 0 || rng.random::<f64>() < (-delta as f64 / temp).exp() {
                                            cost = new_cost;
                                            if cost.total < best_cost { best_cost = cost.total; best_a = a; }
                                            accepted = true;
                                        } else {
                                            let tmp0 = a[w][0]; let tmp1 = a[w][1];
                                            a[w][0] = a[w][2]; a[w][2] = tmp0;
                                            a[w][1] = a[w][3]; a[w][3] = tmp1;
                                            accepted = false;
                                        }
                                    }
                                }
                                6 => {
                                    // Lane pair swap
                                    if idx.full_weeks.is_empty() { accepted = false; } else {
                                        let wi = rng.random_range(0..idx.full_weeks.len());
                                        let w = idx.full_weeks[wi];
                                        let tmp0 = a[w][0]; let tmp2 = a[w][2];
                                        a[w][0] = a[w][1]; a[w][1] = tmp0;
                                        a[w][2] = a[w][3]; a[w][3] = tmp2;
                                        let new_cost = evaluate_masked(&a, &w8, &locked, &baseline, full_schedule_eval);
                                        let delta = new_cost.total as i64 - cost.total as i64;
                                        if delta <= 0 || rng.random::<f64>() < (-delta as f64 / temp).exp() {
                                            cost = new_cost;
                                            if cost.total < best_cost { best_cost = cost.total; best_a = a; }
                                            accepted = true;
                                        } else {
                                            let tmp1 = a[w][1]; let tmp3 = a[w][3];
                                            a[w][1] = a[w][0]; a[w][0] = tmp1;
                                            a[w][3] = a[w][2]; a[w][2] = tmp3;
                                            accepted = false;
                                        }
                                    }
                                }
                                7 => {
                                    // Stay/switch rotation
                                    if idx.active_quads.is_empty() { accepted = false; } else {
                                        let qi = rng.random_range(0..idx.active_quads.len());
                                        let (w, q) = idx.active_quads[qi];
                                        a[w][q].swap(0, 1);
                                        a[w][q].swap(2, 3);
                                        let new_cost = evaluate_masked(&a, &w8, &locked, &baseline, full_schedule_eval);
                                        let delta = new_cost.total as i64 - cost.total as i64;
                                        if delta <= 0 || rng.random::<f64>() < (-delta as f64 / temp).exp() {
                                            cost = new_cost;
                                            if cost.total < best_cost { best_cost = cost.total; best_a = a; }
                                            accepted = true;
                                        } else {
                                            a[w][q].swap(0, 1);
                                            a[w][q].swap(2, 3);
                                            accepted = false;
                                        }
                                    }
                                }
                                8 => {
                                    // Guided matchup
                                    let saved = a;
                                    let did_move = guided_matchup(&mut a, &locked, &idx.active_weeks, full_schedule_eval, &mut rng);
                                    if did_move {
                                        let new_cost = evaluate_masked(&a, &w8, &locked, &baseline, full_schedule_eval);
                                        let delta = new_cost.total as i64 - cost.total as i64;
                                        if delta <= 0 || rng.random::<f64>() < (-delta as f64 / temp).exp() {
                                            cost = new_cost;
                                            if cost.total < best_cost { best_cost = cost.total; best_a = a; }
                                            accepted = true;
                                        } else {
                                            a = saved;
                                            accepted = false;
                                        }
                                    } else { accepted = false; }
                                }
                                9 => {
                                    // Guided lane
                                    let saved = a;
                                    let did_move = guided_lane(&mut a, &locked, &idx.active_quads, full_schedule_eval, scale, &mut rng);
                                    if did_move {
                                        let new_cost = evaluate_masked(&a, &w8, &locked, &baseline, full_schedule_eval);
                                        let delta = new_cost.total as i64 - cost.total as i64;
                                        if delta <= 0 || rng.random::<f64>() < (-delta as f64 / temp).exp() {
                                            cost = new_cost;
                                            if cost.total < best_cost { best_cost = cost.total; best_a = a; }
                                            accepted = true;
                                        } else {
                                            a = saved;
                                            accepted = false;
                                        }
                                    } else { accepted = false; }
                                }
                                _ => {
                                    // Guided early/late (move 10)
                                    let saved = a;
                                    let did_move = guided_early_late(&mut a, &locked, &idx.full_weeks, full_schedule_eval, scale, &mut rng);
                                    if did_move {
                                        let new_cost = evaluate_masked(&a, &w8, &locked, &baseline, full_schedule_eval);
                                        let delta = new_cost.total as i64 - cost.total as i64;
                                        if delta <= 0 || rng.random::<f64>() < (-delta as f64 / temp).exp() {
                                            cost = new_cost;
                                            if cost.total < best_cost { best_cost = cost.total; best_a = a; }
                                            accepted = true;
                                        } else {
                                            a = saved;
                                            accepted = false;
                                        }
                                    } else { accepted = false; }
                                }
                            }

                            if accepted { stats.accepts[move_id] += 1; }
                        }

                        // Parallel tempering: replica swap
                        if num_cores > 1 && i > 0 && i % SWAP_INTERVAL == 0 {
                            let (lock, cvar) = &*swap_pair;
                            let mut buf = lock.lock().unwrap();

                            buf.slots[core_id] = ReplicaSlot { assignment: a, cost: cost.total, best_assignment: best_a, best_cost };
                            buf.checked_in += 1;

                            if buf.checked_in == num_cores {
                                // Last core: perform replica swaps
                                for k in 0..(num_cores - 1) {
                                    let ti = temps[k];
                                    let tj = temps[k + 1];
                                    let ci = buf.slots[k].cost as f64;
                                    let cj = buf.slots[k + 1].cost as f64;
                                    let delta = (1.0 / ti - 1.0 / tj) * (ci - cj);
                                    if delta > 0.0 || rng.random::<f64>() < delta.exp() {
                                        let tmp_a = buf.slots[k].assignment;
                                        let tmp_c = buf.slots[k].cost;
                                        buf.slots[k].assignment = buf.slots[k + 1].assignment;
                                        buf.slots[k].cost = buf.slots[k + 1].cost;
                                        buf.slots[k + 1].assignment = tmp_a;
                                        buf.slots[k + 1].cost = tmp_c;
                                    }
                                }

                                // Track global best from per-core bests
                                for k in 0..num_cores {
                                    if buf.slots[k].best_cost < buf.global_best_cost {
                                        buf.global_best_cost = buf.slots[k].best_cost;
                                        buf.global_best_assignment = buf.slots[k].best_assignment;
                                    }
                                }
                                if buf.global_best_cost < buf.prev_best_cost {
                                    buf.stagnation = 0;
                                    buf.prev_best_cost = buf.global_best_cost;
                                } else {
                                    buf.stagnation += 1;
                                }

                                // Every 100 epochs (~10M iters/core): diversify stagnant cores
                                if buf.epoch > 0 && buf.epoch % 100 == 0 && buf.stagnation > 50 {
                                    let pert = 15 + buf.stagnation as usize / 10;
                                    let mut resets = 0u32;

                                    // Dedup: compare per-core BEST assignments, perturb from global best
                                    let mut already_reset = [false; 64];
                                    for k in 0..num_cores {
                                        for j in (k + 1)..num_cores {
                                            if already_reset[j] { continue; }
                                            if buf.slots[k].best_assignment == buf.slots[j].best_assignment {
                                                buf.slots[j].assignment = buf.global_best_assignment;
                                                perturb(&mut buf.slots[j].assignment, &mut rng, pert);
                                                buf.slots[j].cost = u32::MAX;
                                                already_reset[j] = true;
                                                resets += 1;
                                            }
                                        }
                                    }

                                    // Also re-seed hot replicas that weren't already reset
                                    let hot_start = num_cores * 2 / 3;
                                    for k in hot_start..num_cores {
                                        if already_reset[k] { continue; }
                                        buf.slots[k].assignment = buf.global_best_assignment;
                                        perturb(&mut buf.slots[k].assignment, &mut rng, pert + 5);
                                        buf.slots[k].cost = u32::MAX;
                                        resets += 1;
                                    }

                                    eprintln!(
                                        "[{}] epoch {} | global: {} | stagnation: {} | {} diversifications (pert={})",
                                        now_iso(), buf.epoch, buf.global_best_cost, buf.stagnation, resets, pert,
                                    );
                                }

                                buf.epoch += 1;
                                buf.checked_in = 0;
                                cvar.notify_all();
                            } else {
                                let epoch_before = buf.epoch;
                                loop {
                                    if shutdown.load(Ordering::Relaxed) { break; }
                                    let (guard, _) = cvar.wait_timeout(buf, std::time::Duration::from_millis(500)).unwrap();
                                    buf = guard;
                                    if buf.epoch != epoch_before { break; }
                                }
                            }

                            // Read back (possibly swapped/perturbed) assignment
                            a = buf.slots[core_id].assignment;
                            drop(buf);
                            cost = evaluate_masked(&a, &w8, &locked, &baseline, full_schedule_eval);
                            if cost.total < best_cost { best_cost = cost.total; best_a = a; }

                            if best_cost < global_best.load(Ordering::Relaxed) {
                                let prev = global_best.fetch_min(best_cost, Ordering::Relaxed);
                                if best_cost < prev {
                                    let best_breakdown = evaluate_masked(&best_a, &w8, &locked, &baseline, full_schedule_eval);
                                    eprintln!(
                                        "[{}] core {} new best: {} ★",
                                        now_iso(), core_id, cost_label_masked(&best_breakdown),
                                    );
                                    let mut gb = global_best_assignment.lock().unwrap();
                                    *gb = Some(best_a);

                                    if best_cost < 100 && last_saved.as_ref() != Some(&best_a) {
                                        let ts = chrono::Local::now().format("%Y%m%d-%H%M%S%z");
                                        let filename = format!(
                                            "{}/{:04}-c{}-{}.tsv", results_dir, best_cost, core_id, ts,
                                        );
                                        let _ = fs::write(&filename, assignment_to_tsv(&best_a));
                                        eprintln!("[{}] Saved {}", now_iso(), filename);
                                        last_saved = Some(best_a);
                                    }
                                }
                            }
                        }

                        if i > 0 && i % progress_interval == 0 {
                            let label = if i >= 1_000_000_000 {
                                format!("{:.2}B", i as f64 / 1_000_000_000.0)
                            } else {
                                format!("{}M", i / 1_000_000)
                            };
                            let best_breakdown = evaluate_masked(&best_a, &w8, &locked, &baseline, full_schedule_eval);
                            eprintln!(
                                "[{}] core {} @ {} | best: {} | temp: {:.2}",
                                now_iso(), core_id, label, cost_label_masked(&best_breakdown), temp,
                            );
                        }
                    }

                    // Save best from this run
                    if best_cost < u32::MAX {
                        let prev = global_best.fetch_min(best_cost, Ordering::Relaxed);
                        let marker = if best_cost < prev { " ★" } else { "" };
                        let best_breakdown = evaluate_masked(&best_a, &w8, &locked, &baseline, full_schedule_eval);
                        eprintln!(
                            "[{}] core {} run done | {}{}",
                            now_iso(), core_id, cost_label_masked(&best_breakdown), marker,
                        );

                        let ts = chrono::Local::now().format("%Y%m%d-%H%M%S%z");
                        let filename = format!(
                            "{}/{:04}-c{}-{}.tsv", results_dir, best_cost, core_id, ts,
                        );
                        if last_saved.as_ref() != Some(&best_a) {
                            let _ = fs::write(&filename, assignment_to_tsv(&best_a));
                            eprintln!("[{}] Saved {}", now_iso(), filename);
                            last_saved = Some(best_a);
                        }

                        let mut gb = global_best_assignment.lock().unwrap();
                        if gb.is_none() || best_cost <= global_best.load(Ordering::Relaxed) {
                            *gb = Some(best_a);
                        }
                    }
                }
            })
        })
        .collect();

    for h in handles {
        h.join().unwrap();
    }

    let gb = global_best_assignment.lock().unwrap();
    gb.clone()
}

// ═══════════════════════════════════════════════════════════════════════════
// Main
// ═══════════════════════════════════════════════════════════════════════════

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let mut phase2_files: Vec<String> = Vec::new();
    let mut seed_files: Vec<String> = Vec::new();
    let mut cores_override: Option<usize> = None;
    let mut config_path = "config.toml".to_string();
    let mut in_phase2 = false;
    let mut full_mode = false;
    let mut i = 1;
    while i < args.len() {
        match args[i].as_str() {
            "--phase2" => { in_phase2 = true; }
            "--full" => { full_mode = true; }
            "--cores" => {
                i += 1;
                cores_override = Some(args[i].parse().expect("--cores requires a number"));
            }
            "--config" => {
                i += 1;
                config_path = args[i].clone();
            }
            other => {
                if in_phase2 || full_mode {
                    if other.ends_with(".tsv") {
                        if full_mode { seed_files.push(other.to_string()); }
                        else { phase2_files.push(other.to_string()); }
                    }
                }
            }
        }
        i += 1;
    }

    let config_str = fs::read_to_string(&config_path)
        .unwrap_or_else(|_| {
            eprintln!("[{}] No config.toml found, using defaults", now_iso());
            "[solver]\nt0 = 30.0\n".to_string()
        });
    let config: Config = toml::from_str(&config_str)
        .unwrap_or_else(|e| panic!("Failed to parse {}: {}", config_path, e));

    let config_dir = std::path::Path::new(&config_path).parent().unwrap_or(std::path::Path::new("."));
    let weights_path = config_dir.join(&config.solver.weights_path);
    let weights_str = fs::read_to_string(&weights_path)
        .unwrap_or_else(|e| panic!("Failed to read {}: {}", weights_path.display(), e));
    let weights: Weights = serde_json::from_str(&weights_str)
        .unwrap_or_else(|e| panic!("Failed to parse {}: {}", weights_path.display(), e));

    let available = thread::available_parallelism().map(|n| n.get()).unwrap_or(4);
    let num_cores = cores_override.unwrap_or_else(|| {
        let c = config.solver.cores;
        if c <= 0.0 { available }
        else if c < 1.0 { (available as f64 * c).round().max(1.0) as usize }
        else { (c as usize).min(available) }
    });

    let t0 = config.solver.t0;
    let temp_floor = config.solver.temp_floor;
    let progress_interval = config.solver.progress_interval;

    let shutdown = Arc::new(AtomicBool::new(false));
    {
        let shutdown = Arc::clone(&shutdown);
        ctrlc::set_handler(move || {
            if shutdown.load(Ordering::SeqCst) {
                eprintln!("\n[{}] Force exit.", now_iso());
                std::process::exit(1);
            }
            shutdown.store(true, Ordering::SeqCst);
            eprintln!("\n[{}] Ctrl+C received, finishing up... (press again to force)", now_iso());
        }).expect("Failed to set Ctrl+C handler");
    }

    let phase1_dir = "results/split-sa/phase1";
    let complete_dir = "results/split-sa/complete";
    let full_dir = "results/split-sa/full";

    if full_mode {
        let seeds: Vec<Assignment> = seed_files.iter()
            .filter_map(|path| {
                let content = fs::read_to_string(path)
                    .map_err(|e| eprintln!("Warning: could not read {}: {}", path, e))
                    .ok()?;
                let a = parse_tsv(&content)
                    .or_else(|| { eprintln!("Warning: could not parse {}", path); None })?;
                let full_cost = evaluate(&a, &weights);
                eprintln!("[{}] Loaded seed: {} ({})", now_iso(), path, cost_label(&full_cost));
                Some(a)
            })
            .collect();
        eprintln!("[{}] Full schedule mode: {} seeds, all weeks unlocked", now_iso(), seeds.len());
        let no_lock = [[false; QUADS]; WEEKS];
        let no_baseline = [0i32; TEAMS * TEAMS];
        run_phase("Full", &no_lock, &weights, t0, temp_floor, progress_interval, num_cores, &shutdown, full_dir, &seeds, &no_baseline, true);
    } else if !phase2_files.is_empty() {
        eprintln!("[{}] Phase 2 mode: loading {} seed files", now_iso(), phase2_files.len());
        let seeds: Vec<Assignment> = phase2_files.iter()
            .filter_map(|path| {
                let content = fs::read_to_string(path)
                    .map_err(|e| eprintln!("Warning: could not read {}: {}", path, e))
                    .ok()?;
                let a = parse_tsv(&content)
                    .or_else(|| { eprintln!("Warning: could not parse {}", path); None })?;
                let full_cost = evaluate(&a, &weights);
                eprintln!("[{}] Loaded: {} (full: {})", now_iso(), path, cost_label(&full_cost));
                Some(a)
            })
            .collect();

        if seeds.is_empty() {
            eprintln!("[{}] No valid seed files, exiting", now_iso());
            return;
        }

        let p2_lock = phase2_lock();
        let no_baseline = [0i32; TEAMS * TEAMS];
        run_phase("Phase 2", &p2_lock, &weights, t0, temp_floor, progress_interval, num_cores, &shutdown, complete_dir, &seeds, &no_baseline, true);
    } else {
        eprintln!("[{}] Phase 1 mode: solving weeks 0-6 + half of week 7", now_iso());
        let p1_lock = phase1_lock();
        let no_baseline = [0i32; TEAMS * TEAMS];
        run_phase("Phase 1", &p1_lock, &weights, t0, temp_floor, progress_interval, num_cores, &shutdown, phase1_dir, &[], &no_baseline, false);
    }

    eprintln!("[{}] Done.", now_iso());
}
