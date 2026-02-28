use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{mpsc, Arc};
use std::thread;

use rand::rngs::SmallRng;
use rand::{Rng, SeedableRng};
use solver_core::*;

pub const BATCH_SIZE: u64 = 10_000;
const STATS_RECOMPUTE: u64 = 10_000;
const QUAD_PAIRS: [(usize, usize); 6] = [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)];

const NUM_MOVES: usize = 11;
const BASE_WEIGHTS: [f64; NUM_MOVES] = [
    0.25, 0.15, 0.10, 0.08, 0.06, 0.06, 0.05, 0.06, 0.06, 0.06, 0.07,
];

pub enum WorkerCommand {
    SetState(Assignment),
    SetWeights(Weights),
    SetTemp(f64),
    Shutdown,
}

pub struct WorkerReport {
    pub core_id: usize,
    pub best_assignment: Assignment,
    pub best_cost: u32,
    pub current_assignment: Assignment,
    pub current_cost: u32,
    pub iterations_total: u64,
}

pub struct CpuWorkers {
    pub handles: Vec<thread::JoinHandle<()>>,
    pub commands: Vec<mpsc::Sender<WorkerCommand>>,
    pub reports: mpsc::Receiver<WorkerReport>,
}

pub fn run_cpu_workers(
    num_cores: usize,
    w8: Weights,
    temps: Vec<f64>,
    shutdown: Arc<AtomicBool>,
) -> CpuWorkers {
    let (report_tx, report_rx) = mpsc::channel();
    let mut cmd_txs = Vec::with_capacity(num_cores);
    let mut handles = Vec::with_capacity(num_cores);

    for core_id in 0..num_cores {
        let (cmd_tx, cmd_rx) = mpsc::channel();
        cmd_txs.push(cmd_tx);

        let w8 = w8.clone();
        let temp = temps[core_id];
        let shutdown = Arc::clone(&shutdown);
        let report_tx = report_tx.clone();

        handles.push(thread::spawn(move || {
            worker_loop(core_id, w8, temp, shutdown, cmd_rx, report_tx);
        }));
    }

    CpuWorkers {
        handles,
        commands: cmd_txs,
        reports: report_rx,
    }
}

fn worker_loop(
    core_id: usize,
    initial_w8: Weights,
    initial_temp: f64,
    shutdown: Arc<AtomicBool>,
    cmd_rx: mpsc::Receiver<WorkerCommand>,
    report_tx: mpsc::Sender<WorkerReport>,
) {
    let mut rng = SmallRng::from_os_rng();
    let mut a = random_assignment(&mut rng);
    let mut active_w8 = initial_w8;
    let mut active_temp = initial_temp;
    let mut cost = evaluate(&a, &active_w8);
    let mut best_a = a;
    let mut best_cost = cost.total;
    let mut stats = MoveStats::new();
    let mut iterations_total: u64 = 0;

    loop {
        while let Ok(cmd) = cmd_rx.try_recv() {
            match cmd {
                WorkerCommand::SetState(new_a) => {
                    a = new_a;
                    cost = evaluate(&a, &active_w8);
                    best_a = a;
                    best_cost = cost.total;
                }
                WorkerCommand::SetWeights(new_w8) => {
                    active_w8 = new_w8;
                    cost = evaluate(&a, &active_w8);
                    best_a = a;
                    best_cost = cost.total;
                }
                WorkerCommand::SetTemp(t) => {
                    active_temp = t;
                }
                WorkerCommand::Shutdown => return,
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

            // Exhaustive single-quad-pair search (periodic)
            if i > 0 && i % 100_000 == 0 && cost.total > 0 {
                let ew = rng.random_range(0..WEEKS);
                let (eq1, eq2) = QUAD_PAIRS[rng.random_range(0..QUAD_PAIRS.len())];
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
                    if cost.total < best_cost {
                        best_cost = cost.total;
                        best_a = a;
                    }
                }
            }

            let compound_prob = ((1000.0 - cost.total as f64) / 800.0).clamp(0.0, 0.5);
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
                    let (q1, q2) = QUAD_PAIRS[rng.random_range(0..QUAD_PAIRS.len())];
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
                    if cost.total < best_cost {
                        best_cost = cost.total;
                        best_a = a;
                    }
                } else {
                    a = saved;
                }
            } else {
                let move_id = stats.select(rng.random::<f64>());
                stats.attempts[move_id] += 1;

                let accepted = match move_id {
                    0 => {
                        let w = rng.random_range(0..WEEKS);
                        let (q1, q2) = QUAD_PAIRS[rng.random_range(0..QUAD_PAIRS.len())];
                        let p1 = rng.random_range(0..POS);
                        let p2 = rng.random_range(0..POS);
                        let tmp = a[w][q1][p1];
                        a[w][q1][p1] = a[w][q2][p2];
                        a[w][q2][p2] = tmp;
                        let new_cost = evaluate(&a, &active_w8);
                        let delta = new_cost.total as i64 - cost.total as i64;
                        if sa_accept(delta, active_temp, &mut rng) {
                            cost = new_cost;
                            if cost.total < best_cost { best_cost = cost.total; best_a = a; }
                            true
                        } else {
                            a[w][q2][p2] = a[w][q1][p1];
                            a[w][q1][p1] = tmp;
                            false
                        }
                    }
                    1 => {
                        let w = rng.random_range(0..WEEKS);
                        let q = rng.random_range(0..QUADS);
                        let p1 = rng.random_range(0..POS);
                        let mut p2 = rng.random_range(0..(POS - 1));
                        if p2 >= p1 { p2 += 1; }
                        let tmp = a[w][q][p1];
                        a[w][q][p1] = a[w][q][p2];
                        a[w][q][p2] = tmp;
                        let new_cost = evaluate(&a, &active_w8);
                        let delta = new_cost.total as i64 - cost.total as i64;
                        if sa_accept(delta, active_temp, &mut rng) {
                            cost = new_cost;
                            if cost.total < best_cost { best_cost = cost.total; best_a = a; }
                            true
                        } else {
                            a[w][q][p2] = a[w][q][p1];
                            a[w][q][p1] = tmp;
                            false
                        }
                    }
                    2 => {
                        let team = rng.random_range(0..TEAMS) as u8;
                        let w1 = rng.random_range(0..WEEKS);
                        let mut w2 = rng.random_range(0..(WEEKS - 1));
                        if w2 >= w1 { w2 += 1; }
                        let pos1 = find_team_in_week(&a, w1, team);
                        let pos2 = find_team_in_week(&a, w2, team);
                        if let (Some((q1, p1)), Some((q2, p2))) = (pos1, pos2) {
                            let save = (a[w1][q1][p1], a[w1][q2][p2], a[w2][q1][p1], a[w2][q2][p2]);
                            let other1 = a[w2][q1][p1];
                            let other2 = a[w1][q2][p2];
                            a[w1][q1][p1] = other2;
                            a[w1][q2][p2] = team;
                            a[w2][q2][p2] = other1;
                            a[w2][q1][p1] = team;
                            let new_cost = evaluate(&a, &active_w8);
                            let delta = new_cost.total as i64 - cost.total as i64;
                            if sa_accept(delta, active_temp, &mut rng) {
                                cost = new_cost;
                                if cost.total < best_cost { best_cost = cost.total; best_a = a; }
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
                        let w = rng.random_range(0..WEEKS);
                        let (q1, q2) = QUAD_PAIRS[rng.random_range(0..QUAD_PAIRS.len())];
                        let tmp = a[w][q1];
                        a[w][q1] = a[w][q2];
                        a[w][q2] = tmp;
                        let new_cost = evaluate(&a, &active_w8);
                        let delta = new_cost.total as i64 - cost.total as i64;
                        if sa_accept(delta, active_temp, &mut rng) {
                            cost = new_cost;
                            if cost.total < best_cost { best_cost = cost.total; best_a = a; }
                            true
                        } else {
                            a[w][q2] = a[w][q1];
                            a[w][q1] = tmp;
                            false
                        }
                    }
                    4 => {
                        let w1 = rng.random_range(0..WEEKS);
                        let mut w2 = rng.random_range(0..(WEEKS - 1));
                        if w2 >= w1 { w2 += 1; }
                        let tmp = a[w1];
                        a[w1] = a[w2];
                        a[w2] = tmp;
                        let new_cost = evaluate(&a, &active_w8);
                        let delta = new_cost.total as i64 - cost.total as i64;
                        if sa_accept(delta, active_temp, &mut rng) {
                            cost = new_cost;
                            if cost.total < best_cost { best_cost = cost.total; best_a = a; }
                            true
                        } else {
                            a[w2] = a[w1];
                            a[w1] = tmp;
                            false
                        }
                    }
                    5 => {
                        let w = rng.random_range(0..WEEKS);
                        let tmp0 = a[w][0]; let tmp1 = a[w][1];
                        a[w][0] = a[w][2]; a[w][2] = tmp0;
                        a[w][1] = a[w][3]; a[w][3] = tmp1;
                        let new_cost = evaluate(&a, &active_w8);
                        let delta = new_cost.total as i64 - cost.total as i64;
                        if sa_accept(delta, active_temp, &mut rng) {
                            cost = new_cost;
                            if cost.total < best_cost { best_cost = cost.total; best_a = a; }
                            true
                        } else {
                            let t0 = a[w][0]; let t1 = a[w][1];
                            a[w][0] = a[w][2]; a[w][2] = t0;
                            a[w][1] = a[w][3]; a[w][3] = t1;
                            false
                        }
                    }
                    6 => {
                        let w = rng.random_range(0..WEEKS);
                        let tmp0 = a[w][0]; let tmp2 = a[w][2];
                        a[w][0] = a[w][1]; a[w][1] = tmp0;
                        a[w][2] = a[w][3]; a[w][3] = tmp2;
                        let new_cost = evaluate(&a, &active_w8);
                        let delta = new_cost.total as i64 - cost.total as i64;
                        if sa_accept(delta, active_temp, &mut rng) {
                            cost = new_cost;
                            if cost.total < best_cost { best_cost = cost.total; best_a = a; }
                            true
                        } else {
                            let t1 = a[w][1]; let t3 = a[w][3];
                            a[w][1] = a[w][0]; a[w][0] = t1;
                            a[w][3] = a[w][2]; a[w][2] = t3;
                            false
                        }
                    }
                    7 => {
                        let w = rng.random_range(0..WEEKS);
                        let q = rng.random_range(0..QUADS);
                        a[w][q].swap(0, 1);
                        a[w][q].swap(2, 3);
                        let new_cost = evaluate(&a, &active_w8);
                        let delta = new_cost.total as i64 - cost.total as i64;
                        if sa_accept(delta, active_temp, &mut rng) {
                            cost = new_cost;
                            if cost.total < best_cost { best_cost = cost.total; best_a = a; }
                            true
                        } else {
                            a[w][q].swap(0, 1);
                            a[w][q].swap(2, 3);
                            false
                        }
                    }
                    8 => {
                        let saved = a;
                        if guided_matchup(&mut a, &mut rng) {
                            let new_cost = evaluate(&a, &active_w8);
                            let delta = new_cost.total as i64 - cost.total as i64;
                            if sa_accept(delta, active_temp, &mut rng) {
                                cost = new_cost;
                                if cost.total < best_cost { best_cost = cost.total; best_a = a; }
                                true
                            } else { a = saved; false }
                        } else { false }
                    }
                    9 => {
                        let saved = a;
                        if guided_lane(&mut a, &mut rng) {
                            let new_cost = evaluate(&a, &active_w8);
                            let delta = new_cost.total as i64 - cost.total as i64;
                            if sa_accept(delta, active_temp, &mut rng) {
                                cost = new_cost;
                                if cost.total < best_cost { best_cost = cost.total; best_a = a; }
                                true
                            } else { a = saved; false }
                        } else { false }
                    }
                    _ => {
                        let saved = a;
                        if guided_early_late(&mut a, &mut rng) {
                            let new_cost = evaluate(&a, &active_w8);
                            let delta = new_cost.total as i64 - cost.total as i64;
                            if sa_accept(delta, active_temp, &mut rng) {
                                cost = new_cost;
                                if cost.total < best_cost { best_cost = cost.total; best_a = a; }
                                true
                            } else { a = saved; false }
                        } else { false }
                    }
                };

                if accepted { stats.accepts[move_id] += 1; }
            }
        }
        iterations_total = batch_end;

        let _ = report_tx.send(WorkerReport {
            core_id,
            best_assignment: best_a,
            best_cost,
            current_assignment: a,
            current_cost: cost.total,
            iterations_total,
        });
    }
}

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
    if !found { return false; }

    let week_start = rng.random_range(0..WEEKS);
    for off in 0..WEEKS {
        let w = (week_start + off) % WEEKS;
        let (qa, _) = match find_team_in_week(a, w, ta) { Some(x) => x, None => continue };
        let (qb, pb) = match find_team_in_week(a, w, tb) { Some(x) => x, None => continue };
        if !same_half(qa, qb) || qa == qb { continue; }

        let candidates: Vec<usize> = (0..POS).filter(|&p| a[w][qa][p] != ta).collect();
        if candidates.is_empty() { continue; }
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
            if dev > worst_dev { worst_dev = dev; worst_team = t; }
        }
    }
    if worst_dev < 1.0 { return false; }

    let start = rng.random_range(0..WEEKS * QUADS);
    for off in 0..WEEKS * QUADS {
        let idx = (start + off) % (WEEKS * QUADS);
        let w = idx / QUADS;
        let q = idx % QUADS;
        let mut team_pos = None;
        for p in 0..POS {
            if a[w][q][p] == worst_team as u8 { team_pos = Some(p); break; }
        }
        let tp = match team_pos { Some(p) => p, None => continue };
        let mut swap_pos = rng.random_range(0..(POS - 1));
        if swap_pos >= tp { swap_pos += 1; }
        a[w][q].swap(tp, swap_pos);
        return true;
    }
    false
}

fn guided_early_late(a: &mut Assignment, rng: &mut SmallRng) -> bool {
    let mut early_count = [0i32; TEAMS];
    for w in 0..WEEKS {
        for q in 0..2 {
            for p in 0..POS { early_count[a[w][q][p] as usize] += 1; }
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
    if worst_dev < 1.0 { return false; }

    let start = rng.random_range(0..WEEKS);
    for off in 0..WEEKS {
        let w = (start + off) % WEEKS;
        let team = worst_team as u8;
        let in_early = (0..2).any(|q| (0..POS).any(|p| a[w][q][p] == team));
        if (too_many_early && in_early) || (!too_many_early && !in_early) {
            let tmp0 = a[w][0]; let tmp1 = a[w][1];
            a[w][0] = a[w][2]; a[w][2] = tmp0;
            a[w][1] = a[w][3]; a[w][3] = tmp1;
            return true;
        }
    }
    false
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
