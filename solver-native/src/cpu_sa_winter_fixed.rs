use std::sync::atomic::{AtomicBool, AtomicU32, Ordering};
use std::sync::{mpsc, Arc};
use std::thread;

use rand::rngs::SmallRng;
use rand::{Rng, SeedableRng};
use solver_core::winter_fixed::*;

pub const BATCH_SIZE: u64 = 10_000;
const STATS_RECOMPUTE: u64 = 10_000;
const COOLING_RATE: f64 = 0.999999915;
const MIN_TEMP: f64 = 1.0;

pub enum WinterFixedWorkerCommand {
    SetState(WinterFixedSchedule),
    SetStateWithTemp(WinterFixedSchedule, f64),
    /// Unconditionally reset both current and best state (for island cycling).
    /// The u64 is a generation tag so the main loop can identify which assignment
    /// a report belongs to.
    ResetState(WinterFixedSchedule, u64),
    Sweep,
    Shutdown,
}

pub struct WinterFixedWorkerReport {
    pub core_id: usize,
    pub best_schedule: WinterFixedSchedule,
    pub best_cost: u32,
    pub current_schedule: WinterFixedSchedule,
    pub current_cost: u32,
    pub iterations_total: u64,
    pub current_temp: f64,
    pub sweep_round: u32,
    pub move_rates: [f64; NUM_MOVES],
    pub move_shares: [f64; NUM_MOVES],
    pub generation: u64,
}

pub struct WinterFixedCpuWorkers {
    pub handles: Vec<thread::JoinHandle<()>>,
    pub commands: Vec<mpsc::Sender<WinterFixedWorkerCommand>>,
    pub reports: mpsc::Receiver<WinterFixedWorkerReport>,
    pub live_best_costs: Vec<Arc<AtomicU32>>,
}

pub fn run_winter_fixed_cpu_workers(
    num_cores: usize,
    w8: WinterFixedWeights,
    temps: Vec<f64>,
    shutdown: Arc<AtomicBool>,
) -> WinterFixedCpuWorkers {
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

    WinterFixedCpuWorkers {
        handles,
        commands: cmd_txs,
        reports: report_rx,
        live_best_costs,
    }
}

fn worker_loop(
    core_id: usize,
    w8: WinterFixedWeights,
    initial_temp: f64,
    shutdown: Arc<AtomicBool>,
    cmd_rx: mpsc::Receiver<WinterFixedWorkerCommand>,
    report_tx: mpsc::Sender<WinterFixedWorkerReport>,
    live_best_cost: Arc<AtomicU32>,
) {
    let mut rng = SmallRng::from_os_rng();
    let mut sched = random_fixed_schedule(&mut rng);
    let mut active_temp = initial_temp;
    let mut bd = evaluate_fixed(&sched, &w8);
    let mut cost = bd.total;
    let mut best_sched = sched;
    let mut best_cost = cost;
    let mut stats = MoveStats::new();
    let mut iterations_total: u64 = 0;
    let mut sweep_round: u32 = 0;
    let mut pending_sweep = false;
    let mut generation: u64 = 0;

    loop {
        while let Ok(cmd) = cmd_rx.try_recv() {
            match cmd {
                WinterFixedWorkerCommand::SetState(new_sched) => {
                    sched = new_sched;
                    bd = evaluate_fixed(&sched, &w8);
                    cost = bd.total;
                    if cost < best_cost {
                        best_sched = sched;
                        best_cost = cost;
                        live_best_cost.store(best_cost, Ordering::Relaxed);
                    }
                    active_temp = initial_temp;
                }
                WinterFixedWorkerCommand::SetStateWithTemp(new_sched, temp) => {
                    sched = new_sched;
                    bd = evaluate_fixed(&sched, &w8);
                    cost = bd.total;
                    if cost < best_cost {
                        best_sched = sched;
                        best_cost = cost;
                        live_best_cost.store(best_cost, Ordering::Relaxed);
                    }
                    active_temp = temp;
                }
                WinterFixedWorkerCommand::ResetState(new_sched, gen) => {
                    sched = new_sched;
                    bd = evaluate_fixed(&sched, &w8);
                    cost = bd.total;
                    best_sched = sched;
                    best_cost = cost;
                    live_best_cost.store(best_cost, Ordering::Relaxed);
                    active_temp = initial_temp;
                    generation = gen;
                }
                WinterFixedWorkerCommand::Sweep => {
                    pending_sweep = true;
                }
                WinterFixedWorkerCommand::Shutdown => return,
            }
        }

        if shutdown.load(Ordering::Relaxed) {
            return;
        }

        // Sweep mode
        if pending_sweep {
            pending_sweep = false;
            sweep_round += 1;
            sched = best_sched;
            let improvements = systematic_sweep(&mut sched, &w8, &shutdown);
            bd = evaluate_fixed(&sched, &w8);
            cost = bd.total;
            if cost < best_cost {
                best_cost = cost;
                best_sched = sched;
                live_best_cost.store(best_cost, Ordering::Relaxed);
            }
            let _ = report_tx.send(WinterFixedWorkerReport {
                core_id,
                best_schedule: best_sched,
                best_cost,
                current_schedule: sched,
                current_cost: cost,
                iterations_total,
                current_temp: active_temp,
                sweep_round,
                move_rates: stats.last_rates,
                move_shares: stats.last_shares,
                generation,
            });
            if improvements == 0 {
                sweep_round = 0;
            }
            active_temp = initial_temp;
            continue;
        }

        let batch_end = iterations_total + BATCH_SIZE;
        for i in iterations_total..batch_end {
            if i > 0 && i % STATS_RECOMPUTE == 0 {
                stats.recompute();
            }

            // Exhaustive position-pair search (periodic)
            if i > 0 && i % 100_000 == 0 && cost > 0 {
                let ew = rng.random_range(0..WF_WEEKS);
                let pa = rng.random_range(0..WF_POSITIONS);
                let mut best_delta = 0i64;
                let mut best_pb: Option<usize> = None;
                for pb in 0..WF_POSITIONS {
                    if pb == pa { continue; }
                    sched.mapping[ew].swap(pa, pb);
                    let nc = evaluate_fixed(&sched, &w8).total;
                    let d = nc as i64 - cost as i64;
                    if d < best_delta {
                        best_delta = d;
                        best_pb = Some(pb);
                    }
                    sched.mapping[ew].swap(pa, pb);
                }
                if let Some(pb) = best_pb {
                    sched.mapping[ew].swap(pa, pb);
                    bd = evaluate_fixed(&sched, &w8);
                    cost = bd.total;
                    if cost < best_cost {
                        best_cost = cost;
                        best_sched = sched;
                    }
                }
            }

            let move_id = stats.select(rng.random::<f64>());
            stats.attempts[move_id] += 1;

            let undo = apply_move(&mut sched, move_id, &bd, &mut rng);
            let new_bd = evaluate_fixed(&sched, &w8);
            let new_cost = new_bd.total;
            let delta = new_cost as i64 - cost as i64;

            if sa_accept(delta, active_temp, &mut rng) {
                cost = new_cost;
                bd = new_bd;
                stats.accepts[move_id] += 1;
                if cost < best_cost {
                    best_cost = cost;
                    best_sched = sched;
                }
            } else {
                undo_move(&mut sched, &undo);
            }

            active_temp = (active_temp * COOLING_RATE).max(MIN_TEMP);
        }
        iterations_total = batch_end;
        live_best_cost.store(best_cost, Ordering::Relaxed);

        // Reheat when temperature bottoms out
        if active_temp <= MIN_TEMP + 0.01 {
            sched = best_sched;
            bd = evaluate_fixed(&sched, &w8);
            // Mix of guided and random perturbation
            for _ in 0..3 {
                if rng.random_bool(0.5) {
                    let move_id = pick_move_guided_only(&bd);
                    let _ = apply_move(&mut sched, move_id, &bd, &mut rng);
                    bd = evaluate_fixed(&sched, &w8);
                } else {
                    let _ = apply_move(&mut sched, 0, &bd, &mut rng);
                }
            }
            bd = evaluate_fixed(&sched, &w8);
            cost = bd.total;
            active_temp = initial_temp;
        }

        let _ = report_tx.send(WinterFixedWorkerReport {
            core_id,
            best_schedule: best_sched,
            best_cost,
            current_schedule: sched,
            current_cost: cost,
            iterations_total,
            current_temp: active_temp,
            sweep_round,
            move_rates: stats.last_rates,
            move_shares: stats.last_shares,
            generation,
        });
    }
}

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
