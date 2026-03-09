use std::sync::atomic::{AtomicBool, AtomicU32, Ordering};
use std::sync::{mpsc, Arc};
use std::thread;

use rand::rngs::SmallRng;
use rand::{Rng, SeedableRng};
use solver_core::summer_fixed::*;

pub const BATCH_SIZE: u64 = 50_000;
const STATS_RECOMPUTE: u64 = 10_000;

pub const NUM_MOVES: usize = 7;
pub const MOVE_NAMES: [&str; NUM_MOVES] = [
    "tm_swap", "tog_01", "tog_23", "wk_swap",
    "g_match", "g_slot", "g_lane",
];

pub enum FixedWorkerCommand {
    SetState(FixedSchedule),
    SetTemp(f64),
    Shutdown,
}

pub struct SweepDetail {
    pub n_perturb: u32,
    pub pre_sweep_cost: u32,
    pub post_sweep_cost: u32,
    pub improvements: u32,
}

pub struct FixedWorkerReport {
    pub core_id: usize,
    pub best_sched: FixedSchedule,
    pub best_cost: u32,
    pub current_sched: FixedSchedule,
    pub current_cost: u32,
    pub current_temp: f64,
    pub sweep_round: Option<(u32, u32)>,
    pub sweep_detail: Option<SweepDetail>,
    pub iterations_total: u64,
    pub iterations_since_improve: u64,
    pub move_rates: [f64; NUM_MOVES],
    pub move_shares: [f64; NUM_MOVES],
}

pub struct FixedCpuWorkers {
    pub handles: Vec<thread::JoinHandle<()>>,
    pub commands: Vec<mpsc::Sender<FixedWorkerCommand>>,
    pub reports: mpsc::Receiver<FixedWorkerReport>,
    pub live_best_costs: Vec<Arc<AtomicU32>>,
}

pub fn run_fixed_cpu_workers(
    num_cores: usize,
    w8: FixedWeights,
    temps: Vec<f64>,
    shutdown: Arc<AtomicBool>,
    global_best_cost: Arc<AtomicU32>,
    enable_sweep: bool,
) -> FixedCpuWorkers {
    let (report_tx, report_rx) = mpsc::channel();
    let mut cmd_txs = Vec::with_capacity(num_cores);
    let mut handles = Vec::with_capacity(num_cores);
    let mut live_bests = Vec::with_capacity(num_cores);

    for core_id in 0..num_cores {
        let w8 = w8.clone();
        let shutdown = Arc::clone(&shutdown);
        let global_best = Arc::clone(&global_best_cost);
        let report_tx = report_tx.clone();
        let temp = temps[core_id];
        let (cmd_tx, cmd_rx) = mpsc::channel();
        cmd_txs.push(cmd_tx);
        let live_best = Arc::new(AtomicU32::new(u32::MAX));
        let lb = Arc::clone(&live_best);
        live_bests.push(live_best);

        handles.push(thread::spawn(move || {
            worker_loop(core_id, w8, temp, shutdown, lb, global_best, enable_sweep, report_tx, cmd_rx);
        }));
    }
    drop(report_tx);

    FixedCpuWorkers {
        handles,
        commands: cmd_txs,
        reports: report_rx,
        live_best_costs: live_bests,
    }
}

fn worker_loop(
    core_id: usize,
    w8: FixedWeights,
    initial_temp: f64,
    shutdown: Arc<AtomicBool>,
    live_best_cost: Arc<AtomicU32>,
    global_best_cost: Arc<AtomicU32>,
    enable_sweep: bool,
    report_tx: mpsc::Sender<FixedWorkerReport>,
    cmd_rx: mpsc::Receiver<FixedWorkerCommand>,
) {
    let mut rng = SmallRng::from_os_rng();
    let mut sched = random_fixed_schedule(&mut rng);
    let mut bd = evaluate_fixed(&sched, &w8);
    let mut current_cost = bd.total;
    let mut best_sched = sched;
    let mut best_cost = current_cost;

    let mut temp: f64 = initial_temp;
    let cooling_rate: f64 = 0.99999983;
    let min_temp: f64 = 1.0;
    let mut iterations_since_improve: u64 = 0;
    let mut iterations_total: u64 = 0;

    let mut move_attempts = [0u64; NUM_MOVES];
    let mut move_accepts = [0u64; NUM_MOVES];
    let mut move_selected = [0u64; NUM_MOVES];
    let mut stats_iters: u64 = 0;

    while !shutdown.load(Ordering::Relaxed) {
        // Process commands
        while let Ok(cmd) = cmd_rx.try_recv() {
            match cmd {
                FixedWorkerCommand::SetState(new_sched) => {
                    sched = new_sched;
                    bd = evaluate_fixed(&sched, &w8);
                    current_cost = bd.total;
                    best_cost = current_cost;
                    best_sched = sched;
                    live_best_cost.store(best_cost, Ordering::Relaxed);
                    temp = initial_temp;
                    iterations_since_improve = 0;
                }
                FixedWorkerCommand::SetTemp(t) => {
                    temp = t;
                }
                FixedWorkerCommand::Shutdown => return,
            }
        }

        let is_sweeping = temp <= min_temp + 0.01;
        if is_sweeping && enable_sweep {
            // Run 20 sweep rounds with increasing perturbation, keep overall best
            let mut sweep_best_sched = best_sched;
            let mut sweep_best_cost = best_cost;

            let sweep_global_best_at_start = global_best_cost.load(Ordering::Relaxed);
            // Round 0: no perturbation, rounds 1-19: random 1-3 perturbations
            const SWEEP_ROUNDS: u32 = 20;
            let perturb_counts: [u32; SWEEP_ROUNDS as usize] = [
                0, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3, 3, 3, 3,
            ];
            for round in 0..SWEEP_ROUNDS {
                // Abort sweep if shutting down or GPU found a new global best
                if shutdown.load(Ordering::Relaxed) { break; }
                if global_best_cost.load(Ordering::Relaxed) < sweep_global_best_at_start {
                    break;
                }

                // Start each round from the best known schedule
                sched = sweep_best_sched;
                let n_perturb = perturb_counts[round as usize];

                bd = evaluate_fixed(&sched, &w8);
                for _ in 0..n_perturb {
                    let _ = apply_move(&mut sched, 0, &bd, &mut rng);
                }
                let pre_cost = evaluate_fixed(&sched, &w8).total;

                let improved = systematic_sweep(&mut sched, &w8, &shutdown);
                bd = evaluate_fixed(&sched, &w8);
                current_cost = bd.total;
                iterations_total += 360_000;
                stats_iters += 360_000;

                if current_cost < sweep_best_cost {
                    sweep_best_cost = current_cost;
                    sweep_best_sched = sched;
                }

                // Report progress after each round
                let _ = report_tx.send(FixedWorkerReport {
                    core_id,
                    best_sched: sweep_best_sched,
                    best_cost: sweep_best_cost,
                    current_sched: sched,
                    current_cost,
                    current_temp: temp,
                    sweep_round: Some((round + 1, SWEEP_ROUNDS)),
                    sweep_detail: Some(SweepDetail {
                        n_perturb,
                        pre_sweep_cost: pre_cost,
                        post_sweep_cost: current_cost,
                        improvements: improved,
                    }),
                    iterations_total,
                    iterations_since_improve,
                    move_rates: [0.0; NUM_MOVES],
                    move_shares: [0.0; NUM_MOVES],
                });
            }

            // Adopt the best result from all rounds
            sched = sweep_best_sched;
            bd = evaluate_fixed(&sched, &w8);
            current_cost = bd.total;

            if current_cost < best_cost {
                best_cost = current_cost;
                best_sched = sched;
                live_best_cost.store(best_cost, Ordering::Relaxed);
            }

            // After sweep, perturb from best and reheat for fresh SA exploration
            sched = best_sched;
            bd = evaluate_fixed(&sched, &w8);
            for _ in 0..5 {
                let _ = apply_move(&mut sched, 0, &bd, &mut rng);
            }
            bd = evaluate_fixed(&sched, &w8);
            current_cost = bd.total;
            temp = initial_temp;
            iterations_since_improve = 0;
        } else if is_sweeping {
            // No sweep enabled: perturb from best and reheat
            sched = best_sched;
            bd = evaluate_fixed(&sched, &w8);
            for _ in 0..5 {
                let _ = apply_move(&mut sched, 0, &bd, &mut rng);
            }
            bd = evaluate_fixed(&sched, &w8);
            current_cost = bd.total;
            temp = initial_temp;
            iterations_since_improve = 0;
        } else {
            for _ in 0..BATCH_SIZE {
                let move_id = pick_move(&mut rng, &bd);
                move_selected[move_id] += 1;
                move_attempts[move_id] += 1;

                let undo = apply_move(&mut sched, move_id, &bd, &mut rng);
                let new_bd = evaluate_fixed(&sched, &w8);
                let new_cost = new_bd.total;
                let delta = new_cost as i64 - current_cost as i64;

                if sa_accept(delta, temp, &mut rng) {
                    current_cost = new_cost;
                    bd = new_bd;
                    move_accepts[move_id] += 1;

                    if current_cost < best_cost {
                        best_cost = current_cost;
                        best_sched = sched;
                        iterations_since_improve = 0;
                        live_best_cost.store(best_cost, Ordering::Relaxed);
                    }
                } else {
                    undo_move(&mut sched, &undo);
                }

                temp = (temp * cooling_rate).max(min_temp);
                iterations_since_improve += 1;
                iterations_total += 1;
                stats_iters += 1;
            }
        }

        // Compute move rates
        let mut rates = [0.0f64; NUM_MOVES];
        let mut shares = [0.0f64; NUM_MOVES];
        if stats_iters > 0 {
            for m in 0..NUM_MOVES {
                if move_attempts[m] > 0 {
                    rates[m] = move_accepts[m] as f64 / move_attempts[m] as f64;
                }
                shares[m] = move_selected[m] as f64 / stats_iters as f64;
            }
        }

        let _ = report_tx.send(FixedWorkerReport {
            core_id,
            best_sched,
            best_cost,
            current_sched: sched,
            current_cost,
            current_temp: temp,
            sweep_round: None,
            sweep_detail: None,
            iterations_total,
            iterations_since_improve,
            move_rates: rates,
            move_shares: shares,
        });

        // Reset stats periodically
        if stats_iters >= STATS_RECOMPUTE {
            move_attempts = [0; NUM_MOVES];
            move_accepts = [0; NUM_MOVES];
            move_selected = [0; NUM_MOVES];
            stats_iters = 0;
        }

        // When SA stagnates, trigger a sweep from best
        if iterations_since_improve > 50_000_000 {
            sched = best_sched;
            temp = min_temp;
            bd = evaluate_fixed(&sched, &w8);
            current_cost = bd.total;
            iterations_since_improve = 0;
        }
    }
}

fn sa_accept(delta: i64, temp: f64, rng: &mut SmallRng) -> bool {
    if delta < 0 { return true; }
    if delta == 0 { return rng.random_bool(0.2); }
    let prob = (-delta as f64 / temp).exp();
    rng.random_bool(prob.min(1.0))
}
