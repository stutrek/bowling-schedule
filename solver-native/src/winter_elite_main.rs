use rand::rngs::SmallRng;
use rand::{Rng, SeedableRng};
use crate::cpu_sa_winter_fixed::{self, WinterFixedCpuWorkers, WinterFixedWorkerCommand};
use crate::gpu_sa_loop;
use crate::gpu_setup::create_gpu_resources;
use crate::gpu_types::*;
use crate::gpu_types_winter_fixed::*;
use crate::island_pool::{IslandPool, NUM_ISLANDS, ISLAND_SIZE, REFINEMENT_ITERS};
use crate::output::*;
use crate::output_winter_elite::*;
use solver_core::winter_fixed::*;
use solver_core::winter;
use std::collections::HashSet;
use std::fs;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::time::Instant;

const SAVE_THRESHOLD: u32 = 600;
const VERIFY_INTERVAL_SECS: u64 = 30;
const CPU_TEMP: f64 = 10.0;

fn maybe_save_result(
    sched: &WinterFixedSchedule,
    cost: u32,
    source_label: &str,
    results_dir: &str,
    saved_hashes: &mut HashSet<[u32; WF_ASSIGN_U32S]>,
) {
    let mut out = *sched;
    reassign_commissioners(&mut out);
    let hash = pack_fixed_schedule(&out);
    if saved_hashes.insert(hash) {
        let ts = chrono::Local::now().format("%Y%m%d-%H%M%S%z");
        let filename = format!("{}/{:04}-{}-{}.tsv", results_dir, cost, source_label, ts);
        let _ = fs::write(&filename, fixed_schedule_to_tsv(&out));
    }
}

pub fn run(shutdown: Arc<AtomicBool>, args: &[String]) {
    let no_seed = args.iter().any(|a| a == "--no-seed");
    let dedup = args.iter().any(|a| a == "--dedup");

    let weights_str = fs::read_to_string("../weights.json").expect("Failed to read weights.json");
    let winter_w8: winter::Weights = serde_json::from_str(&weights_str).expect("Invalid weights.json");
    let w8 = WinterFixedWeights {
        matchup_zero: winter_w8.matchup_zero,
        matchup_triple: winter_w8.matchup_triple,
        consecutive_opponents: winter_w8.consecutive_opponents,
        early_late_balance: winter_w8.early_late_balance,
        early_late_alternation: winter_w8.early_late_alternation,
        early_late_consecutive: winter_w8.early_late_consecutive,
        lane_balance: winter_w8.lane_balance,
        lane_switch: winter_w8.lane_switch,
        late_lane_balance: winter_w8.late_lane_balance,
        commissioner_overlap: winter_w8.commissioner_overlap,
        half_season_repeat: winter_w8.half_season_repeat,
    };

    let results_dir = "results/gpu";
    fs::create_dir_all(results_dir).expect("Failed to create results directory");

    let mut seeds: Vec<WinterFixedSchedule> = Vec::new();
    if !no_seed {
        for seed_dir in &["results/split-sa/full", results_dir] {
            if let Ok(entries) = fs::read_dir(seed_dir) {
                for entry in entries.flatten() {
                    let path = entry.path();
                    if path.extension().map_or(false, |e| e == "tsv") {
                        if let Ok(contents) = fs::read_to_string(&path) {
                            if let Some(s) = parse_fixed_tsv(&contents) {
                                seeds.push(s);
                            }
                        }
                    }
                }
            }
        }
    }

    let available_cores = std::thread::available_parallelism().map(|n| n.get()).unwrap_or(4);
    let cpu_cores = available_cores.saturating_sub(2);

    let cpu_temps: Vec<f64> = vec![CPU_TEMP; cpu_cores];

    let cpu_workers = cpu_sa_winter_fixed::run_winter_fixed_cpu_workers(
        cpu_cores,
        w8.clone(),
        cpu_temps,
        Arc::clone(&shutdown),
    );

    let chain_count = detect_chain_count(WF_ASSIGN_U32S);
    let expected_chains = (NUM_ISLANDS * ISLAND_SIZE) as u32;
    assert!(
        chain_count >= expected_chains,
        "GPU supports {} chains but need {} ({} islands × {} chains)",
        chain_count, expected_chains, NUM_ISLANDS, ISLAND_SIZE,
    );
    // Use exactly NUM_ISLANDS * ISLAND_SIZE chains
    let chain_count = expected_chains;

    eprintln!(
        "GPU winter-elite solver: {} chains = {} islands × {} chains/island, {} iters/dispatch",
        chain_count, NUM_ISLANDS, ISLAND_SIZE, ITERS_PER_DISPATCH,
    );
    eprintln!(
        "  ASSIGN_U32S: {}, {} CPU cores at T={:.1}, {} seed files",
        WF_ASSIGN_U32S, cpu_cores, CPU_TEMP, seeds.len(),
    );

    let mut rng = SmallRng::from_os_rng();
    let mut assign_data = vec![0u32; chain_count as usize * WF_ASSIGN_U32S];
    let mut rng_data = vec![0u32; chain_count as usize * 4];
    let mut cost_data = vec![0u32; chain_count as usize];
    let mut best_assign_data = vec![0u32; chain_count as usize * WF_ASSIGN_U32S];
    let mut best_cost_data = vec![u32::MAX; chain_count as usize];

    // All chains start random
    for i in 0..chain_count as usize {
        let s = random_fixed_schedule(&mut rng);
        let packed = pack_fixed_schedule(&s);
        let cost = evaluate_fixed(&s, &w8).total;

        assign_data[i * WF_ASSIGN_U32S..(i + 1) * WF_ASSIGN_U32S].copy_from_slice(&packed);
        best_assign_data[i * WF_ASSIGN_U32S..(i + 1) * WF_ASSIGN_U32S].copy_from_slice(&packed);
        cost_data[i] = cost;
        best_cost_data[i] = cost;

        let seed_val = (i as u64).wrapping_mul(0x9E3779B97F4A7C15) ^ 0xBF58476D1CE4E5B9;
        rng_data[i * 4] = (seed_val & 0xFFFFFFFF) as u32 | 1;
        rng_data[i * 4 + 1] = ((seed_val >> 32) & 0xFFFFFFFF) as u32 | 1;
        rng_data[i * 4 + 2] = rng.random::<u32>() | 1;
        rng_data[i * 4 + 3] = rng.random::<u32>() | 1;
    }

    let gpu_weights = GpuWinterFixedWeights {
        matchup_zero: w8.matchup_zero,
        matchup_triple: w8.matchup_triple,
        consecutive_opponents: w8.consecutive_opponents,
        early_late_balance: w8.early_late_balance as f32,
        early_late_alternation: w8.early_late_alternation,
        early_late_consecutive: w8.early_late_consecutive,
        lane_balance: w8.lane_balance as f32,
        lane_switch: w8.lane_switch as f32,
        late_lane_balance: w8.late_lane_balance as f32,
        commissioner_overlap: w8.commissioner_overlap,
        half_season_repeat: w8.half_season_repeat,
        _pad0: 0,
    };

    let gpu_params = GpuParams {
        iters_per_dispatch: ITERS_PER_DISPATCH,
        chain_count,
        temp_base: GPU_TEMP_MIN as f32,
        temp_step: GPU_TEMP_MAX as f32,
        pod_size: POD_SIZE as u32,
        _pad0: 0, _pad1: 0, _pad2: 0,
    };

    let shader_src = format!(
        "{}\n{}",
        wgsl_consts(),
        include_str!("winter_fixed_solver.wgsl"),
    );

    pollster::block_on(run_gpu(
        assign_data, best_assign_data, cost_data, best_cost_data, rng_data,
        gpu_weights, gpu_params, w8, results_dir.to_string(), shutdown, rng,
        cpu_workers, cpu_cores, shader_src, dedup,
    ));
}

#[allow(clippy::too_many_arguments)]
async fn run_gpu(
    assign_data: Vec<u32>,
    best_assign_data: Vec<u32>,
    cost_data: Vec<u32>,
    best_cost_data: Vec<u32>,
    rng_data: Vec<u32>,
    gpu_weights: GpuWinterFixedWeights,
    gpu_params: GpuParams,
    w8: WinterFixedWeights,
    results_dir: String,
    shutdown: Arc<AtomicBool>,
    mut rng: SmallRng,
    cpu_workers: WinterFixedCpuWorkers,
    cpu_cores: usize,
    shader_src: String,
    dedup: bool,
) {
    let chain_count = gpu_params.chain_count;

    let gpu = create_gpu_resources(
        &assign_data, &best_assign_data, &cost_data, &best_cost_data, &rng_data,
        bytemuck::bytes_of(&gpu_weights), &gpu_params,
        bytemuck::bytes_of(&WF_THRESH_DEFAULT),
        WF_ASSIGN_U32S,
        &shader_src,
        include_str!("winter_fixed_exchange.wgsl"),
    ).await;

    let mut global_best_cost = u32::MAX;
    let mut global_best_schedule: Option<WinterFixedSchedule> = None;
    let mut dispatch_count = 0u64;
    let start_time = Instant::now();
    let mut last_verify = Instant::now();
    let mut last_thresh = WF_THRESH_DEFAULT;

    let mut pool = IslandPool::new(10.0);
    let mut worker_states: Vec<EliteWorkerState> = (0..cpu_cores).map(|_| EliteWorkerState::Idle).collect();
    let mut worker_metas: Vec<EliteWorkerMeta> = (0..cpu_cores).map(|_| EliteWorkerMeta {
        last_report: None,
        prev_iterations: 0,
        prev_iter_time: Instant::now(),
        iters_per_sec: 0,
        best_found_at: Instant::now(),
    }).collect();

    let mut global_best_meta = GlobalBestMeta {
        source: "none".to_string(),
        found_at: Instant::now(),
    };
    let mut saved_hashes: HashSet<[u32; WF_ASSIGN_U32S]> = HashSet::new();
    if let Ok(entries) = fs::read_dir(&results_dir) {
        for entry in entries.flatten() {
            let path = entry.path();
            if path.extension().map_or(false, |e| e == "tsv") {
                if let Ok(contents) = fs::read_to_string(&path) {
                    if let Some(s) = parse_fixed_tsv(&contents) {
                        saved_hashes.insert(pack_fixed_schedule(&s));
                    }
                }
            }
        }
    }
    eprintln!("  Loaded {} existing result hashes for dedup", saved_hashes.len());

    let locked_chains: HashSet<usize> = HashSet::new();
    let mut exchange_swaps_total: u64 = 0;
    let mut exchange_attempts_total: u64 = 0;

    let mut pending_events: Vec<String> = Vec::new();
    let mut last_print = Instant::now();
    let mut last_fresh_table = Instant::now();
    let mut prev_line_count: u32 = 0;

    let display_active = Arc::new(AtomicBool::new(true));
    {
        let da = Arc::clone(&display_active);
        std::thread::spawn(move || {
            use std::io::Read;
            let _ = std::process::Command::new("stty").args(["-icanon", "-echo"]).status();
            let stdin = std::io::stdin();
            for byte in stdin.lock().bytes().flatten() {
                if byte == b'd' {
                    let prev = da.load(Ordering::Relaxed);
                    da.store(!prev, Ordering::Relaxed);
                    if prev {
                        eprintln!("\n[display OFF - press 'd' to re-enable]");
                    } else {
                        eprintln!("[display ON]");
                    }
                }
            }
        });
    }

    loop {
        if shutdown.load(Ordering::Relaxed) { break; }

        // ── Display macro ──
        macro_rules! maybe_print_table {
            () => {
                if !pending_events.is_empty() && !display_active.load(Ordering::Relaxed) {
                    for msg in pending_events.drain(..) {
                        eprintln!("{}", msg);
                    }
                }
                if display_active.load(Ordering::Relaxed) && (last_print.elapsed().as_millis() >= 1000 || !pending_events.is_empty()) {
                    let fresh = last_fresh_table.elapsed().as_secs() >= FRESH_TABLE_INTERVAL_SECS;
                    if !fresh && prev_line_count > 0 {
                        eprint!("\x1b[{}A\r\x1b[J", prev_line_count);
                    } else if fresh {
                        last_fresh_table = Instant::now();
                    }
                    for msg in pending_events.drain(..) {
                        eprintln!("{}", msg);
                    }
                    let mut lines: u32 = 0;
                    if let Some(ref best) = global_best_schedule {
                        let best_bd = evaluate_fixed(best, &w8);
                        let elapsed_s = start_time.elapsed().as_secs_f64().max(0.001);
                        let gpu_ips = (dispatch_count as f64 * ITERS_PER_DISPATCH as f64 * chain_count as f64 / elapsed_s) as u64;
                        let cpu_ips: u64 = worker_metas.iter().map(|m| m.iters_per_sec).sum();
                        print_elite_banner(global_best_cost, &best_bd, &global_best_meta, start_time, gpu_ips, cpu_ips);
                        lines += 2;
                    }
                    {
                        let active: Vec<usize> = pool.islands.iter().enumerate()
                            .filter(|(_, isl)| isl.best_cost < u32::MAX)
                            .map(|(i, _)| i)
                            .collect();
                        let avg_d = pool.sampled_avg_pairwise_distance(&active, &mut rng);
                        let stats = pool.stats(dispatch_count);
                        print_island_summary(&stats, avg_d);
                        lines += 1;
                    }
                    eprintln!();
                    lines += 1;
                    print_elite_table_header();
                    lines += 1;
                    for (i, (state, meta)) in worker_states.iter().zip(worker_metas.iter()).enumerate() {
                        print_elite_worker_row(i, state, meta, REFINEMENT_ITERS);
                        lines += 1;
                    }
                    eprintln!();
                    lines += 1;
                    print_elite_move_stats(&worker_metas);
                    lines += 2;
                    {
                        let island_costs: Vec<u32> = pool.islands.iter().map(|isl| isl.best_cost).collect();
                        let hist = build_histogram(&island_costs, 140);
                        for row in 0..3 {
                            eprintln!("  {} │{}\x1b[K", hist.labels[row], hist.rows[row]);
                        }
                        eprintln!("       └{}\x1b[K", hist.legend);
                        lines += 4;
                    }
                    if exchange_attempts_total > 0 {
                        let rate = exchange_swaps_total as f64 / exchange_attempts_total as f64 * 100.0;
                        let status = if rate > 80.0 { "wasteful" }
                            else if rate > 60.0 { "high" }
                            else if rate > 40.0 { "good" }
                            else if rate > 25.0 { "ok" }
                            else if rate > 10.0 { "low" }
                            else { "awful" };
                        eprintln!("  exchange: {}/{} ({:.1}%) {}\x1b[K",
                            exchange_swaps_total, exchange_attempts_total, rate, status);
                        lines += 1;
                    }
                    eprint!("\x1b[J");
                    prev_line_count = lines;
                    last_print = Instant::now();
                }
            };
        }

        macro_rules! event {
            ($($arg:tt)*) => {
                pending_events.push(format_event($($arg)*));
            };
        }

        // ── 1. GPU dispatch + readback ──
        let readback = gpu_sa_loop::dispatch_and_readback(&gpu, chain_count, &shutdown, dispatch_count);
        let (best_costs, current_costs) = match readback {
            Some(pair) => pair,
            None => continue,
        };
        dispatch_count += 1;

        // ── 2. Per-island: find best chain, update pool ──
        for island_idx in 0..NUM_ISLANDS {
            let start = island_idx * ISLAND_SIZE;
            let end = start + ISLAND_SIZE;
            let mut best_in_island = u32::MAX;
            let mut best_chain = start;
            for ci in start..end {
                if best_costs[ci] < best_in_island {
                    best_in_island = best_costs[ci];
                    best_chain = ci;
                }
            }
            if best_in_island < pool.islands[island_idx].best_cost {
                // Read the actual chain data
                if let Some(packed_vec) = gpu_sa_loop::read_chain_raw(&gpu, best_chain, WF_ASSIGN_U32S) {
                    let mut packed = [0u32; WF_ASSIGN_U32S];
                    packed.copy_from_slice(&packed_vec);
                    let sched = unpack_fixed_schedule(&packed);
                    let verify_cost = evaluate_fixed(&sched, &w8).total;

                    if pool.update_island_best(island_idx, &packed, verify_cost, dispatch_count) {
                        if verify_cost <= SAVE_THRESHOLD {
                            maybe_save_result(&sched, verify_cost, &format!("gpu-i{}", island_idx), &results_dir, &mut saved_hashes);
                        }
                        if verify_cost < global_best_cost {
                            global_best_cost = verify_cost;
                            global_best_schedule = Some(sched);
                            global_best_meta = GlobalBestMeta {
                                source: format!("gpu→island#{}", island_idx),
                                found_at: Instant::now(),
                            };
                            event!(start_time.elapsed(), &format!(
                                "NEW BEST {} from gpu→island#{}", global_best_cost, island_idx));
                        }
                    }
                }
            }
        }

        // ── 3. Replica exchange (within-island only) ──
        let (attempts, swaps) = gpu_sa_loop::replica_exchange(
            &gpu, &current_costs, chain_count, ISLAND_SIZE,
            temp_for_level, &locked_chains, &mut rng, dispatch_count,
        );
        exchange_attempts_total += attempts;
        exchange_swaps_total += swaps;

        maybe_print_table!();

        // ── 4. Drain CPU reports ──
        while let Ok(report) = cpu_workers.reports.try_recv() {
            let cid = report.core_id;
            if cid >= worker_metas.len() { continue; }

            let real_best = evaluate_fixed(&report.best_schedule, &w8).total;

            // Update global best
            if real_best < global_best_cost {
                global_best_cost = real_best;
                global_best_schedule = Some(report.best_schedule);
                let island_label = match &worker_states[cid] {
                    EliteWorkerState::Refining { island_idx, .. } => format!("cpu{}→island#{}", cid, island_idx),
                    EliteWorkerState::Idle => format!("cpu{}", cid),
                };
                global_best_meta = GlobalBestMeta {
                    source: island_label.clone(),
                    found_at: Instant::now(),
                };
                event!(start_time.elapsed(), &format!("NEW BEST {} from {}", global_best_cost, island_label));
            }
            if real_best <= SAVE_THRESHOLD {
                let label = match &worker_states[cid] {
                    EliteWorkerState::Refining { island_idx, .. } => format!("cpu{}-i{}", cid, island_idx),
                    EliteWorkerState::Idle => format!("cpu{}", cid),
                };
                maybe_save_result(&report.best_schedule, real_best, &label, &results_dir, &mut saved_hashes);
            }

            // Check if refinement cycle is complete
            if let EliteWorkerState::Refining { island_idx, start_iters, .. } = &worker_states[cid] {
                let island_idx = *island_idx;
                let start_iters = *start_iters;
                let cycle_iters = report.iterations_total.saturating_sub(start_iters);

                // Skip stale reports from before ResetState was processed.
                // A report needs at least one batch of progress to be from the current island.
                if cycle_iters < 10_000 {
                    // Update meta only, don't touch island
                    worker_metas[cid].last_report = Some(report);
                    continue;
                }

                // Update island best from CPU (only if island hasn't been reset)
                if pool.islands[island_idx].best_cost < u32::MAX {
                    let packed = pack_fixed_schedule(&report.best_schedule);
                    if pool.update_island_best(island_idx, &packed, real_best, dispatch_count) {
                        worker_metas[cid].best_found_at = Instant::now();
                    }
                }

                if cycle_iters >= REFINEMENT_ITERS {
                    // Cycle complete — reseed island and go idle (only if island hasn't been reset)
                    if pool.islands[island_idx].best_cost < u32::MAX {
                        let best_sched = report.best_schedule;
                        let island_start = island_idx * ISLAND_SIZE;
                        let mut assign_bulk = Vec::with_capacity(ISLAND_SIZE * WF_ASSIGN_U32S);
                        let mut cost_bulk = Vec::with_capacity(ISLAND_SIZE);
                        for ci in island_start..island_start + ISLAND_SIZE {
                            let level_in_pod = (ci % TEMP_LEVELS) % POD_SIZE;
                            let t_frac = level_in_pod as f64 / (POD_SIZE - 1).max(1) as f64;
                            let pert = (t_frac * t_frac * 40.0) as usize;
                            let mut s = best_sched;
                            perturb_fixed(&mut s, &mut rng, pert);
                            let p = pack_fixed_schedule(&s);
                            let c = evaluate_fixed(&s, &w8).total;
                            assign_bulk.extend_from_slice(&p);
                            cost_bulk.push(c);
                        }
                        gpu_sa_loop::write_island_raw(&gpu, island_start, &assign_bulk, &cost_bulk, WF_ASSIGN_U32S);
                        pool.mark_refined(island_idx);
                    }
                    worker_states[cid] = EliteWorkerState::Idle;
                }
            }

            // Update meta
            let dt = worker_metas[cid].prev_iter_time.elapsed().as_secs_f64();
            if dt >= 1.0 {
                let di = report.iterations_total.saturating_sub(worker_metas[cid].prev_iterations);
                worker_metas[cid].iters_per_sec = (di as f64 / dt) as u64;
                worker_metas[cid].prev_iterations = report.iterations_total;
                worker_metas[cid].prev_iter_time = Instant::now();
            }
            worker_metas[cid].last_report = Some(report);
        }

        // ── 5. Assign idle workers ──
        let mut busy_islands: Vec<usize> = worker_states.iter()
            .filter_map(|s| s.island_idx())
            .collect();
        for cid in 0..cpu_cores {
            if !matches!(worker_states[cid], EliteWorkerState::Idle) { continue; }

            if let Some(island_idx) = pool.pick_for_refinement(&busy_islands) {
                let island = &pool.islands[island_idx];
                if island.best_cost < u32::MAX {
                    let sched = unpack_fixed_schedule(&island.best_packed);
                    let start_iters = worker_metas[cid].last_report.as_ref()
                        .map(|r| r.iterations_total)
                        .unwrap_or(0);
                    let _ = cpu_workers.commands[cid].send(WinterFixedWorkerCommand::ResetState(sched));
                    worker_states[cid] = EliteWorkerState::Refining {
                        island_idx,
                        start_iters,
                        start_cost: island.best_cost,
                    };
                    busy_islands.push(island_idx);
                }
            }
        }

        // ── 6. Adaptive move thresholds ──
        {
            let mut avg_rates = [0.0f64; NUM_MOVES];
            let mut n = 0usize;
            for meta in worker_metas.iter() {
                if let Some(ref r) = meta.last_report {
                    for m in 0..NUM_MOVES {
                        avg_rates[m] += r.move_rates[m];
                    }
                    n += 1;
                }
            }
            if n > 0 {
                let nf = n as f64;
                let mut weights = [0.0f64; WF_GPU_NUM_MOVES];
                for m in 0..WF_GPU_NUM_MOVES {
                    let rate = avg_rates[m] / nf;
                    weights[m] = WF_GPU_BASE_WEIGHTS[m] * (0.1 + rate);
                }
                let sum: f64 = weights.iter().sum();
                let mut thresh = GpuWinterFixedMoveThresholds { t: [0u32; 8] };
                let mut cum = 0.0;
                for m in 0..WF_GPU_NUM_MOVES {
                    cum += weights[m] / sum;
                    thresh.t[m] = (cum * 100.0).round() as u32;
                }
                thresh.t[WF_GPU_NUM_MOVES - 1] = 100;
                if thresh.t != last_thresh.t {
                    gpu.queue.write_buffer(&gpu.move_thresh_buf, 0, bytemuck::bytes_of(&thresh));
                    last_thresh = thresh;
                }
            }
        }

        // ── 7. Periodic maintenance: dedup + stagnation (only with --dedup) ──
        if dedup {
            let resets = pool.periodic_maintenance(dispatch_count, &[]);
            for (island_idx, reason) in &resets {
                // Evict any CPU worker assigned to this island
                for cid in 0..cpu_cores {
                    if worker_states[cid].island_idx() == Some(*island_idx) {
                        worker_states[cid] = EliteWorkerState::Idle;
                    }
                }
                // Write random schedules to the reset island's chains
                let island_start = island_idx * ISLAND_SIZE;
                let mut assign_bulk = Vec::with_capacity(ISLAND_SIZE * WF_ASSIGN_U32S);
                let mut cost_bulk = Vec::with_capacity(ISLAND_SIZE);
                for _ in 0..ISLAND_SIZE {
                    let s = random_fixed_schedule(&mut rng);
                    let p = pack_fixed_schedule(&s);
                    let c = evaluate_fixed(&s, &w8).total;
                    assign_bulk.extend_from_slice(&p);
                    cost_bulk.push(c);
                }
                gpu_sa_loop::write_island_raw(&gpu, island_start, &assign_bulk, &cost_bulk, WF_ASSIGN_U32S);
                event!(start_time.elapsed(), &format!(
                    "{}: island#{} reset", reason.to_uppercase(), island_idx));
            }

            // ── 8. Adaptive min_distance ──
            pool.maybe_adapt(dispatch_count, &mut rng);
        }

        // ── 9. Periodic verification ──
        if last_verify.elapsed().as_secs() >= VERIFY_INTERVAL_SECS {
            if let Some(ref best) = global_best_schedule {
                let verify_cost = evaluate_fixed(best, &w8);
                if verify_cost.total != global_best_cost {
                    event!(start_time.elapsed(), &format!(
                        "VERIFY: MISMATCH global best {} != cpu eval {}",
                        global_best_cost, verify_cost.total,
                    ));
                }
            }
            last_verify = Instant::now();
        }

        maybe_print_table!();
    }

    // Shutdown
    for cmd_tx in &cpu_workers.commands {
        let _ = cmd_tx.send(WinterFixedWorkerCommand::Shutdown);
    }
    for h in cpu_workers.handles {
        let _ = h.join();
    }

    let _ = std::process::Command::new("stty").args(["icanon", "echo"]).status();

    if let Some(best) = global_best_schedule {
        let final_cost = evaluate_fixed(&best, &w8);
        eprintln!("{}", format_event(start_time.elapsed(), &format!(
            "Final best: {} | {}", global_best_cost, fixed_cost_label(&final_cost),
        )));
    }
}
