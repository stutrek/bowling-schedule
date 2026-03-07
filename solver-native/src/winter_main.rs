use rand::rngs::SmallRng;
use rand::{Rng, SeedableRng};
use crate::cpu_sa_winter::{self as cpu_sa, CpuWorkers, WorkerCommand, NUM_MOVES, MOVE_NAMES};
use crate::gpu_setup::create_gpu_resources;
use crate::gpu_types::*;
use crate::gpu_types_winter::*;
use crate::output::*;
use crate::output_winter::*;
use solver_core::winter::*;
use std::fs;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::time::Instant;

const SYNC_INTERVAL: u64 = 10;

const STAGNATION_DISPATCHES: u64 = 900;
const ESCALATION_DISPATCHES: u64 = 4500;

const VERIFY_INTERVAL_SECS: u64 = 30;

struct ProvenanceTally {
    from_shakeup: u32,
    from_normal: u32,
    from_gpu: u32,
}

struct PartitionState {
    dispatches_since_improvement: u64,
}

fn is_reseeded_chain(global_chain: usize, p_start: usize, chains_per_cpu: usize) -> bool {
    let wgs_per_partition = chains_per_cpu / TEMP_LEVELS;
    let offset_in_partition = (global_chain - p_start) / TEMP_LEVELS;
    offset_in_partition < wgs_per_partition / 2
}

fn reseed_partition_chains(
    queue: &wgpu::Queue,
    assign_buf: &wgpu::Buffer,
    best_assign_buf: &wgpu::Buffer,
    cost_buf: &wgpu::Buffer,
    best_cost_buf: &wgpu::Buffer,
    chain_source: &mut [String],
    w8: &Weights,
    source: &Assignment,
    source_label: &str,
    p_start: usize,
    p_end: usize,
) {
    let packed = pack_assignment(source);
    let cost = evaluate(source, w8).total;
    for idx in p_start..p_end {
        let offset_assign = (idx * ASSIGN_U32S * 4) as u64;
        let offset_cost = (idx * 4) as u64;
        queue.write_buffer(assign_buf, offset_assign, bytemuck::cast_slice(&packed));
        queue.write_buffer(best_assign_buf, offset_assign, bytemuck::cast_slice(&packed));
        queue.write_buffer(cost_buf, offset_cost, bytemuck::bytes_of(&cost));
        queue.write_buffer(best_cost_buf, offset_cost, bytemuck::bytes_of(&cost));
        chain_source[idx] = source_label.to_string();
    }
}

pub fn run(shutdown: Arc<AtomicBool>, args: &[String]) {
    let no_seed = args.iter().any(|a| a == "--no-seed");
    let no_cpu = args.iter().any(|a| a == "--no-cpu");

    let weights_str = fs::read_to_string("../weights.json").expect("Failed to read weights.json");
    let w8: Weights = serde_json::from_str(&weights_str).expect("Invalid weights.json");

    let results_dir = "results/gpu";
    fs::create_dir_all(results_dir).expect("Failed to create results directory");

    let mut seeds: Vec<Assignment> = Vec::new();
    if !no_seed {
        for seed_dir in &["results/split-sa/full", results_dir] {
            if let Ok(entries) = fs::read_dir(seed_dir) {
                for entry in entries.flatten() {
                    let path = entry.path();
                    if path.extension().map_or(false, |e| e == "tsv") {
                        if let Ok(contents) = fs::read_to_string(&path) {
                            if let Some(a) = parse_tsv(&contents) {
                                seeds.push(a);
                            }
                        }
                    }
                }
            }
        }
    }

    let mut best_seed: Option<(Assignment, u32)> = None;
    for a in &seeds {
        let c = evaluate(a, &w8).total;
        if best_seed.as_ref().map_or(true, |(_, bc)| c < *bc) {
            best_seed = Some((*a, c));
        }
    }

    let available_cores = std::thread::available_parallelism().map(|n| n.get()).unwrap_or(4);
    let cpu_cores = if no_cpu { 0 } else { available_cores.saturating_sub(2) };

    let cpu_temps: Vec<f64> = if cpu_cores <= 1 {
        vec![CPU_TEMP_MIN]
    } else {
        (0..cpu_cores)
            .map(|i| CPU_TEMP_MIN + (CPU_TEMP_MAX - CPU_TEMP_MIN) * i as f64 / (cpu_cores - 1) as f64)
            .collect()
    };
    let cpu_temps_display = cpu_temps.clone();

    let cpu_workers = cpu_sa::run_cpu_workers(
        cpu_cores,
        w8.clone(),
        cpu_temps,
        Arc::clone(&shutdown),
    );

    if let Some((ref seed_a, seed_cost)) = best_seed {
        for (i, cmd_tx) in cpu_workers.commands.iter().enumerate() {
            let mut a = *seed_a;
            perturb(&mut a, &mut SmallRng::from_os_rng(), 5 + i * 2);
            let _ = cmd_tx.send(WorkerCommand::SetState(a));
        }
        eprintln!("Best seed: cost {}", seed_cost);
    }

    let chain_count = detect_chain_count(ASSIGN_U32S);
    let num_wgs = chain_count as usize / TEMP_LEVELS;
    let wgs_per_part = if cpu_cores > 0 { num_wgs / cpu_cores } else { num_wgs };
    let pods_per_wg = TEMP_LEVELS / POD_SIZE;
    let total_pods = num_wgs * pods_per_wg;
    eprintln!(
        "GPU solver: {} chains, {} workgroups ({}/partition, {}/2 reseeded), {} iters/dispatch",
        chain_count, num_wgs, wgs_per_part, wgs_per_part, ITERS_PER_DISPATCH,
    );
    eprintln!(
        "  pods: {} per wg × {} wgs = {} total, {} chains/pod, quadratic spacing",
        pods_per_wg, num_wgs, total_pods, POD_SIZE,
    );
    {
        let temps: Vec<String> = (0..POD_SIZE).map(|l| format!("{:.1}", temp_for_level(l))).collect();
        eprintln!("  pod temps: [{}]", temps.join(", "));
    }
    eprintln!(
        "  CPU temps: {:.1}-{:.1}, {} seed files, {} CPU cores",
        cpu_temps_display.first().unwrap_or(&0.0),
        cpu_temps_display.last().unwrap_or(&0.0), seeds.len(), cpu_cores,
    );

    let mut rng = SmallRng::from_os_rng();
    let mut assign_data = vec![0u32; chain_count as usize * ASSIGN_U32S];
    let mut rng_data = vec![0u32; chain_count as usize * 4];
    let mut cost_data = vec![0u32; chain_count as usize];
    let mut best_assign_data = vec![0u32; chain_count as usize * ASSIGN_U32S];
    let mut best_cost_data = vec![u32::MAX; chain_count as usize];

    for i in 0..chain_count as usize {
        let a = if i < seeds.len() { seeds[i] } else { random_assignment(&mut rng) };
        let packed = pack_assignment(&a);
        let cost = evaluate(&a, &w8).total;

        assign_data[i * ASSIGN_U32S..(i + 1) * ASSIGN_U32S].copy_from_slice(&packed);
        best_assign_data[i * ASSIGN_U32S..(i + 1) * ASSIGN_U32S].copy_from_slice(&packed);
        cost_data[i] = cost;
        best_cost_data[i] = cost;

        let seed_val = (i as u64).wrapping_mul(0x9E3779B97F4A7C15) ^ 0xBF58476D1CE4E5B9;
        rng_data[i * 4] = (seed_val & 0xFFFFFFFF) as u32 | 1;
        rng_data[i * 4 + 1] = ((seed_val >> 32) & 0xFFFFFFFF) as u32 | 1;
        rng_data[i * 4 + 2] = rng.random::<u32>() | 1;
        rng_data[i * 4 + 3] = rng.random::<u32>() | 1;
    }

    let gpu_weights = GpuWeights {
        matchup_zero: w8.matchup_zero,
        matchup_triple: w8.matchup_triple,
        consecutive_opponents: w8.consecutive_opponents,
        early_late_balance: w8.early_late_balance as f32,
        early_late_alternation: w8.early_late_alternation,
        lane_balance: w8.lane_balance as f32,
        lane_switch: w8.lane_switch as f32,
        late_lane_balance: w8.late_lane_balance as f32,
        commissioner_overlap: w8.commissioner_overlap,
        half_season_repeat: w8.half_season_repeat,
        _pad0: 0, _pad1: 0,
    };

    let gpu_params = GpuParams {
        iters_per_dispatch: ITERS_PER_DISPATCH,
        chain_count,
        temp_base: GPU_TEMP_MIN as f32,
        temp_step: GPU_TEMP_MAX as f32,
        pod_size: POD_SIZE as u32,
        _pad0: 0, _pad1: 0, _pad2: 0,
    };

    pollster::block_on(run_gpu(
        assign_data, best_assign_data, cost_data, best_cost_data, rng_data,
        gpu_weights, gpu_params, w8, results_dir.to_string(), shutdown, rng,
        cpu_workers, cpu_cores, cpu_temps_display,
    ));
}

#[allow(clippy::too_many_arguments)]
async fn run_gpu(
    assign_data: Vec<u32>,
    best_assign_data: Vec<u32>,
    cost_data: Vec<u32>,
    best_cost_data: Vec<u32>,
    rng_data: Vec<u32>,
    gpu_weights: GpuWeights,
    gpu_params: GpuParams,
    w8: Weights,
    results_dir: String,
    shutdown: Arc<AtomicBool>,
    mut rng: SmallRng,
    cpu_workers: CpuWorkers,
    cpu_cores: usize,
    cpu_temps_display: Vec<f64>,
) {
    let chain_count = gpu_params.chain_count;
    let mut chain_source: Vec<String> = vec!["random".to_string(); chain_count as usize];

    let gpu = create_gpu_resources(
        &assign_data, &best_assign_data, &cost_data, &best_cost_data, &rng_data,
        bytemuck::bytes_of(&gpu_weights), &gpu_params,
        bytemuck::bytes_of(&THRESH_DEFAULT),
        ASSIGN_U32S,
        include_str!("winter_solver.wgsl"),
        include_str!("winter_exchange.wgsl"),
    ).await;

    let mut global_best_cost = u32::MAX;
    let mut global_best_assignment: Option<Assignment> = None;
    let mut dispatch_count = 0u64;
    let start_time = Instant::now();
    let mut last_verify = Instant::now();
    let mut last_thresh = THRESH_DEFAULT;
    let mut partitions: Vec<PartitionState> = (0..cpu_cores)
        .map(|_| PartitionState { dispatches_since_improvement: 0 })
        .collect();

    let mut worker_metas: Vec<WorkerMeta> = (0..cpu_cores).map(|_| WorkerMeta {
        reseeded_at: Instant::now(),
        cost_at_reseed: u32::MAX,
        last_report: None,
        prev_iterations: 0,
        prev_iter_time: Instant::now(),
        iters_per_sec: 0,
    }).collect();

    let mut global_best_meta = GlobalBestMeta {
        source: "none".to_string(),
        found_at: Instant::now(),
    };
    let mut tally = ProvenanceTally { from_shakeup: 0, from_normal: 0, from_gpu: 0 };
    let mut partition_cpu_bests: Vec<u32> = vec![0; cpu_cores];
    let mut partition_gpu_bests: Vec<u32> = vec![0; cpu_cores];
    let mut partition_gpu_seeded: Vec<u32> = vec![0; cpu_cores];
    let mut counts_reset_done = false;
    let mut partition_best_cost: Vec<u32> = vec![u32::MAX; cpu_cores];
    let num_workgroups = chain_count as usize / TEMP_LEVELS;
    let wgs_per_partition = if cpu_cores > 0 { num_workgroups / cpu_cores } else { num_workgroups };
    let mut wg_best_costs: Vec<u32> = vec![u32::MAX; num_workgroups];
    let mut wg_best_levels: Vec<usize> = vec![0; num_workgroups];
    let mut partition_medians: Vec<u32> = vec![0; cpu_cores];
    let mut exchange_swaps_total: u64 = 0;
    let mut exchange_attempts_total: u64 = 0;
    let mut exchange_reset_time = Instant::now();

    let mut pending_events: Vec<String> = Vec::new();
    let mut last_print = Instant::now();
    let mut last_fresh_table = Instant::now();
    let mut prev_line_count: u32 = 0;

    loop {
        if shutdown.load(Ordering::Relaxed) { break; }

        macro_rules! maybe_print_table {
            () => {
                if last_print.elapsed().as_millis() >= 1000 || !pending_events.is_empty() {
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
                    if let Some(ref best) = global_best_assignment {
                        let best_bd = evaluate(best, &w8);
                        print_table_banner(global_best_cost, &best_bd, &global_best_meta, start_time);
                        lines += 2;
                    }
                    print_table_header();
                    lines += 1;
                    for (i, meta) in worker_metas.iter().enumerate() {
                        if let Some(ref report) = meta.last_report {
                            let temp = if i < cpu_temps_display.len() { cpu_temps_display[i] } else { 0.0 };
                            print_cpu_row(i, report, &w8, meta, temp, partitions[i].dispatches_since_improvement);
                            lines += 1;
                        }
                    }
                    {
                        let mut avg_rates = [0.0f64; NUM_MOVES];
                        let mut avg_shares = [0.0f64; NUM_MOVES];
                        let mut n = 0usize;
                        for meta in worker_metas.iter() {
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
                            lines += 2;
                        }
                    }
                    {
                        let part_hdr: Vec<String> = (0..cpu_cores).map(|i| format!("{:>11}", i)).collect();
                        let part_vals: Vec<String> = (0..cpu_cores).map(|i| {
                            let v = format!("{}/{}({})", partition_cpu_bests[i], partition_gpu_bests[i], partition_gpu_seeded[i]);
                            format!("{:>11}", v)
                        }).collect();
                        let gpu_meds: Vec<String> = (0..cpu_cores).map(|pi| {
                            format!("{:>11}", partition_medians[pi])
                        }).collect();
                        let wg_bests: Vec<String> = (0..cpu_cores).map(|pi| {
                            let wg_start = pi * wgs_per_partition;
                            let best = (wg_start..wg_start + wgs_per_partition)
                                .map(|w| wg_best_costs[w])
                                .min().unwrap_or(u32::MAX);
                            format!("{:>11}", best)
                        }).collect();
                        eprintln!("  partition:{}\x1b[K", part_hdr.join(""));
                        eprintln!("  cpu/gpu:  {}\x1b[K", part_vals.join(""));
                        eprintln!("  gpu med:  {}\x1b[K", gpu_meds.join(""));
                        eprintln!("  wg best:  {}\x1b[K", wg_bests.join(""));
                        lines += 4;
                        if exchange_reset_time.elapsed().as_secs() >= 300 {
                            exchange_swaps_total = 0;
                            exchange_attempts_total = 0;
                            exchange_reset_time = Instant::now();
                        }
                        if exchange_attempts_total > 0 {
                            let rate = exchange_swaps_total as f64 / exchange_attempts_total as f64 * 100.0;
                            let status = if rate > 80.0 { "wasteful — temps too close" }
                                else if rate > 60.0 { "high — could widen spacing" }
                                else if rate > 40.0 { "good" }
                                else if rate > 25.0 { "ok" }
                                else if rate > 10.0 { "low — flow is slow" }
                                else { "awful — ladder is broken" };
                            eprintln!("  exchange:  {}/{} ({:.1}%) {}\x1b[K",
                                exchange_swaps_total, exchange_attempts_total, rate, status);
                            lines += 1;
                        }
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

        // 1. GPU dispatch
        let dispatch_start = Instant::now();
        let mut encoder = gpu.device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("SA"),
        });
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("SA Pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&gpu.sa_pipeline);
            pass.set_bind_group(0, &gpu.sa_bg, &[]);
            pass.dispatch_workgroups(gpu.sa_workgroups, 1, 1);
        }
        let per_array_size = chain_count as u64 * 4;
        encoder.copy_buffer_to_buffer(&gpu.best_cost_buf, 0, &gpu.costs_readback_buf, 0, per_array_size);
        encoder.copy_buffer_to_buffer(&gpu.cost_buf, 0, &gpu.costs_readback_buf, per_array_size, per_array_size);
        gpu.queue.submit(Some(encoder.finish()));

        let costs_slice = gpu.costs_readback_buf.slice(..);
        let (tx, rx) = std::sync::mpsc::channel();
        costs_slice.map_async(wgpu::MapMode::Read, move |r| { let _ = tx.send(r); });

        let poll_deadline = Instant::now() + std::time::Duration::from_secs(30);
        let mut poll_ok = false;
        loop {
            match gpu.device.poll(wgpu::PollType::Poll) {
                Ok(status) if status.is_queue_empty() => { poll_ok = true; break; }
                Ok(_) => {
                    if shutdown.load(Ordering::Relaxed) { break; }
                    if Instant::now() > poll_deadline {
                        eprintln!("GPU poll timed out after 30s at dispatch {} — device may be lost", dispatch_count);
                        break;
                    }
                    std::thread::sleep(std::time::Duration::from_millis(1));
                }
                Err(e) => {
                    eprintln!("GPU poll error at dispatch {}: {:?}", dispatch_count, e);
                    break;
                }
            }
        }
        if !poll_ok {
            gpu.costs_readback_buf.unmap();
            continue;
        }
        if rx.recv_timeout(std::time::Duration::from_secs(5)).is_err() {
            eprintln!("GPU map_async recv timed out at dispatch {}", dispatch_count);
            gpu.costs_readback_buf.unmap();
            continue;
        }
        let dispatch_ms = dispatch_start.elapsed().as_millis();
        if dispatch_ms > 5000 {
            eprintln!("WARNING: dispatch {} took {}ms (>5s) — GPU may be starving the system", dispatch_count, dispatch_ms);
        } else if dispatch_count < 5 {
            eprintln!("dispatch {} completed in {}ms", dispatch_count, dispatch_ms);
        }

        // 2. Read costs
        let chains_per_cpu = if cpu_cores > 0 { chain_count as usize / cpu_cores } else { chain_count as usize };
        let mut partition_bests: Vec<(u32, usize)> = vec![(u32::MAX, 0); cpu_cores];
        let current_costs_snapshot: Vec<u32>;
        let _min_cost = {
            let data = costs_slice.get_mapped_range();
            let all: &[u32] = bytemuck::cast_slice(&data);
            let n = chain_count as usize;
            let best_costs = &all[..n];
            let current_costs = &all[n..n * 2];
            current_costs_snapshot = current_costs.to_vec();

            let mut best = u32::MAX;
            for wg in 0..num_workgroups {
                wg_best_costs[wg] = u32::MAX;
            }
            for (i, &c) in best_costs.iter().enumerate() {
                if c < best { best = c; }
                let partition = (i / chains_per_cpu).min(cpu_cores - 1);
                if c < partition_bests[partition].0 {
                    partition_bests[partition] = (c, i);
                }
                let wg = i / TEMP_LEVELS;
                if c < wg_best_costs[wg] {
                    wg_best_costs[wg] = c;
                    wg_best_levels[wg] = i % TEMP_LEVELS;
                }
            }
            for pi in 0..cpu_cores {
                let start = pi * chains_per_cpu;
                let end = ((pi + 1) * chains_per_cpu).min(n);
                partition_medians[pi] = sampled_median(&current_costs[start..end], &mut rng);
            }
            best
        };
        gpu.costs_readback_buf.unmap();

        dispatch_count += 1;
        for p in partitions.iter_mut() { p.dispatches_since_improvement += 1; }

        // 2b. Replica exchange
        {
            let parity = (dispatch_count % 2) as usize;
            let pods_per_wg = TEMP_LEVELS / POD_SIZE;
            let mut swap_pairs: Vec<u32> = Vec::new();
            let mut attempts = 0u64;
            for wg in 0..num_workgroups {
                let wg_base = wg * TEMP_LEVELS;
                for pod in 0..pods_per_wg {
                    let pod_base = wg_base + pod * POD_SIZE;
                    let mut level = parity;
                    while level + 1 < POD_SIZE {
                        let chain_a = pod_base + level;
                        let chain_b = pod_base + level + 1;
                        let cost_a = current_costs_snapshot[chain_a] as f64;
                        let cost_b = current_costs_snapshot[chain_b] as f64;
                        let temp_a = temp_for_level(level);
                        let temp_b = temp_for_level(level + 1);
                        let delta = (1.0 / temp_a - 1.0 / temp_b) * (cost_a - cost_b);
                        attempts += 1;
                        if delta >= 0.0 || rng.random::<f64>() < delta.exp() {
                            swap_pairs.push(chain_a as u32);
                            swap_pairs.push(chain_b as u32);
                        }
                        level += 2;
                    }
                }
            }
            let num_swaps = swap_pairs.len() / 2;
            exchange_attempts_total += attempts;
            exchange_swaps_total += num_swaps as u64;

            if num_swaps > 0 {
                let params_data: [u32; 4] = [num_swaps as u32, 0, 0, 0];
                gpu.queue.write_buffer(&gpu.exchange_params_buf, 0, bytemuck::cast_slice(&params_data));
                gpu.queue.write_buffer(&gpu.swap_pairs_buf, 0, bytemuck::cast_slice(&swap_pairs));

                let mut encoder = gpu.device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
                    label: Some("Exchange"),
                });
                {
                    let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                        label: Some("Exchange Pass"),
                        timestamp_writes: None,
                    });
                    pass.set_pipeline(&gpu.exchange_pipeline);
                    pass.set_bind_group(0, &gpu.exchange_bg, &[]);
                    pass.dispatch_workgroups(((num_swaps as u32) + 63) / 64, 1, 1);
                }
                gpu.queue.submit(Some(encoder.finish()));
            }
        }

        if !counts_reset_done && start_time.elapsed().as_secs() >= 180 {
            counts_reset_done = true;
            for i in 0..cpu_cores {
                partition_cpu_bests[i] = 0;
                partition_gpu_bests[i] = 0;
                partition_gpu_seeded[i] = 0;
            }
            event!(start_time.elapsed(), "RESET partition counters (5-minute warmup)");
        }

        maybe_print_table!();

        // 3. Per-partition GPU feedback + global best tracking
        for pi in 0..cpu_cores {
            let (gpu_part_cost, gpu_part_chain) = partition_bests[pi];
            let cpu_live_best = cpu_workers.live_best_costs[pi].load(Ordering::Relaxed);
            if gpu_part_cost >= cpu_live_best { continue; }

            let mut encoder = gpu.device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("Read Partition Best"),
            });
            let offset = gpu_part_chain as u64 * ASSIGN_U32S as u64 * 4;
            encoder.copy_buffer_to_buffer(&gpu.best_assign_buf, offset, &gpu.assign_readback_buf, 0, (ASSIGN_U32S * 4) as u64);
            gpu.queue.submit(Some(encoder.finish()));

            let assign_slice = gpu.assign_readback_buf.slice(..);
            let (tx, rx) = std::sync::mpsc::channel();
            assign_slice.map_async(wgpu::MapMode::Read, move |r| { let _ = tx.send(r); });
            let poll_deadline = Instant::now() + std::time::Duration::from_secs(30);
            let mut assign_poll_ok = false;
            loop {
                match gpu.device.poll(wgpu::PollType::Poll) {
                    Ok(status) if status.is_queue_empty() => { assign_poll_ok = true; break; }
                    Ok(_) => {
                        if Instant::now() > poll_deadline {
                            eprintln!("GPU assign poll timed out for partition {}", pi);
                            break;
                        }
                        std::thread::sleep(std::time::Duration::from_millis(1));
                    }
                    Err(_) => break,
                }
            }
            if !assign_poll_ok || rx.recv_timeout(std::time::Duration::from_secs(5)).is_err() {
                gpu.assign_readback_buf.unmap();
                continue;
            }

            let packed: [u32; ASSIGN_U32S] = {
                let data = assign_slice.get_mapped_range();
                let slice: &[u32] = bytemuck::cast_slice(&data);
                let mut arr = [0u32; ASSIGN_U32S];
                arr.copy_from_slice(slice);
                arr
            };
            gpu.assign_readback_buf.unmap();

            let assignment = unpack_assignment(&packed);
            let verify_cost = evaluate(&assignment, &w8).total;
            if verify_cost != gpu_part_cost {
                event!(start_time.elapsed(), &format!(
                    "GPU EVAL MISMATCH p{}: gpu={} cpu={}", pi, gpu_part_cost, verify_cost));
            }

            let _ = cpu_workers.commands[pi].send(WorkerCommand::SetState(assignment));
            worker_metas[pi].reseeded_at = Instant::now();
            worker_metas[pi].cost_at_reseed = gpu_part_cost;
            event!(start_time.elapsed(), &format!(
                "GPU SEEDS cpu{}: cost {} (cpu was {})", pi, gpu_part_cost, cpu_live_best));

            let wg_idx = gpu_part_chain / TEMP_LEVELS;
            let local_in_wg = gpu_part_chain % TEMP_LEVELS;
            let pod_in_wg = local_in_wg / POD_SIZE;
            let level_in_pod = local_in_wg % POD_SIZE;
            let p_start = pi * chains_per_cpu;
            let wg_type = if is_reseeded_chain(gpu_part_chain, p_start, chains_per_cpu) { "reseed" } else { "free" };
            let temp_val = temp_for_level(level_in_pod);

            if gpu_part_cost < partition_best_cost[pi] {
                let prev = partition_best_cost[pi];
                partition_best_cost[pi] = gpu_part_cost;
                partitions[pi].dispatches_since_improvement = 0;
                partition_gpu_bests[pi] += 1;
                event!(start_time.elapsed(), &format!(
                    "PARTITION {} NEW BEST {} (was {}) from gpu wg={} pod={} lvl={}/T={:.1} type={} (seed: {})",
                    pi, gpu_part_cost, prev, wg_idx, pod_in_wg, level_in_pod, temp_val, wg_type,
                    chain_source[gpu_part_chain],
                ));
                if gpu_part_cost < 420 {
                    let ts = chrono::Local::now().format("%Y%m%d-%H%M%S%z");
                    let filename = format!("{}/{:04}-gpu-p{}-{}.tsv", results_dir, gpu_part_cost, pi, ts);
                    let mut out = assignment;
                    reassign_commissioners(&mut out);
                    let _ = fs::write(&filename, assignment_to_tsv(&out));
                    event!(start_time.elapsed(), &format!("Saved {}", filename));
                }
            }

            if gpu_part_cost < global_best_cost {
                global_best_cost = gpu_part_cost;
                global_best_assignment = Some(assignment);
                partitions[pi].dispatches_since_improvement = 0;
                tally.from_gpu += 1;
                global_best_meta = GlobalBestMeta {
                    source: format!("gpu-{}-wg{}-pod{}-T{:.1}", wg_type, wg_idx, pod_in_wg, temp_val),
                    found_at: Instant::now(),
                };

                let p_end = ((pi + 1) * chains_per_cpu).min(chain_count as usize);
                reseed_partition_chains(
                    &gpu.queue, &gpu.assign_buf, &gpu.best_assign_buf,
                    &gpu.cost_buf, &gpu.best_cost_buf,
                    &mut chain_source, &w8,
                    &assignment, &format!("gpu-p{}", pi), p_start, p_end,
                );
            }
        }

        // 4. Drain CPU reports
        while let Ok(report) = cpu_workers.reports.try_recv() {
            let cid = report.core_id;
            if cid < worker_metas.len() {
                let real_best = evaluate(&report.best_assignment, &w8).total;
                if real_best < partition_best_cost[cid] {
                    let prev = partition_best_cost[cid];
                    partition_best_cost[cid] = real_best;
                    partitions[cid].dispatches_since_improvement = 0;
                    partition_cpu_bests[cid] += 1;
                    let since_reseed = worker_metas[cid].reseeded_at.elapsed().as_secs();
                    if since_reseed < 30 {
                        partition_gpu_seeded[cid] += 1;
                    }
                    event!(start_time.elapsed(), &format!(
                        "PARTITION {} NEW BEST {} (was {}) from cpu{}{}",
                        cid, real_best, prev, cid,
                        if since_reseed < 30 { " (gpu-seeded)" } else { "" },
                    ));
                    if real_best < 420 {
                        let ts = chrono::Local::now().format("%Y%m%d-%H%M%S%z");
                        let filename = format!("{}/{:04}-cpu{}-{}.tsv", results_dir, real_best, cid, ts);
                        let mut out = report.best_assignment;
                        reassign_commissioners(&mut out);
                        let _ = fs::write(&filename, assignment_to_tsv(&out));
                        event!(start_time.elapsed(), &format!("Saved {}", filename));
                    }
                }
                if real_best < global_best_cost {
                    global_best_cost = real_best;
                    global_best_assignment = Some(report.best_assignment);
                    partitions[cid].dispatches_since_improvement = 0;
                    global_best_meta = GlobalBestMeta {
                        source: format!("cpu{}", cid),
                        found_at: Instant::now(),
                    };

                    let since_reseed = worker_metas[cid].reseeded_at.elapsed().as_secs();
                    let reseed_cost = worker_metas[cid].cost_at_reseed;
                    if since_reseed < 30 {
                        tally.from_shakeup += 1;
                        event!(start_time.elapsed(), &format!(
                            "NEW BEST {} from cpu{} (shook {}s ago, {}->{})",
                            global_best_cost, cid, since_reseed, reseed_cost, global_best_cost,
                        ));
                    } else {
                        tally.from_normal += 1;
                        event!(start_time.elapsed(), &format!(
                            "NEW BEST {} from cpu{} (normal, running {}s)",
                            global_best_cost, cid, since_reseed,
                        ));
                    }

                    let p_start = cid * chains_per_cpu;
                    let p_end = ((cid + 1) * chains_per_cpu).min(chain_count as usize);
                    let source_label = format!("cpu{}", cid);
                    reseed_partition_chains(
                        &gpu.queue, &gpu.assign_buf, &gpu.best_assign_buf,
                        &gpu.cost_buf, &gpu.best_cost_buf,
                        &mut chain_source, &w8,
                        &report.best_assignment, &source_label, p_start, p_end,
                    );
                }
                let dt = worker_metas[cid].prev_iter_time.elapsed().as_secs_f64();
                if dt >= 1.0 {
                    let di = report.iterations_total.saturating_sub(worker_metas[cid].prev_iterations);
                    worker_metas[cid].iters_per_sec = (di as f64 / dt) as u64;
                    worker_metas[cid].prev_iterations = report.iterations_total;
                    worker_metas[cid].prev_iter_time = Instant::now();
                }
                worker_metas[cid].last_report = Some(report);
                maybe_print_table!();
            }
        }

        // 5. Adaptive move thresholds
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
                let base_weights: [f64; NUM_MOVES] = [
                    0.10, 0.25, 0.10, 0.06, 0.06, 0.05, 0.04, 0.06, 0.05, 0.15, 0.08,
                ];
                let mut weights = [0.0f64; NUM_MOVES];
                for m in 0..NUM_MOVES {
                    let rate = avg_rates[m] / nf;
                    weights[m] = base_weights[m] * (0.1 + rate);
                }
                let sum: f64 = weights.iter().sum();
                let mut thresh = GpuMoveThresholds { t: [0u32; 12] };
                let mut cum = 0.0;
                for m in 0..NUM_MOVES {
                    cum += weights[m] / sum;
                    thresh.t[m] = (cum * 100.0).round() as u32;
                }
                thresh.t[NUM_MOVES - 1] = 100;
                if thresh.t != last_thresh.t {
                    gpu.queue.write_buffer(&gpu.move_thresh_buf, 0, bytemuck::bytes_of(&thresh));
                    last_thresh = thresh;
                }
            }
        }

        // 6. Sync interval: per-partition reseeding + stagnation
        maybe_print_table!();
        if dispatch_count % SYNC_INTERVAL == 0 {
            for pi in 0..cpu_cores {
                let seed_assignment = worker_metas[pi].last_report.as_ref()
                    .map(|r| r.best_assignment);
                let seed_assignment = match seed_assignment {
                    Some(a) => a,
                    None => continue,
                };
                let source_label = format!("cpu{}", pi);
                let p_start = pi * chains_per_cpu;
                let p_end = ((pi + 1) * chains_per_cpu).min(chain_count as usize);
                let p_len = p_end - p_start;
                let stag = partitions[pi].dispatches_since_improvement;

                let base_reseed = p_len / 50;
                let reseed_count = if stag < 5 {
                    base_reseed / 2
                } else if stag > STAGNATION_DISPATCHES {
                    base_reseed * 2
                } else {
                    base_reseed
                };

                for _ in 0..reseed_count {
                    let idx = p_start + rng.random_range(0..p_len);
                    let level_in_pod = (idx % TEMP_LEVELS) % POD_SIZE;
                    let t_frac = level_in_pod as f64 / (POD_SIZE - 1).max(1) as f64;
                    let pert = (t_frac * t_frac * 5.0) as usize;
                    let mut a = seed_assignment;
                    perturb(&mut a, &mut rng, pert);
                    let packed = pack_assignment(&a);
                    let cost = evaluate(&a, &w8).total;
                    let offset_assign = (idx * ASSIGN_U32S * 4) as u64;
                    let offset_cost = (idx * 4) as u64;
                    gpu.queue.write_buffer(&gpu.assign_buf, offset_assign, bytemuck::cast_slice(&packed));
                    gpu.queue.write_buffer(&gpu.best_assign_buf, offset_assign, bytemuck::cast_slice(&packed));
                    gpu.queue.write_buffer(&gpu.cost_buf, offset_cost, bytemuck::bytes_of(&cost));
                    gpu.queue.write_buffer(&gpu.best_cost_buf, offset_cost, bytemuck::bytes_of(&cost));
                    chain_source[idx] = source_label.clone();
                }

                if stag >= STAGNATION_DISPATCHES {
                    if stag % STAGNATION_DISPATCHES < SYNC_INTERVAL {
                        event!(start_time.elapsed(), &format!(
                            "SHAKEUP: partition {} (stagnant {} dispatches)", pi, stag));
                    }

                    let inject_count = p_len / 20;
                    for _ in 0..inject_count {
                        let idx = p_start + rng.random_range(0..p_len);
                        let level_in_pod = (idx % TEMP_LEVELS) % POD_SIZE;
                        let t_frac = level_in_pod as f64 / (POD_SIZE - 1).max(1) as f64;
                        let pert = 3 + (t_frac * t_frac * 7.0) as usize;
                        let mut a = seed_assignment;
                        perturb(&mut a, &mut rng, pert);
                        let packed = pack_assignment(&a);
                        let cost = evaluate(&a, &w8).total;
                        let offset_assign = (idx * ASSIGN_U32S * 4) as u64;
                        let offset_cost = (idx * 4) as u64;
                        gpu.queue.write_buffer(&gpu.assign_buf, offset_assign, bytemuck::cast_slice(&packed));
                        gpu.queue.write_buffer(&gpu.best_assign_buf, offset_assign, bytemuck::cast_slice(&packed));
                        gpu.queue.write_buffer(&gpu.cost_buf, offset_cost, bytemuck::bytes_of(&cost));
                        gpu.queue.write_buffer(&gpu.best_cost_buf, offset_cost, bytemuck::bytes_of(&cost));
                        chain_source[idx] = source_label.clone();
                    }
                }

                if stag >= ESCALATION_DISPATCHES && stag % ESCALATION_DISPATCHES < SYNC_INTERVAL {
                    event!(start_time.elapsed(), &format!(
                        "ESCALATED SHAKEUP: partition {} (stagnant {} dispatches)", pi, stag));

                    reseed_partition_chains(
                        &gpu.queue, &gpu.assign_buf, &gpu.best_assign_buf,
                        &gpu.cost_buf, &gpu.best_cost_buf,
                        &mut chain_source, &w8,
                        &seed_assignment, &source_label, p_start, p_end,
                    );

                    let _ = cpu_workers.commands[pi].send(WorkerCommand::SetState(seed_assignment));
                    worker_metas[pi].reseeded_at = Instant::now();
                    worker_metas[pi].cost_at_reseed = evaluate(&seed_assignment, &w8).total;
                }
                maybe_print_table!();
            }
        }

        // 7. Periodic GPU verification
        if last_verify.elapsed().as_secs() >= VERIFY_INTERVAL_SECS {
            if let Some(ref best) = global_best_assignment {
                let verify_cost = evaluate(best, &w8);
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

        // 9. Periodic tally
        if dispatch_count > 0 && dispatch_count % 300 == 0 {
            let total = tally.from_shakeup + tally.from_normal + tally.from_gpu;
            if total > 0 {
                event!(start_time.elapsed(), &format!(
                    "TALLY: {} shakeup, {} normal, {} GPU ({} total)",
                    tally.from_shakeup, tally.from_normal, tally.from_gpu, total));
            }
        }
    }

    // Shutdown
    for cmd_tx in &cpu_workers.commands {
        let _ = cmd_tx.send(WorkerCommand::Shutdown);
    }
    for h in cpu_workers.handles {
        let _ = h.join();
    }

    if let Some(best) = global_best_assignment {
        let final_cost = evaluate(&best, &w8);
        eprintln!("{}", format_event(start_time.elapsed(), &format!(
            "Final best: {} | {}", global_best_cost, cost_label(&final_cost),
        )));
    }
}
