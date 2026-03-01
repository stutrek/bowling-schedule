use rand::rngs::SmallRng;
use rand::{Rng, SeedableRng};
use solver_native::cpu_sa::{self, CpuWorkers, WorkerCommand, NUM_MOVES, MOVE_NAMES};
use solver_native::gpu_setup::create_gpu_resources;
use solver_native::gpu_types::*;
use solver_native::output::*;
use solver_native::*;
use std::fs;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::time::Instant;

const SYNC_INTERVAL: u64 = 10;

const TEMP_LEVELS: usize = 256;

const CPU_TEMP_MIN: f64 = 10.0;
const CPU_TEMP_MAX: f64 = 15.0;

const STAGNATION_DISPATCHES: u64 = 60;
const ESCALATION_DISPATCHES: u64 = 300;

const VERIFY_INTERVAL_SECS: u64 = 30;

struct ProvenanceTally {
    from_shakeup: u32,
    from_normal: u32,
    from_gpu: u32,
}

struct PartitionState {
    dispatches_since_improvement: u64,
    aggressive_logged: bool,
}

/// Reseed every chain in [p_start..p_end) from `source`, with perturbation
/// scaled by each chain's temperature level (cold=1, hot=20).
fn reseed_partition_chains(
    queue: &wgpu::Queue,
    assign_buf: &wgpu::Buffer,
    best_assign_buf: &wgpu::Buffer,
    cost_buf: &wgpu::Buffer,
    best_cost_buf: &wgpu::Buffer,
    chain_source: &mut [String],
    rng: &mut SmallRng,
    w8: &solver_core::Weights,
    source: &solver_core::Assignment,
    source_label: &str,
    p_start: usize,
    p_end: usize,
) {
    for idx in p_start..p_end {
        let temp_frac = (idx % TEMP_LEVELS) as f64 / TEMP_LEVELS as f64;
        let pert = (temp_frac * 5.0) as usize;
        let mut a = *source;
        solver_core::perturb(&mut a, rng, pert);
        let packed = pack_assignment(&a);
        let cost = solver_core::evaluate(&a, w8).total;
        let offset_assign = (idx * ASSIGN_U32S * 4) as u64;
        let offset_cost = (idx * 4) as u64;
        queue.write_buffer(assign_buf, offset_assign, bytemuck::cast_slice(&packed));
        queue.write_buffer(best_assign_buf, offset_assign, bytemuck::cast_slice(&packed));
        queue.write_buffer(cost_buf, offset_cost, bytemuck::bytes_of(&cost));
        queue.write_buffer(best_cost_buf, offset_cost, bytemuck::bytes_of(&cost));
        chain_source[idx] = source_label.to_string();
    }
}

fn main() {
    let shutdown = Arc::new(AtomicBool::new(false));
    {
        let s = Arc::clone(&shutdown);
        ctrlc::set_handler(move || {
            eprintln!("\nShutting down...");
            s.store(true, Ordering::Relaxed);
        })
        .expect("Error setting Ctrl-C handler");
    }

    let args: Vec<String> = std::env::args().collect();
    let no_seed = args.iter().any(|a| a == "--no-seed");
    let no_cpu = args.iter().any(|a| a == "--no-cpu");

    let weights_str = fs::read_to_string("../weights.json").expect("Failed to read weights.json");
    let w8: solver_core::Weights = serde_json::from_str(&weights_str).expect("Invalid weights.json");

    let results_dir = "results/gpu";
    fs::create_dir_all(results_dir).expect("Failed to create results directory");

    let mut seeds: Vec<solver_core::Assignment> = Vec::new();
    if !no_seed {
        for seed_dir in &["results/split-sa/full", results_dir] {
            if let Ok(entries) = fs::read_dir(seed_dir) {
                for entry in entries.flatten() {
                    let path = entry.path();
                    if path.extension().map_or(false, |e| e == "tsv") {
                        if let Ok(contents) = fs::read_to_string(&path) {
                            if let Some(a) = solver_core::parse_tsv(&contents) {
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
        let c = solver_core::evaluate(a, &w8).total;
        if best_seed.as_ref().map_or(true, |(_, bc)| c < *bc) {
            best_seed = Some((*a, c));
        }
    }

    let available_cores = std::thread::available_parallelism().map(|n| n.get()).unwrap_or(4);
    let cpu_cores = if no_cpu { 0 } else { available_cores.saturating_sub(2) };

    let cpu_temps: Vec<f64> = if cpu_cores <= 1 {
        vec![1.0]
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
            solver_core::perturb(&mut a, &mut SmallRng::from_os_rng(), 5 + i * 2);
            let _ = cmd_tx.send(WorkerCommand::SetState(a));
        }
        eprintln!("Best seed: cost {}", seed_cost);
    }

    let chain_count = detect_chain_count();
    eprintln!(
        "GPU solver: {} chains, {} iters/dispatch, {} seed files, {} CPU cores",
        chain_count, ITERS_PER_DISPATCH, seeds.len(), cpu_cores,
    );

    let mut rng = SmallRng::from_os_rng();
    let mut assign_data = vec![0u32; chain_count as usize * ASSIGN_U32S];
    let mut rng_data = vec![0u32; chain_count as usize * 4];
    let mut cost_data = vec![0u32; chain_count as usize];
    let mut best_assign_data = vec![0u32; chain_count as usize * ASSIGN_U32S];
    let mut best_cost_data = vec![u32::MAX; chain_count as usize];

    for i in 0..chain_count as usize {
        let a = if i < seeds.len() { seeds[i] } else { solver_core::random_assignment(&mut rng) };
        let packed = pack_assignment(&a);
        let cost = solver_core::evaluate(&a, &w8).total;

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
        _pad0: 0, _pad1: 0, _pad2: 0,
    };

    let gpu_params = GpuParams {
        iters_per_dispatch: ITERS_PER_DISPATCH,
        chain_count,
        temp_base: CPU_TEMP_MIN as f32,
        temp_step: CPU_TEMP_MAX as f32,
    };

    pollster::block_on(run_gpu(
        assign_data, best_assign_data, cost_data, best_cost_data, rng_data,
        gpu_weights, gpu_params, w8, results_dir.to_string(), shutdown, rng,
        cpu_workers, cpu_cores, cpu_temps_display,
    ));
}

// ═══════════════════════════════════════════════════════════════════════════
// GPU orchestrator loop
// ═══════════════════════════════════════════════════════════════════════════

#[allow(clippy::too_many_arguments)]
async fn run_gpu(
    assign_data: Vec<u32>,
    best_assign_data: Vec<u32>,
    cost_data: Vec<u32>,
    best_cost_data: Vec<u32>,
    rng_data: Vec<u32>,
    gpu_weights: GpuWeights,
    gpu_params: GpuParams,
    w8: solver_core::Weights,
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
        &gpu_weights, &gpu_params,
    ).await;

    // ─── Orchestrator state ───────────────────────────────────────────────
    let mut global_best_cost = u32::MAX;
    let mut global_best_assignment: Option<solver_core::Assignment> = None;
    let mut dispatch_count = 0u64;
    let start_time = Instant::now();
    let mut last_verify = Instant::now();
    let mut last_thresh_regime: u8 = 0;
    let mut partitions: Vec<PartitionState> = (0..cpu_cores)
        .map(|_| PartitionState { dispatches_since_improvement: 0, aggressive_logged: false })
        .collect();
    #[allow(unused_assignments)]
    let mut gpu_median = 0u32;
    #[allow(unused_assignments)]
    let mut gpu_best_cost = 0u32;

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
    let mut partition_best_cost: Vec<u32> = vec![u32::MAX; cpu_cores];
    let mut pending_events: Vec<String> = Vec::new();
    let mut last_print = Instant::now();
    let mut last_fresh_table = Instant::now();
    let mut prev_line_count: u32 = 0;
    let mut gpu_prev_dispatch: u64 = 0;
    let mut gpu_prev_dispatch_time = Instant::now();
    let mut gpu_ips: u64 = 0;

    // ─── Main loop ────────────────────────────────────────────────────────
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
                        let best_bd = solver_core::evaluate(best, &w8);
                        print_table_banner(global_best_cost, &best_bd, &global_best_meta, start_time);
                        lines += 2;
                    }
                    print_table_header();
                    lines += 1;
                    let elapsed = start_time.elapsed();
                    if let Some(ref best) = global_best_assignment {
                        let best_bd = solver_core::evaluate(best, &w8);
                        print_gpu_row(elapsed, gpu_best_cost, gpu_median, &best_bd, gpu_ips);
                        lines += 1;
                    }
                    for (i, meta) in worker_metas.iter().enumerate() {
                        if let Some(ref report) = meta.last_report {
                            let temp = if i < cpu_temps_display.len() { cpu_temps_display[i] } else { 0.0 };
                            print_cpu_row(elapsed, i, report, &w8, meta, temp);
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
                        let part_hdr: Vec<String> = (0..cpu_cores).map(|i| format!("{:>5}", i)).collect();
                        let part_vals: Vec<String> = (0..cpu_cores).map(|i| {
                            format!("{:>2}/{:<2}", partition_cpu_bests[i], partition_gpu_bests[i])
                        }).collect();
                        eprintln!("  partition: {}\x1b[K", part_hdr.join(" "));
                        eprintln!("  cpu/gpu:   {}\x1b[K", part_vals.join(" "));
                        lines += 2;
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
        encoder.copy_buffer_to_buffer(&gpu.best_cost_buf, 0, &gpu.costs_readback_buf, 0, gpu.costs_readback_size);
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

        // 2. Read costs + sample median + per-CPU-partition bests
        let chains_per_cpu = if cpu_cores > 0 { chain_count as usize / cpu_cores } else { chain_count as usize };
        let mut partition_bests: Vec<(u32, usize)> = vec![(u32::MAX, 0); cpu_cores];
        let min_cost = {
            let data = costs_slice.get_mapped_range();
            let costs: &[u32] = bytemuck::cast_slice(&data);
            let mut best = u32::MAX;
            for (i, &c) in costs.iter().enumerate() {
                if c < best { best = c; }
                let partition = (i / chains_per_cpu).min(cpu_cores - 1);
                if c < partition_bests[partition].0 {
                    partition_bests[partition] = (c, i);
                }
            }
            gpu_median = sampled_median(costs, &mut rng);
            best
        };
        gpu.costs_readback_buf.unmap();
        gpu_best_cost = min_cost;

        dispatch_count += 1;
        for p in partitions.iter_mut() { p.dispatches_since_improvement += 1; }
        let gpu_dt = gpu_prev_dispatch_time.elapsed().as_secs_f64();
        if gpu_dt >= 1.0 {
            let gpu_di = (dispatch_count - gpu_prev_dispatch) * ITERS_PER_DISPATCH as u64 * chain_count as u64;
            gpu_ips = (gpu_di as f64 / gpu_dt) as u64;
            gpu_prev_dispatch = dispatch_count;
            gpu_prev_dispatch_time = Instant::now();
        }
        maybe_print_table!();

        // 3. Per-partition GPU feedback + global best tracking
        for pi in 0..cpu_cores {
            let (gpu_part_cost, gpu_part_chain) = partition_bests[pi];
            let cpu_best = worker_metas[pi].last_report.as_ref()
                .map(|r| r.best_cost).unwrap_or(u32::MAX);
            if gpu_part_cost >= cpu_best { continue; }

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

            // Feed to controlling CPU (unperturbed)
            let _ = cpu_workers.commands[pi].send(WorkerCommand::SetState(assignment));
            worker_metas[pi].reseeded_at = Instant::now();
            worker_metas[pi].cost_at_reseed = gpu_part_cost;

            if gpu_part_cost < partition_best_cost[pi] {
                let prev = partition_best_cost[pi];
                partition_best_cost[pi] = gpu_part_cost;
                partition_gpu_bests[pi] += 1;
                event!(start_time.elapsed(), &format!(
                    "PARTITION {} NEW BEST {} (was {}) from gpu chain {} (seed: {})",
                    pi, gpu_part_cost, prev, gpu_part_chain, chain_source[gpu_part_chain],
                ));
            }

            // If this is also a new global best, do global bookkeeping
            if gpu_part_cost < global_best_cost {
                global_best_cost = gpu_part_cost;
                global_best_assignment = Some(assignment);
                partitions[pi].dispatches_since_improvement = 0;
                partitions[pi].aggressive_logged = false;
                tally.from_gpu += 1;
                global_best_meta = GlobalBestMeta {
                    source: format!("gpu({})", chain_source[gpu_part_chain]),
                    found_at: Instant::now(),
                };

                if global_best_cost < 160 {
                    let ts = chrono::Local::now().format("%Y%m%d-%H%M%S%z");
                    let filename = format!("{}/{:04}-gpu-{}.tsv", results_dir, global_best_cost, ts);
                    let mut out = assignment;
                    reassign_commissioners(&mut out);
                    let _ = fs::write(&filename, assignment_to_tsv(&out));
                    event!(start_time.elapsed(), &format!("Saved {}", filename));
                }

                let p_start = pi * chains_per_cpu;
                let p_end = ((pi + 1) * chains_per_cpu).min(chain_count as usize);
                reseed_partition_chains(
                    &gpu.queue, &gpu.assign_buf, &gpu.best_assign_buf,
                    &gpu.cost_buf, &gpu.best_cost_buf,
                    &mut chain_source, &mut rng, &w8,
                    &assignment, &format!("gpu-p{}", pi), p_start, p_end,
                );
            }
        }

        // 4. Drain CPU reports
        while let Ok(report) = cpu_workers.reports.try_recv() {
            let cid = report.core_id;
            if cid < worker_metas.len() {
                let real_best = solver_core::evaluate(&report.best_assignment, &w8).total;
                if real_best < partition_best_cost[cid] {
                    let prev = partition_best_cost[cid];
                    partition_best_cost[cid] = real_best;
                    partition_cpu_bests[cid] += 1;
                    event!(start_time.elapsed(), &format!(
                        "PARTITION {} NEW BEST {} (was {}) from cpu{}",
                        cid, real_best, prev, cid,
                    ));
                }
                if real_best < global_best_cost {
                    global_best_cost = real_best;
                    global_best_assignment = Some(report.best_assignment);
                    partitions[cid].dispatches_since_improvement = 0;
                    partitions[cid].aggressive_logged = false;
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

                    if global_best_cost < 160 {
                        let ts = chrono::Local::now().format("%Y%m%d-%H%M%S%z");
                        let filename = format!("{}/{:04}-cpu{}-{}.tsv", results_dir, global_best_cost, cid, ts);
                        let mut out = report.best_assignment;
                        reassign_commissioners(&mut out);
                        let _ = fs::write(&filename, assignment_to_tsv(&out));
                        event!(start_time.elapsed(), &format!("Saved {}", filename));
                    }

                    // Reseed entire partition from new CPU best
                    let p_start = cid * chains_per_cpu;
                    let p_end = ((cid + 1) * chains_per_cpu).min(chain_count as usize);
                    let source_label = format!("cpu{}", cid);
                    reseed_partition_chains(
                        &gpu.queue, &gpu.assign_buf, &gpu.best_assign_buf,
                        &gpu.cost_buf, &gpu.best_cost_buf,
                        &mut chain_source, &mut rng, &w8,
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

        // 5. Move threshold regime
        let new_regime = if global_best_cost > 500 { 1u8 }
                         else if global_best_cost < 200 { 2u8 }
                         else { 0u8 };
        if new_regime != last_thresh_regime {
            let thresh = match new_regime {
                1 => THRESH_HIGH_COST,
                2 => THRESH_LOW_COST,
                _ => THRESH_DEFAULT,
            };
            gpu.queue.write_buffer(&gpu.move_thresh_buf, 0, bytemuck::bytes_of(&thresh));
            last_thresh_regime = new_regime;
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

                // 6a. Maintenance reseeding across all temp levels
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
                    let temp_frac = (idx % TEMP_LEVELS) as f64 / TEMP_LEVELS as f64;
                    let pert = (temp_frac * 5.0) as usize;
                    let mut a = seed_assignment;
                    solver_core::perturb(&mut a, &mut rng, pert);
                    let packed = pack_assignment(&a);
                    let cost = solver_core::evaluate(&a, &w8).total;
                    let offset_assign = (idx * ASSIGN_U32S * 4) as u64;
                    let offset_cost = (idx * 4) as u64;
                    gpu.queue.write_buffer(&gpu.assign_buf, offset_assign, bytemuck::cast_slice(&packed));
                    gpu.queue.write_buffer(&gpu.best_assign_buf, offset_assign, bytemuck::cast_slice(&packed));
                    gpu.queue.write_buffer(&gpu.cost_buf, offset_cost, bytemuck::bytes_of(&cost));
                    gpu.queue.write_buffer(&gpu.best_cost_buf, offset_cost, bytemuck::bytes_of(&cost));
                    chain_source[idx] = source_label.clone();
                }

                // 6b. Stagnation: extra reseeding with heavier perturbation
                if stag >= STAGNATION_DISPATCHES {
                    if !partitions[pi].aggressive_logged {
                        event!(start_time.elapsed(), &format!(
                            "SHAKEUP: partition {} aggressive (stagnant {} dispatches)", pi, stag));
                        partitions[pi].aggressive_logged = true;
                    }

                    let inject_count = p_len / 20;
                    for _ in 0..inject_count {
                        let idx = p_start + rng.random_range(0..p_len);
                        let temp_frac = (idx % TEMP_LEVELS) as f64 / TEMP_LEVELS as f64;
                        let pert = 3 + (temp_frac * 7.0) as usize;
                        let mut a = seed_assignment;
                        solver_core::perturb(&mut a, &mut rng, pert);
                        let packed = pack_assignment(&a);
                        let cost = solver_core::evaluate(&a, &w8).total;
                        let offset_assign = (idx * ASSIGN_U32S * 4) as u64;
                        let offset_cost = (idx * 4) as u64;
                        gpu.queue.write_buffer(&gpu.assign_buf, offset_assign, bytemuck::cast_slice(&packed));
                        gpu.queue.write_buffer(&gpu.best_assign_buf, offset_assign, bytemuck::cast_slice(&packed));
                        gpu.queue.write_buffer(&gpu.cost_buf, offset_cost, bytemuck::bytes_of(&cost));
                        gpu.queue.write_buffer(&gpu.best_cost_buf, offset_cost, bytemuck::bytes_of(&cost));
                        chain_source[idx] = source_label.clone();
                    }
                }

                // 6c. Escalated shakeup: full partition reseed with heavier perturbation + CPU shake
                if stag >= ESCALATION_DISPATCHES && stag % ESCALATION_DISPATCHES == 0 {
                    event!(start_time.elapsed(), &format!(
                        "ESCALATED SHAKEUP: partition {} (stagnant {} dispatches)", pi, stag));

                    reseed_partition_chains(
                        &gpu.queue, &gpu.assign_buf, &gpu.best_assign_buf,
                        &gpu.cost_buf, &gpu.best_cost_buf,
                        &mut chain_source, &mut rng, &w8,
                        &seed_assignment, &source_label, p_start, p_end,
                    );

                    let _ = cpu_workers.commands[pi].send(WorkerCommand::SetState(seed_assignment));
                    worker_metas[pi].reseeded_at = Instant::now();
                    worker_metas[pi].cost_at_reseed = solver_core::evaluate(&seed_assignment, &w8).total;
                }
                maybe_print_table!();
            }
        }

        // 7. Periodic GPU verification
        if last_verify.elapsed().as_secs() >= VERIFY_INTERVAL_SECS {
            if let Some(ref best) = global_best_assignment {
                let verify_cost = solver_core::evaluate(best, &w8);
                if verify_cost.total != global_best_cost {
                    event!(start_time.elapsed(), &format!(
                        "VERIFY: MISMATCH global best {} != cpu eval {}",
                        global_best_cost, verify_cost.total,
                    ));
                }
            }
            last_verify = Instant::now();
        }

        // 8. Table output (driven by wall clock, not dispatch count)
        maybe_print_table!();

        // 9. Periodic tally + move stats
        if dispatch_count > 0 && dispatch_count % 300 == 0 {
            let total = tally.from_shakeup + tally.from_normal + tally.from_gpu;
            if total > 0 {
                event!(start_time.elapsed(), &format!(
                    "TALLY: {} shakeup, {} normal, {} GPU ({} total)",
                    tally.from_shakeup, tally.from_normal, tally.from_gpu, total));
            }

        }
    }

    // Shutdown: send shutdown to CPU workers and join
    for cmd_tx in &cpu_workers.commands {
        let _ = cmd_tx.send(WorkerCommand::Shutdown);
    }
    for h in cpu_workers.handles {
        let _ = h.join();
    }

    if let Some(best) = global_best_assignment {
        let final_cost = solver_core::evaluate(&best, &w8);
        eprintln!("{}", format_event(start_time.elapsed(), &format!(
            "Final best: {} | {}", global_best_cost, cost_label(&final_cost),
        )));
    }
}
