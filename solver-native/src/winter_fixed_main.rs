use rand::rngs::SmallRng;
use rand::{Rng, SeedableRng};
use crate::cpu_sa_winter_fixed::{self, WinterFixedCpuWorkers, WinterFixedWorkerCommand};
use crate::gpu_setup::create_gpu_resources;
use crate::gpu_types::*;
use crate::gpu_types_winter_fixed::*;
use crate::output::*;
use crate::output_winter_fixed::*;
use solver_core::winter_fixed::*;
use solver_core::winter;
use std::collections::HashSet;
use std::fs;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::time::Instant;

const SYNC_INTERVAL: u64 = 10;
const VERIFY_INTERVAL_SECS: u64 = 30;
const SAVE_THRESHOLD: u32 = 600;
const DIVERSE_CYCLE_ITERS: u64 = 50_000;

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

fn reseed_partition_chains(
    queue: &wgpu::Queue,
    assign_buf: &wgpu::Buffer,
    best_assign_buf: &wgpu::Buffer,
    cost_buf: &wgpu::Buffer,
    best_cost_buf: &wgpu::Buffer,
    chain_source: &mut [String],
    w8: &WinterFixedWeights,
    source: &WinterFixedSchedule,
    source_label: &str,
    p_start: usize,
    p_end: usize,
) {
    let packed = pack_fixed_schedule(source);
    let cost = evaluate_fixed(source, w8).total;
    for idx in p_start..p_end {
        let offset_assign = (idx * WF_ASSIGN_U32S * 4) as u64;
        let offset_cost = (idx * 4) as u64;
        queue.write_buffer(assign_buf, offset_assign, bytemuck::cast_slice(&packed));
        queue.write_buffer(best_assign_buf, offset_assign, bytemuck::cast_slice(&packed));
        queue.write_buffer(cost_buf, offset_cost, bytemuck::bytes_of(&cost));
        queue.write_buffer(best_cost_buf, offset_cost, bytemuck::bytes_of(&cost));
        chain_source[idx] = source_label.to_string();
    }
}

fn read_gpu_chain(
    gpu: &crate::gpu_setup::GpuResources,
    chain_idx: usize,
    w8: &WinterFixedWeights,
) -> Option<(WinterFixedSchedule, u32)> {
    let mut encoder = gpu.device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("Read Chain"),
    });
    let offset = chain_idx as u64 * WF_ASSIGN_U32S as u64 * 4;
    encoder.copy_buffer_to_buffer(
        &gpu.best_assign_buf, offset,
        &gpu.assign_readback_buf, 0,
        (WF_ASSIGN_U32S * 4) as u64,
    );
    gpu.queue.submit(Some(encoder.finish()));

    let slice = gpu.assign_readback_buf.slice(..);
    let (tx, rx) = std::sync::mpsc::channel();
    slice.map_async(wgpu::MapMode::Read, move |r| { let _ = tx.send(r); });
    let deadline = Instant::now() + std::time::Duration::from_secs(10);
    loop {
        match gpu.device.poll(wgpu::PollType::Poll) {
            Ok(status) if status.is_queue_empty() => break,
            Ok(_) => {
                if Instant::now() > deadline {
                    gpu.assign_readback_buf.unmap();
                    return None;
                }
                std::thread::sleep(std::time::Duration::from_millis(1));
            }
            Err(_) => { gpu.assign_readback_buf.unmap(); return None; }
        }
    }
    if rx.recv_timeout(std::time::Duration::from_secs(5)).is_err() {
        gpu.assign_readback_buf.unmap();
        return None;
    }
    let packed: [u32; WF_ASSIGN_U32S] = {
        let data = slice.get_mapped_range();
        let s: &[u32] = bytemuck::cast_slice(&data);
        let mut arr = [0u32; WF_ASSIGN_U32S];
        arr.copy_from_slice(s);
        arr
    };
    gpu.assign_readback_buf.unmap();
    let sched = unpack_fixed_schedule(&packed);
    let cost = evaluate_fixed(&sched, w8).total;
    Some((sched, cost))
}

fn write_gpu_chain(
    gpu: &crate::gpu_setup::GpuResources,
    chain_idx: usize,
    sched: &WinterFixedSchedule,
    cost: u32,
) {
    let packed = pack_fixed_schedule(sched);
    let offset_assign = (chain_idx * WF_ASSIGN_U32S * 4) as u64;
    let offset_cost = (chain_idx * 4) as u64;
    gpu.queue.write_buffer(&gpu.assign_buf, offset_assign, bytemuck::cast_slice(&packed));
    gpu.queue.write_buffer(&gpu.best_assign_buf, offset_assign, bytemuck::cast_slice(&packed));
    gpu.queue.write_buffer(&gpu.cost_buf, offset_cost, bytemuck::bytes_of(&cost));
    gpu.queue.write_buffer(&gpu.best_cost_buf, offset_cost, bytemuck::bytes_of(&cost));
}

pub fn run(shutdown: Arc<AtomicBool>, args: &[String]) {
    let no_seed = args.iter().any(|a| a == "--no-seed");
    let no_cpu = args.iter().any(|a| a == "--no-cpu");
    let do_sweep = args.iter().any(|a| a == "--sweep");
    let diverse = args.iter().any(|a| a == "--diverse");

    let weights_str = fs::read_to_string("../weights.json").expect("Failed to read weights.json");
    let winter_w8: winter::Weights = serde_json::from_str(&weights_str).expect("Invalid weights.json");
    // Convert winter Weights to WinterFixedWeights
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

    // Score and sort seeds by cost (best first)
    let mut scored_seeds: Vec<(WinterFixedSchedule, u32)> = seeds.iter()
        .map(|s| (*s, evaluate_fixed(s, &w8).total))
        .collect();
    scored_seeds.sort_by_key(|(_, c)| *c);

    let available_cores = std::thread::available_parallelism().map(|n| n.get()).unwrap_or(4);
    let cpu_cores = if no_cpu { 0 } else { available_cores.saturating_sub(2) };

    let cpu_temps: Vec<f64> = if diverse {
        vec![10.0; cpu_cores]
    } else if cpu_cores <= 1 {
        vec![CPU_TEMP_MIN]
    } else {
        (0..cpu_cores)
            .map(|i| CPU_TEMP_MIN + (CPU_TEMP_MAX - CPU_TEMP_MIN) * i as f64 / (cpu_cores - 1) as f64)
            .collect()
    };
    let cpu_temps_display = cpu_temps.clone();

    let cpu_workers = cpu_sa_winter_fixed::run_winter_fixed_cpu_workers(
        cpu_cores,
        w8.clone(),
        cpu_temps,
        Arc::clone(&shutdown),
    );

    // Seed only top 3 CPU workers (skip in diverse mode)
    if !diverse {
        let seed_partitions = cpu_cores.min(3).min(scored_seeds.len());
        for i in 0..seed_partitions {
            let (ref seed_s, seed_cost) = scored_seeds[i];
            let _ = cpu_workers.commands[i].send(WinterFixedWorkerCommand::SetState(*seed_s));
            eprintln!("  CPU {} seeded with cost {}", i, seed_cost);
        }
        if seed_partitions > 0 {
            eprintln!("Seeded {} of {} CPU workers", seed_partitions, cpu_cores);
        }
    }

    if do_sweep {
        for cmd in &cpu_workers.commands {
            let _ = cmd.send(WinterFixedWorkerCommand::Sweep);
        }
        eprintln!("Sweep mode enabled for all CPU workers");
    }

    let chain_count = detect_chain_count(WF_ASSIGN_U32S);
    let num_wgs = chain_count as usize / TEMP_LEVELS;
    let wgs_per_part = if cpu_cores > 0 { num_wgs / cpu_cores } else { num_wgs };
    let pods_per_wg = TEMP_LEVELS / POD_SIZE;
    let total_pods = num_wgs * pods_per_wg;
    eprintln!(
        "GPU winter-fixed solver: {} chains, {} workgroups ({}/partition), {} iters/dispatch",
        chain_count, num_wgs, wgs_per_part, ITERS_PER_DISPATCH,
    );
    eprintln!(
        "  pods: {} per wg × {} wgs = {} total, {} chains/pod",
        pods_per_wg, num_wgs, total_pods, POD_SIZE,
    );
    eprintln!(
        "  ASSIGN_U32S: {} (was 48 in winter), chain_count: {} (was {})",
        WF_ASSIGN_U32S, chain_count, detect_chain_count(48),
    );
    {
        let temps: Vec<String> = (0..POD_SIZE).map(|l| format!("{:.1}", temp_for_level(l))).collect();
        eprintln!("  pod temps: [{}]", temps.join(", "));
    }
    eprintln!(
        "  CPU temps: {:.1}-{:.1}, {} seed files, {} CPU cores{}",
        cpu_temps_display.first().unwrap_or(&0.0),
        cpu_temps_display.last().unwrap_or(&0.0), seeds.len(), cpu_cores,
        if diverse { ", DIVERSE mode" } else { "" },
    );

    let mut rng = SmallRng::from_os_rng();
    let mut assign_data = vec![0u32; chain_count as usize * WF_ASSIGN_U32S];
    let mut rng_data = vec![0u32; chain_count as usize * 4];
    let mut cost_data = vec![0u32; chain_count as usize];
    let mut best_assign_data = vec![0u32; chain_count as usize * WF_ASSIGN_U32S];
    let mut best_cost_data = vec![u32::MAX; chain_count as usize];

    // Seed GPU chains: diverse mode = all unique random, default = top-3 partitions from seeds
    let chains_per_part = if cpu_cores > 0 { chain_count as usize / cpu_cores } else { chain_count as usize };
    let seed_chain_limit = if diverse { 0 } else { chains_per_part * cpu_cores.min(3) };
    for i in 0..chain_count as usize {
        let s = if !diverse && i < scored_seeds.len() && i < seed_chain_limit {
            scored_seeds[i].0
        } else {
            random_fixed_schedule(&mut rng)
        };
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

    // Inject template constants into shader
    let shader_src = format!(
        "{}\n{}",
        wgsl_consts(),
        include_str!("winter_fixed_solver.wgsl"),
    );

    pollster::block_on(run_gpu(
        assign_data, best_assign_data, cost_data, best_cost_data, rng_data,
        gpu_weights, gpu_params, w8, results_dir.to_string(), shutdown, rng,
        cpu_workers, cpu_cores, cpu_temps_display, shader_src, diverse,
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
    cpu_temps_display: Vec<f64>,
    shader_src: String,
    diverse: bool,
) {
    let display_active = Arc::new(AtomicBool::new(true));
    let dump_tsv = Arc::new(AtomicBool::new(false));
    {
        let da = Arc::clone(&display_active);
        let dt = Arc::clone(&dump_tsv);
        std::thread::spawn(move || {
            use std::io::Read;
            // Set terminal to raw mode to read single keypresses
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
                } else if byte == b'h' {
                    dt.store(true, Ordering::Relaxed);
                }
            }
        });
    }

    let chain_count = gpu_params.chain_count;
    let mut chain_source: Vec<String> = vec!["random".to_string(); chain_count as usize];

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

    let mut worker_metas: Vec<WinterFixedWorkerMeta> = (0..cpu_cores).map(|_| WinterFixedWorkerMeta {
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
    let mut partition_cpu_bests: Vec<u32> = vec![0; cpu_cores];
    let mut partition_gpu_bests: Vec<u32> = vec![0; cpu_cores];
    let mut counts_reset_done = false;
    let mut partition_best_cost: Vec<u32> = vec![u32::MAX; cpu_cores];
    let mut saved_hashes: HashSet<[u32; WF_ASSIGN_U32S]> = HashSet::new();
    // Pre-populate from existing result files on disk
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
    let num_workgroups = chain_count as usize / TEMP_LEVELS;
    let chains_per_cpu = if cpu_cores > 0 { chain_count as usize / cpu_cores } else { chain_count as usize };
    let mut partition_medians: Vec<u32> = vec![0; cpu_cores];
    let mut best_costs_snapshot: Vec<u32> = vec![u32::MAX; chain_count as usize];
    let mut exchange_swaps_total: u64 = 0;
    let mut exchange_attempts_total: u64 = 0;
    let mut exchange_reset_time = Instant::now();

    // Diverse mode: CPU workers cycle through GPU chains (pull-refine-push)
    let mut diverse_chain_idx: Vec<usize> = (0..cpu_cores).map(|pi| pi * chains_per_cpu).collect();
    let mut diverse_start_iters: Vec<u64> = vec![0; cpu_cores];
    let mut diverse_start_cost: Vec<u32> = vec![u32::MAX; cpu_cores];
    let mut diverse_locked: HashSet<usize> = HashSet::new();
    let mut diverse_seeded: Vec<bool> = vec![false; cpu_cores];
    let mut diverse_pass_done: Vec<usize> = vec![0; cpu_cores];
    let mut diverse_passes_completed: Vec<u64> = vec![0; cpu_cores];

    let mut pending_events: Vec<String> = Vec::new();
    let mut last_print = Instant::now();
    let mut last_fresh_table = Instant::now();
    let mut prev_line_count: u32 = 0;

    loop {
        if shutdown.load(Ordering::Relaxed) { break; }

        macro_rules! maybe_print_table {
            () => {
                // Always drain and print events, but only show the table if display is active
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
                        print_fixed_table_banner(global_best_cost, &best_bd, &global_best_meta, start_time, gpu_ips, cpu_ips);
                        lines += 2;
                    }
                    if diverse {
                        eprintln!(
                            "{:>4} {:>9}  {:>5} {:>5}  {:>5} {:>5} {:>5} {:>5} {:>5} {:>5} {:>5} {:>5} {:>5} {:>5}  {}\x1b[K",
                            "src", "chain", "best", "worst",
                            "match", "consec", "el_bal", "el_alt", "el_con", "lane", "switch", "ll_bal", "comm", "hs_rpt",
                            "state",
                        );
                    } else {
                        print_fixed_table_header();
                    }
                    lines += 1;
                    for (i, meta) in worker_metas.iter().enumerate() {
                        if let Some(ref _report) = meta.last_report {
                            let info_col = if diverse {
                                format!("ch{}", diverse_chain_idx[i])
                            } else {
                                let temp = if i < cpu_temps_display.len() { cpu_temps_display[i] } else { 0.0 };
                                format!("{:.0}/{:.1}", temp, meta.last_report.as_ref().unwrap().current_temp)
                            };
                            let (part_best, part_worst) = if diverse {
                                let p_start = i * chains_per_cpu;
                                let p_end = (p_start + chains_per_cpu).min(best_costs_snapshot.len());
                                let valid = best_costs_snapshot[p_start..p_end].iter()
                                    .copied()
                                    .filter(|&c| c > 0 && c < 1_000_000);
                                let mut lo = u32::MAX;
                                let mut hi = 0u32;
                                for c in valid {
                                    if c < lo { lo = c; }
                                    if c > hi { hi = c; }
                                }
                                (if lo == u32::MAX { 0 } else { lo }, hi)
                            } else {
                                (meta.last_report.as_ref().unwrap().best_cost,
                                 meta.last_report.as_ref().unwrap().current_cost)
                            };
                            print_fixed_cpu_row(i, meta.last_report.as_ref().unwrap(), &w8, meta, &info_col, part_best, part_worst);
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
                        // 3-line GPU best cost histogram across all chains
                        {
                            let hist = build_histogram(&best_costs_snapshot, 140);
                            for row in 0..3 {
                                eprintln!("  {} │{}\x1b[K", hist.labels[row], hist.rows[row]);
                            }
                            eprintln!("       └{}\x1b[K", hist.legend);
                            lines += 4;
                        }
                        if dump_tsv.swap(false, Ordering::Relaxed) {
                            let path = format!("{}/chain_dump_{}.tsv", results_dir,
                                chrono::Local::now().format("%Y%m%d-%H%M%S"));
                            eprintln!("\n[dumping all chains to {}...]\x1b[K", path);
                            let cc = chain_count as usize;
                            let mut lines = Vec::with_capacity(cc + 1);
                            // Header
                            let mut hdr = "chain,score,matchup_balance,consecutive_opponents,early_late_balance,early_late_alternation,early_late_consecutive,lane_balance,lane_switch_balance,late_lane_balance,commissioner_overlap,half_season_repeat".to_string();
                            for i in 0..WF_ASSIGN_U32S {
                                hdr.push_str(&format!(",d{}", i));
                            }
                            lines.push(hdr);
                            let mut read_ok = 0usize;
                            for ci in 0..cc {
                                if let Some((sched, _cost)) = read_gpu_chain(&gpu, ci, &w8) {
                                    let bd = evaluate_fixed(&sched, &w8);
                                    let packed = pack_fixed_schedule(&sched);
                                    let mut line = format!("{},{},{},{},{},{},{},{},{},{},{},{}",
                                        ci, bd.total, bd.matchup_balance, bd.consecutive_opponents,
                                        bd.early_late_balance, bd.early_late_alternation,
                                        bd.early_late_consecutive, bd.lane_balance,
                                        bd.lane_switch_balance, bd.late_lane_balance,
                                        bd.commissioner_overlap, bd.half_season_repeat);
                                    for v in &packed {
                                        line.push_str(&format!(",{}", v));
                                    }
                                    lines.push(line);
                                    read_ok += 1;
                                }
                                if ci % 1000 == 999 {
                                    eprint!("\r  [{}/{}]\x1b[K", ci + 1, cc);
                                }
                            }
                            let content = lines.join("\n");
                            match fs::write(&path, &content) {
                                Ok(()) => eprintln!("\r[chain dump saved: {} chains to {}]\x1b[K", read_ok, path),
                                Err(e) => eprintln!("\r[chain dump failed: {}]\x1b[K", e),
                            }
                        }
                        if exchange_reset_time.elapsed().as_secs() >= 300 {
                            exchange_swaps_total = 0;
                            exchange_attempts_total = 0;
                            exchange_reset_time = Instant::now();
                        }
                        if exchange_attempts_total > 0 {
                            let rate = exchange_swaps_total as f64 / exchange_attempts_total as f64 * 100.0;
                            let status = if rate > 80.0 { "wasteful" }
                                else if rate > 60.0 { "high" }
                                else if rate > 40.0 { "good" }
                                else if rate > 25.0 { "ok" }
                                else if rate > 10.0 { "low" }
                                else { "awful" };
                            eprintln!("  exchange:  {}/{} ({:.1}%) {}\x1b[K",
                                exchange_swaps_total, exchange_attempts_total, rate, status);
                            lines += 1;
                        }
                        if diverse && chains_per_cpu > 0 {
                            let max_done = diverse_pass_done.iter().copied().max().unwrap_or(0);
                            let total_passes: u64 = diverse_passes_completed.iter().sum();
                            let frac = max_done as f64 / chains_per_cpu as f64;
                            let bar_width: usize = 100;
                            let filled = (frac * bar_width as f64) as usize;
                            let bar: String = "█".repeat(filled) + &"░".repeat(bar_width - filled);
                            eprintln!("  cpu pass: [{bar}] {}/{} ({:.0}%) done:{}\x1b[K",
                                max_done, chains_per_cpu, frac * 100.0, total_passes);
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
        if dispatch_count < 5 {
            eprintln!("dispatch {} completed in {}ms", dispatch_count, dispatch_start.elapsed().as_millis());
        }

        // 2. Read costs
        let mut partition_bests: Vec<(u32, usize)> = vec![(u32::MAX, 0); cpu_cores];
        let current_costs_snapshot: Vec<u32>;
        {
            let data = costs_slice.get_mapped_range();
            let all: &[u32] = bytemuck::cast_slice(&data);
            let n = chain_count as usize;
            let best_costs = &all[..n];
            let current_costs = &all[n..n * 2];
            current_costs_snapshot = current_costs.to_vec();
            best_costs_snapshot = best_costs.to_vec();

            // Overlay CPU live bests so partition min/max reflects in-progress work
            if diverse {
                for cid in 0..cpu_cores {
                    let chain = diverse_chain_idx[cid];
                    let cpu_best = cpu_workers.live_best_costs[cid].load(Ordering::Relaxed);
                    if cpu_best > 0 && cpu_best < best_costs_snapshot[chain] {
                        best_costs_snapshot[chain] = cpu_best;
                    }
                }
            }

            for (i, &c) in best_costs.iter().enumerate() {
                let partition = (i / chains_per_cpu).min(cpu_cores - 1);
                if c < partition_bests[partition].0 {
                    partition_bests[partition] = (c, i);
                }
            }
            for pi in 0..cpu_cores {
                let start = pi * chains_per_cpu;
                let end = ((pi + 1) * chains_per_cpu).min(n);
                partition_medians[pi] = sampled_median(&current_costs[start..end], &mut rng);
            }
        }
        gpu.costs_readback_buf.unmap();

        dispatch_count += 1;

        // 2a. Diverse mode: initial seeding of CPU workers from GPU chains
        if diverse {
            for ci in 0..cpu_cores {
                if !diverse_seeded[ci] {
                    let chain_idx = diverse_chain_idx[ci];
                    if let Some((sched, cost)) = read_gpu_chain(&gpu, chain_idx, &w8) {
                        let _ = cpu_workers.commands[ci].send(WinterFixedWorkerCommand::SetState(sched));
                        diverse_start_cost[ci] = cost;
                        diverse_locked.insert(chain_idx);
                        diverse_seeded[ci] = true;
                        // start_iters will be set from first report
                    }
                }
            }
        }

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
                        // Skip pairs involving locked chains (diverse mode CPU refinement)
                        if diverse_locked.contains(&chain_a) || diverse_locked.contains(&chain_b) {
                            level += 2;
                            continue;
                        }
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

            // Swap chain_source tags to match GPU exchange
            for pair in swap_pairs.chunks_exact(2) {
                chain_source.swap(pair[0] as usize, pair[1] as usize);
            }

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
            }
            event!(start_time.elapsed(), "RESET partition counters");
        }

        maybe_print_table!();

        // 3. Per-partition GPU feedback + global best tracking
        //    In diverse mode, don't interrupt CPU cycles — CPU pulls from GPU on its own schedule
        for pi in 0..cpu_cores {
            let (gpu_part_cost, gpu_part_chain) = partition_bests[pi];
            if diverse {
                // Still track global best from GPU, but don't seed CPU
                if gpu_part_cost < partition_best_cost[pi] {
                    let prev = partition_best_cost[pi];
                    partition_best_cost[pi] = gpu_part_cost;
                    partition_gpu_bests[pi] += 1;
                    worker_metas[pi].best_found_at = Instant::now();
                    let level_in_pod = (gpu_part_chain % TEMP_LEVELS) % POD_SIZE;
                    let temp_val = temp_for_level(level_in_pod);
                    event!(start_time.elapsed(), &format!(
                        "PARTITION {} NEW BEST {} (was {}) from gpu T={:.1}",
                        pi, gpu_part_cost, prev, temp_val));
                    if gpu_part_cost <= SAVE_THRESHOLD {
                        if let Some((sched, _)) = read_gpu_chain(&gpu, gpu_part_chain, &w8) {
                            maybe_save_result(&sched, gpu_part_cost, &format!("gpu-p{}", pi), &results_dir, &mut saved_hashes);
                        }
                    }
                }
                if gpu_part_cost < global_best_cost {
                    if let Some((sched, _)) = read_gpu_chain(&gpu, gpu_part_chain, &w8) {
                        global_best_cost = gpu_part_cost;
                        global_best_schedule = Some(sched);
                        let level_in_pod = (gpu_part_chain % TEMP_LEVELS) % POD_SIZE;
                        global_best_meta = GlobalBestMeta {
                            source: format!("gpu-p{}-T{:.1}", pi, temp_for_level(level_in_pod)),
                            found_at: Instant::now(),
                        };
                    }
                }
                continue;
            }
            let cpu_live_best = cpu_workers.live_best_costs[pi].load(Ordering::Relaxed);
            if gpu_part_cost >= cpu_live_best { continue; }

            let mut encoder = gpu.device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("Read Partition Best"),
            });
            let offset = gpu_part_chain as u64 * WF_ASSIGN_U32S as u64 * 4;
            encoder.copy_buffer_to_buffer(&gpu.best_assign_buf, offset, &gpu.assign_readback_buf, 0, (WF_ASSIGN_U32S * 4) as u64);
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

            let packed: [u32; WF_ASSIGN_U32S] = {
                let data = assign_slice.get_mapped_range();
                let slice: &[u32] = bytemuck::cast_slice(&data);
                let mut arr = [0u32; WF_ASSIGN_U32S];
                arr.copy_from_slice(slice);
                arr
            };
            gpu.assign_readback_buf.unmap();

            let schedule = unpack_fixed_schedule(&packed);
            let verify_cost = evaluate_fixed(&schedule, &w8).total;
            if verify_cost != gpu_part_cost {
                event!(start_time.elapsed(), &format!(
                    "GPU EVAL MISMATCH p{}: gpu={} cpu={}", pi, gpu_part_cost, verify_cost));
            }

            let level_in_pod = (gpu_part_chain % TEMP_LEVELS) % POD_SIZE;
            let temp_val = temp_for_level(level_in_pod);

            let is_new_global_best = gpu_part_cost < global_best_cost;

            // Seed CPU: low temp (2.0) for new global best, normal reset otherwise
            if is_new_global_best {
                let _ = cpu_workers.commands[pi].send(WinterFixedWorkerCommand::SetStateWithTemp(schedule, 2.0));
            } else {
                let _ = cpu_workers.commands[pi].send(WinterFixedWorkerCommand::SetState(schedule));
            }

            if gpu_part_cost < partition_best_cost[pi] {
                let prev = partition_best_cost[pi];
                partition_best_cost[pi] = gpu_part_cost;
                partition_gpu_bests[pi] += 1;
                worker_metas[pi].best_found_at = Instant::now();
                event!(start_time.elapsed(), &format!(
                    "PARTITION {} NEW BEST {} (was {}) from gpu T={:.1}",
                    pi, gpu_part_cost, prev, temp_val));

                if gpu_part_cost <= SAVE_THRESHOLD {
                    maybe_save_result(&schedule, gpu_part_cost, &format!("gpu-p{}", pi), &results_dir, &mut saved_hashes);
                }
            }

            if is_new_global_best {
                global_best_cost = gpu_part_cost;
                global_best_schedule = Some(schedule);
                global_best_meta = GlobalBestMeta {
                    source: format!("gpu-p{}-T{:.1}", pi, temp_val),
                    found_at: Instant::now(),
                };

                if !diverse {
                    let p_start = pi * chains_per_cpu;
                    let p_end = ((pi + 1) * chains_per_cpu).min(chain_count as usize);
                    reseed_partition_chains(
                        &gpu.queue, &gpu.assign_buf, &gpu.best_assign_buf,
                        &gpu.cost_buf, &gpu.best_cost_buf,
                        &mut chain_source, &w8,
                        &schedule, &format!("gpu-p{}", pi), p_start, p_end,
                    );
                }
            }
        }

        // 4. Drain CPU reports
        while let Ok(report) = cpu_workers.reports.try_recv() {
            let cid = report.core_id;
            if cid < worker_metas.len() {
                let real_best = evaluate_fixed(&report.best_schedule, &w8).total;

                // Set start_iters from first report for diverse mode
                if diverse && diverse_start_iters[cid] == 0 {
                    diverse_start_iters[cid] = report.iterations_total;
                }

                if real_best < partition_best_cost[cid] {
                    let prev = partition_best_cost[cid];
                    partition_best_cost[cid] = real_best;
                    partition_cpu_bests[cid] += 1;
                    worker_metas[cid].best_found_at = Instant::now();
                    event!(start_time.elapsed(), &format!(
                        "PARTITION {} NEW BEST {} (was {}) from cpu{}", cid, real_best, prev, cid));

                    if real_best <= SAVE_THRESHOLD {
                        maybe_save_result(&report.best_schedule, real_best, &format!("cpu{}", cid), &results_dir, &mut saved_hashes);
                    }
                }
                if real_best < global_best_cost {
                    global_best_cost = real_best;
                    global_best_schedule = Some(report.best_schedule);
                    global_best_meta = GlobalBestMeta {
                        source: format!("cpu{}", cid),
                        found_at: Instant::now(),
                    };

                    event!(start_time.elapsed(), &format!(
                        "NEW BEST {} from cpu{}", global_best_cost, cid));

                    if !diverse {
                        let p_start = cid * chains_per_cpu;
                        let p_end = ((cid + 1) * chains_per_cpu).min(chain_count as usize);
                        let source_label = format!("cpu{}", cid);
                        reseed_partition_chains(
                            &gpu.queue, &gpu.assign_buf, &gpu.best_assign_buf,
                            &gpu.cost_buf, &gpu.best_cost_buf,
                            &mut chain_source, &w8,
                            &report.best_schedule, &source_label, p_start, p_end,
                        );
                    }
                }

                // Diverse mode: cycle to next chain when budget exhausted
                if diverse && report.iterations_total >= diverse_start_iters[cid] + DIVERSE_CYCLE_ITERS {
                    let old_chain = diverse_chain_idx[cid];
                    let old_cost = diverse_start_cost[cid];

                    // Write back only if CPU improved on the chain
                    if real_best < old_cost {
                        write_gpu_chain(&gpu, old_chain, &report.best_schedule, real_best);
                        chain_source[old_chain] = format!("cpu{}", cid);
                    }

                    // Unlock old chain
                    diverse_locked.remove(&old_chain);

                    // Advance to next chain in partition (round-robin)
                    let p_start = cid * chains_per_cpu;
                    let p_end = ((cid + 1) * chains_per_cpu).min(chain_count as usize);
                    let mut next = old_chain + 1;
                    if next >= p_end {
                        next = p_start;
                        diverse_pass_done[cid] = 0;
                        diverse_passes_completed[cid] += 1;
                    } else {
                        diverse_pass_done[cid] += 1;
                    }
                    diverse_chain_idx[cid] = next;

                    // Read next chain from GPU and send to worker
                    if let Some((sched, cost)) = read_gpu_chain(&gpu, next, &w8) {
                        let _ = cpu_workers.commands[cid].send(WinterFixedWorkerCommand::SetState(sched));
                        diverse_start_cost[cid] = cost;
                        diverse_start_iters[cid] = report.iterations_total;
                        diverse_locked.insert(next);
                    }
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

        // 6. Sync interval: per-partition reseeding (skip in diverse mode)
        maybe_print_table!();
        if !diverse && dispatch_count % SYNC_INTERVAL == 0 {
            for pi in 0..cpu_cores {
                let seed_schedule = worker_metas[pi].last_report.as_ref()
                    .map(|r| r.best_schedule);
                let seed_schedule = match seed_schedule {
                    Some(s) => s,
                    None => continue,
                };
                let source_label = format!("cpu{}", pi);
                let p_start = pi * chains_per_cpu;
                let p_end = ((pi + 1) * chains_per_cpu).min(chain_count as usize);
                let p_len = p_end - p_start;

                let reseed_count = p_len / 50;
                for _ in 0..reseed_count {
                    let idx = p_start + rng.random_range(0..p_len);
                    let level_in_pod = (idx % TEMP_LEVELS) % POD_SIZE;
                    let t_frac = level_in_pod as f64 / (POD_SIZE - 1).max(1) as f64;
                    let pert = (t_frac * t_frac * 5.0) as usize;
                    let mut s = seed_schedule;
                    perturb_fixed(&mut s, &mut rng, pert);
                    let packed = pack_fixed_schedule(&s);
                    let cost = evaluate_fixed(&s, &w8).total;
                    let offset_assign = (idx * WF_ASSIGN_U32S * 4) as u64;
                    let offset_cost = (idx * 4) as u64;
                    gpu.queue.write_buffer(&gpu.assign_buf, offset_assign, bytemuck::cast_slice(&packed));
                    gpu.queue.write_buffer(&gpu.best_assign_buf, offset_assign, bytemuck::cast_slice(&packed));
                    gpu.queue.write_buffer(&gpu.cost_buf, offset_cost, bytemuck::bytes_of(&cost));
                    gpu.queue.write_buffer(&gpu.best_cost_buf, offset_cost, bytemuck::bytes_of(&cost));
                    chain_source[idx] = source_label.clone();
                }
                maybe_print_table!();
            }
        }

        // 7. Periodic GPU verification
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

    // Restore terminal settings
    let _ = std::process::Command::new("stty").args(["icanon", "echo"]).status();

    if let Some(best) = global_best_schedule {
        let final_cost = evaluate_fixed(&best, &w8);
        eprintln!("{}", format_event(start_time.elapsed(), &format!(
            "Final best: {} | {}", global_best_cost, fixed_cost_label(&final_cost),
        )));
    }
}
