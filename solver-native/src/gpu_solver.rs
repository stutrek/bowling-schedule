use bytemuck::{Pod, Zeroable};
use rand::rngs::SmallRng;
use rand::{Rng, SeedableRng};
use solver_native::cpu_sa::{self, CpuWorkers, WorkerCommand, WorkerReport};
use solver_native::*;
use std::fs;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant};
use wgpu::util::DeviceExt;

const DEFAULT_CHAIN_COUNT: u32 = 32768;
const MIN_CHAIN_COUNT: u32 = 4096;
const MAX_CHAIN_COUNT: u32 = 131072;
const ITERS_PER_DISPATCH: u32 = 1000;
const ASSIGN_U32S: usize = 48;
const SYNC_INTERVAL: u64 = 10;

const TEMP_LEVELS: usize = 256;
const COLD_THRESHOLD: usize = 64;
const WARM_THRESHOLD: usize = 192;

const STAGNATION_DISPATCHES: u64 = 60;
const ESCALATION_DISPATCHES: u64 = 300;

const VERIFY_INTERVAL_SECS: u64 = 30;


#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
struct GpuWeights {
    matchup_zero: u32,
    matchup_triple: u32,
    consecutive_opponents: u32,
    early_late_balance: f32,
    early_late_alternation: u32,
    lane_balance: f32,
    lane_switch: f32,
    late_lane_balance: f32,
    commissioner_overlap: u32,
    _pad0: u32,
    _pad1: u32,
    _pad2: u32,
}

#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
struct GpuParams {
    iters_per_dispatch: u32,
    chain_count: u32,
    temp_base: f32,
    temp_step: f32,
}

#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
struct GpuMoveThresholds {
    t: [u32; 12],
}

const THRESH_DEFAULT: GpuMoveThresholds = GpuMoveThresholds {
    t: [25, 40, 50, 58, 64, 70, 75, 81, 87, 93, 100, 0],
};
const THRESH_HIGH_COST: GpuMoveThresholds = GpuMoveThresholds {
    t: [30, 40, 52, 62, 69, 75, 79, 82, 88, 94, 100, 0],
};
const THRESH_LOW_COST: GpuMoveThresholds = GpuMoveThresholds {
    t: [20, 38, 46, 52, 57, 62, 67, 74, 82, 91, 100, 0],
};

// ═══════════════════════════════════════════════════════════════════════════
// Orchestrator state
// ═══════════════════════════════════════════════════════════════════════════

struct WorkerMeta {
    reseeded_at: Instant,
    cost_at_reseed: u32,
    last_report: Option<WorkerReport>,
    prev_iterations: u64,
    prev_iter_time: Instant,
    iters_per_sec: u64,
}

struct GlobalBestMeta {
    source: String,
    found_at: Instant,
}

struct ProvenanceTally {
    from_shakeup: u32,
    from_normal: u32,
    from_gpu: u32,
}

// ═══════════════════════════════════════════════════════════════════════════
// Focus mode (orchestrator picks worst constraint)
// ═══════════════════════════════════════════════════════════════════════════

// ═══════════════════════════════════════════════════════════════════════════
// Table output
// ═══════════════════════════════════════════════════════════════════════════

fn fmt_elapsed(d: Duration) -> String {
    let secs = d.as_secs();
    format!("+{:02}:{:02}:{:02}", secs / 3600, (secs % 3600) / 60, secs % 60)
}

const FRESH_TABLE_INTERVAL_SECS: u64 = 300;

fn print_table_banner(
    global_best_cost: u32,
    global_best_bd: &CostBreakdown,
    meta: &GlobalBestMeta,
    start_time: Instant,
) {
    let now = chrono::Local::now().format("%H:%M:%S");
    let age = meta.found_at.elapsed().as_secs();
    let age_str = if age < 60 { format!("{}s ago", age) }
                  else if age < 3600 { format!("{}m{}s ago", age / 60, age % 60) }
                  else { format!("{}h{}m ago", age / 3600, (age % 3600) / 60) };
    eprintln!(
        "── {} {} best={} from {} ({}) ──\x1b[K",
        now, fmt_elapsed(start_time.elapsed()),
        global_best_cost, meta.source, age_str,
    );
    eprintln!("   {}\x1b[K", cost_label(global_best_bd));
}

fn print_table_header() {
    eprintln!(
        "{:>9}  {:>4} {:>5}  {:>5} {:>5}  {:>9} {:>9} {:>9} {:>9} {:>9} {:>9} {:>9} {:>9}  {:>6}  {}\x1b[K",
        "elapsed", "src", "temp", "cur", "best",
        "match", "consec", "el_bal", "el_alt", "lane", "switch", "ll_bal", "comm",
        "it/s", "state",
    );
}

fn print_cpu_row(
    elapsed: Duration,
    core_id: usize,
    report: &WorkerReport,
    w8: &Weights,
    meta: &WorkerMeta,
    temp: f64,
) {
    let cur_bd = evaluate(&report.current_assignment, w8);
    let best_bd = evaluate(&report.best_assignment, w8);
    let since = meta.reseeded_at.elapsed().as_secs();
    let state = if since < 30 { format!("shook+{}s", since) } else { "normal".to_string() };
    let ips = if meta.iters_per_sec > 0 {
        format!("{}k", meta.iters_per_sec / 1000)
    } else {
        "-".to_string()
    };
    eprintln!(
        "{:>9}  cpu{:<1} {:>5.1}  {:>5} {:>5}  {:>4}/{:<4} {:>4}/{:<4} {:>4}/{:<4} {:>4}/{:<4} {:>4}/{:<4} {:>4}/{:<4} {:>4}/{:<4} {:>4}/{:<4}  {:>6}  {}\x1b[K",
        fmt_elapsed(elapsed), core_id, temp,
        cur_bd.total, best_bd.total,
        cur_bd.matchup_balance, best_bd.matchup_balance,
        cur_bd.consecutive_opponents, best_bd.consecutive_opponents,
        cur_bd.early_late_balance, best_bd.early_late_balance,
        cur_bd.early_late_alternation, best_bd.early_late_alternation,
        cur_bd.lane_balance, best_bd.lane_balance,
        cur_bd.lane_switch_balance, best_bd.lane_switch_balance,
        cur_bd.late_lane_balance, best_bd.late_lane_balance,
        cur_bd.commissioner_overlap, best_bd.commissioner_overlap,
        ips,
        state,
    );
}

fn print_gpu_row(
    elapsed: Duration,
    gpu_best_cost: u32,
    gpu_median: u32,
    best_bd: &CostBreakdown,
    gpu_ips: u64,
) {
    let ips = if gpu_ips > 0 {
        format!("{}M", gpu_ips / 1_000_000)
    } else {
        "-".to_string()
    };
    eprintln!(
        "{:>9}  gpu   {:>5}  ~{:<4} {:>5}  {:>4}/{:<4} {:>4}/{:<4} {:>4}/{:<4} {:>4}/{:<4} {:>4}/{:<4} {:>4}/{:<4} {:>4}/{:<4} {:>4}/{:<4}  {:>6}\x1b[K",
        fmt_elapsed(elapsed),
        "-", gpu_median, gpu_best_cost,
        "-", best_bd.matchup_balance,
        "-", best_bd.consecutive_opponents,
        "-", best_bd.early_late_balance,
        "-", best_bd.early_late_alternation,
        "-", best_bd.lane_balance,
        "-", best_bd.lane_switch_balance,
        "-", best_bd.late_lane_balance,
        "-", best_bd.commissioner_overlap,
        ips,
    );
}

fn print_event(elapsed: Duration, msg: &str) {
    eprintln!("{:>9}  >>>  {}", fmt_elapsed(elapsed), msg);
}

// ═══════════════════════════════════════════════════════════════════════════
// GPU helpers (unchanged)
// ═══════════════════════════════════════════════════════════════════════════

fn pack_assignment(a: &solver_core::Assignment) -> [u32; ASSIGN_U32S] {
    let mut packed = [0u32; ASSIGN_U32S];
    for w in 0..solver_core::WEEKS {
        for q in 0..solver_core::QUADS {
            let [pa, pb, pc, pd] = a[w][q];
            packed[w * solver_core::QUADS + q] =
                (pa as u32) | ((pb as u32) << 8) | ((pc as u32) << 16) | ((pd as u32) << 24);
        }
    }
    packed
}

fn unpack_assignment(packed: &[u32; ASSIGN_U32S]) -> solver_core::Assignment {
    let mut a = [[[0u8; solver_core::POS]; solver_core::QUADS]; solver_core::WEEKS];
    for w in 0..solver_core::WEEKS {
        for q in 0..solver_core::QUADS {
            let v = packed[w * solver_core::QUADS + q];
            a[w][q] = [
                (v & 0xFF) as u8,
                ((v >> 8) & 0xFF) as u8,
                ((v >> 16) & 0xFF) as u8,
                ((v >> 24) & 0xFF) as u8,
            ];
        }
    }
    a
}

fn detect_chain_count() -> u32 {
    let instance = wgpu::Instance::default();
    let adapter = pollster::block_on(
        instance.request_adapter(&wgpu::RequestAdapterOptions {
            power_preference: wgpu::PowerPreference::HighPerformance,
            ..Default::default()
        }),
    );
    let adapter = match adapter {
        Ok(a) => a,
        Err(_) => return DEFAULT_CHAIN_COUNT,
    };
    let limits = adapter.limits();
    let bytes_per_chain = (ASSIGN_U32S as u64) * 4;
    let max_chains = limits.max_storage_buffer_binding_size as u64 / bytes_per_chain;
    let chain_count = max_chains.min(MAX_CHAIN_COUNT as u64).max(MIN_CHAIN_COUNT as u64) as u32;
    (chain_count / 256) * 256
}

fn sampled_median(costs: &[u32], rng: &mut SmallRng) -> u32 {
    let n = costs.len();
    if n == 0 { return u32::MAX; }
    let sample_size = 64.min(n);
    let mut samples: Vec<u32> = (0..sample_size).map(|_| costs[rng.random_range(0..n)]).collect();
    samples.sort_unstable();
    samples[samples.len() / 2]
}

// ═══════════════════════════════════════════════════════════════════════════
// Main
// ═══════════════════════════════════════════════════════════════════════════

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
            .map(|i| 0.3 * (7.0 / 0.3f64).powf(i as f64 / (cpu_cores - 1) as f64))
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
        temp_base: 1.0,
        temp_step: 0.15,
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
    let instance = wgpu::Instance::default();
    let adapter = instance
        .request_adapter(&wgpu::RequestAdapterOptions {
            power_preference: wgpu::PowerPreference::HighPerformance,
            ..Default::default()
        })
        .await
        .expect("No GPU adapter found");

    eprintln!("GPU: {} ({:?})", adapter.get_info().name, adapter.get_info().backend);

    let (device, queue) = adapter
        .request_device(&wgpu::DeviceDescriptor {
            label: Some("SA Solver"),
            required_features: wgpu::Features::empty(),
            required_limits: wgpu::Limits::default(),
            ..Default::default()
        })
        .await
        .expect("Failed to create device");

    let sa_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("SA Shader"),
        source: wgpu::ShaderSource::Wgsl(include_str!("solver.wgsl").into()),
    });

    let assign_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("assignments"),
        contents: bytemuck::cast_slice(&assign_data),
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
    });
    let best_assign_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("best_assignments"),
        contents: bytemuck::cast_slice(&best_assign_data),
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
    });
    let cost_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("costs"),
        contents: bytemuck::cast_slice(&cost_data),
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
    });
    let best_cost_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("best_costs"),
        contents: bytemuck::cast_slice(&best_cost_data),
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
    });
    let rng_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("rng_states"),
        contents: bytemuck::cast_slice(&rng_data),
        usage: wgpu::BufferUsages::STORAGE,
    });
    let weights_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("weights"),
        contents: bytemuck::bytes_of(&gpu_weights),
        usage: wgpu::BufferUsages::UNIFORM,
    });
    let params_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("params"),
        contents: bytemuck::bytes_of(&gpu_params),
        usage: wgpu::BufferUsages::UNIFORM,
    });
    let move_thresh_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("move_thresholds"),
        contents: bytemuck::bytes_of(&THRESH_DEFAULT),
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
    });

    let costs_readback_size = (chain_count as usize * 4) as u64;
    let costs_readback_buf = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("costs_readback"),
        size: costs_readback_size,
        usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    let assign_readback_buf = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("assign_readback"),
        size: (ASSIGN_U32S * 4) as u64,
        usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });

    let sa_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("SA BGL"),
        entries: &[
            bgl_storage(0, false), bgl_storage(1, false), bgl_storage(2, false),
            bgl_storage(3, false), bgl_storage(4, false), bgl_uniform(5), bgl_uniform(6),
            bgl_storage(7, true),
        ],
    });

    let sa_bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("SA BG"),
        layout: &sa_bgl,
        entries: &[
            bg_entry(0, &assign_buf), bg_entry(1, &best_assign_buf),
            bg_entry(2, &cost_buf), bg_entry(3, &best_cost_buf),
            bg_entry(4, &rng_buf), bg_entry(5, &weights_buf), bg_entry(6, &params_buf),
            bg_entry(7, &move_thresh_buf),
        ],
    });

    let sa_pl = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some("SA PL"),
        bind_group_layouts: &[&sa_bgl],
        push_constant_ranges: &[],
    });

    let sa_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some("SA Pipeline"),
        layout: Some(&sa_pl),
        module: &sa_shader,
        entry_point: Some("main"),
        compilation_options: Default::default(),
        cache: None,
    });

    let sa_workgroups = (chain_count + 255) / 256;

    // ─── Orchestrator state ───────────────────────────────────────────────
    let mut global_best_cost = u32::MAX;
    let mut global_best_assignment: Option<solver_core::Assignment> = None;
    let mut dispatch_count = 0u64;
    let start_time = Instant::now();
    let mut last_verify = Instant::now();
    let mut dispatches_since_improvement: u64 = 0;
    let mut last_thresh_regime: u8 = 0;
    let mut aggressive_logged = false;
    let mut gpu_median = u32::MAX;
    let mut gpu_best_cost = u32::MAX;

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
    let mut last_print = Instant::now();
    let mut last_fresh_table = Instant::now();
    let mut can_overwrite = false;
    let mut prev_line_count: u32 = 0;

    // ─── Main loop ────────────────────────────────────────────────────────
    loop {
        if shutdown.load(Ordering::Relaxed) { break; }

        macro_rules! maybe_print_table {
            () => {
                if last_print.elapsed().as_millis() >= 1000 {
                    let fresh = !can_overwrite
                        || last_fresh_table.elapsed().as_secs() >= FRESH_TABLE_INTERVAL_SECS;
                    if !fresh && prev_line_count > 0 {
                        eprint!("\x1b[{}A", prev_line_count);
                    } else {
                        last_fresh_table = Instant::now();
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
                    let gpu_ips = if elapsed.as_secs() > 0 {
                        dispatch_count * ITERS_PER_DISPATCH as u64 * chain_count as u64 / elapsed.as_secs()
                    } else { 0 };
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
                    prev_line_count = lines;
                    can_overwrite = true;
                    last_print = Instant::now();
                }
            };
        }

        macro_rules! event {
            ($($arg:tt)*) => {
                print_event($($arg)*);
                can_overwrite = false;
            };
        }

        // 1. GPU dispatch (unchanged)
        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("SA"),
        });
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("SA Pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&sa_pipeline);
            pass.set_bind_group(0, &sa_bg, &[]);
            pass.dispatch_workgroups(sa_workgroups, 1, 1);
        }
        encoder.copy_buffer_to_buffer(&best_cost_buf, 0, &costs_readback_buf, 0, costs_readback_size);
        queue.submit(Some(encoder.finish()));

        let costs_slice = costs_readback_buf.slice(..);
        let (tx, rx) = std::sync::mpsc::channel();
        costs_slice.map_async(wgpu::MapMode::Read, move |r| { let _ = tx.send(r); });
        if device.poll(wgpu::PollType::Wait).is_err() {
            costs_readback_buf.unmap();
            continue;
        }
        if rx.recv().is_err() {
            continue;
        }

        // 2. Read costs + sample median + per-CPU-partition bests
        let chains_per_cpu = if cpu_cores > 0 { chain_count as usize / cpu_cores } else { chain_count as usize };
        let mut partition_bests: Vec<(u32, usize)> = vec![(u32::MAX, 0); cpu_cores]; // (cost, chain_idx)
        let (min_cost, min_chain) = {
            let data = costs_slice.get_mapped_range();
            let costs: &[u32] = bytemuck::cast_slice(&data);
            let mut best = u32::MAX;
            let mut best_idx = 0u32;
            for (i, &c) in costs.iter().enumerate() {
                if c < best { best = c; best_idx = i as u32; }
                let partition = (i / chains_per_cpu).min(cpu_cores - 1);
                if c < partition_bests[partition].0 {
                    partition_bests[partition] = (c, i);
                }
            }
            gpu_median = sampled_median(costs, &mut rng);
            (best, best_idx)
        };
        costs_readback_buf.unmap();
        gpu_best_cost = min_cost;

        dispatch_count += 1;
        dispatches_since_improvement += 1;
        maybe_print_table!();

        // 3. Check GPU improvement
        if min_cost < global_best_cost {
            global_best_cost = min_cost;
            dispatches_since_improvement = 0;
            aggressive_logged = false;
            tally.from_gpu += 1;
            global_best_meta = GlobalBestMeta {
                source: format!("gpu({})", chain_source[min_chain as usize]),
                found_at: Instant::now(),
            };

            let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("Read Best"),
            });
            let offset = min_chain as u64 * ASSIGN_U32S as u64 * 4;
            encoder.copy_buffer_to_buffer(&best_assign_buf, offset, &assign_readback_buf, 0, (ASSIGN_U32S * 4) as u64);
            queue.submit(Some(encoder.finish()));

            let assign_slice = assign_readback_buf.slice(..);
            let (tx, rx) = std::sync::mpsc::channel();
            assign_slice.map_async(wgpu::MapMode::Read, move |r| { let _ = tx.send(r); });
            let _ = device.poll(wgpu::PollType::Wait);
            let _ = rx.recv();

            let packed: [u32; ASSIGN_U32S] = {
                let data = assign_slice.get_mapped_range();
                let slice: &[u32] = bytemuck::cast_slice(&data);
                let mut arr = [0u32; ASSIGN_U32S];
                arr.copy_from_slice(slice);
                arr
            };
            assign_readback_buf.unmap();

            let assignment = unpack_assignment(&packed);
            let cpu_verify = solver_core::evaluate(&assignment, &w8);
            global_best_assignment = Some(assignment);

            let verify_status = if cpu_verify.total == min_cost { "OK" } else { "MISMATCH" };
            let seed_from = &chain_source[min_chain as usize];
            event!(start_time.elapsed(), &format!(
                "NEW BEST {} from gpu chain {} (seed: {}, verify: {}) | {}",
                global_best_cost, min_chain, seed_from, verify_status, cost_label(&cpu_verify),
            ));

            if global_best_cost < 160 {
                let ts = chrono::Local::now().format("%Y%m%d-%H%M%S%z");
                let filename = format!("{}/{:04}-gpu-{}.tsv", results_dir, global_best_cost, ts);
                let mut out = assignment;
                reassign_commissioners(&mut out);
                let _ = fs::write(&filename, assignment_to_tsv(&out));
                event!(start_time.elapsed(), &format!("Saved {}", filename));
            }

            // Burst reseed cold GPU chains from GPU best
            let burst_count = (chain_count as usize) / 20;
            let mut burst_seeded = 0;
            for _ in 0..burst_count * 4 {
                let idx = rng.random_range(0..chain_count as usize);
                if idx % TEMP_LEVELS >= COLD_THRESHOLD { continue; }
                let pert = 1 + (idx % TEMP_LEVELS) * 3 / COLD_THRESHOLD;
                let mut a = assignment;
                solver_core::perturb(&mut a, &mut rng, pert);
                let packed = pack_assignment(&a);
                let cost = solver_core::evaluate(&a, &w8).total;
                queue.write_buffer(&assign_buf, (idx * ASSIGN_U32S * 4) as u64, bytemuck::cast_slice(&packed));
                queue.write_buffer(&cost_buf, (idx * 4) as u64, bytemuck::bytes_of(&cost));
                chain_source[idx] = "gpu".to_string();
                burst_seeded += 1;
                if burst_seeded >= burst_count { break; }
            }

        }

        // 3b. Feed per-partition GPU bests into controlling CPU workers
        for (pi, &(pcost, pidx)) in partition_bests.iter().enumerate() {
            if pcost == u32::MAX { continue; }
            let cpu_best = worker_metas.get(pi)
                .and_then(|m| m.last_report.as_ref())
                .map(|r| solver_core::evaluate(&r.best_assignment, &w8).total)
                .unwrap_or(u32::MAX);
            if pcost < cpu_best {
                // Read back the assignment from GPU
                let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
                    label: Some("Read Partition Best"),
                });
                let offset = pidx as u64 * ASSIGN_U32S as u64 * 4;
                encoder.copy_buffer_to_buffer(&best_assign_buf, offset, &assign_readback_buf, 0, (ASSIGN_U32S * 4) as u64);
                queue.submit(Some(encoder.finish()));

                let assign_slice = assign_readback_buf.slice(..);
                let (tx, rx) = std::sync::mpsc::channel();
                assign_slice.map_async(wgpu::MapMode::Read, move |r| { let _ = tx.send(r); });
                let _ = device.poll(wgpu::PollType::Wait);
                let _ = rx.recv();

                let packed: [u32; ASSIGN_U32S] = {
                    let data = assign_slice.get_mapped_range();
                    let slice: &[u32] = bytemuck::cast_slice(&data);
                    let mut arr = [0u32; ASSIGN_U32S];
                    arr.copy_from_slice(slice);
                    arr
                };
                assign_readback_buf.unmap();

                let assignment = unpack_assignment(&packed);
                let _ = cpu_workers.commands[pi].send(WorkerCommand::SetState(assignment));
                if pi < worker_metas.len() {
                    worker_metas[pi].reseeded_at = Instant::now();
                    worker_metas[pi].cost_at_reseed = pcost;
                }
            }
        }

        // 4. Drain CPU reports
        while let Ok(report) = cpu_workers.reports.try_recv() {
            let cid = report.core_id;
            if cid < worker_metas.len() {
                let real_best = solver_core::evaluate(&report.best_assignment, &w8).total;
                if real_best < global_best_cost {
                    global_best_cost = real_best;
                    global_best_assignment = Some(report.best_assignment);
                    dispatches_since_improvement = 0;
                    aggressive_logged = false;
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

                    // Burst reseed GPU chains from CPU best
                    let burst_count = (chain_count as usize) / 20;
                    let mut burst_seeded = 0;
                    let source_label = format!("cpu{}", cid);
                    for _ in 0..burst_count * 4 {
                        let idx = rng.random_range(0..chain_count as usize);
                        if idx % TEMP_LEVELS >= COLD_THRESHOLD { continue; }
                        let pert = 1 + (idx % TEMP_LEVELS) * 3 / COLD_THRESHOLD;
                        let mut a = report.best_assignment;
                        solver_core::perturb(&mut a, &mut rng, pert);
                        let packed = pack_assignment(&a);
                        let cost = solver_core::evaluate(&a, &w8).total;
                        queue.write_buffer(&assign_buf, (idx * ASSIGN_U32S * 4) as u64, bytemuck::cast_slice(&packed));
                        queue.write_buffer(&cost_buf, (idx * 4) as u64, bytemuck::bytes_of(&cost));
                        chain_source[idx] = source_label.clone();
                        burst_seeded += 1;
                        if burst_seeded >= burst_count { break; }
                    }

                }
                let dt = worker_metas[cid].prev_iter_time.elapsed().as_secs_f64();
                if dt > 0.1 {
                    let di = report.iterations_total.saturating_sub(worker_metas[cid].prev_iterations);
                    worker_metas[cid].iters_per_sec = (di as f64 / dt) as u64;
                    worker_metas[cid].prev_iterations = report.iterations_total;
                    worker_metas[cid].prev_iter_time = Instant::now();
                }
                worker_metas[cid].last_report = Some(report);
            }
        }

        // 4b. Reseed each CPU's GPU partition from that worker's personal best
        for (pi, meta) in worker_metas.iter().enumerate() {
            if let Some(ref report) = meta.last_report {
                let cpu_best = solver_core::evaluate(&report.best_assignment, &w8).total;
                let (gpu_part_best, _) = partition_bests[pi];
                if cpu_best < gpu_part_best {
                    let start = pi * chains_per_cpu;
                    let end = ((pi + 1) * chains_per_cpu).min(chain_count as usize);
                    let reseed_count = (end - start) / 20;
                    let mut reseeded = 0;
                    for _ in 0..reseed_count * 4 {
                        let idx = start + rng.random_range(0..(end - start));
                        if idx % TEMP_LEVELS >= COLD_THRESHOLD { continue; }
                        let pert = 1 + (idx % TEMP_LEVELS) * 3 / COLD_THRESHOLD;
                        let mut a = report.best_assignment;
                        solver_core::perturb(&mut a, &mut rng, pert);
                        let packed = pack_assignment(&a);
                        let cost = solver_core::evaluate(&a, &w8).total;
                        queue.write_buffer(&assign_buf, (idx * ASSIGN_U32S * 4) as u64, bytemuck::cast_slice(&packed));
                        queue.write_buffer(&cost_buf, (idx * 4) as u64, bytemuck::bytes_of(&cost));
                        chain_source[idx] = format!("cpu{}", pi);
                        reseeded += 1;
                        if reseeded >= reseed_count { break; }
                    }
                }
            }
        }

        // 5. Move threshold regime (unchanged logic)
        let new_regime = if global_best_cost > 500 { 1u8 }
                         else if global_best_cost < 200 { 2u8 }
                         else { 0u8 };
        if new_regime != last_thresh_regime {
            let thresh = match new_regime {
                1 => THRESH_HIGH_COST,
                2 => THRESH_LOW_COST,
                _ => THRESH_DEFAULT,
            };
            queue.write_buffer(&move_thresh_buf, 0, bytemuck::bytes_of(&thresh));
            last_thresh_regime = new_regime;
        }

        // 6. Sync interval: GPU reseeding from CPU worker bests + stagnation
        maybe_print_table!();
        if dispatch_count % SYNC_INTERVAL == 0 {
            if let Some(ref best) = global_best_assignment {
                let base_reseed = (chain_count as usize) / 10;
                let reseed_count = if dispatches_since_improvement < 5 {
                    base_reseed / 2
                } else if dispatches_since_improvement > STAGNATION_DISPATCHES {
                    base_reseed * 2
                } else {
                    base_reseed
                };

                // Collect seed sources: global best + each CPU worker's personal best
                let mut seed_sources: Vec<(Assignment, String)> = vec![(*best, global_best_meta.source.clone())];
                for (si, meta) in worker_metas.iter().enumerate() {
                    if let Some(ref report) = meta.last_report {
                        seed_sources.push((report.best_assignment, format!("cpu{}", si)));
                    }
                }

                let mut reseeded = 0;
                let mut attempts = 0;
                let num_sources = seed_sources.len();
                while reseeded < reseed_count && attempts < reseed_count * 4 {
                    attempts += 1;
                    let idx = rng.random_range(0..chain_count as usize);
                    if idx % TEMP_LEVELS >= COLD_THRESHOLD { continue; }
                    let pert_max = 2 + (idx % TEMP_LEVELS) * 3 / COLD_THRESHOLD;
                    let pert = rng.random_range(1..=pert_max);
                    let si = rng.random_range(0..num_sources);
                    let mut a = seed_sources[si].0;
                    solver_core::perturb(&mut a, &mut rng, pert);
                    let packed = pack_assignment(&a);
                    let cost = solver_core::evaluate(&a, &w8).total;
                    queue.write_buffer(&assign_buf, (idx * ASSIGN_U32S * 4) as u64, bytemuck::cast_slice(&packed));
                    queue.write_buffer(&cost_buf, (idx * 4) as u64, bytemuck::bytes_of(&cost));
                    chain_source[idx] = seed_sources[si].1.clone();
                    reseeded += 1;
                }

                if dispatches_since_improvement >= STAGNATION_DISPATCHES {
                    if !aggressive_logged {
                        event!(start_time.elapsed(), &format!(
                            "SHAKEUP: aggressive mode (stagnant {} dispatches)", dispatches_since_improvement,
                        ));
                        aggressive_logged = true;
                    }

                    let shakeup_src = global_best_meta.source.clone();
                    let hot_inject_count = (chain_count as usize) / 100;
                    let mut hot_injected = 0;
                    for _ in 0..hot_inject_count * 4 {
                        let idx = rng.random_range(0..chain_count as usize);
                        if idx % TEMP_LEVELS < WARM_THRESHOLD { continue; }
                        let mut a = *best;
                        let pert = rng.random_range(20..=50);
                        solver_core::perturb(&mut a, &mut rng, pert);
                        let packed = pack_assignment(&a);
                        let cost = solver_core::evaluate(&a, &w8).total;
                        queue.write_buffer(&assign_buf, (idx * ASSIGN_U32S * 4) as u64, bytemuck::cast_slice(&packed));
                        queue.write_buffer(&cost_buf, (idx * 4) as u64, bytemuck::bytes_of(&cost));
                        chain_source[idx] = shakeup_src.clone();
                        hot_injected += 1;
                        if hot_injected >= hot_inject_count { break; }
                    }
                }

                if dispatches_since_improvement >= ESCALATION_DISPATCHES
                    && dispatches_since_improvement % ESCALATION_DISPATCHES == 0
                {
                    event!(start_time.elapsed(), &format!(
                        "ESCALATED SHAKEUP (stagnant {} dispatches)", dispatches_since_improvement,
                    ));

                    let esc_src = global_best_meta.source.clone();
                    let extra_cold = (chain_count as usize) / 5;
                    let mut extra_count = 0;
                    for _ in 0..extra_cold * 4 {
                        let idx = rng.random_range(0..chain_count as usize);
                        if idx % TEMP_LEVELS >= COLD_THRESHOLD { continue; }
                        let mut a = *best;
                        let pert = rng.random_range(10..=20);
                        solver_core::perturb(&mut a, &mut rng, pert);
                        let packed = pack_assignment(&a);
                        let cost = solver_core::evaluate(&a, &w8).total;
                        queue.write_buffer(&assign_buf, (idx * ASSIGN_U32S * 4) as u64, bytemuck::cast_slice(&packed));
                        queue.write_buffer(&cost_buf, (idx * 4) as u64, bytemuck::bytes_of(&cost));
                        chain_source[idx] = esc_src.clone();
                        extra_count += 1;
                        if extra_count >= extra_cold { break; }
                    }

                    let warm_inject = (chain_count as usize) / 50;
                    let mut warm_count = 0;
                    for _ in 0..warm_inject * 4 {
                        let idx = rng.random_range(0..chain_count as usize);
                        let temp_level = idx % TEMP_LEVELS;
                        if temp_level < COLD_THRESHOLD || temp_level >= WARM_THRESHOLD { continue; }
                        let mut a = *best;
                        let pert = rng.random_range(15..=30);
                        solver_core::perturb(&mut a, &mut rng, pert);
                        let packed = pack_assignment(&a);
                        let cost = solver_core::evaluate(&a, &w8).total;
                        queue.write_buffer(&assign_buf, (idx * ASSIGN_U32S * 4) as u64, bytemuck::cast_slice(&packed));
                        queue.write_buffer(&cost_buf, (idx * 4) as u64, bytemuck::bytes_of(&cost));
                        chain_source[idx] = esc_src.clone();
                        warm_count += 1;
                        if warm_count >= warm_inject { break; }
                    }

                    // Reseed ALL CPU workers from best with heavy perturbation
                    for (i, cmd_tx) in cpu_workers.commands.iter().enumerate() {
                        let mut a = *best;
                        solver_core::perturb(&mut a, &mut rng, 15 + i * 3);
                        let _ = cmd_tx.send(WorkerCommand::SetState(a));
                        if i < worker_metas.len() {
                            worker_metas[i].reseeded_at = Instant::now();
                            worker_metas[i].cost_at_reseed = global_best_cost;
                        }
                    }
                }
            }
        }

        // 7. Periodic GPU verification
        if last_verify.elapsed().as_secs() >= VERIFY_INTERVAL_SECS {
            if let Some(ref best) = global_best_assignment {
                let verify_cost = solver_core::evaluate(best, &w8);
                let status = if verify_cost.total == global_best_cost { "OK" } else { "MISMATCH" };
                event!(start_time.elapsed(), &format!(
                    "VERIFY: global best {} == cpu eval {} {}",
                    global_best_cost, verify_cost.total, status,
                ));
            }
            last_verify = Instant::now();
        }

        // 8. Table output (driven by wall clock, not dispatch count)
        maybe_print_table!();

        // 9. Periodic tally
        if dispatch_count > 0 && dispatch_count % 300 == 0 {
            let total = tally.from_shakeup + tally.from_normal + tally.from_gpu;
            if total > 0 {
                event!(start_time.elapsed(), &format!(
                    "TALLY: {} bests from shakeup, {} normal, {} GPU ({} total)",
                    tally.from_shakeup, tally.from_normal, tally.from_gpu, total,
                ));
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
        print_event(start_time.elapsed(), &format!(
            "Final best: {} | {}", global_best_cost, cost_label(&final_cost),
        ));
    }
}

fn bgl_storage(binding: u32, read_only: bool) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry {
        binding,
        visibility: wgpu::ShaderStages::COMPUTE,
        ty: wgpu::BindingType::Buffer {
            ty: wgpu::BufferBindingType::Storage { read_only },
            has_dynamic_offset: false,
            min_binding_size: None,
        },
        count: None,
    }
}

fn bgl_uniform(binding: u32) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry {
        binding,
        visibility: wgpu::ShaderStages::COMPUTE,
        ty: wgpu::BindingType::Buffer {
            ty: wgpu::BufferBindingType::Uniform,
            has_dynamic_offset: false,
            min_binding_size: None,
        },
        count: None,
    }
}

fn bg_entry(binding: u32, buffer: &wgpu::Buffer) -> wgpu::BindGroupEntry<'_> {
    wgpu::BindGroupEntry {
        binding,
        resource: buffer.as_entire_binding(),
    }
}
