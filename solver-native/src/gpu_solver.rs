use bytemuck::{Pod, Zeroable};
use rand::rngs::SmallRng;
use rand::{Rng, SeedableRng};
use solver_native::*;
use std::fs;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::time::Instant;
use wgpu::util::DeviceExt;

const CHAIN_COUNT: u32 = 32768;
const ITERS_PER_DISPATCH: u32 = 1000;
const ASSIGN_U32S: usize = 48;

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

    let weights_str = fs::read_to_string("../weights.json").expect("Failed to read weights.json");
    let w8: solver_core::Weights = serde_json::from_str(&weights_str).expect("Invalid weights.json");

    let results_dir = "results/gpu";
    fs::create_dir_all(results_dir).expect("Failed to create results directory");

    let mut seeds: Vec<solver_core::Assignment> = Vec::new();
    if !no_seed {
        let seed_dir = "results/split-sa/full";
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
    eprintln!(
        "[{}] GPU solver: {} chains, {} iters/dispatch, {} seed files",
        now_iso(), CHAIN_COUNT, ITERS_PER_DISPATCH, seeds.len()
    );

    let mut rng = SmallRng::from_os_rng();
    let mut assign_data = vec![0u32; CHAIN_COUNT as usize * ASSIGN_U32S];
    let mut rng_data = vec![0u32; CHAIN_COUNT as usize * 4];
    let mut cost_data = vec![0u32; CHAIN_COUNT as usize];
    let mut best_assign_data = vec![0u32; CHAIN_COUNT as usize * ASSIGN_U32S];
    let mut best_cost_data = vec![u32::MAX; CHAIN_COUNT as usize];

    for i in 0..CHAIN_COUNT as usize {
        let a = if i < seeds.len() {
            seeds[i]
        } else {
            solver_core::random_assignment(&mut rng)
        };
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
        chain_count: CHAIN_COUNT,
        temp_base: 1.0,
        temp_step: 0.15,
    };

    pollster::block_on(run_gpu(
        assign_data, best_assign_data, cost_data, best_cost_data, rng_data,
        gpu_weights, gpu_params, w8, results_dir.to_string(), shutdown,
    ));
}

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
) {
    let instance = wgpu::Instance::default();
    let adapter = instance
        .request_adapter(&wgpu::RequestAdapterOptions {
            power_preference: wgpu::PowerPreference::HighPerformance,
            ..Default::default()
        })
        .await
        .expect("No GPU adapter found");

    eprintln!(
        "[{}] GPU: {} ({:?})",
        now_iso(), adapter.get_info().name, adapter.get_info().backend
    );

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
        usage: wgpu::BufferUsages::STORAGE,
    });
    let best_assign_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("best_assignments"),
        contents: bytemuck::cast_slice(&best_assign_data),
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
    });
    let cost_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("costs"),
        contents: bytemuck::cast_slice(&cost_data),
        usage: wgpu::BufferUsages::STORAGE,
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

    // Readback buffers: 128KB for all best_costs, 192B for one assignment
    let costs_readback_size = (CHAIN_COUNT as usize * 4) as u64;
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

    // SA pipeline
    let sa_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("SA BGL"),
        entries: &[
            bgl_storage(0, false), bgl_storage(1, false), bgl_storage(2, false),
            bgl_storage(3, false), bgl_storage(4, false), bgl_uniform(5), bgl_uniform(6),
        ],
    });

    let sa_bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("SA BG"),
        layout: &sa_bgl,
        entries: &[
            bg_entry(0, &assign_buf), bg_entry(1, &best_assign_buf),
            bg_entry(2, &cost_buf), bg_entry(3, &best_cost_buf),
            bg_entry(4, &rng_buf), bg_entry(5, &weights_buf), bg_entry(6, &params_buf),
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

    let sa_workgroups = (CHAIN_COUNT + 255) / 256;

    let mut global_best_cost = u32::MAX;
    let mut global_best_assignment: Option<solver_core::Assignment> = None;
    let mut dispatch_count = 0u64;
    let start_time = Instant::now();
    let mut last_print = Instant::now();

    eprintln!("[{}] Starting GPU SA...", now_iso());

    loop {
        if shutdown.load(Ordering::Relaxed) { break; }

        // SA dispatch
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
        // Copy best_costs to readback
        encoder.copy_buffer_to_buffer(&best_cost_buf, 0, &costs_readback_buf, 0, costs_readback_size);
        queue.submit(Some(encoder.finish()));

        // Map and read best_costs
        let costs_slice = costs_readback_buf.slice(..);
        let (tx, rx) = std::sync::mpsc::channel();
        costs_slice.map_async(wgpu::MapMode::Read, move |r| { let _ = tx.send(r); });
        if device.poll(wgpu::PollType::Wait).is_err() {
            eprintln!("[{}] GPU poll timeout, retrying...", now_iso());
            costs_readback_buf.unmap();
            continue;
        }
        if rx.recv().is_err() {
            eprintln!("[{}] Map callback failed, retrying...", now_iso());
            continue;
        }

        let (min_cost, min_chain) = {
            let data = costs_slice.get_mapped_range();
            let costs: &[u32] = bytemuck::cast_slice(&data);
            let mut best = u32::MAX;
            let mut best_idx = 0u32;
            for (i, &c) in costs.iter().enumerate() {
                if c < best { best = c; best_idx = i as u32; }
            }
            (best, best_idx)
        };
        costs_readback_buf.unmap();

        dispatch_count += 1;

        if min_cost < global_best_cost {
            global_best_cost = min_cost;

            // Read back winning assignment
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
            let cpu_cost = solver_core::evaluate(&assignment, &w8);

            eprintln!(
                "[{}] NEW BEST: {} (chain {}, cpu verify: {}) | {}",
                now_iso(), global_best_cost, min_chain, cpu_cost.total, cost_label(&cpu_cost),
            );

            global_best_assignment = Some(assignment);

            if global_best_cost <= 160 {
                let ts = chrono::Local::now().format("%Y%m%d-%H%M%S%z");
                let filename = format!("{}/{:04}-gpu-{}.tsv", results_dir, global_best_cost, ts);
                let mut out = assignment;
                reassign_commissioners(&mut out);
                let _ = fs::write(&filename, assignment_to_tsv(&out));
                eprintln!("[{}] Saved {}", now_iso(), filename);
            }
        }

        if last_print.elapsed().as_secs() >= 2 {
            let total_iters = dispatch_count * ITERS_PER_DISPATCH as u64 * CHAIN_COUNT as u64;
            let elapsed = start_time.elapsed().as_secs_f64();
            let rate = total_iters as f64 / elapsed / 1e6;
            eprintln!(
                "[{}] dispatch {} | {:.2}B total iters | {:.1}M iters/sec | best: {}",
                now_iso(), dispatch_count, total_iters as f64 / 1e9, rate, global_best_cost,
            );
            last_print = Instant::now();
        }
    }

    if let Some(best) = global_best_assignment {
        eprintln!(
            "[{}] Final best: {} | {}",
            now_iso(), global_best_cost, cost_label(&solver_core::evaluate(&best, &w8)),
        );
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
