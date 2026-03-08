//! Integration test: run the WGSL evaluate() on the GPU and compare to CPU.
//! Usage: cargo test -p solver-native --test gpu_eval_summer -- --nocapture

use bytemuck;
use solver_core::summer::*;
use solver_native::gpu_types_summer::unpack_summer_assignment;
use wgpu::util::DeviceExt;

/// Test shader with full SA code: keeps ALL functions to match real compilation.
/// Adds a second entry point for eval-only testing.
fn make_test_shader() -> String {
    let base = include_str!("../src/summer_solver.wgsl");
    // Add a second entry point after the existing main
    let mut shader = base.to_string();
    shader.push_str(
        r#"

@compute @workgroup_size(1)
fn test_eval(@builtin(global_invocation_id) gid: vec3<u32>) {
    var a: array<u32, 200>;
    for (var i = 0u; i < 200u; i++) {
        a[i] = assignments[i];
    }
    costs[0] = evaluate(&a);
}
"#,
    );
    shader
}

fn create_device() -> (wgpu::Device, wgpu::Queue) {
    let instance = wgpu::Instance::new(&wgpu::InstanceDescriptor::default());
    let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
        power_preference: wgpu::PowerPreference::HighPerformance,
        ..Default::default()
    }))
    .expect("no GPU adapter");

    pollster::block_on(adapter.request_device(
        &wgpu::DeviceDescriptor {
            label: Some("test"),
            required_features: wgpu::Features::empty(),
            required_limits: wgpu::Limits::default(),
            ..Default::default()
        },
    ))
    .expect("failed to create device")
}

fn run_gpu_evaluate(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    packed: &[u32; 200],
    w8: &SummerWeights,
) -> u32 {
    let shader_src = make_test_shader();
    let shader_module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("test_eval"),
        source: wgpu::ShaderSource::Wgsl(shader_src.into()),
    });

    // Buffers
    let assign_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("assignments"),
        contents: bytemuck::cast_slice(packed),
        usage: wgpu::BufferUsages::STORAGE,
    });

    // best_assignments - dummy
    let best_assign_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("best_assignments"),
        contents: bytemuck::cast_slice(packed),
        usage: wgpu::BufferUsages::STORAGE,
    });

    // costs buffer - we read back from this
    let costs_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("costs"),
        contents: bytemuck::cast_slice(&[0u32]),
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
    });

    // best_costs - dummy
    let best_costs_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("best_costs"),
        contents: bytemuck::cast_slice(&[0u32]),
        usage: wgpu::BufferUsages::STORAGE,
    });

    // rng - dummy
    let rng_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("rng"),
        contents: bytemuck::cast_slice(&[1u32, 2, 3, 4]),
        usage: wgpu::BufferUsages::STORAGE,
    });

    // weights uniform - must match GpuSummerWeights layout (12 u32s with padding)
    let weights_data: [u32; 12] = [
        w8.matchup_balance,
        w8.lane_switch_consecutive,
        w8.lane_switch_post_break,
        w8.time_gap_large,
        w8.time_gap_consecutive,
        w8.lane_balance,
        w8.commissioner_overlap,
        w8.repeat_matchup_same_night,
        w8.slot_balance,
        0, // pad0
        0, // pad1
        0, // pad2
    ];
    let weights_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("weights"),
        contents: bytemuck::cast_slice(&weights_data),
        usage: wgpu::BufferUsages::UNIFORM,
    });

    // params uniform
    let params_data: [u32; 5] = [
        0, // iters_per_dispatch (unused in test)
        1, // chain_count
        0, // temp_base (f32 bits, unused)
        0, // temp_step (f32 bits, unused)
        1, // pod_size
    ];
    let params_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("params"),
        contents: bytemuck::cast_slice(&params_data),
        usage: wgpu::BufferUsages::UNIFORM,
    });

    // move_thresh
    let thresh_data = [12u32, 20, 28, 32, 36, 44, 52, 58, 66, 72, 77, 82, 90, 96, 100, 100];
    let thresh_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("thresh"),
        contents: bytemuck::cast_slice(&thresh_data),
        usage: wgpu::BufferUsages::STORAGE,
    });

    // Readback buffer
    let readback_buf = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("readback"),
        size: 4,
        usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });

    let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("test_bgl"),
        entries: &[
            bgl_entry(0, wgpu::BufferBindingType::Storage { read_only: false }, None),
            bgl_entry(1, wgpu::BufferBindingType::Storage { read_only: false }, None),
            bgl_entry(2, wgpu::BufferBindingType::Storage { read_only: false }, None),
            bgl_entry(3, wgpu::BufferBindingType::Storage { read_only: false }, None),
            bgl_entry(4, wgpu::BufferBindingType::Storage { read_only: false }, None),
            bgl_entry(5, wgpu::BufferBindingType::Uniform, None),
            bgl_entry(6, wgpu::BufferBindingType::Uniform, None),
            bgl_entry(7, wgpu::BufferBindingType::Storage { read_only: true }, None),
        ],
    });

    let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some("test_pl"),
        bind_group_layouts: &[&bind_group_layout],
        push_constant_ranges: &[],
    });

    let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some("test_pipeline"),
        layout: Some(&pipeline_layout),
        module: &shader_module,
        entry_point: Some("test_eval"),
        compilation_options: Default::default(),
        cache: None,
    });

    let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("test_bg"),
        layout: &bind_group_layout,
        entries: &[
            bg_entry(0, &assign_buf),
            bg_entry(1, &best_assign_buf),
            bg_entry(2, &costs_buf),
            bg_entry(3, &best_costs_buf),
            bg_entry(4, &rng_buf),
            bg_entry(5, &weights_buf),
            bg_entry(6, &params_buf),
            bg_entry(7, &thresh_buf),
        ],
    });

    let mut encoder = device.create_command_encoder(&Default::default());
    {
        let mut pass = encoder.begin_compute_pass(&Default::default());
        pass.set_pipeline(&pipeline);
        pass.set_bind_group(0, &bind_group, &[]);
        pass.dispatch_workgroups(1, 1, 1);
    }
    encoder.copy_buffer_to_buffer(&costs_buf, 0, &readback_buf, 0, 4);
    queue.submit(Some(encoder.finish()));

    let slice = readback_buf.slice(..);
    slice.map_async(wgpu::MapMode::Read, |_| {});
    device.poll(wgpu::PollType::Wait).unwrap();
    let data = slice.get_mapped_range();
    let result = bytemuck::cast_slice::<u8, u32>(&data)[0];
    drop(data);
    readback_buf.unmap();
    result
}

fn bgl_entry(
    binding: u32,
    ty: wgpu::BufferBindingType,
    min_size: Option<std::num::NonZeroU64>,
) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry {
        binding,
        visibility: wgpu::ShaderStages::COMPUTE,
        ty: wgpu::BindingType::Buffer {
            ty,
            has_dynamic_offset: false,
            min_binding_size: min_size,
        },
        count: None,
    }
}

fn bg_entry(binding: u32, buf: &wgpu::Buffer) -> wgpu::BindGroupEntry<'_> {
    wgpu::BindGroupEntry {
        binding,
        resource: buf.as_entire_binding(),
    }
}

#[test]
fn test_gpu_vs_cpu_evaluate() {
    let (device, queue) = create_device();

    let w8 = SummerWeights {
        matchup_balance: 80,
        lane_switch_consecutive: 60,
        lane_switch_post_break: 20,
        time_gap_large: 60,
        time_gap_consecutive: 30,
        lane_balance: 60,
        commissioner_overlap: 30,
        repeat_matchup_same_night: 30,
        slot_balance: 30,
    };

    use rand::SeedableRng;
    use rand::rngs::SmallRng;

    for seed in 0..20u64 {
        let mut rng = SmallRng::seed_from_u64(seed);
        let assignment = random_summer_assignment(&mut rng);
        let cpu_bd = evaluate_summer(&assignment, &w8);
        let cpu_cost = cpu_bd.total;

        // Pack for GPU
        let mut packed = [0u32; 200];
        for w in 0..S_WEEKS {
            for s in 0..S_SLOTS {
                for p in 0..S_PAIRS {
                    let idx = w * S_SLOTS * S_PAIRS + s * S_PAIRS + p;
                    let (t1, t2) = assignment[w][s][p];
                    if t1 == EMPTY {
                        packed[idx] = 0xFFFF;
                    } else {
                        packed[idx] = (t1 as u32) | ((t2 as u32) << 8);
                    }
                }
            }
        }

        let gpu_cost = run_gpu_evaluate(&device, &queue, &packed, &w8);

        println!(
            "seed={:2}: cpu={} gpu={} {}",
            seed,
            cpu_cost,
            gpu_cost,
            if cpu_cost == gpu_cost { "OK" } else { "MISMATCH" }
        );
        assert_eq!(
            cpu_cost, gpu_cost,
            "seed={}: cpu={} gpu={}",
            seed, cpu_cost, gpu_cost
        );
    }
}

/// Run the real SA main entry point for a few iterations, then read back
/// the assignment + cost and verify they agree.
#[test]
fn test_gpu_sa_cost_drift() {
    let (device, queue) = create_device();

    let w8 = SummerWeights {
        matchup_balance: 80,
        lane_switch_consecutive: 60,
        lane_switch_post_break: 20,
        time_gap_large: 60,
        time_gap_consecutive: 30,
        lane_balance: 60,
        commissioner_overlap: 30,
        repeat_matchup_same_night: 30,
        slot_balance: 30,
    };

    use rand::SeedableRng;
    use rand::rngs::SmallRng;
    use rand::Rng;

    let chain_count = 256u32;
    let iters_per_dispatch = 2000u32;

    // Use full shader with real main entry point
    let shader_src = include_str!("../src/summer_solver.wgsl");
    let shader_module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("sa_test"),
        source: wgpu::ShaderSource::Wgsl(shader_src.into()),
    });

    let n = chain_count as usize;
    let mut rng = SmallRng::seed_from_u64(42);

    // Initialize all chains
    let mut assign_data = vec![0u32; n * 200];
    let mut cost_data = vec![0u32; n];
    let mut rng_data = vec![0u32; n * 4];

    for i in 0..n {
        let a = random_summer_assignment(&mut rng);
        let cost = evaluate_summer(&a, &w8).total;
        let mut packed = [0u32; 200];
        for w in 0..S_WEEKS {
            for s in 0..S_SLOTS {
                for p in 0..S_PAIRS {
                    let idx = w * S_SLOTS * S_PAIRS + s * S_PAIRS + p;
                    let (t1, t2) = a[w][s][p];
                    if t1 == EMPTY { packed[idx] = 0xFFFF; }
                    else { packed[idx] = (t1 as u32) | ((t2 as u32) << 8); }
                }
            }
        }
        assign_data[i * 200..(i + 1) * 200].copy_from_slice(&packed);
        cost_data[i] = cost;
        let sv = (i as u64).wrapping_mul(0x9E3779B97F4A7C15) ^ 0xBF58476D1CE4E5B9;
        rng_data[i * 4] = (sv & 0xFFFFFFFF) as u32 | 1;
        rng_data[i * 4 + 1] = ((sv >> 32) & 0xFFFFFFFF) as u32 | 1;
        rng_data[i * 4 + 2] = rng.random::<u32>() | 1;
        rng_data[i * 4 + 3] = rng.random::<u32>() | 1;
    }

    let assign_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("assignments"),
        contents: bytemuck::cast_slice(&assign_data),
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
    });
    let best_assign_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("best_assignments"),
        contents: bytemuck::cast_slice(&assign_data),
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
    });
    let costs_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("costs"),
        contents: bytemuck::cast_slice(&cost_data),
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
    });
    let best_costs_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("best_costs"),
        contents: bytemuck::cast_slice(&cost_data),
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
    });
    let rng_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("rng"),
        contents: bytemuck::cast_slice(&rng_data),
        usage: wgpu::BufferUsages::STORAGE,
    });
    let weights_data: [u32; 12] = [
        w8.matchup_balance, w8.lane_switch_consecutive, w8.lane_switch_post_break,
        w8.time_gap_large, w8.time_gap_consecutive,
        w8.lane_balance, w8.commissioner_overlap, w8.repeat_matchup_same_night,
        w8.slot_balance, 0, 0, 0,
    ];
    let weights_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("weights"),
        contents: bytemuck::cast_slice(&weights_data),
        usage: wgpu::BufferUsages::UNIFORM,
    });
    let temp_base: f32 = 50.0;
    let temp_step: f32 = 50.0;
    let pod_size: u32 = 8;
    let params_data: [u32; 5] = [
        iters_per_dispatch, chain_count,
        temp_base.to_bits(), temp_step.to_bits(), pod_size,
    ];
    let params_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("params"),
        contents: bytemuck::cast_slice(&params_data),
        usage: wgpu::BufferUsages::UNIFORM,
    });
    let thresh_data = [12u32, 20, 28, 32, 36, 44, 52, 58, 66, 72, 77, 82, 90, 96, 100, 100];
    let thresh_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("thresh"),
        contents: bytemuck::cast_slice(&thresh_data),
        usage: wgpu::BufferUsages::STORAGE,
    });

    // Readback buffers for ALL chains
    let per_array_size = (n * 4) as u64;
    let cost_readback = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("cost_rb"), size: per_array_size,
        usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    let best_cost_readback = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("best_cost_rb"), size: per_array_size,
        usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    let assign_readback = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("assign_rb"), size: (n * 200 * 4) as u64,
        usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    let best_assign_readback = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("best_assign_rb"), size: (n * 200 * 4) as u64,
        usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });

    let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("sa_bgl"),
        entries: &[
            bgl_entry(0, wgpu::BufferBindingType::Storage { read_only: false }, None),
            bgl_entry(1, wgpu::BufferBindingType::Storage { read_only: false }, None),
            bgl_entry(2, wgpu::BufferBindingType::Storage { read_only: false }, None),
            bgl_entry(3, wgpu::BufferBindingType::Storage { read_only: false }, None),
            bgl_entry(4, wgpu::BufferBindingType::Storage { read_only: false }, None),
            bgl_entry(5, wgpu::BufferBindingType::Uniform, None),
            bgl_entry(6, wgpu::BufferBindingType::Uniform, None),
            bgl_entry(7, wgpu::BufferBindingType::Storage { read_only: true }, None),
        ],
    });
    let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some("sa_pl"), bind_group_layouts: &[&bind_group_layout], push_constant_ranges: &[],
    });
    let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some("sa_pipeline"), layout: Some(&pipeline_layout),
        module: &shader_module, entry_point: Some("main"),
        compilation_options: Default::default(), cache: None,
    });
    let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("sa_bg"), layout: &bind_group_layout,
        entries: &[
            bg_entry(0, &assign_buf), bg_entry(1, &best_assign_buf),
            bg_entry(2, &costs_buf), bg_entry(3, &best_costs_buf),
            bg_entry(4, &rng_buf), bg_entry(5, &weights_buf),
            bg_entry(6, &params_buf), bg_entry(7, &thresh_buf),
        ],
    });

    // Run SA with 1 workgroup of 256 threads
    let mut encoder = device.create_command_encoder(&Default::default());
    {
        let mut pass = encoder.begin_compute_pass(&Default::default());
        pass.set_pipeline(&pipeline);
        pass.set_bind_group(0, &bind_group, &[]);
        pass.dispatch_workgroups(1, 1, 1);
    }
    encoder.copy_buffer_to_buffer(&costs_buf, 0, &cost_readback, 0, per_array_size);
    encoder.copy_buffer_to_buffer(&best_costs_buf, 0, &best_cost_readback, 0, per_array_size);
    encoder.copy_buffer_to_buffer(&assign_buf, 0, &assign_readback, 0, (n * 200 * 4) as u64);
    encoder.copy_buffer_to_buffer(&best_assign_buf, 0, &best_assign_readback, 0, (n * 200 * 4) as u64);
    queue.submit(Some(encoder.finish()));

    // Read back
    let cur_cost_slice = cost_readback.slice(..);
    let best_cost_slice = best_cost_readback.slice(..);
    let cur_assign_slice = assign_readback.slice(..);
    let best_assign_slice = best_assign_readback.slice(..);
    cur_cost_slice.map_async(wgpu::MapMode::Read, |_| {});
    best_cost_slice.map_async(wgpu::MapMode::Read, |_| {});
    cur_assign_slice.map_async(wgpu::MapMode::Read, |_| {});
    best_assign_slice.map_async(wgpu::MapMode::Read, |_| {});
    device.poll(wgpu::PollType::Wait).unwrap();

    let gpu_cur_costs: Vec<u32> = bytemuck::cast_slice::<u8, u32>(&cur_cost_slice.get_mapped_range()).to_vec();
    let gpu_best_costs: Vec<u32> = bytemuck::cast_slice::<u8, u32>(&best_cost_slice.get_mapped_range()).to_vec();
    let cur_assign_all: Vec<u32> = bytemuck::cast_slice::<u8, u32>(&cur_assign_slice.get_mapped_range()).to_vec();
    let best_assign_all: Vec<u32> = bytemuck::cast_slice::<u8, u32>(&best_assign_slice.get_mapped_range()).to_vec();

    let mut mismatches = 0;
    for i in 0..n {
        // Check current
        let mut cur_packed = [0u32; 200];
        cur_packed.copy_from_slice(&cur_assign_all[i * 200..(i + 1) * 200]);
        let cur_assignment = unpack_summer_assignment(&cur_packed);
        let cpu_cur = evaluate_summer(&cur_assignment, &w8).total;

        // Check best
        let mut best_packed = [0u32; 200];
        best_packed.copy_from_slice(&best_assign_all[i * 200..(i + 1) * 200]);
        let best_assignment = unpack_summer_assignment(&best_packed);
        let cpu_best = evaluate_summer(&best_assignment, &w8).total;

        if gpu_cur_costs[i] != cpu_cur || gpu_best_costs[i] != cpu_best {
            println!(
                "chain {}: cur gpu={} cpu={} diff={} | best gpu={} cpu={} diff={}",
                i, gpu_cur_costs[i], cpu_cur, gpu_cur_costs[i] as i64 - cpu_cur as i64,
                gpu_best_costs[i], cpu_best, gpu_best_costs[i] as i64 - cpu_best as i64,
            );
            mismatches += 1;
        }
    }
    if mismatches == 0 {
        println!("All {} chains OK after {} iterations", n, iters_per_dispatch);
    }
    assert_eq!(mismatches, 0, "{} chains had mismatches", mismatches);
}
