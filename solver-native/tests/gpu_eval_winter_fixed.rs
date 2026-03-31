//! Integration test: run the WGSL evaluate() on the GPU and compare to CPU.
//! Usage: cargo test -p solver-native --test gpu_eval_winter_fixed -- --nocapture

use bytemuck;
use solver_core::winter_fixed::*;
use solver_native::gpu_types_winter_fixed::*;
use solver_native::gpu_types::*;
use wgpu::util::DeviceExt;

fn make_test_shader() -> String {
    let consts = wgsl_consts();
    let base = include_str!("../src/winter_fixed_solver.wgsl");
    let mut shader = format!("{}\n{}", consts, base);
    shader.push_str(
        r#"

@compute @workgroup_size(1)
fn test_eval(@builtin(global_invocation_id) gid: vec3<u32>) {
    var a: array<u32, 25>;
    for (var i = 0u; i < 25u; i++) {
        a[i] = assignments[gid.x * 25u + i];
    }
    costs[gid.x] = evaluate(&a);
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
    packed: &[u32; WF_ASSIGN_U32S],
    w8: &WinterFixedWeights,
) -> u32 {
    let shader_src = make_test_shader();
    let shader_module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("test_eval"),
        source: wgpu::ShaderSource::Wgsl(shader_src.into()),
    });

    let assign_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("assignments"),
        contents: bytemuck::cast_slice(packed),
        usage: wgpu::BufferUsages::STORAGE,
    });

    let best_assign_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("best_assignments"),
        contents: bytemuck::cast_slice(packed),
        usage: wgpu::BufferUsages::STORAGE,
    });

    let costs_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("costs"),
        contents: bytemuck::cast_slice(&[0u32]),
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
    });

    let best_costs_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("best_costs"),
        contents: bytemuck::cast_slice(&[0u32]),
        usage: wgpu::BufferUsages::STORAGE,
    });

    let rng_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("rng"),
        contents: bytemuck::cast_slice(&[1u32, 2, 3, 4]),
        usage: wgpu::BufferUsages::STORAGE,
    });

    let gpu_w8 = GpuWinterFixedWeights {
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
    let weights_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("weights"),
        contents: bytemuck::bytes_of(&gpu_w8),
        usage: wgpu::BufferUsages::UNIFORM,
    });

    let params_data: [u32; 5] = [
        0, // iters_per_dispatch
        1, // chain_count
        0, // temp_base
        0, // temp_step
        1, // pod_size
    ];
    let params_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("params"),
        contents: bytemuck::cast_slice(&params_data),
        usage: wgpu::BufferUsages::UNIFORM,
    });

    let thresh_data = GpuWinterFixedMoveThresholds { t: [30, 40, 46, 54, 62, 72, 90, 100] };
    let thresh_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("thresh"),
        contents: bytemuck::bytes_of(&thresh_data),
        usage: wgpu::BufferUsages::STORAGE,
    });

    let readback_buf = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("readback"),
        size: 4,
        usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });

    let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("test_bgl"),
        entries: &[
            bgl_entry(0, wgpu::BufferBindingType::Storage { read_only: false }),
            bgl_entry(1, wgpu::BufferBindingType::Storage { read_only: false }),
            bgl_entry(2, wgpu::BufferBindingType::Storage { read_only: false }),
            bgl_entry(3, wgpu::BufferBindingType::Storage { read_only: false }),
            bgl_entry(4, wgpu::BufferBindingType::Storage { read_only: false }),
            bgl_entry(5, wgpu::BufferBindingType::Uniform),
            bgl_entry(6, wgpu::BufferBindingType::Uniform),
            bgl_entry(7, wgpu::BufferBindingType::Storage { read_only: true }),
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

fn bgl_entry(binding: u32, ty: wgpu::BufferBindingType) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry {
        binding,
        visibility: wgpu::ShaderStages::COMPUTE,
        ty: wgpu::BindingType::Buffer {
            ty,
            has_dynamic_offset: false,
            min_binding_size: None,
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

    let w8 = WinterFixedWeights {
        matchup_zero: 60,
        matchup_triple: 60,
        consecutive_opponents: 10,
        early_late_balance: 60.0,
        early_late_alternation: 40,
        early_late_consecutive: 5,
        lane_balance: 15.0,
        lane_switch: 10.0,
        late_lane_balance: 15.0,
        commissioner_overlap: 30,
        half_season_repeat: 10,
    };

    use rand::SeedableRng;
    use rand::rngs::SmallRng;

    let mut mismatches = 0;
    for seed in 0..20u64 {
        let mut rng = SmallRng::seed_from_u64(seed);
        let sched = random_fixed_schedule(&mut rng);
        let cpu_bd = evaluate_fixed(&sched, &w8);
        let cpu_cost = cpu_bd.total;

        let packed = pack_fixed_schedule(&sched);
        let gpu_cost = run_gpu_evaluate(&device, &queue, &packed, &w8);

        let status = if cpu_cost == gpu_cost { "OK" } else { mismatches += 1; "MISMATCH" };
        println!(
            "seed={:2}: cpu={:5} gpu={:5} diff={:+5} {} (el_alt={} el_con={})",
            seed, cpu_cost, gpu_cost, gpu_cost as i64 - cpu_cost as i64, status,
            cpu_bd.early_late_alternation, cpu_bd.early_late_consecutive,
        );
    }
    assert_eq!(mismatches, 0, "{} mismatches found", mismatches);
}

/// Run the real SA main entry point for iterations, read back, verify costs match CPU eval.
#[test]
fn test_gpu_sa_cost_drift() {
    let (device, queue) = create_device();

    let w8 = WinterFixedWeights {
        matchup_zero: 60, matchup_triple: 60, consecutive_opponents: 10,
        early_late_balance: 60.0, early_late_alternation: 40, early_late_consecutive: 5,
        lane_balance: 15.0, lane_switch: 10.0, late_lane_balance: 15.0,
        commissioner_overlap: 30, half_season_repeat: 10,
    };

    use rand::SeedableRng;
    use rand::rngs::SmallRng;
    use rand::Rng;

    let chain_count = 256u32;
    let iters_per_dispatch = 2000u32;
    let n = chain_count as usize;
    let assign_u32s = WF_ASSIGN_U32S;
    let mut rng = SmallRng::seed_from_u64(42);

    let shader_src = format!("{}\n{}", wgsl_consts(), include_str!("../src/winter_fixed_solver.wgsl"));
    let shader_module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("sa_test"),
        source: wgpu::ShaderSource::Wgsl(shader_src.into()),
    });

    let mut assign_data = vec![0u32; n * assign_u32s];
    let mut cost_data = vec![0u32; n];
    let mut rng_data = vec![0u32; n * 4];

    for i in 0..n {
        let sched = random_fixed_schedule(&mut rng);
        let cost = evaluate_fixed(&sched, &w8).total;
        let packed = pack_fixed_schedule(&sched);
        assign_data[i * assign_u32s..(i + 1) * assign_u32s].copy_from_slice(&packed);
        cost_data[i] = cost;
        let sv = (i as u64).wrapping_mul(0x9E3779B97F4A7C15) ^ 0xBF58476D1CE4E5B9;
        rng_data[i * 4] = (sv & 0xFFFFFFFF) as u32 | 1;
        rng_data[i * 4 + 1] = ((sv >> 32) & 0xFFFFFFFF) as u32 | 1;
        rng_data[i * 4 + 2] = rng.random::<u32>() | 1;
        rng_data[i * 4 + 3] = rng.random::<u32>() | 1;
    }

    let assign_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("assignments"), contents: bytemuck::cast_slice(&assign_data),
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
    });
    let best_assign_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("best_assignments"), contents: bytemuck::cast_slice(&assign_data),
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
    });
    let costs_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("costs"), contents: bytemuck::cast_slice(&cost_data),
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
    });
    let best_costs_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("best_costs"), contents: bytemuck::cast_slice(&cost_data),
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
    });
    let rng_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("rng"), contents: bytemuck::cast_slice(&rng_data),
        usage: wgpu::BufferUsages::STORAGE,
    });

    let gpu_w8 = GpuWinterFixedWeights {
        matchup_zero: w8.matchup_zero, matchup_triple: w8.matchup_triple,
        consecutive_opponents: w8.consecutive_opponents,
        early_late_balance: w8.early_late_balance as f32,
        early_late_alternation: w8.early_late_alternation,
        early_late_consecutive: w8.early_late_consecutive,
        lane_balance: w8.lane_balance as f32, lane_switch: w8.lane_switch as f32,
        late_lane_balance: w8.late_lane_balance as f32,
        commissioner_overlap: w8.commissioner_overlap,
        half_season_repeat: w8.half_season_repeat, _pad0: 0,
    };
    let weights_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("weights"), contents: bytemuck::bytes_of(&gpu_w8),
        usage: wgpu::BufferUsages::UNIFORM,
    });

    let temp_base: f32 = GPU_TEMP_MIN as f32;
    let temp_step: f32 = GPU_TEMP_MAX as f32;
    let pod_size: u32 = POD_SIZE as u32;
    let params_data: [u32; 5] = [
        iters_per_dispatch, chain_count,
        temp_base.to_bits(), temp_step.to_bits(), pod_size,
    ];
    let params_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("params"), contents: bytemuck::cast_slice(&params_data),
        usage: wgpu::BufferUsages::UNIFORM,
    });

    let thresh_data = WF_THRESH_DEFAULT;
    let thresh_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("thresh"), contents: bytemuck::bytes_of(&thresh_data),
        usage: wgpu::BufferUsages::STORAGE,
    });

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
        label: Some("assign_rb"), size: (n * assign_u32s * 4) as u64,
        usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    let best_assign_readback = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("best_assign_rb"), size: (n * assign_u32s * 4) as u64,
        usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });

    let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("sa_bgl"),
        entries: &[
            bgl_entry(0, wgpu::BufferBindingType::Storage { read_only: false }),
            bgl_entry(1, wgpu::BufferBindingType::Storage { read_only: false }),
            bgl_entry(2, wgpu::BufferBindingType::Storage { read_only: false }),
            bgl_entry(3, wgpu::BufferBindingType::Storage { read_only: false }),
            bgl_entry(4, wgpu::BufferBindingType::Storage { read_only: false }),
            bgl_entry(5, wgpu::BufferBindingType::Uniform),
            bgl_entry(6, wgpu::BufferBindingType::Uniform),
            bgl_entry(7, wgpu::BufferBindingType::Storage { read_only: true }),
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

    // Run SA
    let mut encoder = device.create_command_encoder(&Default::default());
    {
        let mut pass = encoder.begin_compute_pass(&Default::default());
        pass.set_pipeline(&pipeline);
        pass.set_bind_group(0, &bind_group, &[]);
        pass.dispatch_workgroups(1, 1, 1);
    }
    encoder.copy_buffer_to_buffer(&costs_buf, 0, &cost_readback, 0, per_array_size);
    encoder.copy_buffer_to_buffer(&best_costs_buf, 0, &best_cost_readback, 0, per_array_size);
    encoder.copy_buffer_to_buffer(&assign_buf, 0, &assign_readback, 0, (n * assign_u32s * 4) as u64);
    encoder.copy_buffer_to_buffer(&best_assign_buf, 0, &best_assign_readback, 0, (n * assign_u32s * 4) as u64);
    queue.submit(Some(encoder.finish()));

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
        let mut cur_packed = [0u32; WF_ASSIGN_U32S];
        cur_packed.copy_from_slice(&cur_assign_all[i * assign_u32s..(i + 1) * assign_u32s]);
        let cur_sched = unpack_fixed_schedule(&cur_packed);
        let cpu_cur = evaluate_fixed(&cur_sched, &w8).total;

        let mut best_packed = [0u32; WF_ASSIGN_U32S];
        best_packed.copy_from_slice(&best_assign_all[i * assign_u32s..(i + 1) * assign_u32s]);
        let best_sched = unpack_fixed_schedule(&best_packed);
        let cpu_best_bd = evaluate_fixed(&best_sched, &w8);
        let cpu_best = cpu_best_bd.total;

        if gpu_cur_costs[i] != cpu_cur || gpu_best_costs[i] != cpu_best {
            println!(
                "chain {:3}: cur gpu={:5} cpu={:5} diff={:+5} | best gpu={:5} cpu={:5} diff={:+5} (el_con={})",
                i, gpu_cur_costs[i], cpu_cur, gpu_cur_costs[i] as i64 - cpu_cur as i64,
                gpu_best_costs[i], cpu_best, gpu_best_costs[i] as i64 - cpu_best as i64,
                cpu_best_bd.early_late_consecutive,
            );
            mismatches += 1;
        }
    }
    if mismatches == 0 {
        println!("All {} chains OK after {} SA iterations each", n, iters_per_dispatch);
    }
    assert_eq!(mismatches, 0, "{}/{} chains have cost drift", mismatches, n);
}
