use wgpu::util::DeviceExt;

use crate::gpu_types::*;

pub struct GpuResources {
    pub device: wgpu::Device,
    pub queue: wgpu::Queue,
    pub sa_pipeline: wgpu::ComputePipeline,
    pub sa_bg: wgpu::BindGroup,
    pub sa_workgroups: u32,
    pub assign_buf: wgpu::Buffer,
    pub best_assign_buf: wgpu::Buffer,
    pub cost_buf: wgpu::Buffer,
    pub best_cost_buf: wgpu::Buffer,
    pub move_thresh_buf: wgpu::Buffer,
    pub costs_readback_buf: wgpu::Buffer,
    pub assign_readback_buf: wgpu::Buffer,
    pub costs_readback_size: u64,
    pub exchange_pipeline: wgpu::ComputePipeline,
    pub exchange_bg: wgpu::BindGroup,
    pub swap_pairs_buf: wgpu::Buffer,
    pub exchange_params_buf: wgpu::Buffer,
}

pub async fn create_gpu_resources(
    assign_data: &[u32],
    best_assign_data: &[u32],
    cost_data: &[u32],
    best_cost_data: &[u32],
    rng_data: &[u32],
    weights_bytes: &[u8],
    gpu_params: &GpuParams,
    thresh_bytes: &[u8],
    assign_u32s: usize,
    sa_shader_source: &str,
    exchange_shader_source: &str,
) -> GpuResources {
    let chain_count = gpu_params.chain_count;

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

    device.on_uncaptured_error(Box::new(|error| {
        eprintln!("wgpu uncaptured error: {}", error);
    }));

    let sa_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("SA Shader"),
        source: wgpu::ShaderSource::Wgsl(sa_shader_source.into()),
    });

    let assign_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("assignments"),
        contents: bytemuck::cast_slice(assign_data),
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
    });
    let best_assign_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("best_assignments"),
        contents: bytemuck::cast_slice(best_assign_data),
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC | wgpu::BufferUsages::COPY_DST,
    });
    let cost_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("costs"),
        contents: bytemuck::cast_slice(cost_data),
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::COPY_SRC,
    });
    let best_cost_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("best_costs"),
        contents: bytemuck::cast_slice(best_cost_data),
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC | wgpu::BufferUsages::COPY_DST,
    });
    let rng_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("rng_states"),
        contents: bytemuck::cast_slice(rng_data),
        usage: wgpu::BufferUsages::STORAGE,
    });
    let weights_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("weights"),
        contents: weights_bytes,
        usage: wgpu::BufferUsages::UNIFORM,
    });
    let params_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("params"),
        contents: bytemuck::bytes_of(gpu_params),
        usage: wgpu::BufferUsages::UNIFORM,
    });
    let move_thresh_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("move_thresholds"),
        contents: thresh_bytes,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
    });

    let costs_readback_size = (chain_count as usize * 4 * 2) as u64;
    let costs_readback_buf = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("costs_readback"),
        size: costs_readback_size,
        usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    let assign_readback_buf = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("assign_readback"),
        size: (assign_u32s * 4) as u64,
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

    // Exchange (replica swap) pipeline
    let exchange_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("Exchange Shader"),
        source: wgpu::ShaderSource::Wgsl(exchange_shader_source.into()),
    });

    let swap_pairs_buf = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("swap_pairs"),
        size: (MAX_SWAP_PAIRS * 2 * 4) as u64,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });

    let exchange_params_buf = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("exchange_params"),
        size: 16,
        usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });

    let exchange_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("Exchange BGL"),
        entries: &[
            bgl_storage(0, false), bgl_storage(1, false),
            bgl_storage(2, false), bgl_storage(3, false),
            bgl_storage(4, true), bgl_uniform(5),
        ],
    });

    let exchange_bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("Exchange BG"),
        layout: &exchange_bgl,
        entries: &[
            bg_entry(0, &assign_buf), bg_entry(1, &best_assign_buf),
            bg_entry(2, &cost_buf), bg_entry(3, &best_cost_buf),
            bg_entry(4, &swap_pairs_buf), bg_entry(5, &exchange_params_buf),
        ],
    });

    let exchange_pl = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some("Exchange PL"),
        bind_group_layouts: &[&exchange_bgl],
        push_constant_ranges: &[],
    });

    let exchange_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some("Exchange Pipeline"),
        layout: Some(&exchange_pl),
        module: &exchange_shader,
        entry_point: Some("exchange"),
        compilation_options: Default::default(),
        cache: None,
    });

    GpuResources {
        device,
        queue,
        sa_pipeline,
        sa_bg,
        sa_workgroups,
        assign_buf,
        best_assign_buf,
        cost_buf,
        best_cost_buf,
        move_thresh_buf,
        costs_readback_buf,
        assign_readback_buf,
        costs_readback_size,
        exchange_pipeline,
        exchange_bg,
        swap_pairs_buf,
        exchange_params_buf,
    }
}
