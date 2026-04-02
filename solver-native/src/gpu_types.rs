use bytemuck::{Pod, Zeroable};
use rand::rngs::SmallRng;
use rand::Rng;

pub const DEFAULT_CHAIN_COUNT: u32 = 16384;
pub const MIN_CHAIN_COUNT: u32 = 4096;
pub const MAX_CHAIN_COUNT: u32 = 65536;
pub const ITERS_PER_DISPATCH: u32 = 128;

#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
pub struct GpuParams {
    pub iters_per_dispatch: u32,
    pub chain_count: u32,
    pub temp_base: f32,
    pub temp_step: f32,
    pub pod_size: u32,
    pub _pad0: u32,
    pub _pad1: u32,
    pub _pad2: u32,
}

pub const MAX_SWAP_PAIRS: usize = 32768;

pub const GPU_TEMP_MIN: f64 = 3.0;
pub const GPU_TEMP_MAX: f64 = 30.0;
pub const CPU_TEMP_MIN: f64 = 10.0;
pub const CPU_TEMP_MAX: f64 = 15.0;
pub const TEMP_LEVELS: usize = 256;
pub const POD_SIZE: usize = 8;

/// Geometric spacing: uniform 1/T gaps for better replica exchange acceptance.
pub fn temp_for_level(level: usize) -> f64 {
    let t_frac = level as f64 / (POD_SIZE - 1).max(1) as f64;
    GPU_TEMP_MIN * (GPU_TEMP_MAX / GPU_TEMP_MIN).powf(t_frac)
}

pub fn detect_chain_count(assign_u32s: usize) -> u32 {
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
    let bytes_per_chain = (assign_u32s as u64) * 4;
    let max_chains = limits.max_storage_buffer_binding_size as u64 / bytes_per_chain;
    let chain_count = max_chains.min(MAX_CHAIN_COUNT as u64).max(MIN_CHAIN_COUNT as u64) as u32;
    (chain_count / 256) * 256
}

pub fn sampled_median(costs: &[u32], rng: &mut SmallRng) -> u32 {
    let n = costs.len();
    if n == 0 { return u32::MAX; }
    let sample_size = 64.min(n);
    let mut samples: Vec<u32> = (0..sample_size).map(|_| costs[rng.random_range(0..n)]).collect();
    samples.sort_unstable();
    samples[samples.len() / 2]
}

pub fn bgl_storage(binding: u32, read_only: bool) -> wgpu::BindGroupLayoutEntry {
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

pub fn bgl_uniform(binding: u32) -> wgpu::BindGroupLayoutEntry {
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

pub fn bg_entry(binding: u32, buffer: &wgpu::Buffer) -> wgpu::BindGroupEntry<'_> {
    wgpu::BindGroupEntry {
        binding,
        resource: buffer.as_entire_binding(),
    }
}
