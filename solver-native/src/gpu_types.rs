use bytemuck::{Pod, Zeroable};
use rand::rngs::SmallRng;
use rand::Rng;

pub const DEFAULT_CHAIN_COUNT: u32 = 16384;
pub const MIN_CHAIN_COUNT: u32 = 4096;
pub const MAX_CHAIN_COUNT: u32 = 16384;
pub const ITERS_PER_DISPATCH: u32 = 512;
pub const ASSIGN_U32S: usize = 48;

#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
pub struct GpuWeights {
    pub matchup_zero: u32,
    pub matchup_triple: u32,
    pub consecutive_opponents: u32,
    pub early_late_balance: f32,
    pub early_late_alternation: u32,
    pub lane_balance: f32,
    pub lane_switch: f32,
    pub late_lane_balance: f32,
    pub commissioner_overlap: u32,
    pub _pad0: u32,
    pub _pad1: u32,
    pub _pad2: u32,
}

#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
pub struct GpuParams {
    pub iters_per_dispatch: u32,
    pub chain_count: u32,
    pub temp_base: f32,
    pub temp_step: f32,
}

#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
pub struct GpuMoveThresholds {
    pub t: [u32; 12],
}

pub const THRESH_DEFAULT: GpuMoveThresholds = GpuMoveThresholds {
    t: [10, 35, 45, 50, 58, 62, 66, 72, 77, 92, 100, 0],
};
pub const THRESH_HIGH_COST: GpuMoveThresholds = GpuMoveThresholds {
    t: [15, 33, 43, 50, 58, 63, 68, 74, 80, 92, 100, 0],
};
pub const THRESH_LOW_COST: GpuMoveThresholds = GpuMoveThresholds {
    t: [2, 35, 44, 46, 60, 62, 64, 70, 72, 98, 100, 0],
};

pub fn pack_assignment(a: &solver_core::Assignment) -> [u32; ASSIGN_U32S] {
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

pub fn unpack_assignment(packed: &[u32; ASSIGN_U32S]) -> solver_core::Assignment {
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

pub fn detect_chain_count() -> u32 {
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
