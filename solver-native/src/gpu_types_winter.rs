use bytemuck::{Pod, Zeroable};

pub const ASSIGN_U32S: usize = 48;

#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
pub struct GpuWeights {
    pub matchup_zero: u32,
    pub matchup_triple: u32,
    pub consecutive_opponents: u32,
    pub early_late_balance: f32,
    pub early_late_alternation: u32,
    pub early_late_consecutive: u32,
    pub lane_balance: f32,
    pub lane_switch: f32,
    pub late_lane_balance: f32,
    pub commissioner_overlap: u32,
    pub half_season_repeat: u32,
    pub _pad0: u32,
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

pub fn pack_assignment(a: &solver_core::winter::Assignment) -> [u32; ASSIGN_U32S] {
    let mut packed = [0u32; ASSIGN_U32S];
    for w in 0..solver_core::winter::WEEKS {
        for q in 0..solver_core::winter::QUADS {
            let [pa, pb, pc, pd] = a[w][q];
            packed[w * solver_core::winter::QUADS + q] =
                (pa as u32) | ((pb as u32) << 8) | ((pc as u32) << 16) | ((pd as u32) << 24);
        }
    }
    packed
}

pub fn unpack_assignment(packed: &[u32; ASSIGN_U32S]) -> solver_core::winter::Assignment {
    let mut a = [[[0u8; solver_core::winter::POS]; solver_core::winter::QUADS]; solver_core::winter::WEEKS];
    for w in 0..solver_core::winter::WEEKS {
        for q in 0..solver_core::winter::QUADS {
            let v = packed[w * solver_core::winter::QUADS + q];
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
