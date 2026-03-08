use bytemuck::{Pod, Zeroable};
use solver_core::summer::*;

/// GPU packing: 200 logical positions packed 4 per u32 = 50 u32s per chain.
/// Each 8-bit byte holds: left_team[3:0] | right_team[7:4], or 0xFF for empty.
/// Teams 0-11, EMPTY = 0xF. Logical index i → byte (i%4) of u32[i/4].
pub const SUMMER_ASSIGN_U32S: usize = (S_WEEKS * S_SLOTS * S_PAIRS + 3) / 4; // 50

#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
pub struct GpuSummerWeights {
    pub matchup_balance: u32,
    pub lane_switch_consecutive: u32,
    pub lane_switch_post_break: u32,
    pub time_gap_large: u32,
    pub time_gap_consecutive: u32,
    pub lane_balance: u32,
    pub commissioner_overlap: u32,
    pub repeat_matchup_same_night: u32,
    pub slot_balance: u32,
    pub _pad0: u32,
    pub _pad1: u32,
    pub _pad2: u32,
}

#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
pub struct GpuSummerMoveThresholds {
    pub t: [u32; 16], // 11 GPU moves + 5 padding (must be 16-byte aligned)
}

// 11 GPU moves: team_swap(14) matchup_swap(8) opponent_swap(8) lane_swap_week(4)
// slot_swap(4) guided_matchup(14) guided_lane(12) guided_slot(10)
// guided_lane_switch(12) pair_swap_in_slot(6) guided_break_fix(8)
pub const SUMMER_THRESH_DEFAULT: GpuSummerMoveThresholds = GpuSummerMoveThresholds {
    t: [14, 22, 30, 34, 38, 52, 64, 74, 86, 92, 100, 100, 100, 100, 100, 100],
};

/// Number of GPU moves (subset of CPU moves)
pub const GPU_NUM_MOVES: usize = 11;

/// Mapping from GPU move index to CPU move index
pub const GPU_TO_CPU_MOVE: [usize; GPU_NUM_MOVES] = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 12];

/// Base weights for GPU adaptive thresholds
pub const GPU_BASE_WEIGHTS: [f64; GPU_NUM_MOVES] = [
    0.14, 0.08, 0.08, 0.04, 0.04, 0.14, 0.12, 0.10, 0.12, 0.06, 0.08,
];

pub fn pack_summer_assignment(a: &SummerAssignment) -> [u32; SUMMER_ASSIGN_U32S] {
    let mut packed = [0u32; SUMMER_ASSIGN_U32S];
    for w in 0..S_WEEKS {
        for s in 0..S_SLOTS {
            for p in 0..S_PAIRS {
                let li = w * S_SLOTS * S_PAIRS + s * S_PAIRS + p;
                let (t1, t2) = a[w][s][p];
                let val = if t1 == EMPTY {
                    0xFFu32
                } else {
                    (t1 as u32 & 0xF) | ((t2 as u32 & 0xF) << 4)
                };
                let phys = li / 4;
                let shift = (li % 4) * 8;
                packed[phys] |= val << shift;
            }
        }
    }
    packed
}

pub fn unpack_summer_assignment(packed: &[u32; SUMMER_ASSIGN_U32S]) -> SummerAssignment {
    let mut a = [[[(EMPTY, EMPTY); S_PAIRS]; S_SLOTS]; S_WEEKS];
    for w in 0..S_WEEKS {
        for s in 0..S_SLOTS {
            for p in 0..S_PAIRS {
                let li = w * S_SLOTS * S_PAIRS + s * S_PAIRS + p;
                let phys = li / 4;
                let shift = (li % 4) * 8;
                let v = (packed[phys] >> shift) & 0xFF;
                if v == 0xFF {
                    a[w][s][p] = (EMPTY, EMPTY);
                } else {
                    a[w][s][p] = ((v & 0xF) as u8, ((v >> 4) & 0xF) as u8);
                }
            }
        }
    }
    a
}
