use bytemuck::{Pod, Zeroable};
use solver_core::summer::*;

/// GPU packing: 10 weeks × 5 slots × 4 pairs = 200 u32s per chain.
/// Each u32 holds: left_team | (right_team << 8) | 0xFFFF for empty.
/// Slot 4 only uses pairs 2-3 (pairs 0-1 are always empty/invalid).
pub const SUMMER_ASSIGN_U32S: usize = S_WEEKS * S_SLOTS * S_PAIRS; // 200

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
    pub t: [u32; 16], // 15 moves + 1 padding (must be 16-byte aligned)
}

pub const SUMMER_THRESH_DEFAULT: GpuSummerMoveThresholds = GpuSummerMoveThresholds {
    t: [12, 20, 28, 32, 36, 44, 52, 58, 66, 72, 77, 82, 90, 96, 100, 100],
};

pub fn pack_summer_assignment(a: &SummerAssignment) -> [u32; SUMMER_ASSIGN_U32S] {
    let mut packed = [0u32; SUMMER_ASSIGN_U32S];
    for w in 0..S_WEEKS {
        for s in 0..S_SLOTS {
            for p in 0..S_PAIRS {
                let idx = w * S_SLOTS * S_PAIRS + s * S_PAIRS + p;
                let (t1, t2) = a[w][s][p];
                if t1 == EMPTY {
                    packed[idx] = 0xFFFF;
                } else {
                    packed[idx] = (t1 as u32) | ((t2 as u32) << 8);
                }
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
                let idx = w * S_SLOTS * S_PAIRS + s * S_PAIRS + p;
                let v = packed[idx];
                if v == 0xFFFF {
                    a[w][s][p] = (EMPTY, EMPTY);
                } else {
                    a[w][s][p] = ((v & 0xFF) as u8, ((v >> 8) & 0xFF) as u8);
                }
            }
        }
    }
    a
}
