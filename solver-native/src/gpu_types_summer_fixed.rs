use bytemuck::{Pod, Zeroable};
use solver_core::summer_fixed::*;

/// GPU packing: 31 u32s per chain.
/// u32[0..30]: mapping[w][p] stored as byte (p%4) of u32[w*3 + p/4].
/// u32[30]: bits 0-9 = swap_01[0..10], bits 10-19 = swap_23[0..10].
pub const FIXED_ASSIGN_U32S: usize = 31;

#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
pub struct GpuFixedWeights {
    pub matchup_balance: u32,
    pub slot_balance: u32,
    pub lane_balance: u32,
    pub game5_lane_balance: u32,
    pub same_lane_balance: u32,
    pub commissioner_overlap: u32,
    pub _pad0: u32,
    pub _pad1: u32,
}

#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
pub struct GpuFixedMoveThresholds {
    pub t: [u32; 16],
}

/// 7 GPU moves (same as CPU): tm_swap, tog_01, tog_23, wk_swap, g_match, g_slot, g_lane
pub const FIXED_GPU_NUM_MOVES: usize = 7;

pub const FIXED_THRESH_DEFAULT: GpuFixedMoveThresholds = GpuFixedMoveThresholds {
    t: [15, 28, 41, 54, 70, 85, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100],
};

pub const FIXED_GPU_BASE_WEIGHTS: [f64; FIXED_GPU_NUM_MOVES] = [
    0.15, 0.13, 0.13, 0.13, 0.16, 0.15, 0.15,
];

pub fn pack_fixed_schedule(s: &FixedSchedule) -> [u32; FIXED_ASSIGN_U32S] {
    let mut packed = [0u32; FIXED_ASSIGN_U32S];
    for w in 0..SF_WEEKS {
        for p in 0..SF_TEAMS {
            let idx = w * 3 + p / 4;
            let shift = (p % 4) * 8;
            packed[idx] |= (s.mapping[w][p] as u32) << shift;
        }
    }
    let mut flags = 0u32;
    for w in 0..SF_WEEKS {
        if s.swap_01[w] { flags |= 1 << w; }
        if s.swap_23[w] { flags |= 1 << (w + 10); }
    }
    packed[30] = flags;
    packed
}

pub fn unpack_fixed_schedule(packed: &[u32; FIXED_ASSIGN_U32S]) -> FixedSchedule {
    let mut s = FixedSchedule {
        mapping: [[0; SF_TEAMS]; SF_WEEKS],
        swap_01: [false; SF_WEEKS],
        swap_23: [false; SF_WEEKS],
    };
    for w in 0..SF_WEEKS {
        for p in 0..SF_TEAMS {
            let idx = w * 3 + p / 4;
            let shift = (p % 4) * 8;
            s.mapping[w][p] = ((packed[idx] >> shift) & 0xFF) as u8;
        }
    }
    let flags = packed[30];
    for w in 0..SF_WEEKS {
        s.swap_01[w] = (flags & (1 << w)) != 0;
        s.swap_23[w] = (flags & (1 << (w + 10))) != 0;
    }
    s
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::rngs::SmallRng;
    use rand::SeedableRng;

    #[test]
    fn test_pack_unpack_roundtrip() {
        let mut rng = SmallRng::seed_from_u64(42);
        let w8 = FixedWeights {
            matchup_balance: 80, slot_balance: 60, lane_balance: 60,
            game5_lane_balance: 40, same_lane_balance: 40, commissioner_overlap: 30,
        };
        for _ in 0..20 {
            let sched = random_fixed_schedule(&mut rng);
            let packed = pack_fixed_schedule(&sched);
            let unpacked = unpack_fixed_schedule(&packed);
            let cost_orig = evaluate_fixed(&sched, &w8).total;
            let cost_rt = evaluate_fixed(&unpacked, &w8).total;
            assert_eq!(cost_orig, cost_rt, "Pack/unpack roundtrip changed cost");
            assert_eq!(sched.mapping, unpacked.mapping);
            assert_eq!(sched.swap_01, unpacked.swap_01);
            assert_eq!(sched.swap_23, unpacked.swap_23);
        }
    }
}
