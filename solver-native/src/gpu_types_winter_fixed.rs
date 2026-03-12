use bytemuck::{Pod, Zeroable};
use solver_core::winter_fixed::*;

/// GPU packing: 25 u32s per chain.
/// u32[0..24]: mapping[w][pos] stored as 4-bit nibbles, 8 positions per u32.
///   For week w, positions 0..15 are packed into 2 u32s:
///     u32[w*2]   = positions 0-7  (4 bits each)
///     u32[w*2+1] = positions 8-15 (4 bits each)
/// u32[24]: bits 0-11 = lane_swap_early[0..12], bits 12-23 = lane_swap_late[0..12].
pub const WF_ASSIGN_U32S: usize = 25;

#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
pub struct GpuWinterFixedWeights {
    pub matchup_zero: u32,
    pub matchup_triple: u32,
    pub consecutive_opponents: u32,
    pub early_late_balance: f32,
    pub early_late_alternation: u32,
    pub lane_balance: f32,
    pub lane_switch: f32,
    pub late_lane_balance: f32,
    pub commissioner_overlap: u32,
    pub half_season_repeat: u32,
    pub _pad0: u32,
    pub _pad1: u32,
}

#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
pub struct GpuWinterFixedMoveThresholds {
    pub t: [u32; 8],
}

pub const WF_GPU_NUM_MOVES: usize = 8;

/// 8 moves: pos_swap, cross_wk, wk_swap, tog_e, tog_l, g_match, g_lane, g_el
pub const WF_THRESH_DEFAULT: GpuWinterFixedMoveThresholds = GpuWinterFixedMoveThresholds {
    t: [30, 40, 46, 54, 62, 72, 90, 100],
};

pub const WF_GPU_BASE_WEIGHTS: [f64; WF_GPU_NUM_MOVES] = [
    0.30, 0.10, 0.06, 0.08, 0.08, 0.10, 0.18, 0.10,
];

pub fn pack_fixed_schedule(s: &WinterFixedSchedule) -> [u32; WF_ASSIGN_U32S] {
    let mut packed = [0u32; WF_ASSIGN_U32S];
    for w in 0..WF_WEEKS {
        // 16 positions packed as 4-bit nibbles into 2 u32s per week
        for pos in 0..WF_POSITIONS {
            let u32_idx = w * 2 + pos / 8;
            let shift = (pos % 8) * 4;
            packed[u32_idx] |= (s.mapping[w][pos] as u32 & 0xF) << shift;
        }
    }
    // Flag word
    let mut flags = 0u32;
    for w in 0..WF_WEEKS {
        if s.lane_swap_early[w] { flags |= 1 << w; }
        if s.lane_swap_late[w] { flags |= 1 << (w + 12); }
    }
    packed[24] = flags;
    packed
}

pub fn unpack_fixed_schedule(packed: &[u32; WF_ASSIGN_U32S]) -> WinterFixedSchedule {
    let mut s = WinterFixedSchedule {
        mapping: [[0; WF_POSITIONS]; WF_WEEKS],
        lane_swap_early: [false; WF_WEEKS],
        lane_swap_late: [false; WF_WEEKS],
    };
    for w in 0..WF_WEEKS {
        for pos in 0..WF_POSITIONS {
            let u32_idx = w * 2 + pos / 8;
            let shift = (pos % 8) * 4;
            s.mapping[w][pos] = ((packed[u32_idx] >> shift) & 0xF) as u8;
        }
    }
    let flags = packed[24];
    for w in 0..WF_WEEKS {
        s.lane_swap_early[w] = (flags & (1 << w)) != 0;
        s.lane_swap_late[w] = (flags & (1 << (w + 12))) != 0;
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
        let w8 = WinterFixedWeights {
            matchup_zero: 80, matchup_triple: 40,
            consecutive_opponents: 30, early_late_balance: 20.0,
            early_late_alternation: 15, lane_balance: 25.0,
            lane_switch: 15.0, late_lane_balance: 15.0,
            commissioner_overlap: 30, half_season_repeat: 20,
        };
        for _ in 0..20 {
            let sched = random_fixed_schedule(&mut rng);
            let packed = pack_fixed_schedule(&sched);
            let unpacked = unpack_fixed_schedule(&packed);
            let cost_orig = evaluate_fixed(&sched, &w8).total;
            let cost_rt = evaluate_fixed(&unpacked, &w8).total;
            assert_eq!(cost_orig, cost_rt, "Pack/unpack roundtrip changed cost");
            assert_eq!(sched.mapping, unpacked.mapping);
            assert_eq!(sched.lane_swap_early, unpacked.lane_swap_early);
            assert_eq!(sched.lane_swap_late, unpacked.lane_swap_late);
        }
    }
}
