use rand::rngs::SmallRng;
use rand::Rng;
use serde::Deserialize;
use std::sync::atomic::{AtomicBool, Ordering};

pub const WF_TEAMS: usize = 16;
pub const WF_WEEKS: usize = 12;
pub const WF_LANES: usize = 4;
pub const WF_QUADS: usize = 4;
pub const WF_POS_PER_QUAD: usize = 4;
pub const WF_POSITIONS: usize = WF_TEAMS; // 16 positions per week

/// Number of matchup pairs per week: 4 quads × 4 matchups = 16
pub const WF_MATCHUPS_PER_WEEK: usize = 16;

/// Number of unique team pairs: C(16,2) = 120
pub const WF_PAIRS: usize = WF_TEAMS * (WF_TEAMS - 1) / 2;

/// Template matchup entry: two positions that play each other
#[derive(Clone, Copy)]
pub struct MatchupEntry {
    pub pos_a: u8,
    pub pos_b: u8,
    pub quad: u8,
}

/// The 16 matchup entries per week, derived from the fixed quad structure.
/// Each quad [p0, p1, p2, p3] generates: (p0,p1), (p2,p3), (p0,p3), (p2,p1)
pub const MATCHUP_ENTRIES: [MatchupEntry; WF_MATCHUPS_PER_WEEK] = {
    let mut entries = [MatchupEntry { pos_a: 0, pos_b: 0, quad: 0 }; WF_MATCHUPS_PER_WEEK];
    let mut idx = 0;
    let mut q = 0;
    while q < WF_QUADS {
        let base = q * WF_POS_PER_QUAD;
        let p0 = base as u8;
        let p1 = (base + 1) as u8;
        let p2 = (base + 2) as u8;
        let p3 = (base + 3) as u8;
        entries[idx] = MatchupEntry { pos_a: p0, pos_b: p1, quad: q as u8 };
        entries[idx + 1] = MatchupEntry { pos_a: p2, pos_b: p3, quad: q as u8 };
        entries[idx + 2] = MatchupEntry { pos_a: p0, pos_b: p3, quad: q as u8 };
        entries[idx + 3] = MatchupEntry { pos_a: p2, pos_b: p1, quad: q as u8 };
        idx += 4;
        q += 1;
    }
    entries
};

/// Quad index for a position (0-3)
pub const fn quad_of(pos: usize) -> usize { pos / WF_POS_PER_QUAD }

/// Lane offset for a quad: quads 0,2 → lanes 0-1; quads 1,3 → lanes 2-3
pub const fn lane_off_of_quad(q: usize) -> usize { (q % 2) * 2 }

/// Position within its quad (0-3)
pub const fn pos_in_quad(pos: usize) -> usize { pos % WF_POS_PER_QUAD }

/// Whether a position is "early" (quads 0-1) or "late" (quads 2-3)
pub const fn is_early(pos: usize) -> bool { quad_of(pos) < 2 }

/// Whether a position is a "stay" position (positions 0,2 within quad)
pub const fn is_stay(pos: usize) -> bool {
    let p = pos_in_quad(pos);
    p == 0 || p == 2
}

/// The schedule optimized by the solver.
#[derive(Clone, Copy)]
pub struct WinterFixedSchedule {
    pub mapping: [[u8; WF_POSITIONS]; WF_WEEKS],
    pub lane_swap_early: [bool; WF_WEEKS],
    pub lane_swap_late: [bool; WF_WEEKS],
}

#[derive(Deserialize, Clone)]
pub struct WinterFixedWeights {
    pub matchup_zero: u32,
    pub matchup_triple: u32,
    pub consecutive_opponents: u32,
    pub early_late_balance: f64,
    pub early_late_alternation: u32,
    pub early_late_consecutive: u32,
    pub lane_balance: f64,
    pub lane_switch: f64,
    pub late_lane_balance: f64,
    pub commissioner_overlap: u32,
    pub half_season_repeat: u32,
}

#[derive(Clone)]
pub struct WinterFixedCostBreakdown {
    pub matchup_balance: u32,
    pub consecutive_opponents: u32,
    pub early_late_balance: u32,
    pub early_late_alternation: u32,
    pub early_late_consecutive: u32,
    pub lane_balance: u32,
    pub lane_switch_balance: u32,
    pub late_lane_balance: u32,
    pub commissioner_overlap: u32,
    pub half_season_repeat: u32,
    pub total: u32,
}

/// Resolve the effective quad index for a position, applying lane swap flags.
/// Lane swap early: quads 0↔1; Lane swap late: quads 2↔3
fn effective_quad(pos: usize, lane_swap_early: bool, lane_swap_late: bool) -> usize {
    effective_quad_from(quad_of(pos), lane_swap_early, lane_swap_late)
}

/// Resolve effective quad from a quad index directly (avoids redundant quad_of calls).
#[inline(always)]
fn effective_quad_from(q: usize, lane_swap_early: bool, lane_swap_late: bool) -> usize {
    match q {
        0 if lane_swap_early => 1,
        1 if lane_swap_early => 0,
        2 if lane_swap_late => 3,
        3 if lane_swap_late => 2,
        _ => q,
    }
}

pub fn evaluate_fixed(sched: &WinterFixedSchedule, w8: &WinterFixedWeights) -> WinterFixedCostBreakdown {
    let mut matchups = [0i32; WF_TEAMS * WF_TEAMS];
    let mut week_matchup = [0u8; WF_WEEKS * WF_TEAMS * WF_TEAMS];
    let mut lane_counts = [0i32; WF_TEAMS * WF_LANES];
    let mut late_lane_counts = [0i32; WF_TEAMS * WF_LANES];
    let mut stay_count = [0i32; WF_TEAMS];
    let mut early_count = [0i32; WF_TEAMS];
    let mut early_late = [0u8; WF_TEAMS * WF_WEEKS];

    for w in 0..WF_WEEKS {
        let lse = sched.lane_swap_early[w];
        let lsl = sched.lane_swap_late[w];

        // Iterate by quad like the original — 4 iterations with 1 effective_quad call each
        for q in 0..WF_QUADS {
            let eq = effective_quad_from(q, lse, lsl);
            let base = q * WF_POS_PER_QUAD;
            let pa = sched.mapping[w][base] as usize;
            let pb = sched.mapping[w][base + 1] as usize;
            let pc = sched.mapping[w][base + 2] as usize;
            let pd = sched.mapping[w][base + 3] as usize;
            let early: u8 = if eq < 2 { 1 } else { 0 };
            let lo = lane_off_of_quad(eq);

            // Matchups: (pa,pb), (pc,pd), (pa,pd), (pc,pb)
            let pairs: [(usize, usize); 4] = [(pa, pb), (pc, pd), (pa, pd), (pc, pb)];
            for &(t1, t2) in &pairs {
                let lo_t = t1.min(t2);
                let hi_t = t1.max(t2);
                matchups[lo_t * WF_TEAMS + hi_t] += 1;
                week_matchup[w * WF_TEAMS * WF_TEAMS + lo_t * WF_TEAMS + hi_t] = 1;
            }

            // Lane counts: pa=stay(lo×2), pb=split, pc=stay(lo+1×2), pd=split
            lane_counts[pa * WF_LANES + lo] += 2;
            lane_counts[pb * WF_LANES + lo] += 1;
            lane_counts[pb * WF_LANES + lo + 1] += 1;
            lane_counts[pc * WF_LANES + lo + 1] += 2;
            lane_counts[pd * WF_LANES + lo + 1] += 1;
            lane_counts[pd * WF_LANES + lo] += 1;

            if eq >= 2 {
                late_lane_counts[pa * WF_LANES + lo] += 2;
                late_lane_counts[pb * WF_LANES + lo] += 1;
                late_lane_counts[pb * WF_LANES + lo + 1] += 1;
                late_lane_counts[pc * WF_LANES + lo + 1] += 2;
                late_lane_counts[pd * WF_LANES + lo + 1] += 1;
                late_lane_counts[pd * WF_LANES + lo] += 1;
            }

            // Stay: positions 0 and 2 in quad
            stay_count[pa] += 1;
            stay_count[pc] += 1;

            // Early/late
            for &t in &[pa, pb, pc, pd] {
                early_late[t * WF_WEEKS + w] = early;
                if early == 1 {
                    early_count[t] += 1;
                }
            }
        }
    }

    // Matchup balance
    let mut matchup_balance: u32 = 0;
    for i in 0..WF_TEAMS {
        for j in (i + 1)..WF_TEAMS {
            let c = matchups[i * WF_TEAMS + j];
            if c == 0 {
                matchup_balance += w8.matchup_zero;
            } else if c >= 3 {
                matchup_balance += (c - 2) as u32 * w8.matchup_triple;
            }
        }
    }

    // Half-season repeat
    let mut half_season_repeat: u32 = 0;
    const HALF: usize = WF_WEEKS / 2;
    for i in 0..WF_TEAMS {
        for j in (i + 1)..WF_TEAMS {
            let idx = i * WF_TEAMS + j;
            let mut fh = 0u32;
            let mut sh = 0u32;
            for w in 0..HALF {
                fh += week_matchup[w * WF_TEAMS * WF_TEAMS + idx] as u32;
            }
            for w in HALF..WF_WEEKS {
                sh += week_matchup[w * WF_TEAMS * WF_TEAMS + idx] as u32;
            }
            if fh > 1 { half_season_repeat += (fh - 1) * w8.half_season_repeat; }
            if sh > 1 { half_season_repeat += (sh - 1) * w8.half_season_repeat; }
        }
    }

    // Consecutive opponents
    let mut consecutive_opponents: u32 = 0;
    for w in 0..(WF_WEEKS - 1) {
        if w == 4 || w == 5 { continue; }
        let b1 = w * WF_TEAMS * WF_TEAMS;
        let b2 = (w + 1) * WF_TEAMS * WF_TEAMS;
        for i in 0..WF_TEAMS {
            for j in (i + 1)..WF_TEAMS {
                let idx = i * WF_TEAMS + j;
                if week_matchup[b1 + idx] != 0 && week_matchup[b2 + idx] != 0 {
                    consecutive_opponents += w8.consecutive_opponents;
                }
            }
        }
    }

    // Early/late balance
    let mut early_late_balance: u32 = 0;
    let target_e: f64 = WF_WEEKS as f64 / 2.0;
    for t in 0..WF_TEAMS {
        let dev = (early_count[t] as f64 - target_e).abs();
        early_late_balance += (dev * dev * w8.early_late_balance) as u32;
    }

    // Early/late alternation (3 consecutive same)
    let mut early_late_alternation: u32 = 0;
    for t in 0..WF_TEAMS {
        for w in 0..(WF_WEEKS - 2) {
            let base = t * WF_WEEKS;
            if early_late[base + w] == early_late[base + w + 1]
                && early_late[base + w + 1] == early_late[base + w + 2]
            {
                early_late_alternation += w8.early_late_alternation;
            }
        }
    }

    // Early/late consecutive (2 consecutive same)
    let mut early_late_consecutive: u32 = 0;
    for t in 0..WF_TEAMS {
        for w in 0..(WF_WEEKS - 1) {
            let base = t * WF_WEEKS;
            if early_late[base + w] == early_late[base + w + 1] {
                early_late_consecutive += w8.early_late_consecutive;
            }
        }
    }

    // Lane balance
    let mut lane_balance: u32 = 0;
    let target_l: f64 = (WF_WEEKS as f64 * 2.0) / WF_LANES as f64;
    for t in 0..WF_TEAMS {
        for l in 0..WF_LANES {
            lane_balance +=
                ((lane_counts[t * WF_LANES + l] as f64 - target_l).abs() * w8.lane_balance) as u32;
        }
    }

    // Lane switch balance
    let mut lane_switch_balance: u32 = 0;
    let target_stay: f64 = WF_WEEKS as f64 / 2.0;
    for t in 0..WF_TEAMS {
        let dev = (stay_count[t] as f64 - target_stay).abs();
        lane_switch_balance += (dev * w8.lane_switch) as u32;
    }

    // Late lane balance
    let mut late_lane_balance: u32 = 0;
    let late_target_l: f64 = WF_WEEKS as f64 / WF_LANES as f64;
    for t in 0..WF_TEAMS {
        for l in 0..WF_LANES {
            late_lane_balance +=
                ((late_lane_counts[t * WF_LANES + l] as f64 - late_target_l).abs() * w8.late_lane_balance) as u32;
        }
    }

    // Commissioner overlap
    let mut min_overlap = WF_WEEKS as u32;
    for i in 0..WF_TEAMS {
        for j in (i + 1)..WF_TEAMS {
            let mut overlap = 0u32;
            for w in 0..WF_WEEKS {
                if early_late[i * WF_WEEKS + w] == early_late[j * WF_WEEKS + w] {
                    overlap += 1;
                }
            }
            if overlap < min_overlap {
                min_overlap = overlap;
            }
        }
    }
    let commissioner_overlap = w8.commissioner_overlap * min_overlap.saturating_sub(1);

    let total = matchup_balance
        + consecutive_opponents
        + early_late_balance
        + early_late_alternation
        + early_late_consecutive
        + lane_balance
        + lane_switch_balance
        + late_lane_balance
        + commissioner_overlap
        + half_season_repeat;

    WinterFixedCostBreakdown {
        matchup_balance,
        consecutive_opponents,
        early_late_balance,
        early_late_alternation,
        early_late_consecutive,
        lane_balance,
        lane_switch_balance,
        late_lane_balance,
        commissioner_overlap,
        half_season_repeat,
        total,
    }
}

// ── Moves ──

/// Undo token for reversing a move.
pub enum MoveUndo {
    Swap { week: usize, pos_a: usize, pos_b: usize },
    CrossWeek { w1: usize, w2: usize, pos1: usize, pos2: usize },
    WeekSwap { w1: usize, w2: usize },
    ToggleLaneEarly { week: usize },
    ToggleLaneLate { week: usize },
}

pub fn apply_move(
    sched: &mut WinterFixedSchedule,
    move_id: usize,
    _bd: &WinterFixedCostBreakdown,
    rng: &mut SmallRng,
) -> MoveUndo {
    match move_id {
        0 => { // position_swap: swap two positions in one week
            let w = rng.random_range(0..WF_WEEKS);
            let pa = rng.random_range(0..WF_POSITIONS);
            let mut pb = rng.random_range(0..(WF_POSITIONS - 1));
            if pb >= pa { pb += 1; }
            sched.mapping[w].swap(pa, pb);
            MoveUndo::Swap { week: w, pos_a: pa, pos_b: pb }
        }
        1 => { // cross_week_swap: swap a team's position across two weeks
            let team = rng.random_range(0..WF_TEAMS) as u8;
            let w1 = rng.random_range(0..WF_WEEKS);
            let mut w2 = rng.random_range(0..(WF_WEEKS - 1));
            if w2 >= w1 { w2 += 1; }
            let pos1 = match sched.mapping[w1].iter().position(|&t| t == team) {
                Some(p) => p,
                None => { // fallback to position swap
                    let pa = rng.random_range(0..WF_POSITIONS);
                    let mut pb = rng.random_range(0..(WF_POSITIONS - 1));
                    if pb >= pa { pb += 1; }
                    sched.mapping[w1].swap(pa, pb);
                    return MoveUndo::Swap { week: w1, pos_a: pa, pos_b: pb };
                }
            };
            let pos2 = match sched.mapping[w2].iter().position(|&t| t == team) {
                Some(p) => p,
                None => {
                    let pa = rng.random_range(0..WF_POSITIONS);
                    let mut pb = rng.random_range(0..(WF_POSITIONS - 1));
                    if pb >= pa { pb += 1; }
                    sched.mapping[w2].swap(pa, pb);
                    return MoveUndo::Swap { week: w2, pos_a: pa, pos_b: pb };
                }
            };
            // Swap pos1↔pos2 within each week (preserves permutations)
            sched.mapping[w1].swap(pos1, pos2);
            sched.mapping[w2].swap(pos1, pos2);
            MoveUndo::CrossWeek { w1, w2, pos1, pos2 }
        }
        2 => { // week_swap: swap two entire weeks
            let w1 = rng.random_range(0..WF_WEEKS);
            let mut w2 = rng.random_range(0..(WF_WEEKS - 1));
            if w2 >= w1 { w2 += 1; }
            let tmp_m = sched.mapping[w1];
            sched.mapping[w1] = sched.mapping[w2];
            sched.mapping[w2] = tmp_m;
            let tmp_e = sched.lane_swap_early[w1];
            sched.lane_swap_early[w1] = sched.lane_swap_early[w2];
            sched.lane_swap_early[w2] = tmp_e;
            let tmp_l = sched.lane_swap_late[w1];
            sched.lane_swap_late[w1] = sched.lane_swap_late[w2];
            sched.lane_swap_late[w2] = tmp_l;
            MoveUndo::WeekSwap { w1, w2 }
        }
        3 => { // toggle_lane_early
            let w = rng.random_range(0..WF_WEEKS);
            sched.lane_swap_early[w] = !sched.lane_swap_early[w];
            MoveUndo::ToggleLaneEarly { week: w }
        }
        4 => { // toggle_lane_late
            let w = rng.random_range(0..WF_WEEKS);
            sched.lane_swap_late[w] = !sched.lane_swap_late[w];
            MoveUndo::ToggleLaneLate { week: w }
        }
        5 => { // guided_matchup
            guided_matchup(sched, rng)
        }
        6 => { // guided_lane
            guided_lane(sched, rng)
        }
        _ => { // guided_early_late
            guided_early_late(sched, rng)
        }
    }
}

pub fn undo_move(sched: &mut WinterFixedSchedule, undo: &MoveUndo) {
    match undo {
        MoveUndo::Swap { week, pos_a, pos_b } => {
            sched.mapping[*week].swap(*pos_a, *pos_b);
        }
        MoveUndo::CrossWeek { w1, w2, pos1, pos2 } => {
            // Self-inverse: swap pos1↔pos2 within each week again
            sched.mapping[*w1].swap(*pos1, *pos2);
            sched.mapping[*w2].swap(*pos1, *pos2);
        }
        MoveUndo::WeekSwap { w1, w2 } => {
            let tmp_m = sched.mapping[*w1];
            sched.mapping[*w1] = sched.mapping[*w2];
            sched.mapping[*w2] = tmp_m;
            let tmp_e = sched.lane_swap_early[*w1];
            sched.lane_swap_early[*w1] = sched.lane_swap_early[*w2];
            sched.lane_swap_early[*w2] = tmp_e;
            let tmp_l = sched.lane_swap_late[*w1];
            sched.lane_swap_late[*w1] = sched.lane_swap_late[*w2];
            sched.lane_swap_late[*w2] = tmp_l;
        }
        MoveUndo::ToggleLaneEarly { week } => {
            sched.lane_swap_early[*week] = !sched.lane_swap_early[*week];
        }
        MoveUndo::ToggleLaneLate { week } => {
            sched.lane_swap_late[*week] = !sched.lane_swap_late[*week];
        }
    }
}

pub const NUM_MOVES: usize = 8;
pub const MOVE_NAMES: [&str; NUM_MOVES] = [
    "pos_swp", "cross_wk", "wk_swap", "tog_e", "tog_l",
    "g_match", "g_lane", "g_el",
];

pub const BASE_WEIGHTS: [f64; NUM_MOVES] = [
    0.30, 0.10, 0.06, 0.08, 0.08, 0.10, 0.18, 0.10,
];

pub fn pick_move(rng: &mut SmallRng, _bd: &WinterFixedCostBreakdown) -> usize {
    let r: f64 = rng.random();
    let mut cum = 0.0;
    for m in 0..NUM_MOVES {
        cum += BASE_WEIGHTS[m];
        if r < cum { return m; }
    }
    NUM_MOVES - 1
}

fn guided_matchup(sched: &mut WinterFixedSchedule, rng: &mut SmallRng) -> MoveUndo {
    // Find a team pair with 0 matchups and try to bring them together
    let mut matchups = [0u8; WF_TEAMS * WF_TEAMS];
    for w in 0..WF_WEEKS {
        for entry in &MATCHUP_ENTRIES {
            let ta = sched.mapping[w][entry.pos_a as usize] as usize;
            let tb = sched.mapping[w][entry.pos_b as usize] as usize;
            matchups[ta.min(tb) * WF_TEAMS + ta.max(tb)] = 1;
        }
    }

    let start = rng.random_range(0..WF_TEAMS);
    let mut target_a = 0u8;
    let mut target_b = 0u8;
    let mut found = false;
    'outer: for off_i in 0..WF_TEAMS {
        let i = (start + off_i) % WF_TEAMS;
        for j in (i + 1)..WF_TEAMS {
            if matchups[i * WF_TEAMS + j] == 0 {
                target_a = i as u8;
                target_b = j as u8;
                found = true;
                break 'outer;
            }
        }
    }

    if !found {
        // Fallback to position swap
        let w = rng.random_range(0..WF_WEEKS);
        let pa = rng.random_range(0..WF_POSITIONS);
        let mut pb = rng.random_range(0..(WF_POSITIONS - 1));
        if pb >= pa { pb += 1; }
        sched.mapping[w].swap(pa, pb);
        return MoveUndo::Swap { week: w, pos_a: pa, pos_b: pb };
    }

    // Find a week where both teams are in the same half (both early or both late)
    // but different quads, then swap one into the other's quad
    let week_start = rng.random_range(0..WF_WEEKS);
    for off in 0..WF_WEEKS {
        let w = (week_start + off) % WF_WEEKS;
        let pos_a = match sched.mapping[w].iter().position(|&t| t == target_a) {
            Some(p) => p, None => continue,
        };
        let pos_b = match sched.mapping[w].iter().position(|&t| t == target_b) {
            Some(p) => p, None => continue,
        };
        let qa = quad_of(pos_a);
        let qb = quad_of(pos_b);
        let same_half = (qa < 2 && qb < 2) || (qa >= 2 && qb >= 2);
        if same_half && qa != qb {
            // Swap target_b with a random non-target_a position in target_a's quad
            let q_base = qa * WF_POS_PER_QUAD;
            let candidates: Vec<usize> = (q_base..q_base + WF_POS_PER_QUAD)
                .filter(|&p| p != pos_a)
                .collect();
            if candidates.is_empty() { continue; }
            let swap_pos = candidates[rng.random_range(0..candidates.len())];
            sched.mapping[w].swap(pos_b, swap_pos);
            return MoveUndo::Swap { week: w, pos_a: pos_b, pos_b: swap_pos };
        }
    }

    // Fallback
    let w = rng.random_range(0..WF_WEEKS);
    let pa = rng.random_range(0..WF_POSITIONS);
    let mut pb = rng.random_range(0..(WF_POSITIONS - 1));
    if pb >= pa { pb += 1; }
    sched.mapping[w].swap(pa, pb);
    MoveUndo::Swap { week: w, pos_a: pa, pos_b: pb }
}

fn guided_lane(sched: &mut WinterFixedSchedule, rng: &mut SmallRng) -> MoveUndo {
    // Find team with worst lane imbalance, try to fix via toggle or swap
    let mut lane_counts = [0i32; WF_TEAMS * WF_LANES];
    for w in 0..WF_WEEKS {
        let lse = sched.lane_swap_early[w];
        let lsl = sched.lane_swap_late[w];
        for pos in 0..WF_POSITIONS {
            let team = sched.mapping[w][pos] as usize;
            let eq = effective_quad(pos, lse, lsl);
            let lo = lane_off_of_quad(eq);
            let piq = pos_in_quad(pos);
            match piq {
                0 => { lane_counts[team * WF_LANES + lo] += 2; }
                1 => { lane_counts[team * WF_LANES + lo] += 1; lane_counts[team * WF_LANES + lo + 1] += 1; }
                2 => { lane_counts[team * WF_LANES + lo + 1] += 2; }
                _ => { lane_counts[team * WF_LANES + lo + 1] += 1; lane_counts[team * WF_LANES + lo] += 1; }
            }
        }
    }

    let target_l = (WF_WEEKS as f64 * 2.0) / WF_LANES as f64;
    let mut worst_team = 0usize;
    let mut worst_dev = 0.0f64;
    for t in 0..WF_TEAMS {
        for l in 0..WF_LANES {
            let dev = (lane_counts[t * WF_LANES + l] as f64 - target_l).abs();
            if dev > worst_dev { worst_dev = dev; worst_team = t; }
        }
    }

    if worst_dev < 1.0 {
        // Try a toggle instead
        let w = rng.random_range(0..WF_WEEKS);
        if rng.random_bool(0.5) {
            sched.lane_swap_early[w] = !sched.lane_swap_early[w];
            return MoveUndo::ToggleLaneEarly { week: w };
        } else {
            sched.lane_swap_late[w] = !sched.lane_swap_late[w];
            return MoveUndo::ToggleLaneLate { week: w };
        }
    }

    // Find worst_team in a random week and swap its position within the quad
    let start = rng.random_range(0..WF_WEEKS);
    for off in 0..WF_WEEKS {
        let w = (start + off) % WF_WEEKS;
        if let Some(pos) = sched.mapping[w].iter().position(|&t| t == worst_team as u8) {
            let q_base = quad_of(pos) * WF_POS_PER_QUAD;
            let mut swap_pos = q_base + rng.random_range(0..(WF_POS_PER_QUAD - 1));
            if swap_pos >= pos { swap_pos += 1; }
            if swap_pos >= q_base + WF_POS_PER_QUAD { swap_pos = q_base; }
            sched.mapping[w].swap(pos, swap_pos);
            return MoveUndo::Swap { week: w, pos_a: pos, pos_b: swap_pos };
        }
    }

    // Fallback
    let w = rng.random_range(0..WF_WEEKS);
    let pa = rng.random_range(0..WF_POSITIONS);
    let mut pb = rng.random_range(0..(WF_POSITIONS - 1));
    if pb >= pa { pb += 1; }
    sched.mapping[w].swap(pa, pb);
    MoveUndo::Swap { week: w, pos_a: pa, pos_b: pb }
}

fn guided_early_late(sched: &mut WinterFixedSchedule, rng: &mut SmallRng) -> MoveUndo {
    // Find team with worst early/late imbalance
    let mut early_count = [0i32; WF_TEAMS];
    for w in 0..WF_WEEKS {
        let lse = sched.lane_swap_early[w];
        let lsl = sched.lane_swap_late[w];
        for pos in 0..WF_POSITIONS {
            let team = sched.mapping[w][pos] as usize;
            let eq = effective_quad(pos, lse, lsl);
            if eq < 2 { early_count[team] += 1; }
        }
    }

    let target_e = WF_WEEKS as f64 / 2.0;
    let mut worst_team = 0usize;
    let mut worst_dev = 0.0f64;
    let mut too_many_early = false;
    for t in 0..WF_TEAMS {
        let dev = early_count[t] as f64 - target_e;
        if dev.abs() > worst_dev {
            worst_dev = dev.abs();
            worst_team = t;
            too_many_early = dev > 0.0;
        }
    }

    if worst_dev < 1.0 {
        let w = rng.random_range(0..WF_WEEKS);
        let pa = rng.random_range(0..WF_POSITIONS);
        let mut pb = rng.random_range(0..(WF_POSITIONS - 1));
        if pb >= pa { pb += 1; }
        sched.mapping[w].swap(pa, pb);
        return MoveUndo::Swap { week: w, pos_a: pa, pos_b: pb };
    }

    // Find a week and swap this team between early and late positions
    let start = rng.random_range(0..WF_WEEKS);
    for off in 0..WF_WEEKS {
        let w = (start + off) % WF_WEEKS;
        let lse = sched.lane_swap_early[w];
        let lsl = sched.lane_swap_late[w];
        if let Some(pos) = sched.mapping[w].iter().position(|&t| t == worst_team as u8) {
            let eq = effective_quad(pos, lse, lsl);
            let in_early = eq < 2;
            if (too_many_early && in_early) || (!too_many_early && !in_early) {
                // Swap with a random position in the opposite half
                let target_range = if in_early { 8..16 } else { 0..8 };
                let swap_pos = rng.random_range(target_range);
                sched.mapping[w].swap(pos, swap_pos);
                return MoveUndo::Swap { week: w, pos_a: pos, pos_b: swap_pos };
            }
        }
    }

    // Fallback
    let w = rng.random_range(0..WF_WEEKS);
    let pa = rng.random_range(0..WF_POSITIONS);
    let mut pb = rng.random_range(0..(WF_POSITIONS - 1));
    if pb >= pa { pb += 1; }
    sched.mapping[w].swap(pa, pb);
    MoveUndo::Swap { week: w, pos_a: pa, pos_b: pb }
}

// ── Systematic sweep ──

pub fn systematic_sweep(
    sched: &mut WinterFixedSchedule,
    w8: &WinterFixedWeights,
    shutdown: &AtomicBool,
) -> u32 {
    let mut improvements = 0u32;
    let mut best_cost = evaluate_fixed(sched, w8).total;

    'restart: loop {
        if shutdown.load(Ordering::Relaxed) { break; }

        // Phase 1: all single position swaps
        for w in 0..WF_WEEKS {
            for pa in 0..WF_POSITIONS {
                for pb in (pa + 1)..WF_POSITIONS {
                    sched.mapping[w].swap(pa, pb);
                    let c = evaluate_fixed(sched, w8).total;
                    if c < best_cost {
                        best_cost = c;
                        improvements += 1;
                        continue 'restart;
                    }
                    sched.mapping[w].swap(pa, pb);
                }
            }
            if shutdown.load(Ordering::Relaxed) { break; }
        }

        // Phase 2: all lane toggles
        for w in 0..WF_WEEKS {
            sched.lane_swap_early[w] = !sched.lane_swap_early[w];
            let c = evaluate_fixed(sched, w8).total;
            if c < best_cost {
                best_cost = c;
                improvements += 1;
                continue 'restart;
            }
            sched.lane_swap_early[w] = !sched.lane_swap_early[w];

            sched.lane_swap_late[w] = !sched.lane_swap_late[w];
            let c = evaluate_fixed(sched, w8).total;
            if c < best_cost {
                best_cost = c;
                improvements += 1;
                continue 'restart;
            }
            sched.lane_swap_late[w] = !sched.lane_swap_late[w];
        }

        // Phase 3: double swaps (swap + toggle)
        for w in 0..WF_WEEKS {
            for pa in 0..WF_POSITIONS {
                for pb in (pa + 1)..WF_POSITIONS {
                    sched.mapping[w].swap(pa, pb);

                    sched.lane_swap_early[w] = !sched.lane_swap_early[w];
                    let c = evaluate_fixed(sched, w8).total;
                    if c < best_cost {
                        best_cost = c;
                        improvements += 1;
                        continue 'restart;
                    }
                    sched.lane_swap_early[w] = !sched.lane_swap_early[w];

                    sched.lane_swap_late[w] = !sched.lane_swap_late[w];
                    let c = evaluate_fixed(sched, w8).total;
                    if c < best_cost {
                        best_cost = c;
                        improvements += 1;
                        continue 'restart;
                    }
                    sched.lane_swap_late[w] = !sched.lane_swap_late[w];

                    sched.mapping[w].swap(pa, pb);
                }
            }
            if shutdown.load(Ordering::Relaxed) { break; }
        }

        break;
    }

    improvements
}

// ── Random generation and perturbation ──

pub fn random_fixed_schedule(rng: &mut SmallRng) -> WinterFixedSchedule {
    let mut sched = WinterFixedSchedule {
        mapping: [[0u8; WF_POSITIONS]; WF_WEEKS],
        lane_swap_early: [false; WF_WEEKS],
        lane_swap_late: [false; WF_WEEKS],
    };
    for w in 0..WF_WEEKS {
        let mut teams: [u8; WF_TEAMS] = std::array::from_fn(|i| i as u8);
        for i in (1..WF_TEAMS).rev() {
            let j = rng.random_range(0..=i);
            teams.swap(i, j);
        }
        sched.mapping[w] = teams;
        sched.lane_swap_early[w] = rng.random_bool(0.5);
        sched.lane_swap_late[w] = rng.random_bool(0.5);
    }
    sched
}

pub fn perturb_fixed(sched: &mut WinterFixedSchedule, rng: &mut SmallRng, n: usize) {
    for _ in 0..n {
        let w = rng.random_range(0..WF_WEEKS);
        let pa = rng.random_range(0..WF_POSITIONS);
        let mut pb = rng.random_range(0..(WF_POSITIONS - 1));
        if pb >= pa { pb += 1; }
        sched.mapping[w].swap(pa, pb);
    }
}

// ── I/O ──

/// Convert a WinterFixedSchedule to the same TSV format as winter.rs
pub fn fixed_schedule_to_tsv(sched: &WinterFixedSchedule) -> String {
    // Convert to Assignment format and use the same output
    let a = to_assignment(sched);
    crate::winter::assignment_to_tsv(&a)
}

/// Convert WinterFixedSchedule to the old Assignment format
pub fn to_assignment(sched: &WinterFixedSchedule) -> crate::winter::Assignment {
    let mut a = [[[0u8; crate::winter::POS]; crate::winter::QUADS]; crate::winter::WEEKS];
    for w in 0..WF_WEEKS {
        let lse = sched.lane_swap_early[w];
        let lsl = sched.lane_swap_late[w];
        for q in 0..WF_QUADS {
            let base = q * WF_POS_PER_QUAD;
            // Map positions to teams, accounting for lane swaps
            let eq = effective_quad(base, lse, lsl);
            for p in 0..WF_POS_PER_QUAD {
                a[w][eq][p] = sched.mapping[w][base + p];
            }
        }
    }
    a
}

/// Parse a winter TSV into a WinterFixedSchedule
pub fn parse_fixed_tsv(content: &str) -> Option<WinterFixedSchedule> {
    let a = crate::winter::parse_tsv(content)?;
    Some(from_assignment(&a))
}

/// Convert an old Assignment to WinterFixedSchedule
pub fn from_assignment(a: &crate::winter::Assignment) -> WinterFixedSchedule {
    // Try all 4 swap combinations per week and pick the one that matches
    let mut sched = WinterFixedSchedule {
        mapping: [[0u8; WF_POSITIONS]; WF_WEEKS],
        lane_swap_early: [false; WF_WEEKS],
        lane_swap_late: [false; WF_WEEKS],
    };

    for w in 0..WF_WEEKS {
        // No swaps: quad 0 = a[w][0], quad 1 = a[w][1], quad 2 = a[w][2], quad 3 = a[w][3]
        // lane_swap_early: quad 0 = a[w][1], quad 1 = a[w][0]
        // lane_swap_late: quad 2 = a[w][3], quad 3 = a[w][2]
        // The mapping stores positions in template order (quad 0 = pos 0-3, etc.)
        // With no swap, template quad q maps to assignment quad q
        // With lane_swap_early, template quad 0 maps to assignment quad 1 and vice versa
        // We just pick no swaps and directly copy
        for q in 0..WF_QUADS {
            for p in 0..WF_POS_PER_QUAD {
                sched.mapping[w][q * WF_POS_PER_QUAD + p] = a[w][q][p];
            }
        }
        sched.lane_swap_early[w] = false;
        sched.lane_swap_late[w] = false;
    }

    sched
}

pub fn fixed_cost_label(c: &WinterFixedCostBreakdown) -> String {
    format!(
        "total: {:>4} matchup: {:>3} consec: {:>3} el_bal: {:>3} el_alt: {:>3} el_con: {:>3} lane: {:>3} switch: {:>3} ll_bal: {:>3} comm: {:>3} hs_rpt: {:>3}",
        c.total, c.matchup_balance, c.consecutive_opponents,
        c.early_late_balance, c.early_late_alternation, c.early_late_consecutive,
        c.lane_balance, c.lane_switch_balance, c.late_lane_balance,
        c.commissioner_overlap, c.half_season_repeat,
    )
}

pub fn reassign_commissioners(sched: &mut WinterFixedSchedule) {
    let mut a = to_assignment(sched);
    crate::winter::reassign_commissioners(&mut a);
    *sched = from_assignment(&a);
}

/// Generate WGSL constant declarations for the template
pub fn wgsl_consts() -> String {
    let mut lines = Vec::new();

    // Matchup entries
    let pos_a: Vec<String> = MATCHUP_ENTRIES.iter().map(|e| e.pos_a.to_string()).collect();
    let pos_b: Vec<String> = MATCHUP_ENTRIES.iter().map(|e| e.pos_b.to_string()).collect();
    lines.push(format!(
        "const T_POS_A: array<u32, {}> = array<u32, {}>({});",
        WF_MATCHUPS_PER_WEEK, WF_MATCHUPS_PER_WEEK, pos_a.join(", ")
    ));
    lines.push(format!(
        "const T_POS_B: array<u32, {}> = array<u32, {}>({});",
        WF_MATCHUPS_PER_WEEK, WF_MATCHUPS_PER_WEEK, pos_b.join(", ")
    ));

    // Quad of each position
    let quad_of_pos: Vec<String> = (0..WF_POSITIONS).map(|p| quad_of(p).to_string()).collect();
    lines.push(format!(
        "const POS_QUAD: array<u32, {}> = array<u32, {}>({});",
        WF_POSITIONS, WF_POSITIONS, quad_of_pos.join(", ")
    ));

    // Position-in-quad for each position
    let piq: Vec<String> = (0..WF_POSITIONS).map(|p| pos_in_quad(p).to_string()).collect();
    lines.push(format!(
        "const POS_IN_QUAD: array<u32, {}> = array<u32, {}>({});",
        WF_POSITIONS, WF_POSITIONS, piq.join(", ")
    ));

    // Early positions (0-7)
    let early: Vec<String> = (0..WF_POSITIONS).filter(|&p| is_early(p)).map(|p| p.to_string()).collect();
    lines.push(format!("const EARLY_COUNT: u32 = {}u;", early.len()));
    lines.push(format!(
        "const EARLY_POS: array<u32, {}> = array<u32, {}>({});",
        early.len(), early.len(), early.join(", ")
    ));

    // Stay positions
    let stay: Vec<String> = (0..WF_POSITIONS).filter(|&p| is_stay(p)).map(|p| p.to_string()).collect();
    lines.push(format!("const STAY_COUNT: u32 = {}u;", stay.len()));
    lines.push(format!(
        "const STAY_POS: array<u32, {}> = array<u32, {}>({});",
        stay.len(), stay.len(), stay.join(", ")
    ));

    lines.push(format!("const MATCHUPS_PER_WEEK: u32 = {}u;", WF_MATCHUPS_PER_WEEK));

    lines.join("\n")
}

pub fn sa_accept(delta: i64, temp: f64, rng: &mut SmallRng) -> bool {
    if delta < 0 { return true; }
    if delta == 0 { return rng.random_bool(0.2); }
    let prob = (-delta as f64 / temp).exp();
    rng.random_bool(prob.min(1.0))
}
