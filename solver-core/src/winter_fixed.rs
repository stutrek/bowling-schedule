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
        7 => { // guided_early_late
            guided_early_late(sched, rng)
        }
        8 => { // guided_consecutive_opponents
            guided_consecutive(sched, rng)
        }
        9 => { // guided_lane_switch
            guided_lane_switch(sched, rng)
        }
        10 => { // guided_late_lane
            guided_late_lane(sched, rng)
        }
        11 => { // guided_commissioner
            guided_commissioner(sched, rng)
        }
        _ => { // guided_half_season_repeat
            guided_half_season(sched, rng)
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

pub const NUM_MOVES: usize = 13;
pub const MOVE_NAMES: [&str; NUM_MOVES] = [
    "pos_swp", "cross_wk", "wk_swap", "tog_e", "tog_l",
    "g_match", "g_lane", "g_el",
    "g_consec", "g_lswitch", "g_llane", "g_comm", "g_hsrpt",
];

pub const BASE_WEIGHTS: [f64; NUM_MOVES] = [
    0.18, 0.08, 0.04, 0.06, 0.06, 0.08, 0.10, 0.08,
    0.06, 0.06, 0.06, 0.06, 0.08,
];

/// Identify the worst cost component. Returns index 0-9.
pub fn worst_component(bd: &WinterFixedCostBreakdown) -> usize {
    let components = [
        bd.matchup_balance,
        bd.consecutive_opponents,
        bd.early_late_balance,
        bd.early_late_alternation,
        bd.early_late_consecutive,
        bd.lane_balance,
        bd.lane_switch_balance,
        bd.late_lane_balance,
        bd.commissioner_overlap,
        bd.half_season_repeat,
    ];
    let mut worst = 0;
    for i in 1..components.len() {
        if components[i] > components[worst] {
            worst = i;
        }
    }
    worst
}

pub fn pick_move(rng: &mut SmallRng, bd: &WinterFixedCostBreakdown) -> usize {
    // 50% chance: guided move targeting worst component
    // 50% chance: weighted random from all moves (adaptive)
    if rng.random_bool(0.5) {
        pick_move_guided_only(bd)
    } else {
        let r: f64 = rng.random();
        let mut cum = 0.0;
        // Renormalize base weights for random selection
        let sum: f64 = BASE_WEIGHTS.iter().sum();
        for m in 0..NUM_MOVES {
            cum += BASE_WEIGHTS[m] / sum;
            if r < cum { return m; }
        }
        NUM_MOVES - 1
    }
}

/// Pick only a guided move targeting the worst component. For sweep perturbation.
pub fn pick_move_guided_only(bd: &WinterFixedCostBreakdown) -> usize {
    match worst_component(bd) {
        0 => 5,  // matchup → guided matchup
        1 => 8,  // consecutive opponents → guided consecutive
        2 | 3 | 4 => 7, // early/late balance/alternation/consecutive → guided early_late
        5 => 6,  // lane balance → guided lane
        6 => 9,  // lane switch → guided lane switch
        7 => 10, // late lane → guided late lane
        8 => 11, // commissioner → guided commissioner
        9 => 12, // half season repeat → guided half season
        _ => 0,
    }
}

fn guided_matchup(sched: &mut WinterFixedSchedule, rng: &mut SmallRng) -> MoveUndo {
    // Count matchups per pair
    let mut matchups = [0u8; WF_TEAMS * WF_TEAMS];
    let mut week_matchup = [[false; WF_TEAMS * WF_TEAMS]; WF_WEEKS];
    for w in 0..WF_WEEKS {
        for entry in &MATCHUP_ENTRIES {
            let ta = sched.mapping[w][entry.pos_a as usize] as usize;
            let tb = sched.mapping[w][entry.pos_b as usize] as usize;
            let lo = ta.min(tb);
            let hi = ta.max(tb);
            matchups[lo * WF_TEAMS + hi] += 1;
            week_matchup[w][lo * WF_TEAMS + hi] = true;
        }
    }

    // Find worst pair: most over-matched (3+) or under-matched (0)
    let mut worst_lo = 0;
    let mut worst_hi = 1;
    let mut worst_dist: i32 = 0;
    let start = rng.random_range(0..WF_TEAMS);
    for off_i in 0..WF_TEAMS {
        let i = (start + off_i) % WF_TEAMS;
        for j in (i + 1)..WF_TEAMS {
            let c = matchups[i * WF_TEAMS + j] as i32;
            let dist = if c == 0 { 2 } else if c >= 3 { c - 1 } else { 0 };
            if dist > worst_dist {
                worst_dist = dist;
                worst_lo = i;
                worst_hi = j;
            }
        }
    }

    if worst_dist == 0 {
        return fallback_swap(sched, rng);
    }

    let c = matchups[worst_lo * WF_TEAMS + worst_hi];
    if c >= 3 {
        // Over-matched: find a week where they play and swap one out
        let play_weeks: Vec<usize> = (0..WF_WEEKS)
            .filter(|&w| week_matchup[w][worst_lo * WF_TEAMS + worst_hi])
            .collect();
        if play_weeks.is_empty() { return fallback_swap(sched, rng); }
        let w = play_weeks[rng.random_range(0..play_weeks.len())];
        let target = if rng.random_bool(0.5) { worst_lo } else { worst_hi };
        let pos = sched.mapping[w].iter().position(|&t| t as usize == target).unwrap();
        // Swap with a random position in a different quad
        let my_quad = quad_of(pos);
        let candidates: Vec<usize> = (0..WF_POSITIONS).filter(|&p| quad_of(p) != my_quad).collect();
        let swap_pos = candidates[rng.random_range(0..candidates.len())];
        sched.mapping[w].swap(pos, swap_pos);
        return MoveUndo::Swap { week: w, pos_a: pos, pos_b: swap_pos };
    } else {
        // Under-matched (0): find a week where they DON'T play and bring them together
        let mut no_play_weeks: Vec<usize> = (0..WF_WEEKS)
            .filter(|&w| !week_matchup[w][worst_lo * WF_TEAMS + worst_hi])
            .collect();
        if no_play_weeks.is_empty() { no_play_weeks = (0..WF_WEEKS).collect(); }
        let w = no_play_weeks[rng.random_range(0..no_play_weeks.len())];
        let pos_a = sched.mapping[w].iter().position(|&t| t as usize == worst_lo).unwrap();
        let pos_b = sched.mapping[w].iter().position(|&t| t as usize == worst_hi).unwrap();
        let qa = quad_of(pos_a);
        let qb = quad_of(pos_b);
        let same_half = (qa < 2 && qb < 2) || (qa >= 2 && qb >= 2);
        if same_half && qa != qb {
            // Swap target_b into target_a's quad
            let q_base = qa * WF_POS_PER_QUAD;
            let candidates: Vec<usize> = (q_base..q_base + WF_POS_PER_QUAD)
                .filter(|&p| p != pos_a)
                .collect();
            if !candidates.is_empty() {
                let swap_pos = candidates[rng.random_range(0..candidates.len())];
                sched.mapping[w].swap(pos_b, swap_pos);
                return MoveUndo::Swap { week: w, pos_a: pos_b, pos_b: swap_pos };
            }
        }
        // If not in same half, swap one into the other's half
        let target_range = if qa < 2 { 8..16 } else { 0..8 };
        let swap_pos = rng.random_range(target_range);
        sched.mapping[w].swap(pos_b, swap_pos);
        return MoveUndo::Swap { week: w, pos_a: pos_b, pos_b: swap_pos };
    }
}

fn guided_lane(sched: &mut WinterFixedSchedule, rng: &mut SmallRng) -> MoveUndo {
    // Find team with worst lane imbalance
    let mut lane_counts = [0i32; WF_TEAMS * WF_LANES];
    // Track per-team per-week lane contributions for week targeting
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
    let mut worst_lane = 0usize;
    let mut worst_dev = 0.0f64;
    for t in 0..WF_TEAMS {
        for l in 0..WF_LANES {
            let dev = (lane_counts[t * WF_LANES + l] as f64 - target_l).abs();
            if dev > worst_dev { worst_dev = dev; worst_team = t; worst_lane = l; }
        }
    }

    if worst_dev < 1.0 {
        let w = rng.random_range(0..WF_WEEKS);
        if rng.random_bool(0.5) {
            sched.lane_swap_early[w] = !sched.lane_swap_early[w];
            return MoveUndo::ToggleLaneEarly { week: w };
        } else {
            sched.lane_swap_late[w] = !sched.lane_swap_late[w];
            return MoveUndo::ToggleLaneLate { week: w };
        }
    }

    let over = lane_counts[worst_team * WF_LANES + worst_lane] as f64 > target_l;

    // Find weeks where worst_team contributes to the over-represented lane
    let mut good_weeks: Vec<(usize, usize)> = Vec::new(); // (week, pos)
    for w in 0..WF_WEEKS {
        let lse = sched.lane_swap_early[w];
        let lsl = sched.lane_swap_late[w];
        let pos = sched.mapping[w].iter().position(|&t| t as usize == worst_team).unwrap();
        let eq = effective_quad(pos, lse, lsl);
        let lo = lane_off_of_quad(eq);
        let piq = pos_in_quad(pos);
        // Check if this position contributes to worst_lane
        let on_lane = match piq {
            0 => lo == worst_lane,
            1 => lo == worst_lane || lo + 1 == worst_lane,
            2 => lo + 1 == worst_lane,
            _ => lo + 1 == worst_lane || lo == worst_lane,
        };
        if over && on_lane {
            good_weeks.push((w, pos));
        } else if !over && !on_lane {
            good_weeks.push((w, pos));
        }
    }

    if good_weeks.is_empty() {
        return fallback_swap(sched, rng);
    }

    let (w, pos) = good_weeks[rng.random_range(0..good_weeks.len())];
    // Swap within the quad to change lane assignment
    let q_base = quad_of(pos) * WF_POS_PER_QUAD;
    let candidates: Vec<usize> = (q_base..q_base + WF_POS_PER_QUAD)
        .filter(|&p| p != pos)
        .collect();
    let swap_pos = candidates[rng.random_range(0..candidates.len())];
    sched.mapping[w].swap(pos, swap_pos);
    MoveUndo::Swap { week: w, pos_a: pos, pos_b: swap_pos }
}

fn guided_early_late(sched: &mut WinterFixedSchedule, rng: &mut SmallRng) -> MoveUndo {
    // Find team with worst early/late imbalance
    let mut early_count = [0i32; WF_TEAMS];
    let mut team_early_weeks: [Vec<usize>; WF_TEAMS] = std::array::from_fn(|_| Vec::new());
    let mut team_late_weeks: [Vec<usize>; WF_TEAMS] = std::array::from_fn(|_| Vec::new());
    for w in 0..WF_WEEKS {
        let lse = sched.lane_swap_early[w];
        let lsl = sched.lane_swap_late[w];
        for pos in 0..WF_POSITIONS {
            let team = sched.mapping[w][pos] as usize;
            let eq = effective_quad(pos, lse, lsl);
            if eq < 2 {
                early_count[team] += 1;
                team_early_weeks[team].push(w);
            } else {
                team_late_weeks[team].push(w);
            }
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
        return fallback_swap(sched, rng);
    }

    // Find a team with opposite imbalance to swap with
    let swap_target = (0..WF_TEAMS).find(|&t| {
        let dev = early_count[t] as f64 - target_e;
        if too_many_early { dev < -0.5 } else { dev > 0.5 }
    });

    // Pick week where worst_team is in the wrong half
    let weeks = if too_many_early { &team_early_weeks[worst_team] } else { &team_late_weeks[worst_team] };
    if weeks.is_empty() {
        return fallback_swap(sched, rng);
    }
    let w = weeks[rng.random_range(0..weeks.len())];
    let lse = sched.lane_swap_early[w];
    let lsl = sched.lane_swap_late[w];
    let pos = sched.mapping[w].iter().position(|&t| t as usize == worst_team).unwrap();
    let eq = effective_quad(pos, lse, lsl);
    let in_early = eq < 2;

    if let Some(target) = swap_target {
        // Find target's position and swap — prefer if target is in opposite half this week
        let tpos = sched.mapping[w].iter().position(|&t| t as usize == target).unwrap();
        let teq = effective_quad(tpos, lse, lsl);
        let t_in_early = teq < 2;
        if in_early != t_in_early {
            sched.mapping[w].swap(pos, tpos);
            return MoveUndo::Swap { week: w, pos_a: pos, pos_b: tpos };
        }
    }

    // Swap with a random position in the opposite half
    let target_range = if in_early { 8..16 } else { 0..8 };
    let swap_pos = rng.random_range(target_range);
    sched.mapping[w].swap(pos, swap_pos);
    MoveUndo::Swap { week: w, pos_a: pos, pos_b: swap_pos }
}

fn fallback_swap(sched: &mut WinterFixedSchedule, rng: &mut SmallRng) -> MoveUndo {
    let w = rng.random_range(0..WF_WEEKS);
    let pa = rng.random_range(0..WF_POSITIONS);
    let mut pb = rng.random_range(0..(WF_POSITIONS - 1));
    if pb >= pa { pb += 1; }
    sched.mapping[w].swap(pa, pb);
    MoveUndo::Swap { week: w, pos_a: pa, pos_b: pb }
}

/// Guided consecutive opponents: find a pair that plays in consecutive weeks and separate them.
fn guided_consecutive(sched: &mut WinterFixedSchedule, rng: &mut SmallRng) -> MoveUndo {
    let mut week_matchup = [[false; WF_TEAMS * WF_TEAMS]; WF_WEEKS];
    for w in 0..WF_WEEKS {
        for entry in &MATCHUP_ENTRIES {
            let ta = sched.mapping[w][entry.pos_a as usize] as usize;
            let tb = sched.mapping[w][entry.pos_b as usize] as usize;
            let lo = ta.min(tb);
            let hi = ta.max(tb);
            week_matchup[w][lo * WF_TEAMS + hi] = true;
        }
    }

    // Find a consecutive pair
    let mut violations: Vec<(usize, usize, usize)> = Vec::new(); // (week, team_lo, team_hi)
    for w in 0..(WF_WEEKS - 1) {
        if w == 4 || w == 5 { continue; } // skip the gap
        for i in 0..WF_TEAMS {
            for j in (i + 1)..WF_TEAMS {
                let idx = i * WF_TEAMS + j;
                if week_matchup[w][idx] && week_matchup[w + 1][idx] {
                    violations.push((w, i, j));
                }
            }
        }
    }

    if violations.is_empty() {
        return fallback_swap(sched, rng);
    }

    let (w, tlo, thi) = violations[rng.random_range(0..violations.len())];
    // Pick one of the two consecutive weeks and swap one of the teams out of their matchup
    let target_w = if rng.random_bool(0.5) { w } else { w + 1 };
    let target_team = if rng.random_bool(0.5) { tlo } else { thi };
    let pos = sched.mapping[target_w].iter().position(|&t| t as usize == target_team).unwrap();
    // Swap with someone in a different quad (same half to avoid disrupting early/late)
    let my_quad = quad_of(pos);
    let same_half: Vec<usize> = (0..WF_POSITIONS)
        .filter(|&p| quad_of(p) != my_quad && ((quad_of(p) < 2) == (my_quad < 2)))
        .collect();
    if same_half.is_empty() {
        return fallback_swap(sched, rng);
    }
    let swap_pos = same_half[rng.random_range(0..same_half.len())];
    sched.mapping[target_w].swap(pos, swap_pos);
    MoveUndo::Swap { week: target_w, pos_a: pos, pos_b: swap_pos }
}

/// Guided lane switch: balance stay vs split positions per team.
fn guided_lane_switch(sched: &mut WinterFixedSchedule, rng: &mut SmallRng) -> MoveUndo {
    let mut stay_count = [0i32; WF_TEAMS];
    for w in 0..WF_WEEKS {
        for pos in 0..WF_POSITIONS {
            let team = sched.mapping[w][pos] as usize;
            if is_stay(pos) { stay_count[team] += 1; }
        }
    }

    let target_stay = WF_WEEKS as f64 / 2.0;
    let mut worst_team = 0;
    let mut worst_dev = 0.0f64;
    let mut too_many_stay = false;
    for t in 0..WF_TEAMS {
        let dev = stay_count[t] as f64 - target_stay;
        if dev.abs() > worst_dev {
            worst_dev = dev.abs();
            worst_team = t;
            too_many_stay = dev > 0.0;
        }
    }

    if worst_dev < 1.0 {
        return fallback_swap(sched, rng);
    }

    // Find weeks where worst_team is in wrong position type
    let mut good_weeks: Vec<(usize, usize)> = Vec::new();
    for w in 0..WF_WEEKS {
        let pos = sched.mapping[w].iter().position(|&t| t as usize == worst_team).unwrap();
        if (too_many_stay && is_stay(pos)) || (!too_many_stay && !is_stay(pos)) {
            good_weeks.push((w, pos));
        }
    }

    if good_weeks.is_empty() {
        return fallback_swap(sched, rng);
    }

    let (w, pos) = good_weeks[rng.random_range(0..good_weeks.len())];
    // Swap within the same quad with a position of opposite type (stay↔split)
    let q_base = quad_of(pos) * WF_POS_PER_QUAD;
    let candidates: Vec<usize> = (q_base..q_base + WF_POS_PER_QUAD)
        .filter(|&p| p != pos && is_stay(p) != is_stay(pos))
        .collect();
    if candidates.is_empty() {
        return fallback_swap(sched, rng);
    }
    let swap_pos = candidates[rng.random_range(0..candidates.len())];
    sched.mapping[w].swap(pos, swap_pos);
    MoveUndo::Swap { week: w, pos_a: pos, pos_b: swap_pos }
}

/// Guided late lane balance: fix lane imbalance in late games specifically.
fn guided_late_lane(sched: &mut WinterFixedSchedule, rng: &mut SmallRng) -> MoveUndo {
    let mut late_lane_counts = [0i32; WF_TEAMS * WF_LANES];
    for w in 0..WF_WEEKS {
        let lse = sched.lane_swap_early[w];
        let lsl = sched.lane_swap_late[w];
        for q in 0..WF_QUADS {
            let eq = effective_quad_from(q, lse, lsl);
            if eq < 2 { continue; } // only late
            let base = q * WF_POS_PER_QUAD;
            let lo = lane_off_of_quad(eq);
            let pa = sched.mapping[w][base] as usize;
            let pb = sched.mapping[w][base + 1] as usize;
            let pc = sched.mapping[w][base + 2] as usize;
            let pd = sched.mapping[w][base + 3] as usize;
            late_lane_counts[pa * WF_LANES + lo] += 2;
            late_lane_counts[pb * WF_LANES + lo] += 1;
            late_lane_counts[pb * WF_LANES + lo + 1] += 1;
            late_lane_counts[pc * WF_LANES + lo + 1] += 2;
            late_lane_counts[pd * WF_LANES + lo + 1] += 1;
            late_lane_counts[pd * WF_LANES + lo] += 1;
        }
    }

    let target = WF_WEEKS as f64 / WF_LANES as f64;
    let mut worst_team = 0;
    let mut worst_dev = 0.0f64;
    for t in 0..WF_TEAMS {
        for l in 0..WF_LANES {
            let dev = (late_lane_counts[t * WF_LANES + l] as f64 - target).abs();
            if dev > worst_dev { worst_dev = dev; worst_team = t; }
        }
    }

    if worst_dev < 1.0 {
        // Toggle a late lane swap
        let w = rng.random_range(0..WF_WEEKS);
        sched.lane_swap_late[w] = !sched.lane_swap_late[w];
        return MoveUndo::ToggleLaneLate { week: w };
    }

    // Find worst_team in a late quad and swap within that quad
    let mut good_weeks: Vec<(usize, usize)> = Vec::new();
    for w in 0..WF_WEEKS {
        let lse = sched.lane_swap_early[w];
        let lsl = sched.lane_swap_late[w];
        let pos = sched.mapping[w].iter().position(|&t| t as usize == worst_team).unwrap();
        let eq = effective_quad(pos, lse, lsl);
        if eq >= 2 {
            good_weeks.push((w, pos));
        }
    }

    if good_weeks.is_empty() {
        return fallback_swap(sched, rng);
    }

    let (w, pos) = good_weeks[rng.random_range(0..good_weeks.len())];
    let q_base = quad_of(pos) * WF_POS_PER_QUAD;
    let candidates: Vec<usize> = (q_base..q_base + WF_POS_PER_QUAD)
        .filter(|&p| p != pos)
        .collect();
    let swap_pos = candidates[rng.random_range(0..candidates.len())];
    sched.mapping[w].swap(pos, swap_pos);
    MoveUndo::Swap { week: w, pos_a: pos, pos_b: swap_pos }
}

/// Guided commissioner: reduce overlap in early/late patterns between team pairs.
fn guided_commissioner(sched: &mut WinterFixedSchedule, rng: &mut SmallRng) -> MoveUndo {
    // Compute early/late pattern per team
    let mut early_late = [0u8; WF_TEAMS * WF_WEEKS];
    for w in 0..WF_WEEKS {
        let lse = sched.lane_swap_early[w];
        let lsl = sched.lane_swap_late[w];
        for pos in 0..WF_POSITIONS {
            let team = sched.mapping[w][pos] as usize;
            let eq = effective_quad(pos, lse, lsl);
            early_late[team * WF_WEEKS + w] = if eq < 2 { 1 } else { 0 };
        }
    }

    // Find pair with minimum overlap (same as evaluate)
    let mut min_overlap = WF_WEEKS as u32;
    let mut min_i = 0;
    let mut min_j = 1;
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
                min_i = i;
                min_j = j;
            }
        }
    }

    if min_overlap <= 1 {
        return fallback_swap(sched, rng);
    }

    // Find a week where they have the same pattern and flip one of them
    let target = if rng.random_bool(0.5) { min_i } else { min_j };
    let mut same_weeks: Vec<usize> = Vec::new();
    for w in 0..WF_WEEKS {
        if early_late[min_i * WF_WEEKS + w] == early_late[min_j * WF_WEEKS + w] {
            same_weeks.push(w);
        }
    }

    if same_weeks.is_empty() {
        return fallback_swap(sched, rng);
    }

    let w = same_weeks[rng.random_range(0..same_weeks.len())];
    let pos = sched.mapping[w].iter().position(|&t| t as usize == target).unwrap();
    let eq = effective_quad(pos, sched.lane_swap_early[w], sched.lane_swap_late[w]);
    let in_early = eq < 2;
    // Swap to opposite half
    let target_range = if in_early { 8..16 } else { 0..8 };
    let swap_pos = rng.random_range(target_range);
    sched.mapping[w].swap(pos, swap_pos);
    MoveUndo::Swap { week: w, pos_a: pos, pos_b: swap_pos }
}

/// Guided half-season repeat: fix pairs that play more than once in the same half.
fn guided_half_season(sched: &mut WinterFixedSchedule, rng: &mut SmallRng) -> MoveUndo {
    const HALF: usize = WF_WEEKS / 2;
    let mut fh_matchups = [0u8; WF_TEAMS * WF_TEAMS];
    let mut sh_matchups = [0u8; WF_TEAMS * WF_TEAMS];

    for w in 0..WF_WEEKS {
        for entry in &MATCHUP_ENTRIES {
            let ta = sched.mapping[w][entry.pos_a as usize] as usize;
            let tb = sched.mapping[w][entry.pos_b as usize] as usize;
            let lo = ta.min(tb);
            let hi = ta.max(tb);
            let idx = lo * WF_TEAMS + hi;
            if w < HALF { fh_matchups[idx] += 1; }
            else { sh_matchups[idx] += 1; }
        }
    }

    // Find worst violating pair
    let mut worst_pair = (0, 1);
    let mut worst_excess: u8 = 0;
    let mut in_first_half = true;
    for i in 0..WF_TEAMS {
        for j in (i + 1)..WF_TEAMS {
            let idx = i * WF_TEAMS + j;
            if fh_matchups[idx] > 1 && fh_matchups[idx] - 1 > worst_excess {
                worst_excess = fh_matchups[idx] - 1;
                worst_pair = (i, j);
                in_first_half = true;
            }
            if sh_matchups[idx] > 1 && sh_matchups[idx] - 1 > worst_excess {
                worst_excess = sh_matchups[idx] - 1;
                worst_pair = (i, j);
                in_first_half = false;
            }
        }
    }

    if worst_excess == 0 {
        return fallback_swap(sched, rng);
    }

    // Find a week in the offending half where they play and swap one out
    let range = if in_first_half { 0..HALF } else { HALF..WF_WEEKS };
    let mut play_weeks: Vec<usize> = Vec::new();
    for w in range {
        for entry in &MATCHUP_ENTRIES {
            let ta = sched.mapping[w][entry.pos_a as usize] as usize;
            let tb = sched.mapping[w][entry.pos_b as usize] as usize;
            let lo = ta.min(tb);
            let hi = ta.max(tb);
            if lo == worst_pair.0 && hi == worst_pair.1 {
                play_weeks.push(w);
                break;
            }
        }
    }

    if play_weeks.is_empty() {
        return fallback_swap(sched, rng);
    }

    let w = play_weeks[rng.random_range(0..play_weeks.len())];
    let target = if rng.random_bool(0.5) { worst_pair.0 } else { worst_pair.1 };
    let pos = sched.mapping[w].iter().position(|&t| t as usize == target).unwrap();
    let my_quad = quad_of(pos);
    let same_half: Vec<usize> = (0..WF_POSITIONS)
        .filter(|&p| quad_of(p) != my_quad && ((quad_of(p) < 2) == (my_quad < 2)))
        .collect();
    if same_half.is_empty() {
        return fallback_swap(sched, rng);
    }
    let swap_pos = same_half[rng.random_range(0..same_half.len())];
    sched.mapping[w].swap(pos, swap_pos);
    MoveUndo::Swap { week: w, pos_a: pos, pos_b: swap_pos }
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
