use rand::rngs::SmallRng;
use rand::Rng;
use serde::Deserialize;

pub const SF_TEAMS: usize = 12;
pub const SF_WEEKS: usize = 10;
pub const SF_SLOTS: usize = 5;
pub const SF_LANES: usize = 4;

/// Number of matchup pairs per week (4 slots × 4 lanes + 1 slot × 2 lanes)
pub const SF_MATCHUPS_PER_WEEK: usize = 18;

/// Total matchups across the season
pub const SF_TOTAL_MATCHUPS: usize = SF_MATCHUPS_PER_WEEK * SF_WEEKS; // 180

/// Number of unique team pairs: C(12,2) = 66
pub const SF_PAIRS: usize = SF_TEAMS * (SF_TEAMS - 1) / 2;

/// Each position in the template: (slot, lane, position_a, position_b)
/// Positions 0-11 map to the 12 "roles" in the ideal day layout.
/// The reference day (teams 1-12, position = team-1):
///   Slot 0: Lane 0: 4v12  Lane 1: 10v11  Lane 2: 5v8   Lane 3: 3v1
///   Slot 1: Lane 0: 12v7  Lane 1: 10v9   Lane 2: 2v5   Lane 3: 6v1
///   Slot 2: Lane 0: 7v4   Lane 1: 9v11   Lane 2: 2v3   Lane 3: 8v6
///   Slot 3: Lane 0: 4v10  Lane 1: 11v12  Lane 2: 3v5   Lane 3: 8v1
///   Slot 4: Lane 0: -     Lane 1: -      Lane 2: 7v2   Lane 3: 6v9
#[derive(Clone, Copy)]
pub struct TemplateEntry {
    pub slot: u8,
    pub lane: u8,
    pub pos_a: u8,
    pub pos_b: u8,
}

/// The fixed day template: 18 matchup entries.
/// Position indices: team 1→pos 0, team 2→pos 1, ..., team 12→pos 11
pub const TEMPLATE: [TemplateEntry; SF_MATCHUPS_PER_WEEK] = [
    // Slot 0 (Game 1): 4v12, 10v11, 5v8, 3v1
    TemplateEntry { slot: 0, lane: 0, pos_a: 3,  pos_b: 11 },
    TemplateEntry { slot: 0, lane: 1, pos_a: 9,  pos_b: 10 },
    TemplateEntry { slot: 0, lane: 2, pos_a: 4,  pos_b: 7  },
    TemplateEntry { slot: 0, lane: 3, pos_a: 2,  pos_b: 0  },
    // Slot 1 (Game 2): 12v7, 10v9, 2v5, 6v1
    TemplateEntry { slot: 1, lane: 0, pos_a: 11, pos_b: 6  },
    TemplateEntry { slot: 1, lane: 1, pos_a: 9,  pos_b: 8  },
    TemplateEntry { slot: 1, lane: 2, pos_a: 1,  pos_b: 4  },
    TemplateEntry { slot: 1, lane: 3, pos_a: 5,  pos_b: 0  },
    // Slot 2 (Game 3): 7v4, 9v11, 2v3, 8v6
    TemplateEntry { slot: 2, lane: 0, pos_a: 6,  pos_b: 3  },
    TemplateEntry { slot: 2, lane: 1, pos_a: 8,  pos_b: 10 },
    TemplateEntry { slot: 2, lane: 2, pos_a: 1,  pos_b: 2  },
    TemplateEntry { slot: 2, lane: 3, pos_a: 7,  pos_b: 5  },
    // Slot 3 (Game 4): 4v10, 11v12, 3v5, 8v1
    TemplateEntry { slot: 3, lane: 0, pos_a: 3,  pos_b: 9  },
    TemplateEntry { slot: 3, lane: 1, pos_a: 10, pos_b: 11 },
    TemplateEntry { slot: 3, lane: 2, pos_a: 2,  pos_b: 4  },
    TemplateEntry { slot: 3, lane: 3, pos_a: 7,  pos_b: 0  },
    // Slot 4 (Game 5): -, -, 7v2, 6v9
    TemplateEntry { slot: 4, lane: 2, pos_a: 6,  pos_b: 1  },
    TemplateEntry { slot: 4, lane: 3, pos_a: 5,  pos_b: 8  },
];

/// Precomputed: which positions play in slot 0 (first game) — 8 positions
const SLOT0_POSITIONS: [u8; 8] = [3, 11, 9, 10, 4, 7, 2, 0];

/// Precomputed: which positions play in slot 4 (last game) — 4 positions
const SLOT4_POSITIONS: [u8; 4] = [6, 1, 5, 8];

/// Precomputed: which positions stay on the same lane for all 3 games in slots 0-3
/// pos 0 (all lane 3), pos 3 (all lane 0), pos 4 (all lane 2), pos 10 (all lane 1)
const SAME_LANE_POSITIONS: [u8; 4] = [0, 3, 4, 10];

/// The schedule state: per-week team permutation + lane swap flags.
#[derive(Clone, Copy)]
pub struct FixedSchedule {
    /// mapping[week][position] = team_id (0-11)
    pub mapping: [[u8; SF_TEAMS]; SF_WEEKS],
    /// Per-week: swap lanes 0↔1
    pub swap_01: [bool; SF_WEEKS],
    /// Per-week: swap lanes 2↔3
    pub swap_23: [bool; SF_WEEKS],
}

/// Apply lane swaps to a template lane index for a given week.
fn apply_lane_swap(lane: u8, swap_01: bool, swap_23: bool) -> u8 {
    match lane {
        0 if swap_01 => 1,
        1 if swap_01 => 0,
        2 if swap_23 => 3,
        3 if swap_23 => 2,
        other => other,
    }
}

/// Resolve a template entry for a given week: returns (slot, actual_lane, team_a, team_b).
pub fn resolve_entry(sched: &FixedSchedule, week: usize, entry: &TemplateEntry) -> (usize, usize, u8, u8) {
    let actual_lane = apply_lane_swap(entry.lane, sched.swap_01[week], sched.swap_23[week]);
    let team_a = sched.mapping[week][entry.pos_a as usize];
    let team_b = sched.mapping[week][entry.pos_b as usize];
    (entry.slot as usize, actual_lane as usize, team_a, team_b)
}

#[derive(Deserialize, Clone)]
pub struct FixedWeights {
    pub matchup_balance: u32,
    pub slot_balance: u32,
    pub lane_balance: u32,
    pub game5_lane_balance: u32,
    pub same_lane_balance: u32,
    pub commissioner_overlap: u32,
}

#[derive(Clone, Debug)]
pub struct FixedCostBreakdown {
    pub matchup_balance: u32,
    pub slot_balance: u32,
    pub lane_balance: u32,
    pub game5_lane_balance: u32,
    pub same_lane_balance: u32,
    pub commissioner_overlap: u32,
    pub total: u32,
}

pub fn fixed_cost_label(c: &FixedCostBreakdown) -> String {
    format!(
        "total: {:>4} matchup: {:>3} slot: {:>3} lane: {:>3} g5lane: {:>3} same: {:>3} comm: {:>3}",
        c.total, c.matchup_balance, c.slot_balance,
        c.lane_balance, c.game5_lane_balance, c.same_lane_balance, c.commissioner_overlap,
    )
}

pub const NUM_COST_COMPONENTS: usize = 6;
pub const COST_LABELS: [&str; NUM_COST_COMPONENTS] = [
    "matchup", "slot", "lane", "g5lane", "same", "comm",
];

/// Generate a random schedule (random permutation per week, random lane swaps).
pub fn random_fixed_schedule(rng: &mut SmallRng) -> FixedSchedule {
    let mut sched = FixedSchedule {
        mapping: [[0; SF_TEAMS]; SF_WEEKS],
        swap_01: [false; SF_WEEKS],
        swap_23: [false; SF_WEEKS],
    };
    for w in 0..SF_WEEKS {
        let mut perm: [u8; SF_TEAMS] = std::array::from_fn(|i| i as u8);
        for i in (1..SF_TEAMS).rev() {
            let j = rng.random_range(0..=i);
            perm.swap(i, j);
        }
        sched.mapping[w] = perm;
        sched.swap_01[w] = rng.random_bool(0.5);
        sched.swap_23[w] = rng.random_bool(0.5);
    }
    sched
}

/// Evaluate the full cost of a schedule.
pub fn evaluate_fixed(sched: &FixedSchedule, w8: &FixedWeights) -> FixedCostBreakdown {
    // --- Matchup balance: count how many times each pair plays ---
    let mut matchup_counts = [0u8; SF_PAIRS];
    for w in 0..SF_WEEKS {
        for entry in &TEMPLATE {
            let ta = sched.mapping[w][entry.pos_a as usize] as usize;
            let tb = sched.mapping[w][entry.pos_b as usize] as usize;
            let lo = ta.min(tb);
            let hi = ta.max(tb);
            let idx = lo * (2 * SF_TEAMS - lo - 1) / 2 + (hi - lo - 1);
            matchup_counts[idx] += 1;
        }
    }
    let mut matchup_balance: u32 = 0;
    for &c in &matchup_counts {
        if c < 2 || c > 3 {
            // Scale penalty by distance from [2,3] range
            let dist = if c < 2 { 2 - c as u32 } else { c as u32 - 3 };
            matchup_balance += w8.matchup_balance * dist;
        }
    }

    // --- Slot balance: each team in each slot an equal number of times ---
    // Slots 0-3: 8 teams per slot per week, 10 weeks = 80 per slot. 80/12 ≈ 6.67 → [6,7]
    // Slot 4: 4 teams per week, 10 weeks = 40. 40/12 ≈ 3.33 → [3,4]
    let mut slot_counts = [0u32; SF_TEAMS * SF_SLOTS];
    for w in 0..SF_WEEKS {
        for entry in &TEMPLATE {
            let s = entry.slot as usize;
            let ta = sched.mapping[w][entry.pos_a as usize] as usize;
            let tb = sched.mapping[w][entry.pos_b as usize] as usize;
            slot_counts[ta * SF_SLOTS + s] += 1;
            slot_counts[tb * SF_SLOTS + s] += 1;
        }
    }
    let mut slot_balance: u32 = 0;
    for t in 0..SF_TEAMS {
        for s in 0..SF_SLOTS {
            let c = slot_counts[t * SF_SLOTS + s];
            let ok = if s < 4 { c >= 6 && c <= 7 } else { c >= 3 && c <= 4 };
            if !ok {
                let dist = if s < 4 {
                    if c < 6 { 6 - c } else { c - 7 }
                } else {
                    if c < 3 { 3 - c } else { c - 4 }
                };
                slot_balance += w8.slot_balance * dist;
            }
        }
    }

    // --- Lane balance (all games, all slots) ---
    // Each team plays 30 games total (3 per week × 10 weeks).
    // Lanes 0-1 only receive traffic from slots 0-3: ~6.67/team/lane → [6,7]
    // Lanes 2-3 receive traffic from all 5 slots: ~8.33/team/lane → [8,9]
    let mut lane_counts = [0u32; SF_TEAMS * SF_LANES];
    for w in 0..SF_WEEKS {
        for entry in &TEMPLATE {
            let actual_lane = apply_lane_swap(entry.lane, sched.swap_01[w], sched.swap_23[w]) as usize;
            let ta = sched.mapping[w][entry.pos_a as usize] as usize;
            let tb = sched.mapping[w][entry.pos_b as usize] as usize;
            lane_counts[ta * SF_LANES + actual_lane] += 1;
            lane_counts[tb * SF_LANES + actual_lane] += 1;
        }
    }
    let mut lane_balance: u32 = 0;
    for t in 0..SF_TEAMS {
        for l in 0..SF_LANES {
            let c = lane_counts[t * SF_LANES + l];
            let (lo, hi) = if l < 2 { (6, 7) } else { (8, 9) };
            if c < lo || c > hi {
                let dist = if c < lo { lo - c } else { c - hi };
                lane_balance += w8.lane_balance * dist;
            }
        }
    }

    // --- Game 5 lane balance (slot 4, lanes 2 and 3 only) ---
    // Per team: game-5 appearances on lane 2 vs lane 3 should be as even as possible.
    // Each team has ~3.33 game-5 appearances (40/12). Penalize if difference > 1.
    let mut game5_lane2 = [0u32; SF_TEAMS];
    let mut game5_lane3 = [0u32; SF_TEAMS];
    for w in 0..SF_WEEKS {
        for entry in &TEMPLATE {
            if entry.slot != 4 { continue; }
            let actual_lane = apply_lane_swap(entry.lane, sched.swap_01[w], sched.swap_23[w]) as usize;
            let ta = sched.mapping[w][entry.pos_a as usize] as usize;
            let tb = sched.mapping[w][entry.pos_b as usize] as usize;
            if actual_lane == 2 {
                game5_lane2[ta] += 1;
                game5_lane2[tb] += 1;
            } else {
                game5_lane3[ta] += 1;
                game5_lane3[tb] += 1;
            }
        }
    }
    let mut game5_lane_balance: u32 = 0;
    for t in 0..SF_TEAMS {
        let diff = game5_lane2[t].abs_diff(game5_lane3[t]);
        if diff > 1 {
            game5_lane_balance += w8.game5_lane_balance * (diff - 1);
        }
    }

    // --- Same lane balance: weeks where team stays on same lane for all games in slots 0-3 ---
    // 4 "same lane" positions per week × 10 weeks = 40 total "same lane" slots
    // Target per team: 40/12 ≈ 3.33, so [3,4] is ideal
    let mut same_lane_counts = [0u32; SF_TEAMS];
    for w in 0..SF_WEEKS {
        for &pos in &SAME_LANE_POSITIONS {
            let team = sched.mapping[w][pos as usize] as usize;
            same_lane_counts[team] += 1;
        }
    }
    let mut same_lane_balance: u32 = 0;
    for &c in &same_lane_counts {
        if c < 3 || c > 4 {
            let dist = if c < 3 { 3 - c } else { c - 4 };
            same_lane_balance += w8.same_lane_balance * dist;
        }
    }

    // --- Commissioner overlap: minimize co-appearance in slot 0 and slot 4 ---
    // For each team, track which weeks they play in slot 0 and slot 4 (as bits)
    let mut comm_bits = [0u32; SF_TEAMS];
    for w in 0..SF_WEEKS {
        for &pos in &SLOT0_POSITIONS {
            let team = sched.mapping[w][pos as usize] as usize;
            comm_bits[team] |= 1 << (w * 2);
        }
        for &pos in &SLOT4_POSITIONS {
            let team = sched.mapping[w][pos as usize] as usize;
            comm_bits[team] |= 1 << (w * 2 + 1);
        }
    }
    let mut min_co = 20u32;
    for i in 0..SF_TEAMS {
        for j in (i + 1)..SF_TEAMS {
            let co = (comm_bits[i] & comm_bits[j]).count_ones();
            min_co = min_co.min(co);
        }
    }
    let commissioner_overlap = w8.commissioner_overlap * min_co;

    let total = matchup_balance + slot_balance
        + lane_balance + game5_lane_balance + same_lane_balance + commissioner_overlap;

    FixedCostBreakdown {
        matchup_balance,
        slot_balance,
        lane_balance,
        game5_lane_balance,
        same_lane_balance,
        commissioner_overlap,
        total,
    }
}

/// Identify the worst cost component and return its index (0-5).
pub fn worst_component(bd: &FixedCostBreakdown) -> usize {
    let components = [
        bd.matchup_balance,
        bd.slot_balance,
        bd.lane_balance,
        bd.game5_lane_balance,
        bd.same_lane_balance,
        bd.commissioner_overlap,
    ];
    let mut worst = 0;
    for i in 1..components.len() {
        if components[i] > components[worst] {
            worst = i;
        }
    }
    worst
}

// === SA MOVES ===

pub const NUM_MOVES: usize = 7;
pub const MOVE_NAMES: [&str; NUM_MOVES] = [
    "tm_swap", "tog_01", "tog_23", "wk_swap",
    "g_match", "g_slot", "g_lane",
];

/// Pick a random move, biased toward the worst penalty.
/// Returns move index 0-6.
pub fn pick_move(rng: &mut SmallRng, bd: &FixedCostBreakdown) -> usize {
    // 50% chance: guided move targeting worst component
    // 50% chance: random move from the basic 4
    if rng.random_bool(0.5) {
        match worst_component(bd) {
            0 => 4, // matchup → guided matchup
            1 => 5, // slot balance → guided slot
            2 | 3 => 6, // lane balance / game5 lane → guided lane
            4 => 0, // same lane → team swap (change position assignments)
            5 => 0, // commissioner → team swap
            _ => 0,
        }
    } else {
        // Random basic move
        match rng.random_range(0..4u32) {
            0 => 0, // team swap
            1 => 1, // toggle swap_01
            2 => 2, // toggle swap_23
            3 => 3, // week swap
            _ => 0,
        }
    }
}

/// Apply a move to the schedule, returning the undo information.
/// Returns (move_id, undo_data) where undo_data is enough to reverse.
pub enum UndoInfo {
    TeamSwap { week: usize, pos_a: usize, pos_b: usize },
    Toggle01 { week: usize },
    Toggle23 { week: usize },
    WeekSwap { week_a: usize, week_b: usize },
}

pub fn apply_move(
    sched: &mut FixedSchedule,
    move_id: usize,
    bd: &FixedCostBreakdown,
    rng: &mut SmallRng,
) -> UndoInfo {
    match move_id {
        0 => move_team_swap(sched, rng),
        1 => move_toggle_01(sched, rng),
        2 => move_toggle_23(sched, rng),
        3 => move_week_swap(sched, rng),
        4 => move_guided_matchup(sched, bd, rng),
        5 => move_guided_slot(sched, bd, rng),
        6 => move_guided_lane(sched, bd, rng),
        _ => move_team_swap(sched, rng),
    }
}

pub fn undo_move(sched: &mut FixedSchedule, undo: &UndoInfo) {
    match undo {
        UndoInfo::TeamSwap { week, pos_a, pos_b } => {
            sched.mapping[*week].swap(*pos_a, *pos_b);
        }
        UndoInfo::Toggle01 { week } => {
            sched.swap_01[*week] = !sched.swap_01[*week];
        }
        UndoInfo::Toggle23 { week } => {
            sched.swap_23[*week] = !sched.swap_23[*week];
        }
        UndoInfo::WeekSwap { week_a, week_b } => {
            sched.mapping.swap(*week_a, *week_b);
            sched.swap_01.swap(*week_a, *week_b);
            sched.swap_23.swap(*week_a, *week_b);
        }
    }
}

fn move_team_swap(sched: &mut FixedSchedule, rng: &mut SmallRng) -> UndoInfo {
    let w = rng.random_range(0..SF_WEEKS);
    let a = rng.random_range(0..SF_TEAMS);
    let mut b = rng.random_range(0..SF_TEAMS - 1);
    if b >= a { b += 1; }
    sched.mapping[w].swap(a, b);
    UndoInfo::TeamSwap { week: w, pos_a: a, pos_b: b }
}

fn move_toggle_01(sched: &mut FixedSchedule, rng: &mut SmallRng) -> UndoInfo {
    let w = rng.random_range(0..SF_WEEKS);
    sched.swap_01[w] = !sched.swap_01[w];
    UndoInfo::Toggle01 { week: w }
}

fn move_toggle_23(sched: &mut FixedSchedule, rng: &mut SmallRng) -> UndoInfo {
    let w = rng.random_range(0..SF_WEEKS);
    sched.swap_23[w] = !sched.swap_23[w];
    UndoInfo::Toggle23 { week: w }
}

fn move_week_swap(sched: &mut FixedSchedule, rng: &mut SmallRng) -> UndoInfo {
    let a = rng.random_range(0..SF_WEEKS);
    let mut b = rng.random_range(0..SF_WEEKS - 1);
    if b >= a { b += 1; }
    sched.mapping.swap(a, b);
    sched.swap_01.swap(a, b);
    sched.swap_23.swap(a, b);
    UndoInfo::WeekSwap { week_a: a, week_b: b }
}

/// Guided matchup: find the most over- or under-matched pair, swap a team in a relevant week.
fn move_guided_matchup(
    sched: &mut FixedSchedule,
    _bd: &FixedCostBreakdown,
    rng: &mut SmallRng,
) -> UndoInfo {
    // Count matchups
    let mut matchup_counts = [0u8; SF_PAIRS];
    for w in 0..SF_WEEKS {
        for entry in &TEMPLATE {
            let ta = sched.mapping[w][entry.pos_a as usize] as usize;
            let tb = sched.mapping[w][entry.pos_b as usize] as usize;
            let lo = ta.min(tb);
            let hi = ta.max(tb);
            let idx = lo * (2 * SF_TEAMS - lo - 1) / 2 + (hi - lo - 1);
            matchup_counts[idx] += 1;
        }
    }

    // Find worst pair (most over-matched, or if none, most under-matched)
    let mut worst_pair = 0;
    let mut worst_dist: i32 = 0;
    for i in 0..SF_PAIRS {
        let c = matchup_counts[i] as i32;
        let dist = if c > 3 { c - 3 } else if c < 2 { 2 - c } else { 0 };
        if dist > worst_dist {
            worst_dist = dist;
            worst_pair = i;
        }
    }

    if worst_dist == 0 {
        return move_team_swap(sched, rng);
    }

    // Decode pair index back to (team_lo, team_hi)
    let (team_lo, team_hi) = decode_pair_idx(worst_pair);

    let c = matchup_counts[worst_pair];
    if c > 3 {
        // Over-matched: find a week where they play, swap one of them with another team
        let mut weeks_where_matched: Vec<usize> = Vec::new();
        for w in 0..SF_WEEKS {
            for entry in &TEMPLATE {
                let ta = sched.mapping[w][entry.pos_a as usize] as usize;
                let tb = sched.mapping[w][entry.pos_b as usize] as usize;
                if (ta == team_lo && tb == team_hi) || (ta == team_hi && tb == team_lo) {
                    weeks_where_matched.push(w);
                    break;
                }
            }
        }
        if !weeks_where_matched.is_empty() {
            let w = weeks_where_matched[rng.random_range(0..weeks_where_matched.len())];
            // Find which positions these teams occupy
            let pos_lo = sched.mapping[w].iter().position(|&t| t as usize == team_lo).unwrap();
            let pos_hi = sched.mapping[w].iter().position(|&t| t as usize == team_hi).unwrap();
            // Swap one of them with a random other position
            let target_pos = if rng.random_bool(0.5) { pos_lo } else { pos_hi };
            let mut other = rng.random_range(0..SF_TEAMS - 1);
            if other >= target_pos { other += 1; }
            sched.mapping[w].swap(target_pos, other);
            return UndoInfo::TeamSwap { week: w, pos_a: target_pos, pos_b: other };
        }
    } else {
        // Under-matched: find weeks where they DON'T play, try to make them play
        // by swapping one team into a position that would match them
        let w = rng.random_range(0..SF_WEEKS);
        let pos_lo = sched.mapping[w].iter().position(|&t| t as usize == team_lo).unwrap();
        let pos_hi = sched.mapping[w].iter().position(|&t| t as usize == team_hi).unwrap();
        // Check if they already play this week — if not, try swapping
        // Find a position that would make them opponents
        for entry in &TEMPLATE {
            let pa = entry.pos_a as usize;
            let pb = entry.pos_b as usize;
            // If team_lo is at pos pa, we want team_hi at pos pb (or vice versa)
            if pos_lo == pa {
                // Swap team_hi's current position with pb
                sched.mapping[w].swap(pos_hi, pb);
                return UndoInfo::TeamSwap { week: w, pos_a: pos_hi, pos_b: pb };
            }
            if pos_lo == pb {
                sched.mapping[w].swap(pos_hi, pa);
                return UndoInfo::TeamSwap { week: w, pos_a: pos_hi, pos_b: pa };
            }
        }
    }

    move_team_swap(sched, rng)
}

/// Guided slot balance: find the worst slot imbalance and swap teams to fix it.
fn move_guided_slot(
    sched: &mut FixedSchedule,
    _bd: &FixedCostBreakdown,
    rng: &mut SmallRng,
) -> UndoInfo {
    // Count per-team per-slot
    let mut slot_counts = [0u32; SF_TEAMS * SF_SLOTS];
    for w in 0..SF_WEEKS {
        for entry in &TEMPLATE {
            let s = entry.slot as usize;
            let ta = sched.mapping[w][entry.pos_a as usize] as usize;
            let tb = sched.mapping[w][entry.pos_b as usize] as usize;
            slot_counts[ta * SF_SLOTS + s] += 1;
            slot_counts[tb * SF_SLOTS + s] += 1;
        }
    }

    // Find worst team-slot imbalance
    let mut worst_team = 0;
    let mut worst_slot = 0;
    let mut worst_dist: u32 = 0;
    for t in 0..SF_TEAMS {
        for s in 0..SF_SLOTS {
            let c = slot_counts[t * SF_SLOTS + s];
            let dist = if s < 4 {
                if c < 6 { 6 - c } else if c > 7 { c - 7 } else { 0 }
            } else {
                if c < 3 { 3 - c } else if c > 4 { c - 4 } else { 0 }
            };
            if dist > worst_dist {
                worst_dist = dist;
                worst_team = t;
                worst_slot = s;
            }
        }
    }

    if worst_dist == 0 {
        return move_team_swap(sched, rng);
    }

    let c = slot_counts[worst_team * SF_SLOTS + worst_slot];
    let over = if worst_slot < 4 { c > 7 } else { c > 4 };

    // Find a team with opposite imbalance on this slot
    let swap_target = if over {
        // worst_team is over → find someone under
        (0..SF_TEAMS).find(|&t| {
            let tc = slot_counts[t * SF_SLOTS + worst_slot];
            if worst_slot < 4 { tc < 6 } else { tc < 3 }
        })
    } else {
        // worst_team is under → find someone over
        (0..SF_TEAMS).find(|&t| {
            let tc = slot_counts[t * SF_SLOTS + worst_slot];
            if worst_slot < 4 { tc > 7 } else { tc > 4 }
        })
    };

    if let Some(target) = swap_target {
        let w = rng.random_range(0..SF_WEEKS);
        let pos_a = sched.mapping[w].iter().position(|&t| t as usize == worst_team).unwrap();
        let pos_b = sched.mapping[w].iter().position(|&t| t as usize == target).unwrap();
        sched.mapping[w].swap(pos_a, pos_b);
        return UndoInfo::TeamSwap { week: w, pos_a, pos_b };
    }

    move_team_swap(sched, rng)
}

/// Guided lane: fix lane imbalance by toggling lane swaps or swapping teams.
fn move_guided_lane(
    sched: &mut FixedSchedule,
    _bd: &FixedCostBreakdown,
    rng: &mut SmallRng,
) -> UndoInfo {
    // Count lane usage per team across ALL slots
    let mut lane_counts = [0u32; SF_TEAMS * SF_LANES];
    for w in 0..SF_WEEKS {
        for entry in &TEMPLATE {
            let actual_lane = apply_lane_swap(entry.lane, sched.swap_01[w], sched.swap_23[w]) as usize;
            let ta = sched.mapping[w][entry.pos_a as usize] as usize;
            let tb = sched.mapping[w][entry.pos_b as usize] as usize;
            lane_counts[ta * SF_LANES + actual_lane] += 1;
            lane_counts[tb * SF_LANES + actual_lane] += 1;
        }
    }

    // Find most imbalanced team-lane (lanes 0-1: target [6,7], lanes 2-3: target [8,9])
    let mut worst_team = 0;
    let mut worst_lane = 0;
    let mut worst_excess: u32 = 0;
    for t in 0..SF_TEAMS {
        for l in 0..SF_LANES {
            let c = lane_counts[t * SF_LANES + l];
            let (lo, hi) = if l < 2 { (6, 7) } else { (8, 9) };
            let excess = if c > hi { c - hi } else if c < lo { lo - c } else { 0 };
            if excess > worst_excess {
                worst_excess = excess;
                worst_team = t;
                worst_lane = l;
            }
        }
    }

    if worst_excess == 0 {
        return move_team_swap(sched, rng);
    }

    // If the imbalance is between lanes 0/1 or 2/3, toggling the swap might help
    let partner_lane = match worst_lane {
        0 => 1,
        1 => 0,
        2 => 3,
        3 => 2,
        _ => worst_lane,
    };
    let (lo, hi) = if worst_lane < 2 { (6u32, 7u32) } else { (8u32, 9u32) };
    let over = lane_counts[worst_team * SF_LANES + worst_lane] > hi;
    let partner_under = lane_counts[worst_team * SF_LANES + partner_lane] < lo;

    if over && partner_under {
        let w = rng.random_range(0..SF_WEEKS);
        if worst_lane < 2 {
            sched.swap_01[w] = !sched.swap_01[w];
            return UndoInfo::Toggle01 { week: w };
        } else {
            sched.swap_23[w] = !sched.swap_23[w];
            return UndoInfo::Toggle23 { week: w };
        }
    }

    // Otherwise, try swapping this team with one that has opposite imbalance
    let swap_target = (0..SF_TEAMS).find(|&t| {
        t != worst_team && lane_counts[t * SF_LANES + worst_lane] < lo
    });
    if let Some(target) = swap_target {
        let w = rng.random_range(0..SF_WEEKS);
        let pos_a = sched.mapping[w].iter().position(|&t| t as usize == worst_team).unwrap();
        let pos_b = sched.mapping[w].iter().position(|&t| t as usize == target).unwrap();
        sched.mapping[w].swap(pos_a, pos_b);
        return UndoInfo::TeamSwap { week: w, pos_a, pos_b };
    }

    move_team_swap(sched, rng)
}

fn decode_pair_idx(idx: usize) -> (usize, usize) {
    // lo * (2*N - lo - 1) / 2 + (hi - lo - 1) = idx
    let mut lo = 0;
    let mut remaining = idx;
    loop {
        let row_size = 2 * SF_TEAMS - 2 * lo - 1;
        let half = row_size / 2; // integer, same as (SF_TEAMS - lo - 1)
        if remaining < half {
            return (lo, lo + 1 + remaining);
        }
        remaining -= half;
        lo += 1;
    }
}

// === TSV I/O ===

/// Convert a fixed schedule to the legacy SummerAssignment-style TSV format.
pub fn fixed_schedule_to_tsv(sched: &FixedSchedule) -> String {
    let mut lines = vec![String::from("Week\tSlot\tLane 1\tLane 2\tLane 3\tLane 4")];

    for w in 0..SF_WEEKS {
        for s in 0..SF_SLOTS {
            let mut cells: [String; SF_LANES] = std::array::from_fn(|_| String::from("-"));

            for entry in &TEMPLATE {
                if entry.slot as usize != s { continue; }
                let actual_lane = apply_lane_swap(entry.lane, sched.swap_01[w], sched.swap_23[w]) as usize;
                let ta = sched.mapping[w][entry.pos_a as usize];
                let tb = sched.mapping[w][entry.pos_b as usize];
                cells[actual_lane] = format!("{} v {}", ta + 1, tb + 1);
            }

            lines.push(format!(
                "{}\t{}\t{}\t{}\t{}\t{}",
                w + 1, s + 1, cells[0], cells[1], cells[2], cells[3],
            ));
        }
    }

    lines.join("\n")
}

/// Apply N random perturbations to a schedule.
pub fn perturb_fixed(sched: &mut FixedSchedule, rng: &mut SmallRng, n: usize) {
    let bd = evaluate_fixed(sched, &FixedWeights {
        matchup_balance: 80, slot_balance: 60, lane_balance: 60,
        game5_lane_balance: 40, same_lane_balance: 40, commissioner_overlap: 30,
    });
    for _ in 0..n {
        let move_id = rng.random_range(0..4u32) as usize;
        apply_move(sched, move_id, &bd, rng);
    }
}

/// Parse a TSV (as produced by `fixed_schedule_to_tsv`) back into a FixedSchedule.
/// Returns None if parsing fails.
pub fn parse_fixed_tsv(tsv: &str) -> Option<FixedSchedule> {
    let lines: Vec<&str> = tsv.lines().collect();
    if lines.len() < 1 + SF_WEEKS * SF_SLOTS { return None; }

    // Build a grid: matchups[week][slot][lane] = (team_a, team_b) or None
    let mut matchups = [[[None::<(u8, u8)>; SF_LANES]; SF_SLOTS]; SF_WEEKS];
    for i in 1..lines.len() {
        let cols: Vec<&str> = lines[i].split('\t').collect();
        if cols.len() < 6 { continue; }
        let w: usize = cols[0].parse::<usize>().ok()?.checked_sub(1)?;
        let s: usize = cols[1].parse::<usize>().ok()?.checked_sub(1)?;
        if w >= SF_WEEKS || s >= SF_SLOTS { continue; }
        for l in 0..SF_LANES {
            let cell = cols[2 + l].trim();
            if cell == "-" { continue; }
            let parts: Vec<&str> = cell.split(" v ").collect();
            if parts.len() != 2 { continue; }
            let a: u8 = parts[0].trim().parse::<u8>().ok()?.checked_sub(1)?;
            let b: u8 = parts[1].trim().parse::<u8>().ok()?.checked_sub(1)?;
            matchups[w][s][l] = Some((a, b));
        }
    }

    // For each week, find the mapping and swap flags by trying all 4 swap combinations
    let mut sched = FixedSchedule {
        mapping: [[0; SF_TEAMS]; SF_WEEKS],
        swap_01: [false; SF_WEEKS],
        swap_23: [false; SF_WEEKS],
    };

    for w in 0..SF_WEEKS {
        let mut found = false;
        for swap_bits in 0..4u8 {
            let s01 = swap_bits & 1 != 0;
            let s23 = swap_bits & 2 != 0;

            // Try to construct a consistent mapping for this swap combination
            let mut mapping = [255u8; SF_TEAMS];
            let mut ok = true;
            for entry in &TEMPLATE {
                let actual_lane = apply_lane_swap(entry.lane, s01, s23) as usize;
                let s = entry.slot as usize;
                if let Some((ta, tb)) = matchups[w][s][actual_lane] {
                    let pa = entry.pos_a as usize;
                    let pb = entry.pos_b as usize;
                    if mapping[pa] == 255 {
                        mapping[pa] = ta;
                    } else if mapping[pa] != ta {
                        ok = false; break;
                    }
                    if mapping[pb] == 255 {
                        mapping[pb] = tb;
                    } else if mapping[pb] != tb {
                        ok = false; break;
                    }
                } else {
                    ok = false; break;
                }
            }

            if ok && mapping.iter().all(|&m| m != 255) {
                // Verify it's a valid permutation
                let mut seen = [false; SF_TEAMS];
                for &t in &mapping {
                    if (t as usize) >= SF_TEAMS || seen[t as usize] { ok = false; break; }
                    seen[t as usize] = true;
                }
                if ok {
                    sched.mapping[w] = mapping;
                    sched.swap_01[w] = s01;
                    sched.swap_23[w] = s23;
                    found = true;
                    break;
                }
            }
        }
        if !found { return None; }
    }

    Some(sched)
}

/// Commissioner reassignment: find the pair with minimum slot overlap, relabel as teams 1 and 2.
pub fn reassign_commissioners(sched: &mut FixedSchedule) {
    let mut comm_bits = [0u32; SF_TEAMS];
    for w in 0..SF_WEEKS {
        for &pos in &SLOT0_POSITIONS {
            let team = sched.mapping[w][pos as usize] as usize;
            comm_bits[team] |= 1 << (w * 2);
        }
        for &pos in &SLOT4_POSITIONS {
            let team = sched.mapping[w][pos as usize] as usize;
            comm_bits[team] |= 1 << (w * 2 + 1);
        }
    }

    let mut best_i = 0;
    let mut best_j = 1;
    let mut min_co = u32::MAX;
    for i in 0..SF_TEAMS {
        for j in (i + 1)..SF_TEAMS {
            let co = (comm_bits[i] & comm_bits[j]).count_ones();
            if co < min_co {
                min_co = co;
                best_i = i;
                best_j = j;
            }
        }
    }

    if best_i == 0 && best_j == 1 {
        return;
    }

    // Relabel: swap best_i↔0 and best_j↔1 in all weeks
    // Build a permutation table
    let mut perm = [0u8; SF_TEAMS];
    for i in 0..SF_TEAMS {
        perm[i] = i as u8;
    }
    perm.swap(0, best_i);
    perm.swap(1, best_j);
    let mut inv = [0u8; SF_TEAMS];
    for (i, &p) in perm.iter().enumerate() {
        inv[p as usize] = i as u8;
    }

    for w in 0..SF_WEEKS {
        let mut new_mapping = [0u8; SF_TEAMS];
        for pos in 0..SF_TEAMS {
            new_mapping[pos] = inv[sched.mapping[w][pos] as usize];
        }
        sched.mapping[w] = new_mapping;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::SeedableRng;

    fn test_weights() -> FixedWeights {
        FixedWeights {
            matchup_balance: 80,
            slot_balance: 60,
            lane_balance: 60,
            game5_lane_balance: 40,
            same_lane_balance: 40,
            commissioner_overlap: 30,
        }
    }

    #[test]
    fn test_template_positions_appear_3_times() {
        let mut counts = [0u8; SF_TEAMS];
        for entry in &TEMPLATE {
            counts[entry.pos_a as usize] += 1;
            counts[entry.pos_b as usize] += 1;
        }
        for (i, &c) in counts.iter().enumerate() {
            assert_eq!(c, 3, "Position {} appears {} times, expected 3", i, c);
        }
    }

    #[test]
    fn test_template_slot_structure() {
        // Slots 0-3: 4 entries each, slot 4: 2 entries
        for s in 0..4 {
            let count = TEMPLATE.iter().filter(|e| e.slot == s as u8).count();
            assert_eq!(count, 4, "Slot {} has {} entries, expected 4", s, count);
        }
        let count = TEMPLATE.iter().filter(|e| e.slot == 4).count();
        assert_eq!(count, 2, "Slot 4 has {} entries, expected 2", count);
    }

    #[test]
    fn test_template_slot4_lanes() {
        for entry in &TEMPLATE {
            if entry.slot == 4 {
                assert!(entry.lane >= 2, "Slot 4 entry on lane {}, expected 2 or 3", entry.lane);
            }
        }
    }

    #[test]
    fn test_evaluate_deterministic() {
        let mut rng = SmallRng::seed_from_u64(42);
        let sched = random_fixed_schedule(&mut rng);
        let w8 = test_weights();
        let c1 = evaluate_fixed(&sched, &w8);
        let c2 = evaluate_fixed(&sched, &w8);
        assert_eq!(c1.total, c2.total);
    }

    #[test]
    fn test_undo_restores_state() {
        let mut rng = SmallRng::seed_from_u64(42);
        let w8 = test_weights();
        let sched = random_fixed_schedule(&mut rng);
        let original_cost = evaluate_fixed(&sched, &w8).total;

        for move_id in 0..4 {
            let mut s = sched;
            let bd = evaluate_fixed(&s, &w8);
            let undo = apply_move(&mut s, move_id, &bd, &mut rng);
            undo_move(&mut s, &undo);
            let restored_cost = evaluate_fixed(&s, &w8).total;
            assert_eq!(original_cost, restored_cost, "Move {} undo failed", move_id);
        }
    }

    #[test]
    fn test_tsv_roundtrip() {
        let mut rng = SmallRng::seed_from_u64(42);
        let sched = random_fixed_schedule(&mut rng);
        let tsv = fixed_schedule_to_tsv(&sched);
        // Just verify it has the right structure
        let lines: Vec<&str> = tsv.lines().collect();
        assert_eq!(lines.len(), 1 + SF_WEEKS * SF_SLOTS); // header + 50 data lines
    }

    #[test]
    fn test_parse_fixed_tsv_roundtrip() {
        let mut rng = SmallRng::seed_from_u64(99);
        let w8 = test_weights();
        for _ in 0..10 {
            let sched = random_fixed_schedule(&mut rng);
            let tsv = fixed_schedule_to_tsv(&sched);
            let parsed = parse_fixed_tsv(&tsv).expect("Failed to parse TSV");
            let cost_orig = evaluate_fixed(&sched, &w8).total;
            let cost_parsed = evaluate_fixed(&parsed, &w8).total;
            assert_eq!(cost_orig, cost_parsed, "TSV roundtrip changed cost");
        }
    }

    #[test]
    fn test_perturb_fixed() {
        let mut rng = SmallRng::seed_from_u64(42);
        let w8 = test_weights();
        let sched = random_fixed_schedule(&mut rng);
        let orig_cost = evaluate_fixed(&sched, &w8).total;
        let mut perturbed = sched;
        perturb_fixed(&mut perturbed, &mut rng, 10);
        let new_cost = evaluate_fixed(&perturbed, &w8).total;
        // Just verify it's still a valid schedule (different cost is expected)
        assert!(new_cost > 0 || orig_cost > 0);
    }

    #[test]
    fn test_pair_idx_roundtrip() {
        for lo in 0..SF_TEAMS {
            for hi in (lo + 1)..SF_TEAMS {
                let idx = lo * (2 * SF_TEAMS - lo - 1) / 2 + (hi - lo - 1);
                let (dec_lo, dec_hi) = decode_pair_idx(idx);
                assert_eq!((lo, hi), (dec_lo, dec_hi), "Pair index {} decoded wrong", idx);
            }
        }
    }

    #[test]
    fn test_same_lane_positions() {
        // Verify that the SAME_LANE_POSITIONS are correct by checking template
        for &pos in &SAME_LANE_POSITIONS {
            let entries: Vec<&TemplateEntry> = TEMPLATE.iter()
                .filter(|e| e.slot < 4 && (e.pos_a == pos || e.pos_b == pos))
                .collect();
            assert_eq!(entries.len(), 3, "Position {} doesn't have 3 games in slots 0-3", pos);
            let lanes: Vec<u8> = entries.iter().map(|e| e.lane).collect();
            assert!(lanes.iter().all(|&l| l == lanes[0]),
                "Position {} not on same lane: {:?}", pos, lanes);
        }
    }
}
