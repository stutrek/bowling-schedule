use rand::rngs::SmallRng;
use rand::Rng;
use serde::Deserialize;

pub const S_TEAMS: usize = 12;
pub const S_WEEKS: usize = 10;
pub const S_SLOTS: usize = 5;
pub const S_PAIRS: usize = 4;
pub const S_LANES: usize = 4;
pub const S_POSITIONS: usize = 18; // matchup positions per week (4*4 + 1*2), all filled
pub const EMPTY: u8 = 0xFF;

/// [week][slot][lane_pair] -> (left_team, right_team) or (EMPTY, EMPTY)
pub type SummerAssignment = [[[(u8, u8); S_PAIRS]; S_SLOTS]; S_WEEKS];

#[derive(Deserialize, Clone)]
pub struct SummerWeights {
    pub matchup_balance: u32,
    pub lane_switch_consecutive: u32,
    pub lane_switch_post_break: u32,
    pub time_gap_large: u32,
    pub time_gap_consecutive: u32,
    pub lane_balance: u32,
    pub commissioner_overlap: u32,
    pub repeat_matchup_same_night: u32,
    pub slot_balance: u32,
}

#[derive(Clone)]
pub struct SummerCostBreakdown {
    pub matchup_balance: u32,
    pub lane_switches: u32,
    pub time_gaps: u32,
    pub lane_balance: u32,
    pub commissioner_overlap: u32,
    pub repeat_matchup_same_night: u32,
    pub slot_balance: u32,
    pub total: u32,
}

/// Returns true if a lane pair is valid for the given slot.
/// Slots 0-3: all 4 lane pairs valid. Slot 4: only pairs 2 and 3.
pub fn is_valid_position(slot: usize, pair: usize) -> bool {
    if slot < 4 {
        pair < S_PAIRS
    } else {
        pair >= 2 && pair < S_PAIRS
    }
}

/// Lane index for a matchup position. Both teams bowl on the same lane.
pub fn individual_lane(pair: usize, _side: usize) -> usize {
    pair
}

/// Generate a random valid summer assignment.
/// Each team appears exactly 3 times per week, never twice in the same slot.
/// Slots 0-3 have 4 lane pairs, slot 4 has 2 (pairs 2-3). All 18 positions filled.
pub fn random_summer_assignment(rng: &mut SmallRng) -> SummerAssignment {
    let mut a = [[[(EMPTY, EMPTY); S_PAIRS]; S_SLOTS]; S_WEEKS];

    for w in 0..S_WEEKS {
        // 18 valid positions (4*4 + 1*2), all must be filled
        // 18 matchups × 2 teams = 36 = 12 teams × 3

        let mut positions: Vec<(usize, usize)> = Vec::new();
        for s in 0..S_SLOTS {
            for p in 0..S_PAIRS {
                if is_valid_position(s, p) {
                    positions.push((s, p));
                }
            }
        }

        let mut attempts = 0;
        'retry: loop {
            attempts += 1;
            if attempts > 100 { break; }

            let mut remaining = [3u8; S_TEAMS];
            let mut team_slots: [Vec<usize>; S_TEAMS] = Default::default();
            for w_s in a[w].iter_mut() {
                for pos in w_s.iter_mut() {
                    *pos = (EMPTY, EMPTY);
                }
            }

            // Shuffle positions
            for i in (1..positions.len()).rev() {
                let j = rng.random_range(0..=i);
                positions.swap(i, j);
            }

            for &(s, p) in &positions {
                let mut candidates: Vec<u8> = (0..S_TEAMS as u8)
                    .filter(|&t| remaining[t as usize] > 0 && !team_slots[t as usize].contains(&s))
                    .collect();

                if candidates.len() < 2 { continue 'retry; }

                for i in (1..candidates.len()).rev() {
                    let j = rng.random_range(0..=i);
                    candidates.swap(i, j);
                }

                let t1 = candidates[0];
                let t2 = candidates[1];
                a[w][s][p] = (t1, t2);
                remaining[t1 as usize] -= 1;
                remaining[t2 as usize] -= 1;
                team_slots[t1 as usize].push(s);
                team_slots[t2 as usize].push(s);
            }

            if remaining.iter().all(|&r| r == 0) { break; }
        }
    }

    a
}

pub fn evaluate_summer(a: &SummerAssignment, w8: &SummerWeights) -> SummerCostBreakdown {
    let mut matchup_counts = [0i32; S_TEAMS * S_TEAMS];
    let mut week_matchup_counts = [0u8; S_WEEKS * S_TEAMS * S_TEAMS];
    let mut lane_counts = [0i32; S_TEAMS * S_LANES];
    let mut slot_counts = [0i32; S_TEAMS * S_SLOTS];
    // Per-team per-week: which slots they play in (sorted), and which individual lane
    let mut team_week_slots: [[Vec<usize>; S_TEAMS]; S_WEEKS] = Default::default();

    // First pass: collect data
    for w in 0..S_WEEKS {
        for s in 0..S_SLOTS {
            for p in 0..S_PAIRS {
                let (t1, t2) = a[w][s][p];
                if t1 == EMPTY { continue; }

                let t1 = t1 as usize;
                let t2 = t2 as usize;

                // Matchups
                let lo = t1.min(t2);
                let hi = t1.max(t2);
                matchup_counts[lo * S_TEAMS + hi] += 1;
                week_matchup_counts[w * S_TEAMS * S_TEAMS + lo * S_TEAMS + hi] += 1;

                // Lane counts (both teams bowl on the same lane)
                lane_counts[t1 * S_LANES + p] += 1;
                lane_counts[t2 * S_LANES + p] += 1;

                // Slot counts
                slot_counts[t1 * S_SLOTS + s] += 1;
                slot_counts[t2 * S_SLOTS + s] += 1;

                // Track slots per team per week
                team_week_slots[w][t1].push(s);
                team_week_slots[w][t2].push(s);
            }
        }

        for t in 0..S_TEAMS {
            team_week_slots[w][t].sort_unstable();
        }
    }

    // 1. Matchup balance: penalize once per pair outside [2, 3]
    let mut matchup_balance: u32 = 0;
    for i in 0..S_TEAMS {
        for j in (i + 1)..S_TEAMS {
            let c = matchup_counts[i * S_TEAMS + j];
            if c < 2 || c > 3 {
                matchup_balance += w8.matchup_balance;
            }
        }
    }

    // 2. Lane switches
    let mut lane_switches: u32 = 0;
    for w in 0..S_WEEKS {
        for t in 0..S_TEAMS {
            let mut games: Vec<(usize, usize, usize)> = Vec::new();
            for s in 0..S_SLOTS {
                for p in 0..S_PAIRS {
                    let (t1, t2) = a[w][s][p];
                    if t1 as usize == t {
                        games.push((s, p, 0));
                    } else if t2 as usize == t {
                        games.push((s, p, 1));
                    }
                }
            }
            games.sort_by_key(|&(s, _, _)| s);

            for i in 0..(games.len().saturating_sub(1)) {
                let (s1, p1, _) = games[i];
                let (s2, p2, _) = games[i + 1];
                let gap = s2 - s1 - 1;

                if p1 != p2 {
                    if gap == 0 {
                        lane_switches += w8.lane_switch_consecutive;
                    } else {
                        lane_switches += w8.lane_switch_post_break;
                    }
                }
            }
        }
    }

    // 3. Time gaps
    let mut time_gaps: u32 = 0;
    for w in 0..S_WEEKS {
        for t in 0..S_TEAMS {
            let slots = &team_week_slots[w][t];
            if slots.len() < 2 { continue; }

            // Check for 3 consecutive (all gaps = 0)
            if slots.len() == 3 && slots[1] == slots[0] + 1 && slots[2] == slots[1] + 1 {
                time_gaps += w8.time_gap_consecutive;
            }

            for i in 0..(slots.len() - 1) {
                let gap = slots[i + 1] - slots[i] - 1;
                if gap >= 2 {
                    time_gaps += w8.time_gap_large;
                }
            }
        }
    }

    // 4. Lane balance — flat penalty for counts outside [7, 8] (target = 30/4 = 7.5)
    let mut lane_balance: u32 = 0;
    for t in 0..S_TEAMS {
        for l in 0..S_LANES {
            let c = lane_counts[t * S_LANES + l];
            if c < 7 || c > 8 {
                lane_balance += w8.lane_balance;
            }
        }
    }

    // 5. Commissioner overlap — penalize pairs co-appearing in slot 1 or slot 5
    let mut team_week_slot_set: Vec<[bool; S_SLOTS]> = vec![[false; S_SLOTS]; S_TEAMS * S_WEEKS];
    for w in 0..S_WEEKS {
        for t in 0..S_TEAMS {
            for &s in &team_week_slots[w][t] {
                team_week_slot_set[t * S_WEEKS + w][s] = true;
            }
        }
    }

    let mut min_co = u32::MAX;
    for i in 0..S_TEAMS {
        for j in (i + 1)..S_TEAMS {
            let mut co = 0u32;
            for w in 0..S_WEEKS {
                if team_week_slot_set[i * S_WEEKS + w][0]
                    && team_week_slot_set[j * S_WEEKS + w][0]
                {
                    co += 1;
                }
                if team_week_slot_set[i * S_WEEKS + w][4]
                    && team_week_slot_set[j * S_WEEKS + w][4]
                {
                    co += 1;
                }
            }
            if co < min_co {
                min_co = co;
            }
        }
    }
    let commissioner_overlap = w8.commissioner_overlap * min_co;

    // 6. Repeat matchups same night
    let mut repeat_matchup_same_night: u32 = 0;
    for w in 0..S_WEEKS {
        for i in 0..S_TEAMS {
            for j in (i + 1)..S_TEAMS {
                let c = week_matchup_counts[w * S_TEAMS * S_TEAMS + i * S_TEAMS + j];
                if c > 1 {
                    repeat_matchup_same_night += (c - 1) as u32 * w8.repeat_matchup_same_night;
                }
            }
        }
    }

    // 7. Slot balance — flat penalty for counts outside [floor, ceil] of target
    //    slots 0-3: target 20/3 ≈ 6.67 → [6, 7] ok
    //    slot 4:    target 10/3 ≈ 3.33 → [3, 4] ok
    let mut slot_balance: u32 = 0;
    for t in 0..S_TEAMS {
        for s in 0..S_SLOTS {
            let c = slot_counts[t * S_SLOTS + s];
            let ok = if s < 4 { c == 6 || c == 7 } else { c == 3 || c == 4 };
            if !ok {
                slot_balance += w8.slot_balance;
            }
        }
    }

    let total = matchup_balance
        + lane_switches
        + time_gaps
        + lane_balance
        + commissioner_overlap
        + repeat_matchup_same_night
        + slot_balance;

    SummerCostBreakdown {
        matchup_balance,
        lane_switches,
        time_gaps,
        lane_balance,
        commissioner_overlap,
        repeat_matchup_same_night,
        slot_balance,
        total,
    }
}

pub fn perturb_summer(a: &mut SummerAssignment, rng: &mut SmallRng, n: usize) {
    for _ in 0..n {
        let w = rng.random_range(0..S_WEEKS);
        // Pick two filled positions in different slots and swap one team from each
        let mut filled: Vec<(usize, usize)> = Vec::new();
        for s in 0..S_SLOTS {
            for p in 0..S_PAIRS {
                if a[w][s][p].0 != EMPTY {
                    filled.push((s, p));
                }
            }
        }
        if filled.len() < 2 { continue; }

        let i1 = rng.random_range(0..filled.len());
        let mut i2 = rng.random_range(0..(filled.len() - 1));
        if i2 >= i1 { i2 += 1; }

        let (s1, p1) = filled[i1];
        let (s2, p2) = filled[i2];
        if s1 == s2 { continue; } // same slot, skip

        // Pick which side to swap from each (0=left, 1=right)
        let side1 = rng.random_range(0..2usize);
        let side2 = rng.random_range(0..2usize);

        let t1 = if side1 == 0 { a[w][s1][p1].0 } else { a[w][s1][p1].1 };
        let t2 = if side2 == 0 { a[w][s2][p2].0 } else { a[w][s2][p2].1 };

        // Check that t1 isn't already in slot s2, and t2 isn't already in slot s1
        let t1_in_s2 = (0..S_PAIRS).any(|pp| {
            let (a1, a2) = a[w][s2][pp];
            (a1 == t1 || a2 == t1) && !(pp == p2 && ((side2 == 0 && a1 == t2) || (side2 == 1 && a2 == t2)))
        });
        let t2_in_s1 = (0..S_PAIRS).any(|pp| {
            let (a1, a2) = a[w][s1][pp];
            (a1 == t2 || a2 == t2) && !(pp == p1 && ((side1 == 0 && a1 == t1) || (side1 == 1 && a2 == t1)))
        });
        if t1_in_s2 || t2_in_s1 { continue; }

        // Perform swap
        if side1 == 0 { a[w][s1][p1].0 = t2; } else { a[w][s1][p1].1 = t2; }
        if side2 == 0 { a[w][s2][p2].0 = t1; } else { a[w][s2][p2].1 = t1; }
    }
}

pub fn summer_cost_label(c: &SummerCostBreakdown) -> String {
    format!(
        "total: {:>4} matchup: {:>3} lane_sw: {:>3} gaps: {:>3} lane: {:>3} comm: {:>3} repeat: {:>3} slot: {:>3}",
        c.total, c.matchup_balance, c.lane_switches, c.time_gaps,
        c.lane_balance, c.commissioner_overlap, c.repeat_matchup_same_night,
        c.slot_balance,
    )
}

pub fn summer_assignment_to_tsv(a: &SummerAssignment) -> String {
    let mut lines = vec![String::from("Week\tSlot\tLane 1\tLane 2\tLane 3\tLane 4")];

    for w in 0..S_WEEKS {
        for s in 0..S_SLOTS {
            let mut cells: [String; S_PAIRS] = Default::default();
            for p in 0..S_PAIRS {
                let (t1, t2) = a[w][s][p];
                if t1 == EMPTY {
                    cells[p] = String::from("-");
                } else {
                    cells[p] = format!("{} v {}", t1 + 1, t2 + 1);
                }
            }
            lines.push(format!(
                "{}\t{}\t{}\t{}\t{}\t{}",
                w + 1, s + 1, cells[0], cells[1], cells[2], cells[3],
            ));
        }
    }

    lines.join("\n")
}

pub fn parse_summer_tsv(content: &str) -> Option<SummerAssignment> {
    let mut a: SummerAssignment = [[[(EMPTY, EMPTY); S_PAIRS]; S_SLOTS]; S_WEEKS];
    let lines: Vec<&str> = content.lines().collect();
    if lines.len() < 2 { return None; }

    for w in 0..S_WEEKS {
        for s in 0..S_SLOTS {
            let line_idx = 1 + w * S_SLOTS + s;
            if line_idx >= lines.len() { return None; }
            let cols: Vec<&str> = lines[line_idx].split('\t').collect();
            if cols.len() < 6 { return None; }

            for p in 0..S_PAIRS {
                let cell = cols[2 + p].trim();
                if cell == "-" {
                    a[w][s][p] = (EMPTY, EMPTY);
                } else {
                    let parts: Vec<&str> = cell.split(" v ").collect();
                    if parts.len() != 2 { return None; }
                    let t1 = parts[0].trim().parse::<u8>().ok()? - 1;
                    let t2 = parts[1].trim().parse::<u8>().ok()? - 1;
                    a[w][s][p] = (t1, t2);
                }
            }
        }
    }
    Some(a)
}

pub fn reassign_summer_commissioners(a: &mut SummerAssignment) {
    // Build slot sets for each team-week
    let mut team_week_slots: [[Vec<usize>; S_TEAMS]; S_WEEKS] = Default::default();
    for w in 0..S_WEEKS {
        for s in 0..S_SLOTS {
            for p in 0..S_PAIRS {
                let (t1, t2) = a[w][s][p];
                if t1 != EMPTY {
                    team_week_slots[w][t1 as usize].push(s);
                    team_week_slots[w][t2 as usize].push(s);
                }
            }
        }
    }

    // Find pair with minimum shared-slot overlap
    let mut best_i = 0usize;
    let mut best_j = 1usize;
    let mut min_overlap = (S_WEEKS * S_SLOTS) as u32;

    for i in 0..S_TEAMS {
        for j in (i + 1)..S_TEAMS {
            let mut overlap = 0u32;
            for w in 0..S_WEEKS {
                for &si in &team_week_slots[w][i] {
                    if team_week_slots[w][j].contains(&si) {
                        overlap += 1;
                    }
                }
            }
            if overlap < min_overlap {
                min_overlap = overlap;
                best_i = i;
                best_j = j;
            }
        }
    }

    if best_i == 0 && best_j == 1 {
        return;
    }

    // Build permutation to swap best_i->0 and best_j->1
    let mut perm: [u8; S_TEAMS] = std::array::from_fn(|i| i as u8);
    perm.swap(0, best_i);
    perm.swap(1, best_j);
    let mut inv = [0u8; S_TEAMS];
    for (i, &p) in perm.iter().enumerate() {
        inv[p as usize] = i as u8;
    }

    for w in 0..S_WEEKS {
        for s in 0..S_SLOTS {
            for p in 0..S_PAIRS {
                let (t1, t2) = a[w][s][p];
                if t1 != EMPTY {
                    a[w][s][p] = (inv[t1 as usize], inv[t2 as usize]);
                }
            }
        }
    }
}

/// Flatten a SummerAssignment into a byte vector for serialization.
/// Layout: for each week, for each slot, for each pair: left_team, right_team (2 bytes).
pub fn summer_assignment_to_flat(a: &SummerAssignment) -> Vec<u8> {
    let mut flat = Vec::with_capacity(S_WEEKS * S_SLOTS * S_PAIRS * 2);
    for w in 0..S_WEEKS {
        for s in 0..S_SLOTS {
            for p in 0..S_PAIRS {
                flat.push(a[w][s][p].0);
                flat.push(a[w][s][p].1);
            }
        }
    }
    flat
}

/// Reconstruct a SummerAssignment from a flat byte vector.
pub fn flat_to_summer_assignment(flat: &[u8]) -> SummerAssignment {
    let mut a = [[[(EMPTY, EMPTY); S_PAIRS]; S_SLOTS]; S_WEEKS];
    for w in 0..S_WEEKS {
        for s in 0..S_SLOTS {
            for p in 0..S_PAIRS {
                let idx = (w * S_SLOTS * S_PAIRS + s * S_PAIRS + p) * 2;
                if idx + 1 < flat.len() {
                    a[w][s][p] = (flat[idx], flat[idx + 1]);
                }
            }
        }
    }
    a
}

/// GPU-equivalent evaluation: mirrors the WGSL shader logic exactly.
/// Uses packed byte arrays and the same loop structure as the GPU shader.
pub fn evaluate_summer_gpu_style(a: &SummerAssignment, w8: &SummerWeights) -> SummerCostBreakdown {
    // --- Pass 1: matchup balance + repeat matchup same night ---
    let mut matchup_total = [0u32; 17]; // packed bytes for C(12,2)=66 pairs

    fn pair_idx(lo: usize, hi: usize) -> usize {
        lo * (2 * S_TEAMS - lo - 1) / 2 + (hi - lo - 1)
    }
    fn inc_packed(m: &mut [u32; 17], lo: usize, hi: usize) {
        let idx = pair_idx(lo, hi);
        m[idx / 4] += 1 << ((idx % 4) * 8);
    }
    fn get_packed(m: &[u32; 17], lo: usize, hi: usize) -> u32 {
        let idx = pair_idx(lo, hi);
        (m[idx / 4] >> ((idx % 4) * 8)) & 0xFF
    }

    let mut repeat_matchup_same_night: u32 = 0;
    for w in 0..S_WEEKS {
        let mut week_m = [0u32; 17];
        for s in 0..S_SLOTS {
            for p in 0..S_PAIRS {
                let (t1, t2) = a[w][s][p];
                if t1 == EMPTY { continue; }
                let (t1, t2) = (t1 as usize, t2 as usize);
                let lo = t1.min(t2);
                let hi = t1.max(t2);
                inc_packed(&mut matchup_total, lo, hi);
                inc_packed(&mut week_m, lo, hi);
            }
        }
        for i in 0..S_TEAMS {
            for j in (i + 1)..S_TEAMS {
                let c = get_packed(&week_m, i, j);
                if c > 1 {
                    repeat_matchup_same_night += (c - 1) * w8.repeat_matchup_same_night;
                }
            }
        }
    }

    let mut matchup_balance: u32 = 0;
    for i in 0..S_TEAMS {
        for j in (i + 1)..S_TEAMS {
            let c = get_packed(&matchup_total, i, j);
            if c < 2 || c > 3 {
                matchup_balance += w8.matchup_balance;
            }
        }
    }

    // --- Pass 2: lane switches + time gaps (per team per week) ---
    let mut lane_switches: u32 = 0;
    let mut time_gaps: u32 = 0;

    for w in 0..S_WEEKS {
        for t in 0..S_TEAMS {
            let mut game_slots = [0u32; 3];
            let mut game_pairs = [0u32; 3];
            let mut gc: u32 = 0;

            for s in 0..S_SLOTS {
                for p in 0..S_PAIRS {
                    let (t1, t2) = a[w][s][p];
                    if t1 == EMPTY { continue; }
                    if t1 as usize == t || t2 as usize == t {
                        if gc < 3 {
                            game_slots[gc as usize] = s as u32;
                            game_pairs[gc as usize] = p as u32;
                            gc += 1;
                        }
                    }
                }
            }

            // Time gaps
            if gc == 3 && game_slots[1] == game_slots[0] + 1 && game_slots[2] == game_slots[1] + 1 {
                time_gaps += w8.time_gap_consecutive;
            }
            if gc >= 2 {
                for g in 0..(gc - 1) as usize {
                    let gap = game_slots[g + 1] - game_slots[g] - 1;
                    if gap >= 2 {
                        time_gaps += w8.time_gap_large;
                    }
                }
            }

            // Lane switches
            if gc >= 2 {
                for g in 0..(gc - 1) as usize {
                    let gap = game_slots[g + 1] - game_slots[g] - 1;
                    if game_pairs[g] != game_pairs[g + 1] {
                        if gap == 0 {
                            lane_switches += w8.lane_switch_consecutive;
                        } else {
                            lane_switches += w8.lane_switch_post_break;
                        }
                    }
                }
            }
        }
    }

    // --- Pass 3: lane balance ---
    let mut lane_counts = [0u32; 12]; // packed: 12 teams × 4 lanes, 4 per u32
    for w in 0..S_WEEKS {
        for s in 0..S_SLOTS {
            for p in 0..S_PAIRS {
                let (t1, t2) = a[w][s][p];
                if t1 == EMPTY { continue; }
                let (t1, t2) = (t1 as usize, t2 as usize);
                let l1 = t1 * S_LANES + p;
                lane_counts[l1 / 4] += 1 << ((l1 % 4) * 8);
                let l2 = t2 * S_LANES + p;
                lane_counts[l2 / 4] += 1 << ((l2 % 4) * 8);
            }
        }
    }

    let mut lane_balance: u32 = 0;
    for t in 0..S_TEAMS {
        for l in 0..S_LANES {
            let li = t * S_LANES + l;
            let count = (lane_counts[li / 4] >> ((li % 4) * 8)) & 0xFF;
            if count < 7 || count > 8 {
                lane_balance += w8.lane_balance;
            }
        }
    }

    // --- Pass 4: slot balance ---
    let mut slot_counts_packed = [0u32; 15]; // packed: 12 teams × 5 slots, 4 per u32
    for w in 0..S_WEEKS {
        for s in 0..S_SLOTS {
            for p in 0..S_PAIRS {
                let (t1, t2) = a[w][s][p];
                if t1 == EMPTY { continue; }
                let (t1, t2) = (t1 as usize, t2 as usize);
                let si1 = t1 * S_SLOTS + s;
                slot_counts_packed[si1 / 4] += 1 << ((si1 % 4) * 8);
                let si2 = t2 * S_SLOTS + s;
                slot_counts_packed[si2 / 4] += 1 << ((si2 % 4) * 8);
            }
        }
    }

    let mut slot_balance: u32 = 0;
    for t in 0..S_TEAMS {
        for s in 0..S_SLOTS {
            let si = t * S_SLOTS + s;
            let count = (slot_counts_packed[si / 4] >> ((si % 4) * 8)) & 0xFF;
            let ok = if s < 4 { count == 6 || count == 7 } else { count == 3 || count == 4 };
            if !ok {
                slot_balance += w8.slot_balance;
            }
        }
    }

    // --- Pass 5: commissioner overlap (slot 1/5 co-appearance) ---
    // 1 u32 per team: bit w*2 = plays in slot 0, bit w*2+1 = plays in slot 4
    let mut team_comm_bits = [0u32; S_TEAMS];
    for w in 0..S_WEEKS {
        for p in 0..S_PAIRS {
            let (t1, t2) = a[w][0][p];
            if t1 != EMPTY {
                let bit = w * 2;
                team_comm_bits[t1 as usize] |= 1u32 << bit;
                team_comm_bits[t2 as usize] |= 1u32 << bit;
            }
        }
        for p in 2..S_PAIRS {
            let (t1, t2) = a[w][4][p];
            if t1 != EMPTY {
                let bit = w * 2 + 1;
                team_comm_bits[t1 as usize] |= 1u32 << bit;
                team_comm_bits[t2 as usize] |= 1u32 << bit;
            }
        }
    }

    let mut min_co = 20u32;
    for i in 0..S_TEAMS {
        for j in (i + 1)..S_TEAMS {
            let co = (team_comm_bits[i] & team_comm_bits[j]).count_ones();
            min_co = min_co.min(co);
        }
    }
    let commissioner_overlap = w8.commissioner_overlap * min_co;

    let total = matchup_balance + lane_switches + time_gaps + lane_balance
        + commissioner_overlap + repeat_matchup_same_night + slot_balance;

    SummerCostBreakdown {
        matchup_balance,
        lane_switches,
        time_gaps,
        lane_balance,
        commissioner_overlap,
        repeat_matchup_same_night,
        slot_balance,
        total,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::SeedableRng;

    fn test_weights() -> SummerWeights {
        SummerWeights {
            matchup_balance: 80,
            lane_switch_consecutive: 60,
            lane_switch_post_break: 20,
            time_gap_large: 60,
            time_gap_consecutive: 30,
            lane_balance: 60,
            commissioner_overlap: 30,
            repeat_matchup_same_night: 30,
            slot_balance: 30,
        }
    }

    #[test]
    fn test_evaluate_deterministic() {
        let mut rng = SmallRng::seed_from_u64(42);
        let a = random_summer_assignment(&mut rng);
        let w8 = test_weights();
        let c1 = evaluate_summer(&a, &w8);
        let c2 = evaluate_summer(&a, &w8);
        assert_eq!(c1.total, c2.total);
    }

    #[test]
    fn test_cpu_vs_gpu_style_evaluation() {
        let w8 = test_weights();
        for seed in 0..20 {
            let mut rng = SmallRng::seed_from_u64(seed);
            let a = random_summer_assignment(&mut rng);
            let cpu = evaluate_summer(&a, &w8);
            let gpu = evaluate_summer_gpu_style(&a, &w8);
            assert_eq!(cpu.matchup_balance, gpu.matchup_balance,
                "seed={seed}: matchup_balance CPU={} GPU={}", cpu.matchup_balance, gpu.matchup_balance);
            assert_eq!(cpu.lane_switches, gpu.lane_switches,
                "seed={seed}: lane_switches CPU={} GPU={}", cpu.lane_switches, gpu.lane_switches);
            assert_eq!(cpu.time_gaps, gpu.time_gaps,
                "seed={seed}: time_gaps CPU={} GPU={}", cpu.time_gaps, gpu.time_gaps);
            assert_eq!(cpu.lane_balance, gpu.lane_balance,
                "seed={seed}: lane_balance CPU={} GPU={}", cpu.lane_balance, gpu.lane_balance);
            assert_eq!(cpu.commissioner_overlap, gpu.commissioner_overlap,
                "seed={seed}: commissioner_overlap CPU={} GPU={}", cpu.commissioner_overlap, gpu.commissioner_overlap);
            assert_eq!(cpu.repeat_matchup_same_night, gpu.repeat_matchup_same_night,
                "seed={seed}: repeat_matchup_same_night CPU={} GPU={}", cpu.repeat_matchup_same_night, gpu.repeat_matchup_same_night);
            assert_eq!(cpu.slot_balance, gpu.slot_balance,
                "seed={seed}: slot_balance CPU={} GPU={}", cpu.slot_balance, gpu.slot_balance);
            assert_eq!(cpu.total, gpu.total,
                "seed={seed}: total CPU={} GPU={}", cpu.total, gpu.total);
        }
    }

    #[test]
    fn test_assignment_validity() {
        let mut rng = SmallRng::seed_from_u64(123);
        let a = random_summer_assignment(&mut rng);
        for w in 0..S_WEEKS {
            let mut team_count = [0u8; S_TEAMS];
            for s in 0..S_SLOTS {
                let mut teams_in_slot: Vec<u8> = Vec::new();
                for p in 0..S_PAIRS {
                    let (t1, t2) = a[w][s][p];
                    if t1 == EMPTY { continue; }
                    assert!((t1 as usize) < S_TEAMS, "Invalid team {t1}");
                    assert!((t2 as usize) < S_TEAMS, "Invalid team {t2}");
                    assert_ne!(t1, t2, "Team playing itself");
                    assert!(!teams_in_slot.contains(&t1),
                        "Team {t1} appears twice in week {w} slot {s}");
                    assert!(!teams_in_slot.contains(&t2),
                        "Team {t2} appears twice in week {w} slot {s}");
                    teams_in_slot.push(t1);
                    teams_in_slot.push(t2);
                    team_count[t1 as usize] += 1;
                    team_count[t2 as usize] += 1;
                }
                if s == 4 {
                    assert_eq!(a[w][s][0], (EMPTY, EMPTY));
                    assert_eq!(a[w][s][1], (EMPTY, EMPTY));
                }
            }
            for t in 0..S_TEAMS {
                assert_eq!(team_count[t], 3,
                    "Week {w}: team {t} has {} games, expected 3", team_count[t]);
            }
        }
    }
}
