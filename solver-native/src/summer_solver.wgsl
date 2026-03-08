// Summer league SA solver shader
// 12 teams, 10 weeks, 5 slots, 4 lane pairs
// Assignment: 200 u32s (10 weeks × 5 slots × 4 pairs)
// Each u32: left_team | (right_team << 8), or 0xFFFF for empty

const TEAMS: u32 = 12u;
const WEEKS: u32 = 10u;
const SLOTS: u32 = 5u;
const PAIRS: u32 = 4u;
const LANES: u32 = 4u;
const ASSIGN_SIZE: u32 = 200u;
const EMPTY: u32 = 0xFFu;
const EMPTY_POS: u32 = 0xFFFFu;
const NUM_PAIRS_C2: u32 = 66u; // C(12,2)
const MATCHUP_U32S: u32 = 17u; // ceil(66/4)

struct Weights {
    matchup_balance: u32,
    lane_switch_consecutive: u32,
    lane_switch_post_break: u32,
    third_game_diff_lane: u32,
    time_gap_large: u32,
    time_gap_consecutive: u32,
    lane_balance: u32,
    commissioner_overlap: u32,
    repeat_matchup_same_night: u32,
    slot_balance: u32,
}

struct Params {
    iters_per_dispatch: u32,
    chain_count: u32,
    temp_base: f32,
    temp_step: f32,
    pod_size: u32,
}

@group(0) @binding(0) var<storage, read_write> assignments: array<u32>;
@group(0) @binding(1) var<storage, read_write> best_assignments: array<u32>;
@group(0) @binding(2) var<storage, read_write> costs: array<u32>;
@group(0) @binding(3) var<storage, read_write> best_costs: array<u32>;
@group(0) @binding(4) var<storage, read_write> rng_states: array<u32>;
@group(0) @binding(5) var<uniform> weights: Weights;
@group(0) @binding(6) var<uniform> params: Params;
@group(0) @binding(7) var<storage, read> move_thresh: array<u32, 12>;

// ═══════════════════════════════════════════════════════════════════════
// RNG: xoshiro128++
// ═══════════════════════════════════════════════════════════════════════

fn rotl(x: u32, k: u32) -> u32 {
    return (x << k) | (x >> (32u - k));
}

fn rng_next(s: ptr<function, array<u32, 4>>) -> u32 {
    let result = rotl((*s)[0] + (*s)[3], 7u) + (*s)[0];
    let t = (*s)[1] << 9u;
    (*s)[2] ^= (*s)[0];
    (*s)[3] ^= (*s)[1];
    (*s)[1] ^= (*s)[2];
    (*s)[0] ^= (*s)[3];
    (*s)[2] ^= t;
    (*s)[3] = rotl((*s)[3], 11u);
    return result;
}

fn rng_range(s: ptr<function, array<u32, 4>>, n: u32) -> u32 {
    return rng_next(s) % n;
}

fn rng_f32(s: ptr<function, array<u32, 4>>) -> f32 {
    return f32(rng_next(s) >> 8u) / 16777216.0;
}

// ═══════════════════════════════════════════════════════════════════════
// Assignment access helpers
// ═══════════════════════════════════════════════════════════════════════

fn pos_idx(w: u32, s: u32, p: u32) -> u32 {
    return w * SLOTS * PAIRS + s * PAIRS + p;
}

fn get_pos(a: ptr<function, array<u32, 200>>, w: u32, s: u32, p: u32) -> u32 {
    return (*a)[pos_idx(w, s, p)];
}

fn get_left(a: ptr<function, array<u32, 200>>, w: u32, s: u32, p: u32) -> u32 {
    return (*a)[pos_idx(w, s, p)] & 0xFFu;
}

fn get_right(a: ptr<function, array<u32, 200>>, w: u32, s: u32, p: u32) -> u32 {
    return ((*a)[pos_idx(w, s, p)] >> 8u) & 0xFFu;
}

fn is_empty(a: ptr<function, array<u32, 200>>, w: u32, s: u32, p: u32) -> bool {
    return (*a)[pos_idx(w, s, p)] == EMPTY_POS;
}

fn is_valid_pos(s: u32, p: u32) -> bool {
    return s < 4u || p >= 2u;
}

fn set_pos(a: ptr<function, array<u32, 200>>, w: u32, s: u32, p: u32, left: u32, right: u32) {
    (*a)[pos_idx(w, s, p)] = (left & 0xFFu) | ((right & 0xFFu) << 8u);
}

fn set_empty(a: ptr<function, array<u32, 200>>, w: u32, s: u32, p: u32) {
    (*a)[pos_idx(w, s, p)] = EMPTY_POS;
}

fn set_side(a: ptr<function, array<u32, 200>>, w: u32, s: u32, p: u32, side: u32, team: u32) {
    let idx = pos_idx(w, s, p);
    if (side == 0u) {
        (*a)[idx] = ((*a)[idx] & 0xFF00u) | (team & 0xFFu);
    } else {
        (*a)[idx] = ((*a)[idx] & 0x00FFu) | ((team & 0xFFu) << 8u);
    }
}

fn get_side(a: ptr<function, array<u32, 200>>, w: u32, s: u32, p: u32, side: u32) -> u32 {
    if (side == 0u) { return get_left(a, w, s, p); }
    return get_right(a, w, s, p);
}

// ═══════════════════════════════════════════════════════════════════════
// Matchup pair index (for C(12,2) = 66 pairs)
// ═══════════════════════════════════════════════════════════════════════

fn pair_idx(lo: u32, hi: u32) -> u32 {
    return lo * (2u * TEAMS - lo - 1u) / 2u + (hi - lo - 1u);
}

fn inc_packed(m: ptr<function, array<u32, 17>>, lo: u32, hi: u32) {
    if (lo >= hi) { return; } // guard against self-matchup corruption
    let idx = pair_idx(lo, hi);
    if (idx >= 66u) { return; } // guard against OOB
    (*m)[idx / 4u] += (1u << ((idx % 4u) * 8u));
}

fn get_packed(m: ptr<function, array<u32, 17>>, lo: u32, hi: u32) -> u32 {
    if (lo >= hi) { return 0u; } // guard against self-matchup corruption
    let idx = pair_idx(lo, hi);
    if (idx >= 66u) { return 0u; } // guard against OOB
    return ((*m)[idx / 4u] >> ((idx % 4u) * 8u)) & 0xFFu;
}

// ═══════════════════════════════════════════════════════════════════════
// Check if team is in a slot (excluding one pair)
// ═══════════════════════════════════════════════════════════════════════

fn team_in_slot(a: ptr<function, array<u32, 200>>, w: u32, slot: u32, team: u32, exclude_pair: u32) -> bool {
    for (var p = 0u; p < PAIRS; p++) {
        if (p == exclude_pair) { continue; }
        if (is_empty(a, w, slot, p)) { continue; }
        if (get_left(a, w, slot, p) == team || get_right(a, w, slot, p) == team) { return true; }
    }
    return false;
}

fn team_in_slot_any(a: ptr<function, array<u32, 200>>, w: u32, slot: u32, team: u32) -> bool {
    for (var p = 0u; p < PAIRS; p++) {
        if (is_empty(a, w, slot, p)) { continue; }
        if (get_left(a, w, slot, p) == team || get_right(a, w, slot, p) == team) { return true; }
    }
    return false;
}

// ═══════════════════════════════════════════════════════════════════════
// Evaluation
// ═══════════════════════════════════════════════════════════════════════

fn evaluate(a: ptr<function, array<u32, 200>>) -> u32 {
    var total = 0u;

    // --- Pass 1: matchup balance + repeat matchup same night ---
    var matchup_total: array<u32, 17>;
    for (var i = 0u; i < MATCHUP_U32S; i++) { matchup_total[i] = 0u; }

    for (var w = 0u; w < WEEKS; w++) {
        var week_m: array<u32, 17>;
        for (var i = 0u; i < MATCHUP_U32S; i++) { week_m[i] = 0u; }

        for (var s = 0u; s < SLOTS; s++) {
            for (var p = 0u; p < PAIRS; p++) {
                if (is_empty(a, w, s, p)) { continue; }
                let t1 = get_left(a, w, s, p);
                let t2 = get_right(a, w, s, p);
                if (t1 >= TEAMS || t2 >= TEAMS) { continue; }
                let lo = min(t1, t2);
                let hi = max(t1, t2);
                inc_packed(&matchup_total, lo, hi);
                inc_packed(&week_m, lo, hi);
            }
        }

        // Repeat matchup same night
        for (var i = 0u; i < TEAMS; i++) {
            for (var j = i + 1u; j < TEAMS; j++) {
                let c = get_packed(&week_m, i, j);
                if (c > 1u) {
                    total += (c - 1u) * weights.repeat_matchup_same_night;
                }
            }
        }
    }

    // Matchup balance: penalize once per pair outside [2, 3]
    for (var i = 0u; i < TEAMS; i++) {
        for (var j = i + 1u; j < TEAMS; j++) {
            let c = get_packed(&matchup_total, i, j);
            if (c < 2u || c > 3u) {
                total += weights.matchup_balance;
            }
        }
    }

    // --- Pass 2: lane switches + time gaps (per team per week) ---
    for (var w = 0u; w < WEEKS; w++) {
        for (var t = 0u; t < TEAMS; t++) {
            var game_slots: array<u32, 3>;
            var game_pairs: array<u32, 3>;
            var gc = 0u;

            for (var s = 0u; s < SLOTS; s++) {
                for (var p = 0u; p < PAIRS; p++) {
                    if (is_empty(a, w, s, p)) { continue; }
                    if (get_left(a, w, s, p) == t || get_right(a, w, s, p) == t) {
                        if (gc < 3u) {
                            game_slots[gc] = s;
                            game_pairs[gc] = p;
                            gc += 1u;
                        }
                    }
                }
            }

            // Time gaps
            if (gc == 3u && game_slots[1] == game_slots[0] + 1u && game_slots[2] == game_slots[1] + 1u) {
                total += weights.time_gap_consecutive;
            }
            if (gc >= 2u) {
            for (var g = 0u; g < gc - 1u; g++) {
                let gap = game_slots[g + 1u] - game_slots[g] - 1u;
                if (gap >= 2u) {
                    total += weights.time_gap_large;
                }
            }

            // Lane switches (lane = pair, both teams bowl on same lane)
            for (var g = 0u; g < gc - 1u; g++) {
                let gap = game_slots[g + 1u] - game_slots[g] - 1u;
                if (game_pairs[g] != game_pairs[g + 1u]) {
                    if (gap == 0u) {
                        total += weights.lane_switch_consecutive;
                    } else {
                        total += weights.lane_switch_post_break;
                    }
                }
            }
            // One game on different lane from the other two
            if (gc == 3u) {
                var ndiff = 0u;
                if (game_pairs[0u] != game_pairs[1u]) { ndiff += 1u; }
                if (game_pairs[0u] != game_pairs[2u]) { ndiff += 1u; }
                if (game_pairs[1u] != game_pairs[2u]) { ndiff += 1u; }
                if (ndiff == 2u) {
                    total += weights.third_game_diff_lane;
                }
            }
            } // end if gc >= 2
        }
    }

    // --- Pass 3: lane balance ---
    // 12 teams × 4 lanes, packed 4 per u32 = 12 u32s
    var lane_counts: array<u32, 12>;
    for (var i = 0u; i < 12u; i++) { lane_counts[i] = 0u; }

    for (var w = 0u; w < WEEKS; w++) {
        for (var s = 0u; s < SLOTS; s++) {
            for (var p = 0u; p < PAIRS; p++) {
                if (is_empty(a, w, s, p)) { continue; }
                let t1 = get_left(a, w, s, p);
                let t2 = get_right(a, w, s, p);
                if (t1 >= TEAMS || t2 >= TEAMS) { continue; }
                // Both teams bowl on the same lane (= pair index)
                let l1 = t1 * LANES + p;
                lane_counts[l1 / 4u] += (1u << ((l1 % 4u) * 8u));
                let l2 = t2 * LANES + p;
                lane_counts[l2 / 4u] += (1u << ((l2 % 4u) * 8u));
            }
        }
    }

    // lane balance: flat penalty for counts outside [7, 8] (target = 30/4 = 7.5)
    for (var t = 0u; t < TEAMS; t++) {
        for (var l = 0u; l < LANES; l++) {
            let li = t * LANES + l;
            let count = (lane_counts[li / 4u] >> ((li % 4u) * 8u)) & 0xFFu;
            if (count < 7u || count > 8u) {
                total += weights.lane_balance;
            }
        }
    }

    // --- Pass 4: slot balance ---
    var slot_counts: array<u32, 15>;
    for (var i = 0u; i < 15u; i++) { slot_counts[i] = 0u; }

    for (var w = 0u; w < WEEKS; w++) {
        for (var s = 0u; s < SLOTS; s++) {
            for (var p = 0u; p < PAIRS; p++) {
                if (is_empty(a, w, s, p)) { continue; }
                let t1 = get_left(a, w, s, p);
                let t2 = get_right(a, w, s, p);
                if (t1 >= TEAMS || t2 >= TEAMS) { continue; }
                let si1 = t1 * SLOTS + s;
                slot_counts[si1 / 4u] += (1u << ((si1 % 4u) * 8u));
                let si2 = t2 * SLOTS + s;
                slot_counts[si2 / 4u] += (1u << ((si2 % 4u) * 8u));
            }
        }
    }

    // slot balance: flat penalty for counts outside [floor, ceil] of target
    // slots 0-3: target 20/3 ≈ 6.67 → [6, 7] ok
    // slot 4:    target 10/3 ≈ 3.33 → [3, 4] ok
    for (var t = 0u; t < TEAMS; t++) {
        for (var s = 0u; s < SLOTS; s++) {
            let si = t * SLOTS + s;
            let count = (slot_counts[si / 4u] >> ((si % 4u) * 8u)) & 0xFFu;
            var ok = false;
            if (s < 4u) { ok = count == 6u || count == 7u; } else { ok = count == 3u || count == 4u; }
            if (!ok) {
                total += weights.slot_balance;
            }
        }
    }

    // --- Pass 5: commissioner overlap (slot 1/5 co-appearance) ---
    // 1 u32 per team: bit w*2 = plays in slot 0, bit w*2+1 = plays in slot 4
    var team_comm_bits: array<u32, 12>;
    for (var i = 0u; i < 12u; i++) { team_comm_bits[i] = 0u; }

    for (var w = 0u; w < WEEKS; w++) {
        // Slot 0: all pairs
        for (var p = 0u; p < PAIRS; p++) {
            if (!is_empty(a, w, 0u, p)) {
                let t1 = get_left(a, w, 0u, p);
                let t2 = get_right(a, w, 0u, p);
                if (t1 < TEAMS && t2 < TEAMS) {
                    let bit = w * 2u;
                    team_comm_bits[t1] |= (1u << bit);
                    team_comm_bits[t2] |= (1u << bit);
                }
            }
        }
        // Slot 4: pairs 2 and 3 only
        for (var p = 2u; p < PAIRS; p++) {
            if (!is_empty(a, w, 4u, p)) {
                let t1 = get_left(a, w, 4u, p);
                let t2 = get_right(a, w, 4u, p);
                if (t1 < TEAMS && t2 < TEAMS) {
                    let bit = w * 2u + 1u;
                    team_comm_bits[t1] |= (1u << bit);
                    team_comm_bits[t2] |= (1u << bit);
                }
            }
        }
    }

    var min_co = 20u;
    for (var i = 0u; i < TEAMS; i++) {
        for (var j = i + 1u; j < TEAMS; j++) {
            let co = countOneBits(team_comm_bits[i] & team_comm_bits[j]);
            min_co = min(min_co, co);
        }
    }
    total += weights.commissioner_overlap * min_co;

    return total;
}

// ═══════════════════════════════════════════════════════════════════════
// SA helpers
// ═══════════════════════════════════════════════════════════════════════

fn sa_accept(delta: i32, temp: f32, s: ptr<function, array<u32, 4>>) -> bool {
    if (delta < 0) { return true; }
    if (delta == 0) { return rng_f32(s) < 0.2; }
    return rng_f32(s) < exp(f32(-delta) / temp);
}

fn write_best(a: ptr<function, array<u32, 200>>, base: u32) {
    for (var i = 0u; i < ASSIGN_SIZE; i++) {
        best_assignments[base + i] = (*a)[i];
    }
}

fn save_all(a: ptr<function, array<u32, 200>>) -> array<u32, 200> {
    var s: array<u32, 200>;
    for (var i = 0u; i < ASSIGN_SIZE; i++) { s[i] = (*a)[i]; }
    return s;
}

fn restore_all(a: ptr<function, array<u32, 200>>, s: ptr<function, array<u32, 200>>) {
    for (var i = 0u; i < ASSIGN_SIZE; i++) { (*a)[i] = (*s)[i]; }
}

// ═══════════════════════════════════════════════════════════════════════
// Moves
// ═══════════════════════════════════════════════════════════════════════

// Move 0: team_swap — swap one team from each of two filled positions in different slots
fn move_team_swap(a: ptr<function, array<u32, 200>>, s: ptr<function, array<u32, 4>>) -> bool {
    let w = rng_range(s, WEEKS);
    // Pick two random filled positions
    let s1 = rng_range(s, SLOTS);
    let p1 = rng_range(s, PAIRS);
    if (is_empty(a, w, s1, p1)) { return false; }
    let s2 = rng_range(s, SLOTS);
    if (s2 == s1) { return false; }
    let p2 = rng_range(s, PAIRS);
    if (is_empty(a, w, s2, p2)) { return false; }

    let side1 = rng_range(s, 2u);
    let side2 = rng_range(s, 2u);
    let t1 = get_side(a, w, s1, p1, side1);
    let t2 = get_side(a, w, s2, p2, side2);

    if (team_in_slot(a, w, s2, t1, p2)) { return false; }
    if (team_in_slot(a, w, s1, t2, p1)) { return false; }

    // Check we don't create same-team matchups
    let opp1 = get_side(a, w, s1, p1, 1u - side1);
    let opp2 = get_side(a, w, s2, p2, 1u - side2);
    if (opp1 == t2 || opp2 == t1) { return false; }

    set_side(a, w, s1, p1, side1, t2);
    set_side(a, w, s2, p2, side2, t1);
    return true;
}

// Move 1: matchup_swap — swap two entire matchups between different slots
fn move_matchup_swap(a: ptr<function, array<u32, 200>>, s: ptr<function, array<u32, 4>>) -> bool {
    let w = rng_range(s, WEEKS);
    // Pick two filled positions in different slots
    let s1 = rng_range(s, SLOTS);
    let p1 = rng_range(s, PAIRS);
    if (!is_valid_pos(s1, p1)) { return false; }
    if (is_empty(a, w, s1, p1)) { return false; }

    let s2 = rng_range(s, SLOTS);
    if (s2 == s1) { return false; }
    let p2 = rng_range(s, PAIRS);
    if (!is_valid_pos(s2, p2)) { return false; }
    if (is_empty(a, w, s2, p2)) { return false; }

    let a1 = get_left(a, w, s1, p1);
    let b1 = get_right(a, w, s1, p1);
    let a2 = get_left(a, w, s2, p2);
    let b2 = get_right(a, w, s2, p2);

    // Check: no team from matchup 1 already in slot 2 (excluding swap partner)
    if (team_in_slot(a, w, s2, a1, p2)) { return false; }
    if (team_in_slot(a, w, s2, b1, p2)) { return false; }
    if (team_in_slot(a, w, s1, a2, p1)) { return false; }
    if (team_in_slot(a, w, s1, b2, p1)) { return false; }

    // Swap entire matchups
    set_pos(a, w, s1, p1, a2, b2);
    set_pos(a, w, s2, p2, a1, b1);
    return true;
}

// Move 3: opponent_swap — swap one team between two matchups in the same slot
fn move_opponent_swap(a: ptr<function, array<u32, 200>>, s: ptr<function, array<u32, 4>>) -> bool {
    let w = rng_range(s, WEEKS);
    let sl = rng_range(s, SLOTS);
    let p1 = rng_range(s, PAIRS);
    if (is_empty(a, w, sl, p1)) { return false; }
    var p2 = rng_range(s, PAIRS - 1u);
    if (p2 >= p1) { p2 += 1u; }
    if (is_empty(a, w, sl, p2)) { return false; }

    let side1 = rng_range(s, 2u);
    let side2 = rng_range(s, 2u);
    let t1 = get_side(a, w, sl, p1, side1);
    let t2 = get_side(a, w, sl, p2, side2);

    // Check we don't create same-team matchups
    let opp1 = get_side(a, w, sl, p1, 1u - side1);
    let opp2 = get_side(a, w, sl, p2, 1u - side2);
    if (opp1 == t2 || opp2 == t1) { return false; }

    set_side(a, w, sl, p1, side1, t2);
    set_side(a, w, sl, p2, side2, t1);
    return true;
}

// Move 3: lane_swap_week — swap two entire lanes across all slots in a week
fn move_lane_swap_week(a: ptr<function, array<u32, 200>>, s: ptr<function, array<u32, 4>>) {
    let w = rng_range(s, WEEKS);
    let p1 = rng_range(s, PAIRS);
    var p2 = rng_range(s, PAIRS - 1u);
    if (p2 >= p1) { p2 += 1u; }
    for (var sl = 0u; sl < SLOTS; sl++) {
        if (is_valid_pos(sl, p1) && is_valid_pos(sl, p2)) {
            let tmp = (*a)[pos_idx(w, sl, p1)];
            (*a)[pos_idx(w, sl, p1)] = (*a)[pos_idx(w, sl, p2)];
            (*a)[pos_idx(w, sl, p2)] = tmp;
        }
    }
}

// Move 5: slot_swap — swap two full slots (0-3 only) within a week
fn move_slot_swap(a: ptr<function, array<u32, 200>>, s: ptr<function, array<u32, 4>>) {
    let w = rng_range(s, WEEKS);
    let s1 = rng_range(s, 4u);
    var s2 = rng_range(s, 3u);
    if (s2 >= s1) { s2 += 1u; }
    for (var p = 0u; p < PAIRS; p++) {
        let tmp = (*a)[pos_idx(w, s1, p)];
        (*a)[pos_idx(w, s1, p)] = (*a)[pos_idx(w, s2, p)];
        (*a)[pos_idx(w, s2, p)] = tmp;
    }
}

// Move 6: guided_matchup — find under-matched pair and try to create matchup
fn move_guided_matchup(a: ptr<function, array<u32, 200>>, s: ptr<function, array<u32, 4>>) -> bool {
    // Count matchups
    var m: array<u32, 17>;
    for (var i = 0u; i < MATCHUP_U32S; i++) { m[i] = 0u; }
    for (var w = 0u; w < WEEKS; w++) {
        for (var sl = 0u; sl < SLOTS; sl++) {
            for (var p = 0u; p < PAIRS; p++) {
                if (is_empty(a, w, sl, p)) { continue; }
                let t1 = get_left(a, w, sl, p);
                let t2 = get_right(a, w, sl, p);
                inc_packed(&m, min(t1, t2), max(t1, t2));
            }
        }
    }

    // Find pair with count < 2
    let start = rng_range(s, TEAMS);
    var ta = 0u; var tb = 0u; var found = false;
    for (var off = 0u; off < TEAMS; off++) {
        if (found) { break; }
        let i = (start + off) % TEAMS;
        for (var j = i + 1u; j < TEAMS; j++) {
            if (get_packed(&m, i, j) < 2u) {
                ta = i; tb = j; found = true; break;
            }
        }
    }
    if (!found) { return false; }

    // Find a week where both play in different slots, swap to create matchup
    let wstart = rng_range(s, WEEKS);
    for (var off = 0u; off < WEEKS; off++) {
        let w = (wstart + off) % WEEKS;
        // Find ta's position
        var sa = 99u; var pa = 99u; var side_a = 99u;
        var sb = 99u; var pb = 99u; var side_b = 99u;
        for (var sl = 0u; sl < SLOTS; sl++) {
            for (var p = 0u; p < PAIRS; p++) {
                if (is_empty(a, w, sl, p)) { continue; }
                if (get_left(a, w, sl, p) == ta && sa == 99u) { sa = sl; pa = p; side_a = 0u; }
                else if (get_right(a, w, sl, p) == ta && sa == 99u) { sa = sl; pa = p; side_a = 1u; }
                if (get_left(a, w, sl, p) == tb && sb == 99u) { sb = sl; pb = p; side_b = 0u; }
                else if (get_right(a, w, sl, p) == tb && sb == 99u) { sb = sl; pb = p; side_b = 1u; }
            }
        }
        if (sa == 99u || sb == 99u || sa == sb) { continue; }

        // Swap tb's opponent into ta's matchup, putting tb there
        let opp = get_side(a, w, sa, pa, 1u - side_a);
        if (team_in_slot(a, w, sa, tb, pa)) { continue; }
        if (team_in_slot(a, w, sb, opp, pb)) { continue; }
        // Check we don't create same-team matchup at (sb, pb)
        let opp_b = get_side(a, w, sb, pb, 1u - side_b);
        if (opp_b == opp) { continue; }

        set_side(a, w, sa, pa, 1u - side_a, tb);
        set_side(a, w, sb, pb, side_b, opp);
        return true;
    }
    return false;
}

// Move 6: guided_lane — find worst lane imbalance, opponent-swap to fix
fn move_guided_lane(a: ptr<function, array<u32, 200>>, s: ptr<function, array<u32, 4>>) -> bool {
    var lc: array<u32, 12>;
    for (var i = 0u; i < 12u; i++) { lc[i] = 0u; }

    for (var w = 0u; w < WEEKS; w++) {
        for (var sl = 0u; sl < SLOTS; sl++) {
            for (var p = 0u; p < PAIRS; p++) {
                if (is_empty(a, w, sl, p)) { continue; }
                let t1 = get_left(a, w, sl, p);
                let t2 = get_right(a, w, sl, p);
                let l1 = t1 * LANES + p;
                lc[l1 / 4u] += (1u << ((l1 % 4u) * 8u));
                let l2 = t2 * LANES + p;
                lc[l2 / 4u] += (1u << ((l2 % 4u) * 8u));
            }
        }
    }

    let tgt = f32(WEEKS * 3u) / f32(LANES);
    var worst_team = 0u; var worst_lane = 0u; var worst_dev = 0.0; var worst_over = false;
    for (var t = 0u; t < TEAMS; t++) {
        for (var l = 0u; l < LANES; l++) {
            let li = t * LANES + l;
            let count = (lc[li / 4u] >> ((li % 4u) * 8u)) & 0xFFu;
            let dev = f32(count) - tgt;
            if (abs(dev) > worst_dev) { worst_dev = abs(dev); worst_team = t; worst_lane = l; worst_over = dev > 0.0; }
        }
    }
    if (worst_dev < 1.0) { return false; }

    // Find worst_team on the over-represented lane and opponent-swap to a different lane
    let wstart = rng_range(s, WEEKS);
    for (var off = 0u; off < WEEKS; off++) {
        let w = (wstart + off) % WEEKS;
        // Find worst_team's position on the source lane
        var found_sl = 99u; var found_p = 99u; var found_side = 99u;
        for (var sl = 0u; sl < SLOTS; sl++) {
            for (var p = 0u; p < PAIRS; p++) {
                if (is_empty(a, w, sl, p)) { continue; }
                let matches_lane = (worst_over && p == worst_lane) || (!worst_over && p != worst_lane);
                if (!matches_lane) { continue; }
                if (get_left(a, w, sl, p) == worst_team) { found_sl = sl; found_p = p; found_side = 0u; break; }
                if (get_right(a, w, sl, p) == worst_team) { found_sl = sl; found_p = p; found_side = 1u; break; }
            }
            if (found_sl != 99u) { break; }
        }
        if (found_sl == 99u) { continue; }

        // Find another matchup in the same slot on a different lane to swap into
        let pstart = rng_range(s, PAIRS);
        for (var poff = 0u; poff < PAIRS; poff++) {
            let tgt_p = (pstart + poff) % PAIRS;
            if (tgt_p == found_p) { continue; }
            if (is_empty(a, w, found_sl, tgt_p)) { continue; }

            let swap_side = rng_range(s, 2u);
            let their_team = get_side(a, w, found_sl, tgt_p, swap_side);
            let my_opp = get_side(a, w, found_sl, found_p, 1u - found_side);
            let their_opp = get_side(a, w, found_sl, tgt_p, 1u - swap_side);
            if (my_opp == their_team || their_opp == worst_team) { continue; }

            set_side(a, w, found_sl, found_p, found_side, their_team);
            set_side(a, w, found_sl, tgt_p, swap_side, worst_team);
            return true;
        }
    }
    return false;
}

// Move 8: guided_lane_switch — find team with worst lane switches, fix via opponent-swap
fn move_guided_lane_switch(a: ptr<function, array<u32, 200>>, s: ptr<function, array<u32, 4>>) -> bool {
    let w = rng_range(s, WEEKS);

    // Find team with worst lane-switch penalty in this week
    var worst_team = 0u;
    var worst_penalty = 0u;
    let t_start = rng_range(s, TEAMS);

    for (var t_off = 0u; t_off < TEAMS; t_off++) {
        let t = (t_start + t_off) % TEAMS;
        var gs: array<u32, 3>;
        var gp: array<u32, 3>;
        var gc = 0u;
        for (var sl = 0u; sl < SLOTS; sl++) {
            for (var p = 0u; p < PAIRS; p++) {
                if (is_empty(a, w, sl, p)) { continue; }
                if (get_left(a, w, sl, p) == t || get_right(a, w, sl, p) == t) {
                    if (gc < 3u) { gs[gc] = sl; gp[gc] = p; gc += 1u; }
                }
            }
        }
        if (gc < 2u) { continue; }

        var pen = 0u;
        for (var g = 0u; g < gc - 1u; g++) {
            if (gp[g] != gp[g + 1u]) {
                let gap = gs[g + 1u] - gs[g] - 1u;
                if (gap == 0u) { pen += weights.lane_switch_consecutive; }
                else { pen += weights.lane_switch_post_break; }
            }
        }
        if (gc == 3u) {
            var ld = 0u;
            if (gp[0u] != gp[1u]) { ld += 1u; }
            if (gp[0u] != gp[2u]) { ld += 1u; }
            if (gp[1u] != gp[2u]) { ld += 1u; }
            if (ld == 2u) { pen += weights.third_game_diff_lane; }
        }
        if (pen > worst_penalty) { worst_penalty = pen; worst_team = t; }
    }

    if (worst_penalty == 0u) { return false; }

    // Re-find games for worst_team
    var gs: array<u32, 3>;
    var gp: array<u32, 3>;
    var gside: array<u32, 3>;
    var gc = 0u;
    for (var sl = 0u; sl < SLOTS; sl++) {
        for (var p = 0u; p < PAIRS; p++) {
            if (is_empty(a, w, sl, p)) { continue; }
            if (get_left(a, w, sl, p) == worst_team) {
                if (gc < 3u) { gs[gc] = sl; gp[gc] = p; gside[gc] = 0u; gc += 1u; }
            } else if (get_right(a, w, sl, p) == worst_team) {
                if (gc < 3u) { gs[gc] = sl; gp[gc] = p; gside[gc] = 1u; gc += 1u; }
            }
        }
    }

    // Find worst transition
    var worst_idx = 0u;
    var worst_trans = 0u;
    for (var g = 0u; g < gc - 1u; g++) {
        if (gp[g] == gp[g + 1u]) { continue; }
        let gap = gs[g + 1u] - gs[g] - 1u;
        var pen = 0u;
        if (gap == 0u) { pen = weights.lane_switch_consecutive; } else { pen = weights.lane_switch_post_break; }
        if (pen > worst_trans) { worst_trans = pen; worst_idx = g; }
    }

    // Try to fix: opponent-swap team from one pair to match the other
    let pick = rng_range(s, 2u);
    var fix_s = gs[worst_idx]; var fix_p = gp[worst_idx]; var fix_side = gside[worst_idx]; var target_p = gp[worst_idx + 1u];
    if (pick == 1u) { fix_s = gs[worst_idx + 1u]; fix_p = gp[worst_idx + 1u]; fix_side = gside[worst_idx + 1u]; target_p = gp[worst_idx]; }

    if (!is_empty(a, w, fix_s, target_p)) {
        let swap_side = rng_range(s, 2u);
        let their_team = get_side(a, w, fix_s, target_p, swap_side);
        let my_opp = get_side(a, w, fix_s, fix_p, 1u - fix_side);
        let their_opp = get_side(a, w, fix_s, target_p, 1u - swap_side);
        if (my_opp == their_team || their_opp == worst_team) { return false; }

        set_side(a, w, fix_s, fix_p, fix_side, their_team);
        set_side(a, w, fix_s, target_p, swap_side, worst_team);
        return true;
    }
    return false;
}

// Move 7: guided_slot — find worst slot imbalance, team_swap into target slot
fn move_guided_slot(a: ptr<function, array<u32, 200>>, s: ptr<function, array<u32, 4>>) -> bool {
    var sc: array<u32, 15>;
    for (var i = 0u; i < 15u; i++) { sc[i] = 0u; }

    for (var w = 0u; w < WEEKS; w++) {
        for (var sl = 0u; sl < SLOTS; sl++) {
            for (var p = 0u; p < PAIRS; p++) {
                if (is_empty(a, w, sl, p)) { continue; }
                let t1 = get_left(a, w, sl, p);
                let t2 = get_right(a, w, sl, p);
                let si1 = t1 * SLOTS + sl;
                sc[si1 / 4u] += (1u << ((si1 % 4u) * 8u));
                let si2 = t2 * SLOTS + sl;
                sc[si2 / 4u] += (1u << ((si2 % 4u) * 8u));
            }
        }
    }

    let slot4_target = f32(4u * WEEKS) / f32(TEAMS);
    let slot03_target = (f32(WEEKS * 3u) - slot4_target) / 4.0;

    var worst_team = 0u; var worst_slot = 0u; var worst_dev = 0.0;
    for (var t = 0u; t < TEAMS; t++) {
        for (var sl = 0u; sl < SLOTS; sl++) {
            let si = t * SLOTS + sl;
            let count = (sc[si / 4u] >> ((si % 4u) * 8u)) & 0xFFu;
            var tgt = slot03_target;
            if (sl == 4u) { tgt = slot4_target; }
            let dev = tgt - f32(count); // positive = underrepresented
            if (dev > worst_dev) { worst_dev = dev; worst_team = t; worst_slot = sl; }
        }
    }
    if (worst_dev < 1.0) { return false; }

    // Find worst_team NOT in worst_slot, team_swap with someone in worst_slot
    let wstart = rng_range(s, WEEKS);
    for (var off = 0u; off < WEEKS; off++) {
        let w = (wstart + off) % WEEKS;

        // Find worst_team in a different slot (the "from" position)
        var sf = 99u; var pf = 99u; var side_f = 99u;
        for (var sl = 0u; sl < SLOTS; sl++) {
            if (sl == worst_slot) { continue; }
            for (var p = 0u; p < PAIRS; p++) {
                if (is_empty(a, w, sl, p)) { continue; }
                if (get_left(a, w, sl, p) == worst_team) {
                    sf = sl; pf = p; side_f = 0u;
                } else if (get_right(a, w, sl, p) == worst_team) {
                    sf = sl; pf = p; side_f = 1u;
                }
            }
        }
        if (sf == 99u) { continue; }

        // Find a matchup in worst_slot to swap with
        let pstart = rng_range(s, PAIRS);
        for (var poff = 0u; poff < PAIRS; poff++) {
            let pt = (pstart + poff) % PAIRS;
            if (!is_valid_pos(worst_slot, pt)) { continue; }
            if (is_empty(a, w, worst_slot, pt)) { continue; }

            let side_t = rng_range(s, 2u);
            let their_team = get_side(a, w, worst_slot, pt, side_t);

            // Check constraints
            if (team_in_slot(a, w, worst_slot, worst_team, pt)) { continue; }
            if (team_in_slot(a, w, sf, their_team, pf)) { continue; }
            let my_opp = get_side(a, w, sf, pf, 1u - side_f);
            let their_opp = get_side(a, w, worst_slot, pt, 1u - side_t);
            if (my_opp == their_team || their_opp == worst_team) { continue; }

            // Perform team swap
            set_side(a, w, sf, pf, side_f, their_team);
            set_side(a, w, worst_slot, pt, side_t, worst_team);
            return true;
        }
    }
    return false;
}

// Move 9: pair_swap_in_slot — swap two entire matchups between lane pairs in same slot
fn move_pair_swap_in_slot(a: ptr<function, array<u32, 200>>, s: ptr<function, array<u32, 4>>) -> bool {
    let w = rng_range(s, WEEKS);
    let sl = rng_range(s, SLOTS);
    let p1 = rng_range(s, PAIRS);
    if (!is_valid_pos(sl, p1)) { return false; }
    if (is_empty(a, w, sl, p1)) { return false; }
    var p2 = rng_range(s, PAIRS - 1u);
    if (p2 >= p1) { p2 += 1u; }
    if (!is_valid_pos(sl, p2)) { return false; }
    if (is_empty(a, w, sl, p2)) { return false; }

    let tmp = (*a)[pos_idx(w, sl, p1)];
    (*a)[pos_idx(w, sl, p1)] = (*a)[pos_idx(w, sl, p2)];
    (*a)[pos_idx(w, sl, p2)] = tmp;
    return true;
}

// Move 10: guided_lane_cross_slot — fix lane imbalance via team_swap across slots
fn move_guided_lane_cross_slot(a: ptr<function, array<u32, 200>>, s: ptr<function, array<u32, 4>>) -> bool {
    var lc: array<u32, 12>;
    for (var i = 0u; i < 12u; i++) { lc[i] = 0u; }

    for (var w = 0u; w < WEEKS; w++) {
        for (var sl = 0u; sl < SLOTS; sl++) {
            for (var p = 0u; p < PAIRS; p++) {
                if (is_empty(a, w, sl, p)) { continue; }
                let t1 = get_left(a, w, sl, p);
                let t2 = get_right(a, w, sl, p);
                let l1 = t1 * LANES + p;
                lc[l1 / 4u] += (1u << ((l1 % 4u) * 8u));
                let l2 = t2 * LANES + p;
                lc[l2 / 4u] += (1u << ((l2 % 4u) * 8u));
            }
        }
    }

    let tgt = f32(WEEKS * 3u) / f32(LANES);
    var worst_team = 0u; var worst_lane = 0u; var worst_dev = 0.0; var worst_over = false;
    for (var t = 0u; t < TEAMS; t++) {
        for (var l = 0u; l < LANES; l++) {
            let li = t * LANES + l;
            let count = (lc[li / 4u] >> ((li % 4u) * 8u)) & 0xFFu;
            let dev = f32(count) - tgt;
            if (abs(dev) > worst_dev) { worst_dev = abs(dev); worst_team = t; worst_lane = l; worst_over = dev > 0.0; }
        }
    }
    if (worst_dev < 1.0) { return false; }

    let w = rng_range(s, WEEKS);
    // Find worst_team on source lane
    var sf = 99u; var pf = 99u; var side_f = 99u;
    for (var sl = 0u; sl < SLOTS; sl++) {
        for (var p = 0u; p < PAIRS; p++) {
            if (is_empty(a, w, sl, p)) { continue; }
            let matches = (worst_over && p == worst_lane) || (!worst_over && p != worst_lane);
            if (!matches) { continue; }
            if (get_left(a, w, sl, p) == worst_team) { sf = sl; pf = p; side_f = 0u; }
            else if (get_right(a, w, sl, p) == worst_team) { sf = sl; pf = p; side_f = 1u; }
        }
    }
    if (sf == 99u) { return false; }

    // Find target in different slot on different lane
    let sl_start = rng_range(s, SLOTS);
    for (var sl_off = 0u; sl_off < SLOTS; sl_off++) {
        let st = (sl_start + sl_off) % SLOTS;
        if (st == sf) { continue; }
        let p_start = rng_range(s, PAIRS);
        for (var p_off = 0u; p_off < PAIRS; p_off++) {
            let pt = (p_start + p_off) % PAIRS;
            if (!is_valid_pos(st, pt)) { continue; }
            if (is_empty(a, w, st, pt)) { continue; }
            let on_target = (worst_over && pt != worst_lane) || (!worst_over && pt == worst_lane);
            if (!on_target) { continue; }

            let side_t = rng_range(s, 2u);
            let their_team = get_side(a, w, st, pt, side_t);

            if (team_in_slot(a, w, st, worst_team, pt)) { continue; }
            if (team_in_slot(a, w, sf, their_team, pf)) { continue; }
            let my_opp = get_side(a, w, sf, pf, 1u - side_f);
            let their_opp = get_side(a, w, st, pt, 1u - side_t);
            if (my_opp == their_team || their_opp == worst_team) { continue; }

            set_side(a, w, sf, pf, side_f, their_team);
            set_side(a, w, st, pt, side_t, worst_team);
            return true;
        }
    }
    return false;
}

// Move 11: lane_chase — multi-week compound move to fix worst lane imbalance
fn move_lane_chase(a: ptr<function, array<u32, 200>>, s: ptr<function, array<u32, 4>>) -> bool {
    // Compute lane counts (packed: 12 teams × 4 lanes = 48 bytes = 12 u32s)
    var lc: array<u32, 12>;
    for (var i = 0u; i < 12u; i++) { lc[i] = 0u; }

    for (var w = 0u; w < WEEKS; w++) {
        for (var sl = 0u; sl < SLOTS; sl++) {
            for (var p = 0u; p < PAIRS; p++) {
                if (is_empty(a, w, sl, p)) { continue; }
                let t1 = get_left(a, w, sl, p);
                let t2 = get_right(a, w, sl, p);
                let l1 = t1 * LANES + p;
                lc[l1 / 4u] += (1u << ((l1 % 4u) * 8u));
                let l2 = t2 * LANES + p;
                lc[l2 / 4u] += (1u << ((l2 % 4u) * 8u));
            }
        }
    }

    let tgt = f32(WEEKS * 3u) / f32(LANES);
    var worst_team = 0u; var worst_lane = 0u; var worst_dev = 0.0; var worst_over = false;
    for (var t = 0u; t < TEAMS; t++) {
        for (var l = 0u; l < LANES; l++) {
            let li = t * LANES + l;
            let count = (lc[li / 4u] >> ((li % 4u) * 8u)) & 0xFFu;
            let dev = f32(count) - tgt;
            if (abs(dev) > worst_dev) { worst_dev = abs(dev); worst_team = t; worst_lane = l; worst_over = dev > 0.0; }
        }
    }
    if (worst_dev < 1.0) { return false; }

    var changed = false;
    let w_start = rng_range(s, WEEKS);

    for (var w_off = 0u; w_off < WEEKS; w_off++) {
        let w = (w_start + w_off) % WEEKS;

        // Find worst_team on source lane in this week
        var sf = 99u; var pf = 99u; var side_f = 99u;
        for (var sl = 0u; sl < SLOTS; sl++) {
            for (var p = 0u; p < PAIRS; p++) {
                if (is_empty(a, w, sl, p)) { continue; }
                let matches = (worst_over && p == worst_lane) || (!worst_over && p != worst_lane);
                if (!matches) { continue; }
                if (get_left(a, w, sl, p) == worst_team) { sf = sl; pf = p; side_f = 0u; }
                else if (get_right(a, w, sl, p) == worst_team) { sf = sl; pf = p; side_f = 1u; }
            }
        }
        if (sf == 99u) { continue; }

        // Strategy 1: pair-swap within same slot
        var did_pair_swap = false;
        for (var tp = 0u; tp < PAIRS; tp++) {
            if (tp == pf) { continue; }
            if (!is_valid_pos(sf, tp)) { continue; }
            if (is_empty(a, w, sf, tp)) { continue; }
            let on_target = (worst_over && tp != worst_lane) || (!worst_over && tp == worst_lane);
            if (!on_target) { continue; }

            let tmp = (*a)[pos_idx(w, sf, pf)];
            (*a)[pos_idx(w, sf, pf)] = (*a)[pos_idx(w, sf, tp)];
            (*a)[pos_idx(w, sf, tp)] = tmp;
            changed = true;
            did_pair_swap = true;
            break;
        }
        if (did_pair_swap) { continue; }

        // Strategy 2: team_swap across slots
        var did_team_swap = false;
        for (var st = 0u; st < SLOTS; st++) {
            if (st == sf) { continue; }
            if (did_team_swap) { break; }
            for (var pt = 0u; pt < PAIRS; pt++) {
                if (!is_valid_pos(st, pt)) { continue; }
                if (is_empty(a, w, st, pt)) { continue; }
                let on_target = (worst_over && pt != worst_lane) || (!worst_over && pt == worst_lane);
                if (!on_target) { continue; }

                let side_t = rng_range(s, 2u);
                let their_team = get_side(a, w, st, pt, side_t);

                if (team_in_slot(a, w, st, worst_team, pt)) { continue; }
                if (team_in_slot(a, w, sf, their_team, pf)) { continue; }
                let my_opp = get_side(a, w, sf, pf, 1u - side_f);
                let their_opp = get_side(a, w, st, pt, 1u - side_t);
                if (my_opp == their_team || their_opp == worst_team) { continue; }

                set_side(a, w, sf, pf, side_f, their_team);
                set_side(a, w, st, pt, side_t, worst_team);
                changed = true;
                did_team_swap = true;
                break;
            }
        }
    }

    return changed;
}

// ═══════════════════════════════════════════════════════════════════════
// Main SA kernel
// ═══════════════════════════════════════════════════════════════════════

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let tid = gid.x;
    if (tid >= params.chain_count) { return; }

    let base = tid * ASSIGN_SIZE;
    let rng_base = tid * 4u;

    var a: array<u32, 200>;
    for (var i = 0u; i < ASSIGN_SIZE; i++) {
        a[i] = assignments[base + i];
    }

    var rng: array<u32, 4>;
    for (var i = 0u; i < 4u; i++) {
        rng[i] = rng_states[rng_base + i];
    }

    var cost = costs[tid];
    var best_cost = best_costs[tid];

    let level_in_pod = tid % params.pod_size;
    let t_frac = f32(level_in_pod) / f32(max(params.pod_size - 1u, 1u));
    let temp = params.temp_base * pow(params.temp_step / params.temp_base, t_frac);

    for (var iter = 0u; iter < params.iters_per_dispatch; iter++) {
        if (best_cost == 0u) { break; }

        let move_id = rng_range(&rng, 100u);

        if (move_id < move_thresh[0]) {
            // 0: team_swap
            var sq = save_all(&a);
            let did = move_team_swap(&a, &rng);
            if (did) {
                let new_cost = evaluate(&a);
                let delta = i32(new_cost) - i32(cost);
                if (sa_accept(delta, temp, &rng)) {
                    cost = new_cost;
                    if (cost < best_cost) { best_cost = cost; write_best(&a, base); }
                } else { restore_all(&a, &sq); }
            } else { restore_all(&a, &sq); }
        } else if (move_id < move_thresh[1]) {
            // 1: matchup_swap
            var sq = save_all(&a);
            let did = move_matchup_swap(&a, &rng);
            if (did) {
                let new_cost = evaluate(&a);
                let delta = i32(new_cost) - i32(cost);
                if (sa_accept(delta, temp, &rng)) {
                    cost = new_cost;
                    if (cost < best_cost) { best_cost = cost; write_best(&a, base); }
                } else { restore_all(&a, &sq); }
            }
        } else if (move_id < move_thresh[2]) {
            // 2: opponent_swap
            var sq = save_all(&a);
            let did = move_opponent_swap(&a, &rng);
            if (did) {
                let new_cost = evaluate(&a);
                let delta = i32(new_cost) - i32(cost);
                if (sa_accept(delta, temp, &rng)) {
                    cost = new_cost;
                    if (cost < best_cost) { best_cost = cost; write_best(&a, base); }
                } else { restore_all(&a, &sq); }
            }
        } else if (move_id < move_thresh[3]) {
            // 3: lane_swap_week
            var sq = save_all(&a);
            move_lane_swap_week(&a, &rng);
            let new_cost = evaluate(&a);
            let delta = i32(new_cost) - i32(cost);
            if (sa_accept(delta, temp, &rng)) {
                cost = new_cost;
                if (cost < best_cost) { best_cost = cost; write_best(&a, base); }
            } else { restore_all(&a, &sq); }
        } else if (move_id < move_thresh[4]) {
            // 4: slot_swap
            var sq = save_all(&a);
            move_slot_swap(&a, &rng);
            let new_cost = evaluate(&a);
            let delta = i32(new_cost) - i32(cost);
            if (sa_accept(delta, temp, &rng)) {
                cost = new_cost;
                if (cost < best_cost) { best_cost = cost; write_best(&a, base); }
            } else { restore_all(&a, &sq); }
        } else if (move_id < move_thresh[5]) {
            // 5: guided_matchup
            var sq = save_all(&a);
            let did = move_guided_matchup(&a, &rng);
            if (did) {
                let new_cost = evaluate(&a);
                let delta = i32(new_cost) - i32(cost);
                if (sa_accept(delta, temp, &rng)) {
                    cost = new_cost;
                    if (cost < best_cost) { best_cost = cost; write_best(&a, base); }
                } else { restore_all(&a, &sq); }
            }
        } else if (move_id < move_thresh[6]) {
            // 6: guided_lane
            var sq = save_all(&a);
            let did = move_guided_lane(&a, &rng);
            if (did) {
                let new_cost = evaluate(&a);
                let delta = i32(new_cost) - i32(cost);
                if (sa_accept(delta, temp, &rng)) {
                    cost = new_cost;
                    if (cost < best_cost) { best_cost = cost; write_best(&a, base); }
                } else { restore_all(&a, &sq); }
            }
        } else if (move_id < move_thresh[7]) {
            // 7: guided_slot
            var sq = save_all(&a);
            let did = move_guided_slot(&a, &rng);
            if (did) {
                let new_cost = evaluate(&a);
                let delta = i32(new_cost) - i32(cost);
                if (sa_accept(delta, temp, &rng)) {
                    cost = new_cost;
                    if (cost < best_cost) { best_cost = cost; write_best(&a, base); }
                } else { restore_all(&a, &sq); }
            }
        } else if (move_id < move_thresh[8]) {
            // 8: guided_lane_switch
            var sq = save_all(&a);
            let did = move_guided_lane_switch(&a, &rng);
            if (did) {
                let new_cost = evaluate(&a);
                let delta = i32(new_cost) - i32(cost);
                if (sa_accept(delta, temp, &rng)) {
                    cost = new_cost;
                    if (cost < best_cost) { best_cost = cost; write_best(&a, base); }
                } else { restore_all(&a, &sq); }
            }
        } else if (move_id < move_thresh[9]) {
            // 9: pair_swap_in_slot
            var sq = save_all(&a);
            let did = move_pair_swap_in_slot(&a, &rng);
            if (did) {
                let new_cost = evaluate(&a);
                let delta = i32(new_cost) - i32(cost);
                if (sa_accept(delta, temp, &rng)) {
                    cost = new_cost;
                    if (cost < best_cost) { best_cost = cost; write_best(&a, base); }
                } else { restore_all(&a, &sq); }
            }
        } else if (move_id < move_thresh[10]) {
            // 10: guided_lane_cross_slot
            var sq = save_all(&a);
            let did = move_guided_lane_cross_slot(&a, &rng);
            if (did) {
                let new_cost = evaluate(&a);
                let delta = i32(new_cost) - i32(cost);
                if (sa_accept(delta, temp, &rng)) {
                    cost = new_cost;
                    if (cost < best_cost) { best_cost = cost; write_best(&a, base); }
                } else { restore_all(&a, &sq); }
            }
        } else {
            // 11: lane_chase
            var sq = save_all(&a);
            let did = move_lane_chase(&a, &rng);
            if (did) {
                let new_cost = evaluate(&a);
                let delta = i32(new_cost) - i32(cost);
                if (sa_accept(delta, temp, &rng)) {
                    cost = new_cost;
                    if (cost < best_cost) { best_cost = cost; write_best(&a, base); }
                } else { restore_all(&a, &sq); }
            }
        }
    }

    // Write back state
    for (var i = 0u; i < ASSIGN_SIZE; i++) {
        assignments[base + i] = a[i];
    }
    for (var i = 0u; i < 4u; i++) {
        rng_states[rng_base + i] = rng[i];
    }
    costs[tid] = cost;
    best_costs[tid] = best_cost;
}
