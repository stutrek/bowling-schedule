// Summer-fixed SA solver shader
// 12 teams, 10 weeks, fixed day template with 18 matchup entries
// Assignment: 31 u32s per chain
//   u32[0..30]: mapping[w][p] — byte (p%4) of u32[w*3 + p/4], team 0-11
//   u32[30]: bits 0-9 = swap_01[w], bits 10-19 = swap_23[w]

const TEAMS: u32 = 12u;
const WEEKS: u32 = 10u;
const SLOTS: u32 = 5u;
const LANES: u32 = 4u;
const ASSIGN_SIZE: u32 = 31u;
const TEMPLATE_SIZE: u32 = 18u;
const NUM_PAIRS: u32 = 66u;
const MATCHUP_U32S: u32 = 9u; // ceil(66/8) = 9, packing 8 nibbles per u32

// Template arrays (injected at runtime)
// TEMPLATE_CONSTS_PLACEHOLDER

struct Weights {
    matchup_balance: u32,
    slot_balance: u32,
    lane_balance: u32,
    game5_lane_balance: u32,
    same_lane_balance: u32,
    commissioner_overlap: u32,
    matchup_spacing: u32,
    _pad0: u32,
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
@group(0) @binding(7) var<storage, read> move_thresh: array<u32, 16>;

// Undo info: move_type + up to 3 values
var<private> undo_type: u32;
var<private> undo_a: u32;
var<private> undo_b: u32;
var<private> undo_c: u32;

// RNG: xoshiro128++
var<private> rng: vec4<u32>;

fn rng_load(tid: u32) {
    let b = tid * 4u;
    rng = vec4(rng_states[b], rng_states[b + 1u], rng_states[b + 2u], rng_states[b + 3u]);
}

fn rng_store(tid: u32) {
    let b = tid * 4u;
    rng_states[b] = rng.x;
    rng_states[b + 1u] = rng.y;
    rng_states[b + 2u] = rng.z;
    rng_states[b + 3u] = rng.w;
}

fn rotl(x: u32, k: u32) -> u32 {
    return (x << k) | (x >> (32u - k));
}

fn rng_next() -> u32 {
    let result = rotl(rng.x + rng.w, 7u) + rng.x;
    let t = rng.y << 9u;
    rng.z ^= rng.x;
    rng.w ^= rng.y;
    rng.y ^= rng.z;
    rng.x ^= rng.w;
    rng.z ^= t;
    rng.w = rotl(rng.w, 11u);
    return result;
}

fn rng_range(n: u32) -> u32 {
    return rng_next() % n;
}

fn rng_bool() -> bool {
    return (rng_next() & 1u) == 0u;
}

// --- Assignment access helpers ---

fn get_team(base: u32, w: u32, p: u32) -> u32 {
    let idx = base + w * 3u + p / 4u;
    let shift = (p % 4u) * 8u;
    return (assignments[idx] >> shift) & 0xFFu;
}

fn set_team(base: u32, w: u32, p: u32, team: u32) {
    let idx = base + w * 3u + p / 4u;
    let shift = (p % 4u) * 8u;
    let mask = ~(0xFFu << shift);
    assignments[idx] = (assignments[idx] & mask) | (team << shift);
}

fn get_swap01(base: u32, w: u32) -> bool {
    return (assignments[base + 30u] & (1u << w)) != 0u;
}

fn get_swap23(base: u32, w: u32) -> bool {
    return (assignments[base + 30u] & (1u << (w + 10u))) != 0u;
}

fn toggle_swap01(base: u32, w: u32) {
    assignments[base + 30u] ^= (1u << w);
}

fn toggle_swap23(base: u32, w: u32) {
    assignments[base + 30u] ^= (1u << (w + 10u));
}

fn apply_lane_swap(lane: u32, s01: bool, s23: bool) -> u32 {
    if (lane == 0u && s01) { return 1u; }
    if (lane == 1u && s01) { return 0u; }
    if (lane == 2u && s23) { return 3u; }
    if (lane == 3u && s23) { return 2u; }
    return lane;
}

// --- Pair index ---

fn pair_idx(a: u32, b: u32) -> u32 {
    let lo = min(a, b);
    let hi = max(a, b);
    return lo * (2u * TEAMS - lo - 1u) / 2u + (hi - lo - 1u);
}

// --- Cost evaluation ---

fn evaluate(base: u32) -> u32 {
    var cost: u32 = 0u;

    // 1. Matchup balance: 66 pair counts packed as nibbles (4 bits each), 8 per u32
    var mc: array<u32, 9>;
    for (var i = 0u; i < 9u; i++) { mc[i] = 0u; }

    for (var w = 0u; w < WEEKS; w++) {
        for (var e = 0u; e < TEMPLATE_SIZE; e++) {
            let ta = get_team(base, w, T_POS_A[e]);
            let tb = get_team(base, w, T_POS_B[e]);
            let pi = pair_idx(ta, tb);
            let word = pi / 8u;
            let nibble = (pi % 8u) * 4u;
            mc[word] += 1u << nibble;
        }
    }
    for (var i = 0u; i < NUM_PAIRS; i++) {
        let word = i / 8u;
        let nibble = (i % 8u) * 4u;
        let c = (mc[word] >> nibble) & 0xFu;
        if (c < 2u) { cost += weights.matchup_balance * (2u - c); }
        else if (c > 3u) { cost += weights.matchup_balance * (c - 3u); }
    }

    // 2. Slot balance: 12 teams × 5 slots = 60 counters, packed 4 per u32, 8 bits each
    var sc: array<u32, 15>;
    for (var i = 0u; i < 15u; i++) { sc[i] = 0u; }

    for (var w = 0u; w < WEEKS; w++) {
        for (var e = 0u; e < TEMPLATE_SIZE; e++) {
            let s = T_SLOT[e];
            let ta = get_team(base, w, T_POS_A[e]);
            let tb = get_team(base, w, T_POS_B[e]);
            let idx_a = ta * 5u + s;
            let idx_b = tb * 5u + s;
            sc[idx_a / 4u] += 1u << ((idx_a % 4u) * 8u);
            sc[idx_b / 4u] += 1u << ((idx_b % 4u) * 8u);
        }
    }
    for (var t = 0u; t < TEAMS; t++) {
        for (var s = 0u; s < SLOTS; s++) {
            let idx = t * 5u + s;
            let c = (sc[idx / 4u] >> ((idx % 4u) * 8u)) & 0xFFu;
            if (s < 4u) {
                if (c < 6u) { cost += weights.slot_balance * (6u - c); }
                else if (c > 7u) { cost += weights.slot_balance * (c - 7u); }
            } else {
                if (c < 3u) { cost += weights.slot_balance * (3u - c); }
                else if (c > 4u) { cost += weights.slot_balance * (c - 4u); }
            }
        }
    }

    // 3. Lane balance: 12 teams × 4 lanes = 48 counters, packed 4 per u32
    var lc: array<u32, 12>;
    for (var i = 0u; i < 12u; i++) { lc[i] = 0u; }

    for (var w = 0u; w < WEEKS; w++) {
        let s01 = get_swap01(base, w);
        let s23 = get_swap23(base, w);
        for (var e = 0u; e < TEMPLATE_SIZE; e++) {
            let actual_lane = apply_lane_swap(T_LANE[e], s01, s23);
            let ta = get_team(base, w, T_POS_A[e]);
            let tb = get_team(base, w, T_POS_B[e]);
            let idx_a = ta * 4u + actual_lane;
            let idx_b = tb * 4u + actual_lane;
            lc[idx_a / 4u] += 1u << ((idx_a % 4u) * 8u);
            lc[idx_b / 4u] += 1u << ((idx_b % 4u) * 8u);
        }
    }
    for (var t = 0u; t < TEAMS; t++) {
        for (var l = 0u; l < LANES; l++) {
            let idx = t * 4u + l;
            let c = (lc[idx / 4u] >> ((idx % 4u) * 8u)) & 0xFFu;
            if (l < 2u) {
                if (c < 7u) { cost += weights.lane_balance * (7u - c); }
                else if (c > 7u) { cost += weights.lane_balance * (c - 7u); }
            } else {
                if (c < 8u) { cost += weights.lane_balance * (8u - c); }
                else if (c > 8u) { cost += weights.lane_balance * (c - 8u); }
            }
        }
    }

    // 4. Game 5 lane balance: per team, |lane2_count - lane3_count| > 1 penalized
    // Reuse lane counts from above but only for slot 4 entries
    var g5l2: array<u32, 3>;  // 12 counters packed 4 per u32
    var g5l3: array<u32, 3>;
    for (var i = 0u; i < 3u; i++) { g5l2[i] = 0u; g5l3[i] = 0u; }

    for (var w = 0u; w < WEEKS; w++) {
        let s01 = get_swap01(base, w);
        let s23 = get_swap23(base, w);
        // Template entries 16 and 17 are slot 4
        for (var e = 16u; e < 18u; e++) {
            let actual_lane = apply_lane_swap(T_LANE[e], s01, s23);
            let ta = get_team(base, w, T_POS_A[e]);
            let tb = get_team(base, w, T_POS_B[e]);
            if (actual_lane == 2u) {
                g5l2[ta / 4u] += 1u << ((ta % 4u) * 8u);
                g5l2[tb / 4u] += 1u << ((tb % 4u) * 8u);
            } else {
                g5l3[ta / 4u] += 1u << ((ta % 4u) * 8u);
                g5l3[tb / 4u] += 1u << ((tb % 4u) * 8u);
            }
        }
    }
    for (var t = 0u; t < TEAMS; t++) {
        let c2 = (g5l2[t / 4u] >> ((t % 4u) * 8u)) & 0xFFu;
        let c3 = (g5l3[t / 4u] >> ((t % 4u) * 8u)) & 0xFFu;
        let diff = max(c2, c3) - min(c2, c3);
        if (diff > 1u) {
            cost += weights.game5_lane_balance * (diff - 1u);
        }
    }

    // 5. Same lane balance: count SAME_LANE_POS per team, penalize outside [3,4]
    var slc: array<u32, 3>;
    for (var i = 0u; i < 3u; i++) { slc[i] = 0u; }

    for (var w = 0u; w < WEEKS; w++) {
        for (var i = 0u; i < SAME_LANE_COUNT; i++) {
            let team = get_team(base, w, SAME_LANE_POS[i]);
            slc[team / 4u] += 1u << ((team % 4u) * 8u);
        }
    }
    for (var t = 0u; t < TEAMS; t++) {
        let c = (slc[t / 4u] >> ((t % 4u) * 8u)) & 0xFFu;
        if (c < 3u) { cost += weights.same_lane_balance * (3u - c); }
        else if (c > 4u) { cost += weights.same_lane_balance * (c - 4u); }
    }

    // 6. Commissioner overlap: comm_bits per team, min pair overlap
    var cb: array<u32, 12>;
    for (var i = 0u; i < 12u; i++) { cb[i] = 0u; }

    for (var w = 0u; w < WEEKS; w++) {
        for (var i = 0u; i < SLOT0_COUNT; i++) {
            let team = get_team(base, w, SLOT0_POS[i]);
            cb[team] |= 1u << (w * 2u);
        }
        for (var i = 0u; i < SLOT4_COUNT; i++) {
            let team = get_team(base, w, SLOT4_POS[i]);
            cb[team] |= 1u << (w * 2u + 1u);
        }
    }
    var min_co: u32 = 20u;
    for (var i = 0u; i < TEAMS; i++) {
        for (var j = i + 1u; j < TEAMS; j++) {
            let co = countOneBits(cb[i] & cb[j]);
            min_co = min(min_co, co);
        }
    }
    cost += weights.commissioner_overlap * min_co;

    // 7. Matchup spacing: penalize pairs whose matchups are too close together
    // 2 total matchups → need 4+ weeks apart; 3 total → need 2+ weeks apart
    // Per-pair week bitmask: 10 weeks fit in 10 bits per pair
    // Pack pair week-bits: 3 pairs per u32 (10 bits each), need ceil(66/3) = 22 u32s
    var pw: array<u32, 22>;
    for (var i = 0u; i < 22u; i++) { pw[i] = 0u; }

    for (var w = 0u; w < WEEKS; w++) {
        for (var e = 0u; e < TEMPLATE_SIZE; e++) {
            let ta = get_team(base, w, T_POS_A[e]);
            let tb = get_team(base, w, T_POS_B[e]);
            let pi = pair_idx(ta, tb);
            let word = pi / 3u;
            let slot_shift = (pi % 3u) * 10u;
            pw[word] |= (1u << w) << slot_shift;
        }
    }

    // Check spacing for each pair using matchup counts from section 1
    for (var i = 0u; i < NUM_PAIRS; i++) {
        let c = (mc[i / 8u] >> ((i % 8u) * 4u)) & 0xFu;
        if (c < 2u) { continue; }
        let min_gap = select(2u, 4u, c == 2u);
        let word = i / 3u;
        let slot_shift = (i % 3u) * 10u;
        let bits = (pw[word] >> slot_shift) & 0x3FFu; // 10 bits of week presence

        // Extract ordered week indices and check gaps
        var prev_week: u32 = 0u;
        var found_first: bool = false;
        for (var w = 0u; w < WEEKS; w++) {
            if ((bits & (1u << w)) != 0u) {
                if (found_first) {
                    let gap = w - prev_week;
                    if (gap < min_gap) {
                        cost += weights.matchup_spacing;
                    }
                }
                prev_week = w;
                found_first = true;
            }
        }
    }

    return cost;
}

// --- Moves ---

// Move 0: Team swap — swap two positions in a random week
fn move_team_swap(base: u32) {
    let w = rng_range(WEEKS);
    let a = rng_range(TEAMS);
    var b = rng_range(TEAMS - 1u);
    if (b >= a) { b += 1u; }

    let ta = get_team(base, w, a);
    let tb = get_team(base, w, b);
    set_team(base, w, a, tb);
    set_team(base, w, b, ta);

    undo_type = 0u;
    undo_a = w;
    undo_b = a;
    undo_c = b;
}

// Move 1: Toggle swap_01
fn move_toggle_01(base: u32) {
    let w = rng_range(WEEKS);
    toggle_swap01(base, w);
    undo_type = 1u;
    undo_a = w;
}

// Move 2: Toggle swap_23
fn move_toggle_23(base: u32) {
    let w = rng_range(WEEKS);
    toggle_swap23(base, w);
    undo_type = 2u;
    undo_a = w;
}

// Move 3: Week swap — swap all data for two weeks
fn move_week_swap(base: u32) {
    let wa = rng_range(WEEKS);
    var wb = rng_range(WEEKS - 1u);
    if (wb >= wa) { wb += 1u; }

    // Swap 3 u32s of mapping
    let ba = base + wa * 3u;
    let bb = base + wb * 3u;
    for (var i = 0u; i < 3u; i++) {
        let tmp = assignments[ba + i];
        assignments[ba + i] = assignments[bb + i];
        assignments[bb + i] = tmp;
    }
    // Swap swap_01 bits
    let flags = assignments[base + 30u];
    let bit_a01 = (flags >> wa) & 1u;
    let bit_b01 = (flags >> wb) & 1u;
    let bit_a23 = (flags >> (wa + 10u)) & 1u;
    let bit_b23 = (flags >> (wb + 10u)) & 1u;
    var new_flags = flags;
    // Clear and set swap_01 bits
    new_flags = new_flags & ~(1u << wa) & ~(1u << wb);
    new_flags = new_flags | (bit_b01 << wa) | (bit_a01 << wb);
    // Clear and set swap_23 bits
    new_flags = new_flags & ~(1u << (wa + 10u)) & ~(1u << (wb + 10u));
    new_flags = new_flags | (bit_b23 << (wa + 10u)) | (bit_a23 << (wb + 10u));
    assignments[base + 30u] = new_flags;

    undo_type = 3u;
    undo_a = wa;
    undo_b = wb;
}

// Move 4: Guided matchup — find worst pair, swap a team in a relevant week
fn move_guided_matchup(base: u32) {
    // Count matchups (reuse evaluate's approach)
    var mc: array<u32, 9>;
    for (var i = 0u; i < 9u; i++) { mc[i] = 0u; }
    for (var w = 0u; w < WEEKS; w++) {
        for (var e = 0u; e < TEMPLATE_SIZE; e++) {
            let ta = get_team(base, w, T_POS_A[e]);
            let tb = get_team(base, w, T_POS_B[e]);
            let pi = pair_idx(ta, tb);
            mc[pi / 8u] += 1u << ((pi % 8u) * 4u);
        }
    }

    // Find worst pair
    var worst_pi: u32 = 0u;
    var worst_dist: u32 = 0u;
    for (var i = 0u; i < NUM_PAIRS; i++) {
        let c = (mc[i / 8u] >> ((i % 8u) * 4u)) & 0xFu;
        var dist: u32 = 0u;
        if (c < 2u) { dist = 2u - c; }
        else if (c > 3u) { dist = c - 3u; }
        if (dist > worst_dist) {
            worst_dist = dist;
            worst_pi = i;
        }
    }

    if (worst_dist == 0u) {
        move_team_swap(base);
        return;
    }

    // Just do a targeted team swap in a random week
    let w = rng_range(WEEKS);
    let a = rng_range(TEAMS);
    var b = rng_range(TEAMS - 1u);
    if (b >= a) { b += 1u; }
    let ta = get_team(base, w, a);
    let tb = get_team(base, w, b);
    set_team(base, w, a, tb);
    set_team(base, w, b, ta);
    undo_type = 0u;
    undo_a = w;
    undo_b = a;
    undo_c = b;
}

// Move 5: Guided slot — find worst slot imbalance, swap teams
fn move_guided_slot(base: u32) {
    var sc: array<u32, 15>;
    for (var i = 0u; i < 15u; i++) { sc[i] = 0u; }
    for (var w = 0u; w < WEEKS; w++) {
        for (var e = 0u; e < TEMPLATE_SIZE; e++) {
            let s = T_SLOT[e];
            let ta = get_team(base, w, T_POS_A[e]);
            let tb = get_team(base, w, T_POS_B[e]);
            sc[(ta * 5u + s) / 4u] += 1u << (((ta * 5u + s) % 4u) * 8u);
            sc[(tb * 5u + s) / 4u] += 1u << (((tb * 5u + s) % 4u) * 8u);
        }
    }

    var worst_team: u32 = 0u;
    var worst_dist: u32 = 0u;
    for (var t = 0u; t < TEAMS; t++) {
        for (var s = 0u; s < SLOTS; s++) {
            let idx = t * 5u + s;
            let c = (sc[idx / 4u] >> ((idx % 4u) * 8u)) & 0xFFu;
            var dist: u32 = 0u;
            if (s < 4u) {
                if (c < 6u) { dist = 6u - c; } else if (c > 7u) { dist = c - 7u; }
            } else {
                if (c < 3u) { dist = 3u - c; } else if (c > 4u) { dist = c - 4u; }
            }
            if (dist > worst_dist) {
                worst_dist = dist;
                worst_team = t;
            }
        }
    }

    if (worst_dist == 0u) {
        move_team_swap(base);
        return;
    }

    // Swap worst_team with a random other team in a random week
    let w = rng_range(WEEKS);
    var pos_a: u32 = 0u;
    for (var p = 0u; p < TEAMS; p++) {
        if (get_team(base, w, p) == worst_team) { pos_a = p; }
    }
    var pos_b = rng_range(TEAMS - 1u);
    if (pos_b >= pos_a) { pos_b += 1u; }

    let ta = get_team(base, w, pos_a);
    let tb = get_team(base, w, pos_b);
    set_team(base, w, pos_a, tb);
    set_team(base, w, pos_b, ta);
    undo_type = 0u;
    undo_a = w;
    undo_b = pos_a;
    undo_c = pos_b;
}

// Move 6: Guided lane — fix lane imbalance
fn move_guided_lane(base: u32) {
    var lc: array<u32, 12>;
    for (var i = 0u; i < 12u; i++) { lc[i] = 0u; }
    for (var w = 0u; w < WEEKS; w++) {
        let s01 = get_swap01(base, w);
        let s23 = get_swap23(base, w);
        for (var e = 0u; e < TEMPLATE_SIZE; e++) {
            let actual_lane = apply_lane_swap(T_LANE[e], s01, s23);
            let ta = get_team(base, w, T_POS_A[e]);
            let tb = get_team(base, w, T_POS_B[e]);
            lc[(ta * 4u + actual_lane) / 4u] += 1u << (((ta * 4u + actual_lane) % 4u) * 8u);
            lc[(tb * 4u + actual_lane) / 4u] += 1u << (((tb * 4u + actual_lane) % 4u) * 8u);
        }
    }

    var worst_team: u32 = 0u;
    var worst_lane: u32 = 0u;
    var worst_excess: u32 = 0u;
    for (var t = 0u; t < TEAMS; t++) {
        for (var l = 0u; l < LANES; l++) {
            let idx = t * 4u + l;
            let c = (lc[idx / 4u] >> ((idx % 4u) * 8u)) & 0xFFu;
            var excess: u32 = 0u;
            if (l < 2u) {
                if (c < 7u) { excess = 7u - c; } else if (c > 7u) { excess = c - 7u; }
            } else {
                if (c < 8u) { excess = 8u - c; } else if (c > 8u) { excess = c - 8u; }
            }
            if (excess > worst_excess) {
                worst_excess = excess;
                worst_team = t;
                worst_lane = l;
            }
        }
    }

    if (worst_excess == 0u) {
        move_team_swap(base);
        return;
    }

    // 50% chance: toggle lane swap, 50% chance: team swap
    if (rng_bool()) {
        let w = rng_range(WEEKS);
        if (worst_lane < 2u) {
            toggle_swap01(base, w);
            undo_type = 1u;
            undo_a = w;
        } else {
            toggle_swap23(base, w);
            undo_type = 2u;
            undo_a = w;
        }
    } else {
        let w = rng_range(WEEKS);
        var pos_a: u32 = 0u;
        for (var p = 0u; p < TEAMS; p++) {
            if (get_team(base, w, p) == worst_team) { pos_a = p; }
        }
        var pos_b = rng_range(TEAMS - 1u);
        if (pos_b >= pos_a) { pos_b += 1u; }
        let ta = get_team(base, w, pos_a);
        let tb = get_team(base, w, pos_b);
        set_team(base, w, pos_a, tb);
        set_team(base, w, pos_b, ta);
        undo_type = 0u;
        undo_a = w;
        undo_b = pos_a;
        undo_c = pos_b;
    }
}

fn apply_move(base: u32, move_type: u32) {
    switch move_type {
        case 0u: { move_team_swap(base); }
        case 1u: { move_toggle_01(base); }
        case 2u: { move_toggle_23(base); }
        case 3u: { move_week_swap(base); }
        case 4u: { move_guided_matchup(base); }
        case 5u: { move_guided_slot(base); }
        case 6u: { move_guided_lane(base); }
        default: { move_team_swap(base); }
    }
}

fn undo(base: u32) {
    switch undo_type {
        case 0u: {
            // Team swap: re-swap
            let w = undo_a;
            let a = undo_b;
            let b = undo_c;
            let ta = get_team(base, w, a);
            let tb = get_team(base, w, b);
            set_team(base, w, a, tb);
            set_team(base, w, b, ta);
        }
        case 1u: {
            // Toggle swap_01: re-toggle
            toggle_swap01(base, undo_a);
        }
        case 2u: {
            // Toggle swap_23: re-toggle
            toggle_swap23(base, undo_a);
        }
        case 3u: {
            // Week swap: re-swap
            let wa = undo_a;
            let wb = undo_b;
            let ba = base + wa * 3u;
            let bb = base + wb * 3u;
            for (var i = 0u; i < 3u; i++) {
                let tmp = assignments[ba + i];
                assignments[ba + i] = assignments[bb + i];
                assignments[bb + i] = tmp;
            }
            let flags = assignments[base + 30u];
            let bit_a01 = (flags >> wa) & 1u;
            let bit_b01 = (flags >> wb) & 1u;
            let bit_a23 = (flags >> (wa + 10u)) & 1u;
            let bit_b23 = (flags >> (wb + 10u)) & 1u;
            var new_flags = flags;
            new_flags = new_flags & ~(1u << wa) & ~(1u << wb);
            new_flags = new_flags | (bit_b01 << wa) | (bit_a01 << wb);
            new_flags = new_flags & ~(1u << (wa + 10u)) & ~(1u << (wb + 10u));
            new_flags = new_flags | (bit_b23 << (wa + 10u)) | (bit_a23 << (wb + 10u));
            assignments[base + 30u] = new_flags;
        }
        default: {}
    }
}

// --- SA acceptance ---

fn sa_accept(delta: i32, temp: f32) -> bool {
    if (delta < 0) { return true; }
    if (delta == 0) { return (rng_range(5u) == 0u); }
    let prob = exp(f32(-delta) / temp);
    let threshold = u32(prob * 4294967295.0);
    return rng_next() < threshold;
}

// --- Main entry ---

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let tid = gid.x;
    if (tid >= params.chain_count) { return; }

    rng_load(tid);
    let base = tid * ASSIGN_SIZE;
    var current_cost = costs[tid];

    let level_in_pod = tid % params.pod_size;
    let t_frac = f32(level_in_pod) / f32(max(params.pod_size - 1u, 1u));
    let temp = params.temp_base * pow(params.temp_step / params.temp_base, t_frac);

    for (var iter = 0u; iter < params.iters_per_dispatch; iter++) {
        let r = rng_range(100u);
        var move_type: u32 = 0u;
        if (r < move_thresh[0]) { move_type = 0u; }
        else if (r < move_thresh[1]) { move_type = 1u; }
        else if (r < move_thresh[2]) { move_type = 2u; }
        else if (r < move_thresh[3]) { move_type = 3u; }
        else if (r < move_thresh[4]) { move_type = 4u; }
        else if (r < move_thresh[5]) { move_type = 5u; }
        else { move_type = 6u; }

        apply_move(base, move_type);
        let new_cost = evaluate(base);
        let delta = i32(new_cost) - i32(current_cost);

        if (sa_accept(delta, temp)) {
            current_cost = new_cost;
            if (current_cost < best_costs[tid]) {
                best_costs[tid] = current_cost;
                for (var i = 0u; i < ASSIGN_SIZE; i++) {
                    best_assignments[base + i] = assignments[base + i];
                }
            }
        } else {
            undo(base);
        }
    }

    costs[tid] = current_cost;
    rng_store(tid);
}
