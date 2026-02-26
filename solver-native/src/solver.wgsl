// Constants matching solver-core
const TEAMS: u32 = 16u;
const LANES: u32 = 4u;
const WEEKS: u32 = 12u;
const QUADS: u32 = 4u;
const POS: u32 = 4u;
const ASSIGN_SIZE: u32 = 48u; // WEEKS * QUADS

struct Weights {
    matchup_zero: u32,
    matchup_triple: u32,
    consecutive_opponents: u32,
    early_late_balance: f32,
    early_late_alternation: u32,
    lane_balance: f32,
    lane_switch: f32,
    late_lane_balance: f32,
    commissioner_overlap: u32,
}

struct Params {
    iters_per_dispatch: u32,
    chain_count: u32,
    temp_base: f32,
    temp_step: f32,
}

@group(0) @binding(0) var<storage, read_write> assignments: array<u32>;
@group(0) @binding(1) var<storage, read_write> best_assignments: array<u32>;
@group(0) @binding(2) var<storage, read_write> costs: array<u32>;
@group(0) @binding(3) var<storage, read_write> best_costs: array<u32>;
@group(0) @binding(4) var<storage, read_write> rng_states: array<u32>;
@group(0) @binding(5) var<uniform> weights: Weights;
@group(0) @binding(6) var<uniform> params: Params;

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
// Assignment access helpers (packed 4xu8 in u32)
// ═══════════════════════════════════════════════════════════════════════

fn get_team(a: ptr<function, array<u32, 48>>, w: u32, q: u32, p: u32) -> u32 {
    let packed = (*a)[w * QUADS + q];
    return (packed >> (p * 8u)) & 0xFFu;
}

fn set_team(a: ptr<function, array<u32, 48>>, w: u32, q: u32, p: u32, team: u32) {
    let idx = w * QUADS + q;
    let shift = p * 8u;
    let mask = ~(0xFFu << shift);
    (*a)[idx] = ((*a)[idx] & mask) | ((team & 0xFFu) << shift);
}

fn swap_positions(a: ptr<function, array<u32, 48>>, w1: u32, q1: u32, p1: u32, w2: u32, q2: u32, p2: u32) {
    let t1 = get_team(a, w1, q1, p1);
    let t2 = get_team(a, w2, q2, p2);
    set_team(a, w1, q1, p1, t2);
    set_team(a, w2, q2, p2, t1);
}

fn get_quad(a: ptr<function, array<u32, 48>>, w: u32, q: u32) -> u32 {
    return (*a)[w * QUADS + q];
}

fn set_quad(a: ptr<function, array<u32, 48>>, w: u32, q: u32, val: u32) {
    (*a)[w * QUADS + q] = val;
}

// ═══════════════════════════════════════════════════════════════════════
// Branchless cost function
// ═══════════════════════════════════════════════════════════════════════

fn evaluate(a: ptr<function, array<u32, 48>>) -> u32 {
    var matchups: array<i32, 256>;
    var lane_counts: array<i32, 64>;
    var late_lane_counts: array<i32, 64>;
    var stay_count: array<i32, 16>;
    var early_count: array<i32, 16>;
    var early_late: array<u32, 192>;

    for (var i = 0u; i < 256u; i++) { matchups[i] = 0; }
    for (var i = 0u; i < 64u; i++) { lane_counts[i] = 0; late_lane_counts[i] = 0; }
    for (var i = 0u; i < 16u; i++) { stay_count[i] = 0; early_count[i] = 0; }
    for (var i = 0u; i < 192u; i++) { early_late[i] = 0u; }

    // Phase 1: accumulate statistics
    for (var w = 0u; w < WEEKS; w++) {
        for (var q = 0u; q < QUADS; q++) {
            let pa = get_team(a, w, q, 0u);
            let pb = get_team(a, w, q, 1u);
            let pc = get_team(a, w, q, 2u);
            let pd = get_team(a, w, q, 3u);
            let early = u32(q < 2u);
            let lane_off = (q % 2u) * 2u;

            // Matchups: (pa,pb), (pc,pd), (pa,pd), (pc,pb)
            let lo0 = min(pa, pb); let hi0 = max(pa, pb);
            let lo1 = min(pc, pd); let hi1 = max(pc, pd);
            let lo2 = min(pa, pd); let hi2 = max(pa, pd);
            let lo3 = min(pc, pb); let hi3 = max(pc, pb);
            matchups[lo0 * TEAMS + hi0] += 1;
            matchups[lo1 * TEAMS + hi1] += 1;
            matchups[lo2 * TEAMS + hi2] += 1;
            matchups[lo3 * TEAMS + hi3] += 1;

            // Lane counts
            lane_counts[pa * LANES + lane_off] += 2;
            lane_counts[pb * LANES + lane_off] += 1;
            lane_counts[pb * LANES + lane_off + 1u] += 1;
            lane_counts[pc * LANES + lane_off + 1u] += 2;
            lane_counts[pd * LANES + lane_off + 1u] += 1;
            lane_counts[pd * LANES + lane_off] += 1;

            // Late lane counts (q >= 2)
            let is_late = u32(q >= 2u);
            late_lane_counts[pa * LANES + lane_off] += i32(is_late) * 2;
            late_lane_counts[pb * LANES + lane_off] += i32(is_late);
            late_lane_counts[pb * LANES + lane_off + 1u] += i32(is_late);
            late_lane_counts[pc * LANES + lane_off + 1u] += i32(is_late) * 2;
            late_lane_counts[pd * LANES + lane_off + 1u] += i32(is_late);
            late_lane_counts[pd * LANES + lane_off] += i32(is_late);

            // Stay count (positions 0 and 2)
            stay_count[pa] += 1;
            stay_count[pc] += 1;

            // Early/late tracking
            early_late[pa * WEEKS + w] = early;
            early_late[pb * WEEKS + w] = early;
            early_late[pc * WEEKS + w] = early;
            early_late[pd * WEEKS + w] = early;
            early_count[pa] += i32(early);
            early_count[pb] += i32(early);
            early_count[pc] += i32(early);
            early_count[pd] += i32(early);
        }
    }

    // Phase 2: compute penalties (branchless)
    var total = 0u;

    // Matchup balance
    for (var i = 0u; i < TEAMS; i++) {
        for (var j = i + 1u; j < TEAMS; j++) {
            let c = matchups[i * TEAMS + j];
            total += u32(c == 0) * weights.matchup_zero;
            total += u32(max(c, 2) - 2) * weights.matchup_triple;
        }
    }

    // Consecutive opponents (bitset approach)
    for (var w = 0u; w < WEEKS - 1u; w++) {
        var pairs_w: array<u32, 4>;
        var pairs_w1: array<u32, 4>;
        for (var k = 0u; k < 4u; k++) { pairs_w[k] = 0u; pairs_w1[k] = 0u; }

        for (var q = 0u; q < QUADS; q++) {
            let pa0 = get_team(a, w, q, 0u);
            let pb0 = get_team(a, w, q, 1u);
            let pc0 = get_team(a, w, q, 2u);
            let pd0 = get_team(a, w, q, 3u);
            // Set pair bits for week w
            set_pair_bit(&pairs_w, min(pa0, pb0), max(pa0, pb0));
            set_pair_bit(&pairs_w, min(pc0, pd0), max(pc0, pd0));
            set_pair_bit(&pairs_w, min(pa0, pd0), max(pa0, pd0));
            set_pair_bit(&pairs_w, min(pc0, pb0), max(pc0, pb0));

            let pa1 = get_team(a, w + 1u, q, 0u);
            let pb1 = get_team(a, w + 1u, q, 1u);
            let pc1 = get_team(a, w + 1u, q, 2u);
            let pd1 = get_team(a, w + 1u, q, 3u);
            set_pair_bit(&pairs_w1, min(pa1, pb1), max(pa1, pb1));
            set_pair_bit(&pairs_w1, min(pc1, pd1), max(pc1, pd1));
            set_pair_bit(&pairs_w1, min(pa1, pd1), max(pa1, pd1));
            set_pair_bit(&pairs_w1, min(pc1, pb1), max(pc1, pb1));
        }

        for (var k = 0u; k < 4u; k++) {
            total += countOneBits(pairs_w[k] & pairs_w1[k]) * weights.consecutive_opponents;
        }
    }

    // Early/late balance
    let target_e = f32(WEEKS) / 2.0;
    for (var t = 0u; t < TEAMS; t++) {
        let dev = abs(f32(early_count[t]) - target_e);
        total += u32(dev * dev * weights.early_late_balance);
    }

    // Early/late alternation (3 consecutive same)
    for (var t = 0u; t < TEAMS; t++) {
        for (var w = 0u; w < WEEKS - 2u; w++) {
            let e0 = early_late[t * WEEKS + w];
            let e1 = early_late[t * WEEKS + w + 1u];
            let e2 = early_late[t * WEEKS + w + 2u];
            let same = (1u - min(e0 ^ e1, 1u)) * (1u - min(e1 ^ e2, 1u));
            total += same * weights.early_late_alternation;
        }
    }

    // Lane balance
    let target_l = f32(WEEKS) * 2.0 / f32(LANES);
    for (var t = 0u; t < TEAMS; t++) {
        for (var l = 0u; l < LANES; l++) {
            total += u32(abs(f32(lane_counts[t * LANES + l]) - target_l) * weights.lane_balance);
        }
    }

    // Lane switch balance
    let target_stay = f32(WEEKS) / 2.0;
    for (var t = 0u; t < TEAMS; t++) {
        let dev = abs(f32(stay_count[t]) - target_stay);
        total += u32(dev * weights.lane_switch);
    }

    // Late lane balance
    let late_target_l = f32(WEEKS) / f32(LANES);
    for (var t = 0u; t < TEAMS; t++) {
        for (var l = 0u; l < LANES; l++) {
            total += u32(abs(f32(late_lane_counts[t * LANES + l]) - late_target_l) * weights.late_lane_balance);
        }
    }

    // Commissioner overlap
    var min_overlap = WEEKS;
    for (var i = 0u; i < TEAMS; i++) {
        for (var j = i + 1u; j < TEAMS; j++) {
            var overlap = 0u;
            for (var w = 0u; w < WEEKS; w++) {
                overlap += u32(early_late[i * WEEKS + w] == early_late[j * WEEKS + w]);
            }
            min_overlap = min(min_overlap, overlap);
        }
    }
    let sub = u32(max(i32(min_overlap) - 1, 0));
    total += weights.commissioner_overlap * sub;

    return total;
}

fn set_pair_bit(bits: ptr<function, array<u32, 4>>, lo: u32, hi: u32) {
    let pair_idx = lo * (2u * TEAMS - lo - 1u) / 2u + (hi - lo - 1u);
    let word = pair_idx / 32u;
    let bit = pair_idx % 32u;
    (*bits)[word] |= (1u << bit);
}

// ═══════════════════════════════════════════════════════════════════════
// Move types
// ═══════════════════════════════════════════════════════════════════════

fn move_inter_quad_swap(a: ptr<function, array<u32, 48>>, s: ptr<function, array<u32, 4>>) {
    let w = rng_range(s, WEEKS);
    let q1 = rng_range(s, QUADS);
    var q2 = rng_range(s, QUADS - 1u);
    if (q2 >= q1) { q2 += 1u; }
    let p1 = rng_range(s, POS);
    let p2 = rng_range(s, POS);
    swap_positions(a, w, q1, p1, w, q2, p2);
}

fn move_intra_quad_swap(a: ptr<function, array<u32, 48>>, s: ptr<function, array<u32, 4>>) {
    let w = rng_range(s, WEEKS);
    let q = rng_range(s, QUADS);
    let p1 = rng_range(s, POS);
    var p2 = rng_range(s, POS - 1u);
    if (p2 >= p1) { p2 += 1u; }
    swap_positions(a, w, q, p1, w, q, p2);
}

fn move_quad_swap(a: ptr<function, array<u32, 48>>, s: ptr<function, array<u32, 4>>) {
    let w = rng_range(s, WEEKS);
    let q1 = rng_range(s, QUADS);
    var q2 = rng_range(s, QUADS - 1u);
    if (q2 >= q1) { q2 += 1u; }
    let tmp = get_quad(a, w, q1);
    set_quad(a, w, q1, get_quad(a, w, q2));
    set_quad(a, w, q2, tmp);
}

fn move_week_swap(a: ptr<function, array<u32, 48>>, s: ptr<function, array<u32, 4>>) {
    let w1 = rng_range(s, WEEKS);
    var w2 = rng_range(s, WEEKS - 1u);
    if (w2 >= w1) { w2 += 1u; }
    for (var q = 0u; q < QUADS; q++) {
        let tmp = get_quad(a, w1, q);
        set_quad(a, w1, q, get_quad(a, w2, q));
        set_quad(a, w2, q, tmp);
    }
}

fn move_early_late_flip(a: ptr<function, array<u32, 48>>, s: ptr<function, array<u32, 4>>) {
    let w = rng_range(s, WEEKS);
    let q0 = get_quad(a, w, 0u);
    let q1 = get_quad(a, w, 1u);
    set_quad(a, w, 0u, get_quad(a, w, 2u));
    set_quad(a, w, 1u, get_quad(a, w, 3u));
    set_quad(a, w, 2u, q0);
    set_quad(a, w, 3u, q1);
}

fn move_lane_pair_swap(a: ptr<function, array<u32, 48>>, s: ptr<function, array<u32, 4>>) {
    let w = rng_range(s, WEEKS);
    let q0 = get_quad(a, w, 0u);
    let q2 = get_quad(a, w, 2u);
    set_quad(a, w, 0u, get_quad(a, w, 1u));
    set_quad(a, w, 1u, q0);
    set_quad(a, w, 2u, get_quad(a, w, 3u));
    set_quad(a, w, 3u, q2);
}

fn move_stay_switch(a: ptr<function, array<u32, 48>>, s: ptr<function, array<u32, 4>>) {
    let w = rng_range(s, WEEKS);
    let q = rng_range(s, QUADS);
    swap_positions(a, w, q, 0u, w, q, 1u);
    swap_positions(a, w, q, 2u, w, q, 3u);
}

fn find_team_in_week(a: ptr<function, array<u32, 48>>, w: u32, team: u32) -> vec2<u32> {
    for (var q = 0u; q < QUADS; q++) {
        for (var p = 0u; p < POS; p++) {
            if (get_team(a, w, q, p) == team) {
                return vec2<u32>(q, p);
            }
        }
    }
    return vec2<u32>(0xFFFFFFFFu, 0xFFFFFFFFu);
}

fn move_cross_week_swap(a: ptr<function, array<u32, 48>>, s: ptr<function, array<u32, 4>>) -> bool {
    let team = rng_range(s, TEAMS);
    let w1 = rng_range(s, WEEKS);
    var w2 = rng_range(s, WEEKS - 1u);
    if (w2 >= w1) { w2 += 1u; }

    let pos1 = find_team_in_week(a, w1, team);
    let pos2 = find_team_in_week(a, w2, team);
    if (pos1.x == 0xFFFFFFFFu || pos2.x == 0xFFFFFFFFu) { return false; }

    let qi1 = pos1.x; let pi1 = pos1.y;
    let qi2 = pos2.x; let pi2 = pos2.y;

    let other1 = get_team(a, w2, qi1, pi1);
    let other2 = get_team(a, w1, qi2, pi2);
    set_team(a, w1, qi1, pi1, other2);
    set_team(a, w1, qi2, pi2, team);
    set_team(a, w2, qi2, pi2, other1);
    set_team(a, w2, qi1, pi1, team);
    return true;
}

fn move_guided_matchup(a: ptr<function, array<u32, 48>>, s: ptr<function, array<u32, 4>>) -> bool {
    var match_seen: array<u32, 8>; // 120 pairs packed as bits (4 u32s would suffice, use 8 for safety)
    for (var k = 0u; k < 8u; k++) { match_seen[k] = 0u; }

    for (var w = 0u; w < WEEKS; w++) {
        for (var q = 0u; q < QUADS; q++) {
            let pa = get_team(a, w, q, 0u);
            let pb = get_team(a, w, q, 1u);
            let pc = get_team(a, w, q, 2u);
            let pd = get_team(a, w, q, 3u);
            set_pair_bit_8(&match_seen, min(pa, pb), max(pa, pb));
            set_pair_bit_8(&match_seen, min(pc, pd), max(pc, pd));
            set_pair_bit_8(&match_seen, min(pa, pd), max(pa, pd));
            set_pair_bit_8(&match_seen, min(pc, pb), max(pc, pb));
        }
    }

    let start_i = rng_range(s, TEAMS);
    var ta = 0u;
    var tb = 0u;
    var found = false;
    for (var off_i = 0u; off_i < TEAMS; off_i++) {
        if (found) { break; }
        let i = (start_i + off_i) % TEAMS;
        for (var j = i + 1u; j < TEAMS; j++) {
            if (!get_pair_bit_8(&match_seen, i, j)) {
                ta = i; tb = j;
                found = true;
                break;
            }
        }
    }
    if (!found) { return false; }

    let week_start = rng_range(s, WEEKS);
    for (var off = 0u; off < WEEKS; off++) {
        let w = (week_start + off) % WEEKS;
        let pos_a = find_team_in_week(a, w, ta);
        let pos_b = find_team_in_week(a, w, tb);
        if (pos_a.x == 0xFFFFFFFFu || pos_b.x == 0xFFFFFFFFu) { continue; }
        let qa = pos_a.x;
        let qb = pos_b.x;
        let pb = pos_b.y;
        let same_half = (qa < 2u && qb < 2u) || (qa >= 2u && qb >= 2u);
        if (!same_half || qa == qb) { continue; }

        var swap_pos = rng_range(s, POS - 1u);
        let first_non_ta = find_non_team_pos(a, w, qa, ta, s);
        if (first_non_ta == 0xFFFFFFFFu) { continue; }
        swap_positions(a, w, qa, first_non_ta, w, qb, pb);
        return true;
    }
    return false;
}

fn find_non_team_pos(a: ptr<function, array<u32, 48>>, w: u32, q: u32, team: u32, s: ptr<function, array<u32, 4>>) -> u32 {
    var candidates: array<u32, 4>;
    var count = 0u;
    for (var p = 0u; p < POS; p++) {
        if (get_team(a, w, q, p) != team) {
            candidates[count] = p;
            count += 1u;
        }
    }
    if (count == 0u) { return 0xFFFFFFFFu; }
    return candidates[rng_range(s, count)];
}

fn set_pair_bit_8(bits: ptr<function, array<u32, 8>>, lo: u32, hi: u32) {
    let pair_idx = lo * (2u * TEAMS - lo - 1u) / 2u + (hi - lo - 1u);
    let word = pair_idx / 32u;
    let bit = pair_idx % 32u;
    (*bits)[word] |= (1u << bit);
}

fn get_pair_bit_8(bits: ptr<function, array<u32, 8>>, lo: u32, hi: u32) -> bool {
    let pair_idx = lo * (2u * TEAMS - lo - 1u) / 2u + (hi - lo - 1u);
    let word = pair_idx / 32u;
    let bit = pair_idx % 32u;
    return ((*bits)[word] & (1u << bit)) != 0u;
}

fn move_guided_lane(a: ptr<function, array<u32, 48>>, s: ptr<function, array<u32, 4>>) -> bool {
    var lc: array<i32, 64>;
    for (var i = 0u; i < 64u; i++) { lc[i] = 0; }

    for (var w = 0u; w < WEEKS; w++) {
        for (var q = 0u; q < QUADS; q++) {
            let pa = get_team(a, w, q, 0u);
            let pb = get_team(a, w, q, 1u);
            let pc = get_team(a, w, q, 2u);
            let pd = get_team(a, w, q, 3u);
            let lo = (q % 2u) * 2u;
            lc[pa * LANES + lo] += 2;
            lc[pb * LANES + lo] += 1;
            lc[pb * LANES + lo + 1u] += 1;
            lc[pc * LANES + lo + 1u] += 2;
            lc[pd * LANES + lo + 1u] += 1;
            lc[pd * LANES + lo] += 1;
        }
    }

    let target_l = f32(WEEKS) * 2.0 / f32(LANES);
    var worst_team = 0u;
    var worst_dev = 0.0;
    for (var t = 0u; t < TEAMS; t++) {
        for (var l = 0u; l < LANES; l++) {
            let dev = abs(f32(lc[t * LANES + l]) - target_l);
            if (dev > worst_dev) { worst_dev = dev; worst_team = t; }
        }
    }
    if (worst_dev < 1.0) { return false; }

    let start = rng_range(s, WEEKS * QUADS);
    for (var off = 0u; off < WEEKS * QUADS; off++) {
        let idx = (start + off) % (WEEKS * QUADS);
        let w = idx / QUADS;
        let q = idx % QUADS;
        var team_pos = 0xFFFFFFFFu;
        for (var p = 0u; p < POS; p++) {
            if (get_team(a, w, q, p) == worst_team) { team_pos = p; break; }
        }
        if (team_pos == 0xFFFFFFFFu) { continue; }

        var sp = rng_range(s, POS - 1u);
        if (sp >= team_pos) { sp += 1u; }
        swap_positions(a, w, q, team_pos, w, q, sp);
        return true;
    }
    return false;
}

fn move_guided_early_late(a: ptr<function, array<u32, 48>>, s: ptr<function, array<u32, 4>>) -> bool {
    var ec: array<i32, 16>;
    for (var i = 0u; i < 16u; i++) { ec[i] = 0; }

    for (var w = 0u; w < WEEKS; w++) {
        for (var q = 0u; q < 2u; q++) {
            for (var p = 0u; p < POS; p++) {
                ec[get_team(a, w, q, p)] += 1;
            }
        }
    }

    let target_e = f32(WEEKS) / 2.0;
    var worst_team = 0u;
    var worst_dev = 0.0;
    var too_many_early = false;
    for (var t = 0u; t < TEAMS; t++) {
        let dev = f32(ec[t]) - target_e;
        if (abs(dev) > worst_dev) {
            worst_dev = abs(dev);
            worst_team = t;
            too_many_early = dev > 0.0;
        }
    }
    if (worst_dev < 1.0) { return false; }

    let start = rng_range(s, WEEKS);
    for (var off = 0u; off < WEEKS; off++) {
        let w = (start + off) % WEEKS;
        let team = worst_team;
        var in_early = false;
        for (var q = 0u; q < 2u; q++) {
            for (var p = 0u; p < POS; p++) {
                if (get_team(a, w, q, p) == team) { in_early = true; }
            }
        }

        if ((too_many_early && in_early) || (!too_many_early && !in_early)) {
            move_early_late_flip(a, s);
            return true;
        }
    }
    return false;
}

fn write_best(a: ptr<function, array<u32, 48>>, base: u32) {
    for (var i = 0u; i < ASSIGN_SIZE; i++) {
        best_assignments[base + i] = (*a)[i];
    }
}

fn save_all(a: ptr<function, array<u32, 48>>) -> array<u32, 48> {
    var s: array<u32, 48>;
    for (var i = 0u; i < ASSIGN_SIZE; i++) { s[i] = (*a)[i]; }
    return s;
}

fn restore_all(a: ptr<function, array<u32, 48>>, s: ptr<function, array<u32, 48>>) {
    for (var i = 0u; i < ASSIGN_SIZE; i++) { (*a)[i] = (*s)[i]; }
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

    // Load assignment into private memory
    var a: array<u32, 48>;
    for (var i = 0u; i < ASSIGN_SIZE; i++) {
        a[i] = assignments[base + i];
    }

    // Load RNG state
    var rng: array<u32, 4>;
    for (var i = 0u; i < 4u; i++) {
        rng[i] = rng_states[rng_base + i];
    }

    var cost = costs[tid];
    var best_cost = best_costs[tid];

    let temp = params.temp_base + f32(tid % 64u) * params.temp_step;

    // SA loop — each move saves only affected values for targeted undo
    for (var iter = 0u; iter < params.iters_per_dispatch; iter++) {
        if (best_cost == 0u) { break; }

        let move_id = rng_range(&rng, 100u);

        // Save affected state, apply move, evaluate, accept/reject, undo if needed
        if (move_id < 25u) {
            // inter-quad swap: affects 2 quads in 1 week
            let w = rng_range(&rng, WEEKS);
            let q1 = rng_range(&rng, QUADS);
            var q2 = rng_range(&rng, QUADS - 1u);
            if (q2 >= q1) { q2 += 1u; }
            let p1 = rng_range(&rng, POS);
            let p2 = rng_range(&rng, POS);
            swap_positions(&a, w, q1, p1, w, q2, p2);
            let new_cost = evaluate(&a);
            let delta = i32(new_cost) - i32(cost);
            if (delta <= 0 || rng_f32(&rng) < exp(f32(-delta) / temp)) {
                cost = new_cost;
                if (cost < best_cost) { best_cost = cost; write_best(&a, base); }
            } else {
                swap_positions(&a, w, q1, p1, w, q2, p2);
            }
        } else if (move_id < 40u) {
            // intra-quad swap
            let w = rng_range(&rng, WEEKS);
            let q = rng_range(&rng, QUADS);
            let p1 = rng_range(&rng, POS);
            var p2 = rng_range(&rng, POS - 1u);
            if (p2 >= p1) { p2 += 1u; }
            swap_positions(&a, w, q, p1, w, q, p2);
            let new_cost = evaluate(&a);
            let delta = i32(new_cost) - i32(cost);
            if (delta <= 0 || rng_f32(&rng) < exp(f32(-delta) / temp)) {
                cost = new_cost;
                if (cost < best_cost) { best_cost = cost; write_best(&a, base); }
            } else {
                swap_positions(&a, w, q, p1, w, q, p2);
            }
        } else if (move_id < 50u) {
            // cross-week swap: save 4 affected cells
            let team = rng_range(&rng, TEAMS);
            let w1 = rng_range(&rng, WEEKS);
            var w2 = rng_range(&rng, WEEKS - 1u);
            if (w2 >= w1) { w2 += 1u; }
            let pos1 = find_team_in_week(&a, w1, team);
            let pos2 = find_team_in_week(&a, w2, team);
            if (pos1.x != 0xFFFFFFFFu && pos2.x != 0xFFFFFFFFu) {
                let qi1 = pos1.x; let pi1 = pos1.y;
                let qi2 = pos2.x; let pi2 = pos2.y;
                let s0 = get_team(&a, w1, qi1, pi1);
                let s1 = get_team(&a, w1, qi2, pi2);
                let s2 = get_team(&a, w2, qi1, pi1);
                let s3 = get_team(&a, w2, qi2, pi2);
                let other1 = get_team(&a, w2, qi1, pi1);
                let other2 = get_team(&a, w1, qi2, pi2);
                set_team(&a, w1, qi1, pi1, other2);
                set_team(&a, w1, qi2, pi2, team);
                set_team(&a, w2, qi2, pi2, other1);
                set_team(&a, w2, qi1, pi1, team);
                let new_cost = evaluate(&a);
                let delta = i32(new_cost) - i32(cost);
                if (delta <= 0 || rng_f32(&rng) < exp(f32(-delta) / temp)) {
                    cost = new_cost;
                    if (cost < best_cost) { best_cost = cost; write_best(&a, base); }
                } else {
                    set_team(&a, w1, qi1, pi1, s0);
                    set_team(&a, w1, qi2, pi2, s1);
                    set_team(&a, w2, qi1, pi1, s2);
                    set_team(&a, w2, qi2, pi2, s3);
                }
            }
        } else if (move_id < 58u) {
            // quad swap: save 2 quads
            let w = rng_range(&rng, WEEKS);
            let q1 = rng_range(&rng, QUADS);
            var q2 = rng_range(&rng, QUADS - 1u);
            if (q2 >= q1) { q2 += 1u; }
            let tmp = get_quad(&a, w, q1);
            set_quad(&a, w, q1, get_quad(&a, w, q2));
            set_quad(&a, w, q2, tmp);
            let new_cost = evaluate(&a);
            let delta = i32(new_cost) - i32(cost);
            if (delta <= 0 || rng_f32(&rng) < exp(f32(-delta) / temp)) {
                cost = new_cost;
                if (cost < best_cost) { best_cost = cost; write_best(&a, base); }
            } else {
                let tmp2 = get_quad(&a, w, q1);
                set_quad(&a, w, q1, get_quad(&a, w, q2));
                set_quad(&a, w, q2, tmp2);
            }
        } else if (move_id < 64u) {
            // week swap: save 2 weeks (8 quads)
            let w1 = rng_range(&rng, WEEKS);
            var w2 = rng_range(&rng, WEEKS - 1u);
            if (w2 >= w1) { w2 += 1u; }
            for (var q = 0u; q < QUADS; q++) {
                let tmp = get_quad(&a, w1, q);
                set_quad(&a, w1, q, get_quad(&a, w2, q));
                set_quad(&a, w2, q, tmp);
            }
            let new_cost = evaluate(&a);
            let delta = i32(new_cost) - i32(cost);
            if (delta <= 0 || rng_f32(&rng) < exp(f32(-delta) / temp)) {
                cost = new_cost;
                if (cost < best_cost) { best_cost = cost; write_best(&a, base); }
            } else {
                for (var q = 0u; q < QUADS; q++) {
                    let tmp = get_quad(&a, w1, q);
                    set_quad(&a, w1, q, get_quad(&a, w2, q));
                    set_quad(&a, w2, q, tmp);
                }
            }
        } else if (move_id < 70u) {
            // early/late flip: save 4 quads
            let w = rng_range(&rng, WEEKS);
            let sq0 = get_quad(&a, w, 0u); let sq1 = get_quad(&a, w, 1u);
            let sq2 = get_quad(&a, w, 2u); let sq3 = get_quad(&a, w, 3u);
            set_quad(&a, w, 0u, sq2); set_quad(&a, w, 1u, sq3);
            set_quad(&a, w, 2u, sq0); set_quad(&a, w, 3u, sq1);
            let new_cost = evaluate(&a);
            let delta = i32(new_cost) - i32(cost);
            if (delta <= 0 || rng_f32(&rng) < exp(f32(-delta) / temp)) {
                cost = new_cost;
                if (cost < best_cost) { best_cost = cost; write_best(&a, base); }
            } else {
                set_quad(&a, w, 0u, sq0); set_quad(&a, w, 1u, sq1);
                set_quad(&a, w, 2u, sq2); set_quad(&a, w, 3u, sq3);
            }
        } else if (move_id < 75u) {
            // lane pair swap: save 4 quads
            let w = rng_range(&rng, WEEKS);
            let sq0 = get_quad(&a, w, 0u); let sq1 = get_quad(&a, w, 1u);
            let sq2 = get_quad(&a, w, 2u); let sq3 = get_quad(&a, w, 3u);
            set_quad(&a, w, 0u, sq1); set_quad(&a, w, 1u, sq0);
            set_quad(&a, w, 2u, sq3); set_quad(&a, w, 3u, sq2);
            let new_cost = evaluate(&a);
            let delta = i32(new_cost) - i32(cost);
            if (delta <= 0 || rng_f32(&rng) < exp(f32(-delta) / temp)) {
                cost = new_cost;
                if (cost < best_cost) { best_cost = cost; write_best(&a, base); }
            } else {
                set_quad(&a, w, 0u, sq0); set_quad(&a, w, 1u, sq1);
                set_quad(&a, w, 2u, sq2); set_quad(&a, w, 3u, sq3);
            }
        } else if (move_id < 81u) {
            // stay/switch
            let w = rng_range(&rng, WEEKS);
            let q = rng_range(&rng, QUADS);
            swap_positions(&a, w, q, 0u, w, q, 1u);
            swap_positions(&a, w, q, 2u, w, q, 3u);
            let new_cost = evaluate(&a);
            let delta = i32(new_cost) - i32(cost);
            if (delta <= 0 || rng_f32(&rng) < exp(f32(-delta) / temp)) {
                cost = new_cost;
                if (cost < best_cost) { best_cost = cost; write_best(&a, base); }
            } else {
                swap_positions(&a, w, q, 0u, w, q, 1u);
                swap_positions(&a, w, q, 2u, w, q, 3u);
            }
        } else if (move_id < 87u) {
            var sq = save_all(&a);
            let did = move_guided_matchup(&a, &rng);
            if (did) {
                let new_cost = evaluate(&a);
                let delta = i32(new_cost) - i32(cost);
                if (delta <= 0 || rng_f32(&rng) < exp(f32(-delta) / temp)) {
                    cost = new_cost;
                    if (cost < best_cost) { best_cost = cost; write_best(&a, base); }
                } else {
                    restore_all(&a, &sq);
                }
            }
        } else if (move_id < 93u) {
            var sq = save_all(&a);
            let did = move_guided_lane(&a, &rng);
            if (did) {
                let new_cost = evaluate(&a);
                let delta = i32(new_cost) - i32(cost);
                if (delta <= 0 || rng_f32(&rng) < exp(f32(-delta) / temp)) {
                    cost = new_cost;
                    if (cost < best_cost) { best_cost = cost; write_best(&a, base); }
                } else {
                    restore_all(&a, &sq);
                }
            }
        } else {
            var sq = save_all(&a);
            let did = move_guided_early_late(&a, &rng);
            if (did) {
                let new_cost = evaluate(&a);
                let delta = i32(new_cost) - i32(cost);
                if (delta <= 0 || rng_f32(&rng) < exp(f32(-delta) / temp)) {
                    cost = new_cost;
                    if (cost < best_cost) { best_cost = cost; write_best(&a, base); }
                } else {
                    restore_all(&a, &sq);
                }
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
