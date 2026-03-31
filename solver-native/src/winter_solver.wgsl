// Constants matching solver-core
const TEAMS: u32 = 16u;
const LANES: u32 = 4u;
const WEEKS: u32 = 12u;
const QUADS: u32 = 4u;
const POS: u32 = 4u;
const ASSIGN_SIZE: u32 = 48u;

struct Weights {
    matchup_zero: u32,
    matchup_triple: u32,
    consecutive_opponents: u32,
    early_late_balance: f32,
    early_late_alternation: u32,
    early_late_consecutive: u32,
    lane_balance: f32,
    lane_switch: f32,
    late_lane_balance: f32,
    commissioner_overlap: u32,
    half_season_repeat: u32,
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
// Packed matchup helpers (120 pairs as u8 in 30 u32s)
// ═══════════════════════════════════════════════════════════════════════

fn matchup_pair_idx(lo: u32, hi: u32) -> u32 {
    return lo * (2u * TEAMS - lo - 1u) / 2u + (hi - lo - 1u);
}

fn inc_matchup(m: ptr<function, array<u32, 30>>, lo: u32, hi: u32) {
    let idx = matchup_pair_idx(lo, hi);
    (*m)[idx / 4u] += (1u << ((idx % 4u) * 8u));
}

fn get_matchup(m: ptr<function, array<u32, 30>>, lo: u32, hi: u32) -> u32 {
    let idx = matchup_pair_idx(lo, hi);
    return ((*m)[idx / 4u] >> ((idx % 4u) * 8u)) & 0xFFu;
}

// ═══════════════════════════════════════════════════════════════════════
// Bitset helpers for consecutive opponents + guided matchup move
// ═══════════════════════════════════════════════════════════════════════

fn set_pair_bit(bits: ptr<function, array<u32, 4>>, lo: u32, hi: u32) {
    let pair_idx = matchup_pair_idx(lo, hi);
    (*bits)[pair_idx / 32u] |= (1u << (pair_idx % 32u));
}

// ═══════════════════════════════════════════════════════════════════════
// Cost function — Pass 1: matchups + early/late
// Scratch: fh/sh matchups 60 u32 (240B) + el 16 u32 (64B) = 304B
// ═══════════════════════════════════════════════════════════════════════

fn eval_matchups_early_late(a: ptr<function, array<u32, 48>>) -> u32 {
    var fh: array<u32, 30>;
    var sh: array<u32, 30>;
    var el: array<u32, 16>;
    for (var i = 0u; i < 30u; i++) { fh[i] = 0u; sh[i] = 0u; }
    for (var i = 0u; i < 16u; i++) { el[i] = 0u; }

    let half = WEEKS / 2u;
    for (var w = 0u; w < WEEKS; w++) {
        let week_bit = 1u << w;
        for (var q = 0u; q < QUADS; q++) {
            let pa = get_team(a, w, q, 0u);
            let pb = get_team(a, w, q, 1u);
            let pc = get_team(a, w, q, 2u);
            let pd = get_team(a, w, q, 3u);

            if (w < half) {
                inc_matchup(&fh, min(pa, pb), max(pa, pb));
                inc_matchup(&fh, min(pc, pd), max(pc, pd));
                inc_matchup(&fh, min(pa, pd), max(pa, pd));
                inc_matchup(&fh, min(pc, pb), max(pc, pb));
            } else {
                inc_matchup(&sh, min(pa, pb), max(pa, pb));
                inc_matchup(&sh, min(pc, pd), max(pc, pd));
                inc_matchup(&sh, min(pa, pd), max(pa, pd));
                inc_matchup(&sh, min(pc, pb), max(pc, pb));
            }

            if (q < 2u) {
                el[pa] |= week_bit;
                el[pb] |= week_bit;
                el[pc] |= week_bit;
                el[pd] |= week_bit;
            }
        }
    }

    var total = 0u;

    // Matchup balance + half-season repeat
    for (var i = 0u; i < TEAMS; i++) {
        for (var j = i + 1u; j < TEAMS; j++) {
            let fc = get_matchup(&fh, i, j);
            let sc = get_matchup(&sh, i, j);
            let c = fc + sc;
            total += u32(c == 0u) * weights.matchup_zero;
            total += u32(max(c, 2u) - 2u) * weights.matchup_triple;
            total += u32(max(fc, 1u) - 1u) * weights.half_season_repeat;
            total += u32(max(sc, 1u) - 1u) * weights.half_season_repeat;
        }
    }

    // Consecutive opponents (bitset per week-pair, temporary 8 u32s)
    for (var w = 0u; w < WEEKS - 1u; w++) {
        if (w == 4u || w == 5u) { continue; }
        var pw: array<u32, 4>;
        var pw1: array<u32, 4>;
        for (var k = 0u; k < 4u; k++) { pw[k] = 0u; pw1[k] = 0u; }

        for (var q = 0u; q < QUADS; q++) {
            let a0 = get_team(a, w, q, 0u);
            let b0 = get_team(a, w, q, 1u);
            let c0 = get_team(a, w, q, 2u);
            let d0 = get_team(a, w, q, 3u);
            set_pair_bit(&pw, min(a0, b0), max(a0, b0));
            set_pair_bit(&pw, min(c0, d0), max(c0, d0));
            set_pair_bit(&pw, min(a0, d0), max(a0, d0));
            set_pair_bit(&pw, min(c0, b0), max(c0, b0));

            let a1 = get_team(a, w + 1u, q, 0u);
            let b1 = get_team(a, w + 1u, q, 1u);
            let c1 = get_team(a, w + 1u, q, 2u);
            let d1 = get_team(a, w + 1u, q, 3u);
            set_pair_bit(&pw1, min(a1, b1), max(a1, b1));
            set_pair_bit(&pw1, min(c1, d1), max(c1, d1));
            set_pair_bit(&pw1, min(a1, d1), max(a1, d1));
            set_pair_bit(&pw1, min(c1, b1), max(c1, b1));
        }

        for (var k = 0u; k < 4u; k++) {
            total += countOneBits(pw[k] & pw1[k]) * weights.consecutive_opponents;
        }
    }

    // Early/late balance (popcount replaces early_count array)
    let target_e = f32(WEEKS) / 2.0;
    let week_mask = (1u << WEEKS) - 1u;
    for (var t = 0u; t < TEAMS; t++) {
        let ec = countOneBits(el[t] & week_mask);
        let dev = abs(f32(ec) - target_e);
        total += u32(dev * dev * weights.early_late_balance);
    }

    // Early/late alternation (3 consecutive same — bit extraction)
    for (var t = 0u; t < TEAMS; t++) {
        for (var w = 0u; w < WEEKS - 2u; w++) {
            let e0 = (el[t] >> w) & 1u;
            let e1 = (el[t] >> (w + 1u)) & 1u;
            let e2 = (el[t] >> (w + 2u)) & 1u;
            let same = (1u - min(e0 ^ e1, 1u)) * (1u - min(e1 ^ e2, 1u));
            total += same * weights.early_late_alternation;
        }
    }

    // Early/late consecutive (2 consecutive same)
    for (var t = 0u; t < TEAMS; t++) {
        for (var w = 0u; w < WEEKS - 1u; w++) {
            let e0 = (el[t] >> w) & 1u;
            let e1 = (el[t] >> (w + 1u)) & 1u;
            total += (1u - min(e0 ^ e1, 1u)) * weights.early_late_consecutive;
        }
    }

    // Commissioner overlap (XNOR + popcount)
    var min_overlap = WEEKS;
    for (var i = 0u; i < TEAMS; i++) {
        for (var j = i + 1u; j < TEAMS; j++) {
            let same_bits = ~(el[i] ^ el[j]) & week_mask;
            let overlap = countOneBits(same_bits);
            min_overlap = min(min_overlap, overlap);
        }
    }
    total += weights.commissioner_overlap * u32(max(i32(min_overlap) - 1, 0));

    return total;
}

// ═══════════════════════════════════════════════════════════════════════
// Cost function — Pass 2: lanes + stay/switch
// Scratch: lc 16 u32 (64B) + llc 16 u32 (64B) + sc 4 u32 (16B) = 144B
// ═══════════════════════════════════════════════════════════════════════

fn eval_lanes(a: ptr<function, array<u32, 48>>) -> u32 {
    var lc: array<u32, 16>;
    var llc: array<u32, 16>;
    var sc: array<u32, 4>;
    for (var i = 0u; i < 16u; i++) { lc[i] = 0u; llc[i] = 0u; }
    for (var i = 0u; i < 4u; i++) { sc[i] = 0u; }

    for (var w = 0u; w < WEEKS; w++) {
        for (var q = 0u; q < QUADS; q++) {
            let pa = get_team(a, w, q, 0u);
            let pb = get_team(a, w, q, 1u);
            let pc = get_team(a, w, q, 2u);
            let pd = get_team(a, w, q, 3u);
            let lo = (q % 2u) * 2u;
            let is_late = u32(q >= 2u);

            lc[pa] += 2u << (lo * 8u);
            lc[pb] += 1u << (lo * 8u);
            lc[pb] += 1u << ((lo + 1u) * 8u);
            lc[pc] += 2u << ((lo + 1u) * 8u);
            lc[pd] += 1u << ((lo + 1u) * 8u);
            lc[pd] += 1u << (lo * 8u);

            llc[pa] += is_late * (2u << (lo * 8u));
            llc[pb] += is_late * (1u << (lo * 8u));
            llc[pb] += is_late * (1u << ((lo + 1u) * 8u));
            llc[pc] += is_late * (2u << ((lo + 1u) * 8u));
            llc[pd] += is_late * (1u << ((lo + 1u) * 8u));
            llc[pd] += is_late * (1u << (lo * 8u));

            sc[pa / 4u] += 1u << ((pa % 4u) * 8u);
            sc[pc / 4u] += 1u << ((pc % 4u) * 8u);
        }
    }

    var total = 0u;

    // Lane balance
    let target_l = f32(WEEKS) * 2.0 / f32(LANES);
    for (var t = 0u; t < TEAMS; t++) {
        for (var l = 0u; l < LANES; l++) {
            let count = (lc[t] >> (l * 8u)) & 0xFFu;
            total += u32(abs(f32(count) - target_l) * weights.lane_balance);
        }
    }

    // Late lane balance
    let late_target = f32(WEEKS) / f32(LANES);
    for (var t = 0u; t < TEAMS; t++) {
        for (var l = 0u; l < LANES; l++) {
            let count = (llc[t] >> (l * 8u)) & 0xFFu;
            total += u32(abs(f32(count) - late_target) * weights.late_lane_balance);
        }
    }

    // Lane switch balance
    let target_stay = f32(WEEKS) / 2.0;
    for (var t = 0u; t < TEAMS; t++) {
        let count = (sc[t / 4u] >> ((t % 4u) * 8u)) & 0xFFu;
        let dev = abs(f32(count) - target_stay);
        total += u32(dev * weights.lane_switch);
    }

    return total;
}

// ═══════════════════════════════════════════════════════════════════════
// Combined evaluate
// ═══════════════════════════════════════════════════════════════════════

fn evaluate(a: ptr<function, array<u32, 48>>) -> u32 {
    return eval_matchups_early_late(a) + eval_lanes(a);
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
    var match_seen: array<u32, 4>;
    for (var k = 0u; k < 4u; k++) { match_seen[k] = 0u; }

    for (var w = 0u; w < WEEKS; w++) {
        for (var q = 0u; q < QUADS; q++) {
            let pa = get_team(a, w, q, 0u);
            let pb = get_team(a, w, q, 1u);
            let pc = get_team(a, w, q, 2u);
            let pd = get_team(a, w, q, 3u);
            set_pair_bit(&match_seen, min(pa, pb), max(pa, pb));
            set_pair_bit(&match_seen, min(pc, pd), max(pc, pd));
            set_pair_bit(&match_seen, min(pa, pd), max(pa, pd));
            set_pair_bit(&match_seen, min(pc, pb), max(pc, pb));
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
            let idx = matchup_pair_idx(i, j);
            if ((match_seen[idx / 32u] & (1u << (idx % 32u))) == 0u) {
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

fn move_guided_lane(a: ptr<function, array<u32, 48>>, s: ptr<function, array<u32, 4>>) -> bool {
    var lc: array<u32, 16>;
    for (var i = 0u; i < 16u; i++) { lc[i] = 0u; }

    for (var w = 0u; w < WEEKS; w++) {
        for (var q = 0u; q < QUADS; q++) {
            let pa = get_team(a, w, q, 0u);
            let pb = get_team(a, w, q, 1u);
            let pc = get_team(a, w, q, 2u);
            let pd = get_team(a, w, q, 3u);
            let lo = (q % 2u) * 2u;
            lc[pa] += 2u << (lo * 8u);
            lc[pb] += 1u << (lo * 8u);
            lc[pb] += 1u << ((lo + 1u) * 8u);
            lc[pc] += 2u << ((lo + 1u) * 8u);
            lc[pd] += 1u << ((lo + 1u) * 8u);
            lc[pd] += 1u << (lo * 8u);
        }
    }

    let target_l = f32(WEEKS) * 2.0 / f32(LANES);
    var worst_team = 0u;
    var worst_dev = 0.0;
    for (var t = 0u; t < TEAMS; t++) {
        for (var l = 0u; l < LANES; l++) {
            let dev = abs(f32((lc[t] >> (l * 8u)) & 0xFFu) - target_l);
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

// ═══════════════════════════════════════════════════════════════════════
// SA helpers
// ═══════════════════════════════════════════════════════════════════════

fn sa_accept(delta: i32, temp: f32, s: ptr<function, array<u32, 4>>) -> bool {
    if (delta < 0) { return true; }
    if (delta == 0) { return rng_f32(s) < 0.2; }
    return rng_f32(s) < exp(f32(-delta) / temp);
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

    var a: array<u32, 48>;
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

        // Compound move: bundle multiple inter-quad swaps when cost is low
        let compound_prob = clamp((1000.0 - f32(cost)) / 800.0, 0.0, 0.5);
        if (rng_f32(&rng) < compound_prob) {
            var sq = save_all(&a);
            let max_n = u32(clamp((800.0 - f32(cost)) / 50.0, 4.0, 12.0));
            let n_swaps = 2u + rng_range(&rng, max_n - 1u);
            for (var k = 0u; k < n_swaps; k++) {
                let cw = rng_range(&rng, WEEKS);
                let cq1 = rng_range(&rng, QUADS);
                var cq2 = rng_range(&rng, QUADS - 1u);
                if (cq2 >= cq1) { cq2 += 1u; }
                let cp1 = rng_range(&rng, POS);
                let cp2 = rng_range(&rng, POS);
                swap_positions(&a, cw, cq1, cp1, cw, cq2, cp2);
            }
            let new_cost = evaluate(&a);
            let delta = i32(new_cost) - i32(cost);
            if (sa_accept(delta, temp, &rng)) {
                cost = new_cost;
                if (cost < best_cost) { best_cost = cost; write_best(&a, base); }
            } else {
                restore_all(&a, &sq);
            }
        } else {

        let move_id = rng_range(&rng, 100u);

        if (move_id < move_thresh[0]) {
            let w = rng_range(&rng, WEEKS);
            let q1 = rng_range(&rng, QUADS);
            var q2 = rng_range(&rng, QUADS - 1u);
            if (q2 >= q1) { q2 += 1u; }
            let p1 = rng_range(&rng, POS);
            let p2 = rng_range(&rng, POS);
            swap_positions(&a, w, q1, p1, w, q2, p2);
            let new_cost = evaluate(&a);
            let delta = i32(new_cost) - i32(cost);
            if (sa_accept(delta, temp, &rng)) {
                cost = new_cost;
                if (cost < best_cost) { best_cost = cost; write_best(&a, base); }
            } else {
                swap_positions(&a, w, q1, p1, w, q2, p2);
            }
        } else if (move_id < move_thresh[1]) {
            let w = rng_range(&rng, WEEKS);
            let q = rng_range(&rng, QUADS);
            let p1 = rng_range(&rng, POS);
            var p2 = rng_range(&rng, POS - 1u);
            if (p2 >= p1) { p2 += 1u; }
            swap_positions(&a, w, q, p1, w, q, p2);
            let new_cost = evaluate(&a);
            let delta = i32(new_cost) - i32(cost);
            if (sa_accept(delta, temp, &rng)) {
                cost = new_cost;
                if (cost < best_cost) { best_cost = cost; write_best(&a, base); }
            } else {
                swap_positions(&a, w, q, p1, w, q, p2);
            }
        } else if (move_id < move_thresh[2]) {
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
                if (sa_accept(delta, temp, &rng)) {
                    cost = new_cost;
                    if (cost < best_cost) { best_cost = cost; write_best(&a, base); }
                } else {
                    set_team(&a, w1, qi1, pi1, s0);
                    set_team(&a, w1, qi2, pi2, s1);
                    set_team(&a, w2, qi1, pi1, s2);
                    set_team(&a, w2, qi2, pi2, s3);
                }
            }
        } else if (move_id < move_thresh[3]) {
            let w = rng_range(&rng, WEEKS);
            let q1 = rng_range(&rng, QUADS);
            var q2 = rng_range(&rng, QUADS - 1u);
            if (q2 >= q1) { q2 += 1u; }
            let tmp = get_quad(&a, w, q1);
            set_quad(&a, w, q1, get_quad(&a, w, q2));
            set_quad(&a, w, q2, tmp);
            let new_cost = evaluate(&a);
            let delta = i32(new_cost) - i32(cost);
            if (sa_accept(delta, temp, &rng)) {
                cost = new_cost;
                if (cost < best_cost) { best_cost = cost; write_best(&a, base); }
            } else {
                let tmp2 = get_quad(&a, w, q1);
                set_quad(&a, w, q1, get_quad(&a, w, q2));
                set_quad(&a, w, q2, tmp2);
            }
        } else if (move_id < move_thresh[4]) {
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
            if (sa_accept(delta, temp, &rng)) {
                cost = new_cost;
                if (cost < best_cost) { best_cost = cost; write_best(&a, base); }
            } else {
                for (var q = 0u; q < QUADS; q++) {
                    let tmp = get_quad(&a, w1, q);
                    set_quad(&a, w1, q, get_quad(&a, w2, q));
                    set_quad(&a, w2, q, tmp);
                }
            }
        } else if (move_id < move_thresh[5]) {
            let w = rng_range(&rng, WEEKS);
            let sq0 = get_quad(&a, w, 0u); let sq1 = get_quad(&a, w, 1u);
            let sq2 = get_quad(&a, w, 2u); let sq3 = get_quad(&a, w, 3u);
            set_quad(&a, w, 0u, sq2); set_quad(&a, w, 1u, sq3);
            set_quad(&a, w, 2u, sq0); set_quad(&a, w, 3u, sq1);
            let new_cost = evaluate(&a);
            let delta = i32(new_cost) - i32(cost);
            if (sa_accept(delta, temp, &rng)) {
                cost = new_cost;
                if (cost < best_cost) { best_cost = cost; write_best(&a, base); }
            } else {
                set_quad(&a, w, 0u, sq0); set_quad(&a, w, 1u, sq1);
                set_quad(&a, w, 2u, sq2); set_quad(&a, w, 3u, sq3);
            }
        } else if (move_id < move_thresh[6]) {
            let w = rng_range(&rng, WEEKS);
            let sq0 = get_quad(&a, w, 0u); let sq1 = get_quad(&a, w, 1u);
            let sq2 = get_quad(&a, w, 2u); let sq3 = get_quad(&a, w, 3u);
            set_quad(&a, w, 0u, sq1); set_quad(&a, w, 1u, sq0);
            set_quad(&a, w, 2u, sq3); set_quad(&a, w, 3u, sq2);
            let new_cost = evaluate(&a);
            let delta = i32(new_cost) - i32(cost);
            if (sa_accept(delta, temp, &rng)) {
                cost = new_cost;
                if (cost < best_cost) { best_cost = cost; write_best(&a, base); }
            } else {
                set_quad(&a, w, 0u, sq0); set_quad(&a, w, 1u, sq1);
                set_quad(&a, w, 2u, sq2); set_quad(&a, w, 3u, sq3);
            }
        } else if (move_id < move_thresh[7]) {
            let w = rng_range(&rng, WEEKS);
            let q = rng_range(&rng, QUADS);
            swap_positions(&a, w, q, 0u, w, q, 1u);
            swap_positions(&a, w, q, 2u, w, q, 3u);
            let new_cost = evaluate(&a);
            let delta = i32(new_cost) - i32(cost);
            if (sa_accept(delta, temp, &rng)) {
                cost = new_cost;
                if (cost < best_cost) { best_cost = cost; write_best(&a, base); }
            } else {
                swap_positions(&a, w, q, 0u, w, q, 1u);
                swap_positions(&a, w, q, 2u, w, q, 3u);
            }
        } else if (move_id < move_thresh[8]) {
            var sq = save_all(&a);
            let did = move_guided_matchup(&a, &rng);
            if (did) {
                let new_cost = evaluate(&a);
                let delta = i32(new_cost) - i32(cost);
                if (sa_accept(delta, temp, &rng)) {
                    cost = new_cost;
                    if (cost < best_cost) { best_cost = cost; write_best(&a, base); }
                } else {
                    restore_all(&a, &sq);
                }
            }
        } else if (move_id < move_thresh[9]) {
            var sq = save_all(&a);
            let did = move_guided_lane(&a, &rng);
            if (did) {
                let new_cost = evaluate(&a);
                let delta = i32(new_cost) - i32(cost);
                if (sa_accept(delta, temp, &rng)) {
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
                if (sa_accept(delta, temp, &rng)) {
                    cost = new_cost;
                    if (cost < best_cost) { best_cost = cost; write_best(&a, base); }
                } else {
                    restore_all(&a, &sq);
                }
            }
        }

        } // end compound else

        // Exhaustive quad-pair search on last iteration of dispatch
        if (iter == params.iters_per_dispatch - 1u && cost > 0u) {
            let ew = rng_range(&rng, WEEKS);
            let eq1 = rng_range(&rng, QUADS);
            var eq2 = rng_range(&rng, QUADS - 1u);
            if (eq2 >= eq1) { eq2 += 1u; }
            var best_delta = 0;
            var best_ep1 = 0xFFFFFFFFu;
            var best_ep2 = 0xFFFFFFFFu;
            for (var ep1 = 0u; ep1 < POS; ep1++) {
                for (var ep2 = 0u; ep2 < POS; ep2++) {
                    swap_positions(&a, ew, eq1, ep1, ew, eq2, ep2);
                    let nc = evaluate(&a);
                    let d = i32(nc) - i32(cost);
                    if (d < best_delta) {
                        best_delta = d;
                        best_ep1 = ep1;
                        best_ep2 = ep2;
                    }
                    swap_positions(&a, ew, eq1, ep1, ew, eq2, ep2);
                }
            }
            if (best_ep1 != 0xFFFFFFFFu) {
                swap_positions(&a, ew, eq1, best_ep1, ew, eq2, best_ep2);
                cost = u32(i32(cost) + best_delta);
                if (cost < best_cost) { best_cost = cost; write_best(&a, base); }
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
