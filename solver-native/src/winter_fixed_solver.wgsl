// Winter Fixed GPU Solver — template-based with 4-bit packing (25 u32s per chain)
// Template constants (T_POS_A, T_POS_B, POS_QUAD, POS_IN_QUAD, etc.) are injected above.

const TEAMS: u32 = 16u;
const LANES: u32 = 4u;
const WEEKS: u32 = 12u;
const POSITIONS: u32 = 16u;
const QUADS: u32 = 4u;
const ASSIGN_SIZE: u32 = 25u;
const FLAG_WORD: u32 = 24u;

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
@group(0) @binding(7) var<storage, read> move_thresh: array<u32, 8>;

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
// 4-bit team access (8 teams per u32, 2 u32s per week)
// ═══════════════════════════════════════════════════════════════════════

fn get_team(a: ptr<function, array<u32, 25>>, w: u32, pos: u32) -> u32 {
    let idx = w * 2u + pos / 8u;
    let shift = (pos % 8u) * 4u;
    return ((*a)[idx] >> shift) & 0xFu;
}

fn set_team(a: ptr<function, array<u32, 25>>, w: u32, pos: u32, team: u32) {
    let idx = w * 2u + pos / 8u;
    let shift = (pos % 8u) * 4u;
    let mask = ~(0xFu << shift);
    (*a)[idx] = ((*a)[idx] & mask) | ((team & 0xFu) << shift);
}

fn swap_positions(a: ptr<function, array<u32, 25>>, w: u32, pa: u32, pb: u32) {
    let ta = get_team(a, w, pa);
    let tb = get_team(a, w, pb);
    set_team(a, w, pa, tb);
    set_team(a, w, pb, ta);
}

fn get_lane_swap_early(a: ptr<function, array<u32, 25>>, w: u32) -> bool {
    return ((*a)[FLAG_WORD] & (1u << w)) != 0u;
}

fn get_lane_swap_late(a: ptr<function, array<u32, 25>>, w: u32) -> bool {
    return ((*a)[FLAG_WORD] & (1u << (w + 12u))) != 0u;
}

fn toggle_lane_swap_early(a: ptr<function, array<u32, 25>>, w: u32) {
    (*a)[FLAG_WORD] ^= (1u << w);
}

fn toggle_lane_swap_late(a: ptr<function, array<u32, 25>>, w: u32) {
    (*a)[FLAG_WORD] ^= (1u << (w + 12u));
}

// Effective quad after lane swap: early swaps 0↔1, late swaps 2↔3
fn effective_quad(pos_quad: u32, lse: bool, lsl: bool) -> u32 {
    if (pos_quad == 0u && lse) { return 1u; }
    if (pos_quad == 1u && lse) { return 0u; }
    if (pos_quad == 2u && lsl) { return 3u; }
    if (pos_quad == 3u && lsl) { return 2u; }
    return pos_quad;
}

// ═══════════════════════════════════════════════════════════════════════
// Packed matchup helpers (120 pairs)
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

fn set_pair_bit(bits: ptr<function, array<u32, 4>>, lo: u32, hi: u32) {
    let pair_idx = matchup_pair_idx(lo, hi);
    (*bits)[pair_idx / 32u] |= (1u << (pair_idx % 32u));
}

// ═══════════════════════════════════════════════════════════════════════
// Cost function — Pass 1: matchups + early/late
// ═══════════════════════════════════════════════════════════════════════

fn eval_matchups_early_late(a: ptr<function, array<u32, 25>>) -> u32 {
    var fh: array<u32, 30>;
    var sh: array<u32, 30>;
    var el: array<u32, 16>;
    for (var i = 0u; i < 30u; i++) { fh[i] = 0u; sh[i] = 0u; }
    for (var i = 0u; i < 16u; i++) { el[i] = 0u; }

    let half = WEEKS / 2u;
    for (var w = 0u; w < WEEKS; w++) {
        let lse = get_lane_swap_early(a, w);
        let lsl = get_lane_swap_late(a, w);
        let week_bit = 1u << w;

        // Iterate by quad: matchups + early/late in one pass
        for (var q = 0u; q < QUADS; q++) {
            let eq = effective_quad(q, lse, lsl);
            let base = q * 4u;
            let pa = get_team(a, w, base);
            let pb = get_team(a, w, base + 1u);
            let pc = get_team(a, w, base + 2u);
            let pd = get_team(a, w, base + 3u);

            // 4 matchups per quad: (pa,pb), (pc,pd), (pa,pd), (pc,pb)
            if (w < half) {
                inc_matchup(&fh, min(pa,pb), max(pa,pb));
                inc_matchup(&fh, min(pc,pd), max(pc,pd));
                inc_matchup(&fh, min(pa,pd), max(pa,pd));
                inc_matchup(&fh, min(pc,pb), max(pc,pb));
            } else {
                inc_matchup(&sh, min(pa,pb), max(pa,pb));
                inc_matchup(&sh, min(pc,pd), max(pc,pd));
                inc_matchup(&sh, min(pa,pd), max(pa,pd));
                inc_matchup(&sh, min(pc,pb), max(pc,pb));
            }

            // Early/late
            if (eq < 2u) {
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

    // Consecutive opponents
    for (var w = 0u; w < WEEKS - 1u; w++) {
        if (w == 4u || w == 5u) { continue; }
        var pw: array<u32, 4>;
        var pw1: array<u32, 4>;
        for (var k = 0u; k < 4u; k++) { pw[k] = 0u; pw1[k] = 0u; }

        for (var q = 0u; q < QUADS; q++) {
            let base = q * 4u;
            let a0 = get_team(a, w, base);
            let b0 = get_team(a, w, base + 1u);
            let c0 = get_team(a, w, base + 2u);
            let d0 = get_team(a, w, base + 3u);
            set_pair_bit(&pw, min(a0,b0), max(a0,b0));
            set_pair_bit(&pw, min(c0,d0), max(c0,d0));
            set_pair_bit(&pw, min(a0,d0), max(a0,d0));
            set_pair_bit(&pw, min(c0,b0), max(c0,b0));

            let a1 = get_team(a, w + 1u, base);
            let b1 = get_team(a, w + 1u, base + 1u);
            let c1 = get_team(a, w + 1u, base + 2u);
            let d1 = get_team(a, w + 1u, base + 3u);
            set_pair_bit(&pw1, min(a1,b1), max(a1,b1));
            set_pair_bit(&pw1, min(c1,d1), max(c1,d1));
            set_pair_bit(&pw1, min(a1,d1), max(a1,d1));
            set_pair_bit(&pw1, min(c1,b1), max(c1,b1));
        }

        for (var k = 0u; k < 4u; k++) {
            total += countOneBits(pw[k] & pw1[k]) * weights.consecutive_opponents;
        }
    }

    // Early/late balance
    let target_e = f32(WEEKS) / 2.0;
    let week_mask = (1u << WEEKS) - 1u;
    for (var t = 0u; t < TEAMS; t++) {
        let ec = countOneBits(el[t] & week_mask);
        let dev = abs(f32(ec) - target_e);
        total += u32(dev * dev * weights.early_late_balance);
    }

    // Early/late alternation (3 consecutive same)
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

    // Commissioner overlap
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
// ═══════════════════════════════════════════════════════════════════════

fn eval_lanes(a: ptr<function, array<u32, 25>>) -> u32 {
    var lc: array<u32, 16>;   // lane counts packed: 4 lanes × 8 bits per team
    var llc: array<u32, 16>;  // late lane counts
    var sc: array<u32, 4>;    // stay counts packed: 4 teams × 8 bits per u32
    for (var i = 0u; i < 16u; i++) { lc[i] = 0u; llc[i] = 0u; }
    for (var i = 0u; i < 4u; i++) { sc[i] = 0u; }

    for (var w = 0u; w < WEEKS; w++) {
        let lse = get_lane_swap_early(a, w);
        let lsl = get_lane_swap_late(a, w);

        for (var q = 0u; q < QUADS; q++) {
            let eq = effective_quad(q, lse, lsl);
            let lo = (eq % 2u) * 2u;
            let is_late = u32(eq >= 2u);
            let base = q * 4u;
            let pa = get_team(a, w, base);
            let pb = get_team(a, w, base + 1u);
            let pc = get_team(a, w, base + 2u);
            let pd = get_team(a, w, base + 3u);

            // Lane counts: pa=stay(lo×2), pb=split, pc=stay(lo+1×2), pd=split
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

            // Stay count: positions 0, 2 in quad
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

fn evaluate(a: ptr<function, array<u32, 25>>) -> u32 {
    return eval_matchups_early_late(a) + eval_lanes(a);
}

// ═══════════════════════════════════════════════════════════════════════
// SA helpers
// ═══════════════════════════════════════════════════════════════════════

fn sa_accept(delta: i32, temp: f32, s: ptr<function, array<u32, 4>>) -> bool {
    if (delta < 0) { return true; }
    if (delta == 0) { return rng_f32(s) < 0.2; }
    return rng_f32(s) < exp(f32(-delta) / temp);
}

fn write_best(a: ptr<function, array<u32, 25>>, base: u32) {
    for (var i = 0u; i < ASSIGN_SIZE; i++) {
        best_assignments[base + i] = (*a)[i];
    }
}

fn save_all(a: ptr<function, array<u32, 25>>) -> array<u32, 25> {
    var s: array<u32, 25>;
    for (var i = 0u; i < ASSIGN_SIZE; i++) { s[i] = (*a)[i]; }
    return s;
}

fn restore_all(a: ptr<function, array<u32, 25>>, s: ptr<function, array<u32, 25>>) {
    for (var i = 0u; i < ASSIGN_SIZE; i++) { (*a)[i] = (*s)[i]; }
}

// ═══════════════════════════════════════════════════════════════════════
// Find team position in a week
// ═══════════════════════════════════════════════════════════════════════

fn find_team_pos(a: ptr<function, array<u32, 25>>, w: u32, team: u32) -> u32 {
    for (var p = 0u; p < POSITIONS; p++) {
        if (get_team(a, w, p) == team) {
            return p;
        }
    }
    return 0xFFFFFFFFu;
}

// ═══════════════════════════════════════════════════════════════════════
// Move: guided matchup
// ═══════════════════════════════════════════════════════════════════════

fn move_guided_matchup(a: ptr<function, array<u32, 25>>, s: ptr<function, array<u32, 4>>) -> bool {
    var match_seen: array<u32, 4>;
    for (var k = 0u; k < 4u; k++) { match_seen[k] = 0u; }

    for (var w = 0u; w < WEEKS; w++) {
        for (var mi = 0u; mi < MATCHUPS_PER_WEEK; mi++) {
            let ta = get_team(a, w, T_POS_A[mi]);
            let tb = get_team(a, w, T_POS_B[mi]);
            set_pair_bit(&match_seen, min(ta, tb), max(ta, tb));
        }
    }

    let start_i = rng_range(s, TEAMS);
    var target_a = 0u;
    var target_b = 0u;
    var found = false;
    for (var off_i = 0u; off_i < TEAMS; off_i++) {
        if (found) { break; }
        let i = (start_i + off_i) % TEAMS;
        for (var j = i + 1u; j < TEAMS; j++) {
            let idx = matchup_pair_idx(i, j);
            if ((match_seen[idx / 32u] & (1u << (idx % 32u))) == 0u) {
                target_a = i; target_b = j;
                found = true;
                break;
            }
        }
    }
    if (!found) { return false; }

    let week_start = rng_range(s, WEEKS);
    for (var off = 0u; off < WEEKS; off++) {
        let w = (week_start + off) % WEEKS;
        let pa = find_team_pos(a, w, target_a);
        let pb = find_team_pos(a, w, target_b);
        if (pa == 0xFFFFFFFFu || pb == 0xFFFFFFFFu) { continue; }
        let qa = POS_QUAD[pa];
        let qb = POS_QUAD[pb];
        let same_half = (qa < 2u && qb < 2u) || (qa >= 2u && qb >= 2u);
        if (!same_half || qa == qb) { continue; }

        // Swap target_b with a random non-target_a position in target_a's quad
        let q_base = qa * 4u;
        var candidates: array<u32, 4>;
        var count = 0u;
        for (var p = q_base; p < q_base + 4u; p++) {
            if (p != pa) {
                candidates[count] = p;
                count += 1u;
            }
        }
        if (count == 0u) { continue; }
        let swap_pos = candidates[rng_range(s, count)];
        swap_positions(a, w, pb, swap_pos);
        return true;
    }
    return false;
}

// ═══════════════════════════════════════════════════════════════════════
// Move: guided lane
// ═══════════════════════════════════════════════════════════════════════

fn move_guided_lane(a: ptr<function, array<u32, 25>>, s: ptr<function, array<u32, 4>>) -> bool {
    var lc: array<u32, 16>;
    for (var i = 0u; i < 16u; i++) { lc[i] = 0u; }

    for (var w = 0u; w < WEEKS; w++) {
        let lse = get_lane_swap_early(a, w);
        let lsl = get_lane_swap_late(a, w);
        for (var pos = 0u; pos < POSITIONS; pos++) {
            let team = get_team(a, w, pos);
            let eq = effective_quad(POS_QUAD[pos], lse, lsl);
            let lo = (eq % 2u) * 2u;
            let piq = POS_IN_QUAD[pos];
            switch piq {
                case 0u: { lc[team] += 2u << (lo * 8u); }
                case 1u: { lc[team] += 1u << (lo * 8u); lc[team] += 1u << ((lo + 1u) * 8u); }
                case 2u: { lc[team] += 2u << ((lo + 1u) * 8u); }
                default: { lc[team] += 1u << ((lo + 1u) * 8u); lc[team] += 1u << (lo * 8u); }
            }
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

    // Swap worst_team within its quad in some week
    let start = rng_range(s, WEEKS);
    for (var off = 0u; off < WEEKS; off++) {
        let w = (start + off) % WEEKS;
        let pos = find_team_pos(a, w, worst_team);
        if (pos == 0xFFFFFFFFu) { continue; }
        let q_base = POS_QUAD[pos] * 4u;
        var sp = q_base + rng_range(s, 3u);
        if (sp >= pos) { sp += 1u; }
        if (sp >= q_base + 4u) { sp = q_base; }
        swap_positions(a, w, pos, sp);
        return true;
    }
    return false;
}

// ═══════════════════════════════════════════════════════════════════════
// Move: guided early/late
// ═══════════════════════════════════════════════════════════════════════

fn move_guided_early_late(a: ptr<function, array<u32, 25>>, s: ptr<function, array<u32, 4>>) -> bool {
    var ec: array<i32, 16>;
    for (var i = 0u; i < 16u; i++) { ec[i] = 0; }

    for (var w = 0u; w < WEEKS; w++) {
        let lse = get_lane_swap_early(a, w);
        let lsl = get_lane_swap_late(a, w);
        for (var pos = 0u; pos < POSITIONS; pos++) {
            let team = get_team(a, w, pos);
            let eq = effective_quad(POS_QUAD[pos], lse, lsl);
            if (eq < 2u) {
                ec[team] += 1;
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

    // Swap between early (pos 0-7) and late (pos 8-15)
    let start = rng_range(s, WEEKS);
    for (var off = 0u; off < WEEKS; off++) {
        let w = (start + off) % WEEKS;
        let pos = find_team_pos(a, w, worst_team);
        if (pos == 0xFFFFFFFFu) { continue; }
        let lse = get_lane_swap_early(a, w);
        let lsl = get_lane_swap_late(a, w);
        let eq = effective_quad(POS_QUAD[pos], lse, lsl);
        let in_early = eq < 2u;
        if ((too_many_early && in_early) || (!too_many_early && !in_early)) {
            let swap_pos = select(rng_range(s, 8u), 8u + rng_range(s, 8u), in_early);
            swap_positions(a, w, pos, swap_pos);
            return true;
        }
    }
    return false;
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

    var a: array<u32, 25>;
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
            // position_swap: swap two positions in one week
            let w = rng_range(&rng, WEEKS);
            let pa = rng_range(&rng, POSITIONS);
            var pb = rng_range(&rng, POSITIONS - 1u);
            if (pb >= pa) { pb += 1u; }
            swap_positions(&a, w, pa, pb);
            let new_cost = evaluate(&a);
            let delta = i32(new_cost) - i32(cost);
            if (sa_accept(delta, temp, &rng)) {
                cost = new_cost;
                if (cost < best_cost) { best_cost = cost; write_best(&a, base); }
            } else {
                swap_positions(&a, w, pa, pb);
            }
        } else if (move_id < move_thresh[1]) {
            // cross_week_swap
            let team = rng_range(&rng, TEAMS);
            let w1 = rng_range(&rng, WEEKS);
            var w2 = rng_range(&rng, WEEKS - 1u);
            if (w2 >= w1) { w2 += 1u; }
            let pos1 = find_team_pos(&a, w1, team);
            let pos2 = find_team_pos(&a, w2, team);
            if (pos1 != 0xFFFFFFFFu && pos2 != 0xFFFFFFFFu && pos1 != pos2) {
                // Save the 4 cells for undo
                let s0 = get_team(&a, w1, pos1);
                let s1 = get_team(&a, w1, pos2);
                let s2 = get_team(&a, w2, pos1);
                let s3 = get_team(&a, w2, pos2);
                // Swap pos1↔pos2 within each week (preserves permutations)
                set_team(&a, w1, pos1, s1);
                set_team(&a, w1, pos2, s0);
                set_team(&a, w2, pos1, s3);
                set_team(&a, w2, pos2, s2);
                let new_cost = evaluate(&a);
                let delta = i32(new_cost) - i32(cost);
                if (sa_accept(delta, temp, &rng)) {
                    cost = new_cost;
                    if (cost < best_cost) { best_cost = cost; write_best(&a, base); }
                } else {
                    set_team(&a, w1, pos1, s0);
                    set_team(&a, w1, pos2, s1);
                    set_team(&a, w2, pos1, s2);
                    set_team(&a, w2, pos2, s3);
                }
            }
        } else if (move_id < move_thresh[2]) {
            // week_swap
            let w1 = rng_range(&rng, WEEKS);
            var w2 = rng_range(&rng, WEEKS - 1u);
            if (w2 >= w1) { w2 += 1u; }
            // Swap all positions + flags between weeks
            for (var off = 0u; off < 2u; off++) {
                let tmp = a[w1 * 2u + off];
                a[w1 * 2u + off] = a[w2 * 2u + off];
                a[w2 * 2u + off] = tmp;
            }
            // Swap flag bits
            let e1 = (a[FLAG_WORD] >> w1) & 1u;
            let e2 = (a[FLAG_WORD] >> w2) & 1u;
            if (e1 != e2) {
                a[FLAG_WORD] ^= (1u << w1) | (1u << w2);
            }
            let l1 = (a[FLAG_WORD] >> (w1 + 12u)) & 1u;
            let l2 = (a[FLAG_WORD] >> (w2 + 12u)) & 1u;
            if (l1 != l2) {
                a[FLAG_WORD] ^= (1u << (w1 + 12u)) | (1u << (w2 + 12u));
            }
            let new_cost = evaluate(&a);
            let delta = i32(new_cost) - i32(cost);
            if (sa_accept(delta, temp, &rng)) {
                cost = new_cost;
                if (cost < best_cost) { best_cost = cost; write_best(&a, base); }
            } else {
                // Undo
                for (var off = 0u; off < 2u; off++) {
                    let tmp = a[w1 * 2u + off];
                    a[w1 * 2u + off] = a[w2 * 2u + off];
                    a[w2 * 2u + off] = tmp;
                }
                if (e1 != e2) {
                    a[FLAG_WORD] ^= (1u << w1) | (1u << w2);
                }
                if (l1 != l2) {
                    a[FLAG_WORD] ^= (1u << (w1 + 12u)) | (1u << (w2 + 12u));
                }
            }
        } else if (move_id < move_thresh[3]) {
            // toggle_lane_early
            let w = rng_range(&rng, WEEKS);
            toggle_lane_swap_early(&a, w);
            let new_cost = evaluate(&a);
            let delta = i32(new_cost) - i32(cost);
            if (sa_accept(delta, temp, &rng)) {
                cost = new_cost;
                if (cost < best_cost) { best_cost = cost; write_best(&a, base); }
            } else {
                toggle_lane_swap_early(&a, w);
            }
        } else if (move_id < move_thresh[4]) {
            // toggle_lane_late
            let w = rng_range(&rng, WEEKS);
            toggle_lane_swap_late(&a, w);
            let new_cost = evaluate(&a);
            let delta = i32(new_cost) - i32(cost);
            if (sa_accept(delta, temp, &rng)) {
                cost = new_cost;
                if (cost < best_cost) { best_cost = cost; write_best(&a, base); }
            } else {
                toggle_lane_swap_late(&a, w);
            }
        } else if (move_id < move_thresh[5]) {
            // guided_matchup
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
        } else if (move_id < move_thresh[6]) {
            // guided_lane
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
            // guided_early_late
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

        // Exhaustive position-pair search on last iteration
        if (iter == params.iters_per_dispatch - 1u && cost > 0u) {
            let ew = rng_range(&rng, WEEKS);
            let epa = rng_range(&rng, POSITIONS);
            var best_delta = 0;
            var best_epb = 0xFFFFFFFFu;
            for (var epb = 0u; epb < POSITIONS; epb++) {
                if (epb == epa) { continue; }
                swap_positions(&a, ew, epa, epb);
                let nc = evaluate(&a);
                let d = i32(nc) - i32(cost);
                if (d < best_delta) {
                    best_delta = d;
                    best_epb = epb;
                }
                swap_positions(&a, ew, epa, epb);
            }
            if (best_epb != 0xFFFFFFFFu) {
                swap_positions(&a, ew, epa, best_epb);
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
