use rand::rngs::SmallRng;
use rand::{Rng, SeedableRng};
use std::fs;
use std::io::{self, BufRead};
use std::sync::atomic::{AtomicBool, AtomicU32, AtomicU64, AtomicU8, AtomicUsize, Ordering};
use std::sync::Arc;
use std::thread;
use std::time::Instant;

use solver_native::*;

// ═══════════════════════════════════════════════════════════════════════════
// Phase 1: CSP for early/late patterns
// ═══════════════════════════════════════════════════════════════════════════

fn enumerate_valid_patterns() -> Vec<[bool; WEEKS]> {
    let mut patterns = Vec::new();
    for mask in 0u16..(1 << WEEKS) {
        if mask.count_ones() != 6 {
            continue;
        }
        let mut pattern = [false; WEEKS];
        for w in 0..WEEKS {
            pattern[w] = (mask >> w) & 1 == 1;
        }
        let mut valid = true;
        for w in 0..(WEEKS - 2) {
            if pattern[w] == pattern[w + 1] && pattern[w + 1] == pattern[w + 2] {
                valid = false;
                break;
            }
        }
        if valid {
            patterns.push(pattern);
        }
    }
    patterns
}

struct EarlyLateCsp {
    patterns: Vec<[bool; WEEKS]>,
    order: Vec<Vec<usize>>,
    early_counts: [u8; WEEKS],
    team_patterns: [usize; TEAMS],
}

impl EarlyLateCsp {
    fn new(patterns: Vec<[bool; WEEKS]>, rng: &mut SmallRng) -> Self {
        let n = patterns.len();
        let order = (0..TEAMS)
            .map(|_| {
                let mut indices: Vec<usize> = (0..n).collect();
                for i in (1..indices.len()).rev() {
                    let j = rng.random_range(0..=i);
                    indices.swap(i, j);
                }
                indices
            })
            .collect();
        EarlyLateCsp {
            patterns,
            order,
            early_counts: [0; WEEKS],
            team_patterns: [0; TEAMS],
        }
    }

    fn solve(&mut self, team: usize) -> bool {
        if team == TEAMS {
            return self.early_counts.iter().all(|&c| c == 8);
        }

        let remaining_after = (TEAMS - team - 1) as u8;

        for idx in 0..self.order[team].len() {
            let pi = self.order[team][idx];
            let pattern = self.patterns[pi];

            let mut feasible = true;
            for w in 0..WEEKS {
                let new_count = self.early_counts[w] + pattern[w] as u8;
                if new_count > 8 || (8 - new_count) > remaining_after {
                    feasible = false;
                    break;
                }
            }
            if !feasible {
                continue;
            }

            let mut pairs_ok = true;
            for prev in 0..team {
                let prev_pat = &self.patterns[self.team_patterns[prev]];
                let mut reachable = false;
                for w in 0..COVERAGE_WEEKS {
                    if pattern[w] == prev_pat[w] {
                        reachable = true;
                        break;
                    }
                }
                if !reachable {
                    pairs_ok = false;
                    break;
                }
            }
            if !pairs_ok {
                continue;
            }

            for w in 0..WEEKS {
                self.early_counts[w] += pattern[w] as u8;
            }
            self.team_patterns[team] = pi;

            if self.solve(team + 1) {
                return true;
            }

            for w in 0..WEEKS {
                self.early_counts[w] -= pattern[w] as u8;
            }
        }
        false
    }

    fn build_halves(&self) -> [([u8; 8], [u8; 8]); WEEKS] {
        let mut halves = [([0u8; 8], [0u8; 8]); WEEKS];
        for w in 0..WEEKS {
            let mut ei = 0;
            let mut li = 0;
            for t in 0..TEAMS {
                if self.patterns[self.team_patterns[t]][w] {
                    halves[w].0[ei] = t as u8;
                    ei += 1;
                } else {
                    halves[w].1[li] = t as u8;
                    li += 1;
                }
            }
        }
        halves
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// Phase 2: Lane-aware backtracking CSP
// ═══════════════════════════════════════════════════════════════════════════

fn pair_idx(a: u8, b: u8) -> usize {
    a.min(b) as usize * TEAMS + a.max(b) as usize
}

fn compute_matchups(teams: &[u8; 4], config: u8) -> [(u8, u8); 4] {
    let [a, b, c, d] = *teams;
    let (p0, p1, p2, p3) = match config {
        0 => (a, c, b, d),
        1 => (a, b, c, d),
        2 => (a, b, d, c),
        _ => unreachable!(),
    };
    [
        (p0.min(p1), p0.max(p1)),
        (p2.min(p3), p2.max(p3)),
        (p0.min(p3), p0.max(p3)),
        (p2.min(p1), p2.max(p1)),
    ]
}

fn quad_positions(teams: &[u8; 4], config: u8) -> [u8; 4] {
    let [a, b, c, d] = *teams;
    match config {
        0 => [a, c, b, d],
        1 => [a, b, c, d],
        2 => [a, b, d, c],
        _ => unreachable!(),
    }
}

/// All 8 arrangements of a quad that preserve matchups:
/// 4 flip variants (identity, swap 0↔2, swap 1↔3, swap both)
/// × 2 orientation variants (identity, toggle stayer/switcher)
fn lane_variants(base: [u8; 4]) -> [[u8; 4]; 8] {
    let [p0, p1, p2, p3] = base;
    [
        [p0, p1, p2, p3],
        [p2, p1, p0, p3],
        [p0, p3, p2, p1],
        [p2, p3, p0, p1],
        [p1, p0, p3, p2],
        [p3, p0, p1, p2],
        [p1, p2, p3, p0],
        [p3, p2, p1, p0],
    ]
}

fn generate_partitions(group: &[u8; 8]) -> Vec<([u8; 4], [u8; 4])> {
    let mut result = Vec::with_capacity(70);
    let n = group.len();
    for b in 1..n {
        for c in (b + 1)..n {
            for d in (c + 1)..n {
                let mut qa = [group[0], group[b], group[c], group[d]];
                let mut qb = [0u8; 4];
                let mut bi = 0;
                for i in 1..n {
                    if i != b && i != c && i != d {
                        qb[bi] = group[i];
                        bi += 1;
                    }
                }
                qa.sort();
                qb.sort();
                result.push((qa, qb));
                result.push((qb, qa));
            }
        }
    }
    result
}

struct CspState {
    matchup_counts: [u8; TEAMS * TEAMS],
    last_week_matchups: [bool; TEAMS * TEAMS],
    lane_counts: [i16; TEAMS * LANES],
    stay_counts: [i16; TEAMS],
    assignment: Assignment,
}

fn apply_quad_lanes(state: &mut CspState, pos: &[u8; 4], q_idx: usize) {
    let lo = (q_idx % 2) * 2;
    let [p0, p1, p2, p3] = *pos;
    state.lane_counts[p0 as usize * LANES + lo] += 2;
    state.lane_counts[p1 as usize * LANES + lo] += 1;
    state.lane_counts[p1 as usize * LANES + lo + 1] += 1;
    state.lane_counts[p2 as usize * LANES + lo + 1] += 2;
    state.lane_counts[p3 as usize * LANES + lo + 1] += 1;
    state.lane_counts[p3 as usize * LANES + lo] += 1;
    state.stay_counts[p0 as usize] += 1;
    state.stay_counts[p2 as usize] += 1;
}

fn undo_quad_lanes(state: &mut CspState, pos: &[u8; 4], q_idx: usize) {
    let lo = (q_idx % 2) * 2;
    let [p0, p1, p2, p3] = *pos;
    state.lane_counts[p0 as usize * LANES + lo] -= 2;
    state.lane_counts[p1 as usize * LANES + lo] -= 1;
    state.lane_counts[p1 as usize * LANES + lo + 1] -= 1;
    state.lane_counts[p2 as usize * LANES + lo + 1] -= 2;
    state.lane_counts[p3 as usize * LANES + lo + 1] -= 1;
    state.lane_counts[p3 as usize * LANES + lo] -= 1;
    state.stay_counts[p0 as usize] -= 1;
    state.stay_counts[p2 as usize] -= 1;
}

/// Max lane/stay counts allowed during the first COVERAGE_WEEKS,
/// reserving at least 1 for the remaining weeks.
const COVERAGE_LANE_CAP: i16 = TARGET_LANE - 1;
const COVERAGE_STAY_CAP: i16 = TARGET_STAY - 1;

/// Quick check after applying one quad's lanes: prune if any affected team
/// already exceeds the lane or stay target by more than the allowed slack.
fn quad_lanes_ok(state: &CspState, pos: &[u8; 4], q_idx: usize, week: usize) -> bool {
    let lo = (q_idx % 2) * 2;
    let lane_cap = if week < COVERAGE_WEEKS { COVERAGE_LANE_CAP } else { TARGET_LANE + LANE_SLACK };
    let stay_cap = if week < COVERAGE_WEEKS { COVERAGE_STAY_CAP } else { TARGET_STAY + STAY_SLACK };
    for &p in pos {
        let t = p as usize;
        if state.lane_counts[t * LANES + lo] > lane_cap { return false; }
        if state.lane_counts[t * LANES + lo + 1] > lane_cap { return false; }
        if state.stay_counts[t] > stay_cap { return false; }
    }
    true
}

struct HalfChoice {
    qa_teams: [u8; 4],
    qa_mc: u8,
    qb_teams: [u8; 4],
    qb_mc: u8,
    matchups: [(u8, u8); 8],
}

/// Week before COVERAGE_WEEKS where repeats become unavoidable (128 slots, 120 pairs).
const REPEAT_WEEK: usize = COVERAGE_WEEKS - 1;

fn enumerate_half_choices(group: &[u8; 8], state: &CspState, week: usize) -> Vec<HalfChoice> {
    let max_count: u8 = if week < REPEAT_WEEK { 1 } else { 2 };
    let partitions = generate_partitions(group);
    let mut choices = Vec::new();

    for (qa, qb) in &partitions {
        for mc_a in 0..3u8 {
            let m_a = compute_matchups(qa, mc_a);
            let mut ok = true;
            for &(lo, hi) in &m_a {
                let idx = pair_idx(lo, hi);
                if state.matchup_counts[idx] >= max_count
                    || (BAN_REPEAT_OPPONENTS && state.last_week_matchups[idx])
                {
                    ok = false;
                    break;
                }
            }
            if !ok { continue; }

            for mc_b in 0..3u8 {
                let m_b = compute_matchups(qb, mc_b);
                let mut ok = true;
                for &(lo, hi) in &m_b {
                    let idx = pair_idx(lo, hi);
                    if state.matchup_counts[idx] >= max_count
                        || (BAN_REPEAT_OPPONENTS && state.last_week_matchups[idx])
                    {
                        ok = false;
                        break;
                    }
                }
                if !ok { continue; }

                let mut cross_ok = true;
                for &(la, ha) in &m_a {
                    for &(lb, hb) in &m_b {
                        if la == lb && ha == hb && state.matchup_counts[pair_idx(la, ha)] >= 1 {
                            cross_ok = false;
                            break;
                        }
                    }
                    if !cross_ok { break; }
                }
                if !cross_ok { continue; }

                let mut matchups = [(0u8, 0u8); 8];
                matchups[..4].copy_from_slice(&m_a);
                matchups[4..].copy_from_slice(&m_b);
                choices.push(HalfChoice {
                    qa_teams: *qa, qa_mc: mc_a,
                    qb_teams: *qb, qb_mc: mc_b,
                    matchups,
                });
            }
        }
    }
    choices
}

fn precompute_pair_futures(halves: &[([u8; 8], [u8; 8]); WEEKS]) -> [[bool; WEEKS]; TEAMS * TEAMS] {
    let mut futures = [[false; WEEKS]; TEAMS * TEAMS];
    for w in 0..WEEKS {
        for group in [&halves[w].0, &halves[w].1] {
            for i in 0..8 {
                for j in (i + 1)..8 {
                    futures[pair_idx(group[i], group[j])][w] = true;
                }
            }
        }
    }
    futures
}

const LANE_SLACK: i16 = 6;
const STAY_SLACK: i16 = 6;
const BAN_REPEAT_OPPONENTS: bool = false;
const TARGET_LANE: i16 = (WEEKS as i16 * 2) / LANES as i16;
const TARGET_STAY: i16 = WEEKS as i16 / 2;

/// All 120 pairs must be covered by this week (0-indexed, exclusive).
const COVERAGE_WEEKS: usize = 8;

fn forward_check(
    state: &CspState,
    week: usize,
    pair_futures: &[[bool; WEEKS]; TEAMS * TEAMS],
) -> bool {
    let remaining = (WEEKS - week - 1) as i16;

    // Coverage deadline: uncovered pairs must be reachable within the first COVERAGE_WEEKS
    let coverage_horizon = if week < COVERAGE_WEEKS { COVERAGE_WEEKS } else { WEEKS };
    for i in 0..TEAMS {
        for j in (i + 1)..TEAMS {
            let idx = i * TEAMS + j;
            if state.matchup_counts[idx] == 0 {
                let mut reachable = false;
                for w in (week + 1)..coverage_horizon {
                    if pair_futures[idx][w] {
                        reachable = true;
                        break;
                    }
                }
                if !reachable {
                    return false;
                }
            }
        }
    }

    let lane_cap = if week < COVERAGE_WEEKS { COVERAGE_LANE_CAP } else { TARGET_LANE + LANE_SLACK };
    let stay_cap = if week < COVERAGE_WEEKS { COVERAGE_STAY_CAP } else { TARGET_STAY + STAY_SLACK };
    for t in 0..TEAMS {
        for l in 0..LANES {
            let c = state.lane_counts[t * LANES + l];
            if c > lane_cap { return false; }
            if c + remaining * 2 < TARGET_LANE - LANE_SLACK { return false; }
        }
        let s = state.stay_counts[t];
        if s > stay_cap { return false; }
        if s + remaining < TARGET_STAY - STAY_SLACK { return false; }
    }

    true
}

fn shuffle<T>(v: &mut [T], rng: &mut SmallRng) {
    for i in (1..v.len()).rev() {
        let j = rng.random_range(0..=i);
        v.swap(i, j);
    }
}

fn lane_variant_score(state: &CspState, v: &[u8; 4], q_idx: usize) -> i32 {
    let lo = (q_idx % 2) * 2;
    let [p0, p1, p2, p3] = *v;
    let mut s = 0i32;
    let tl = TARGET_LANE as i32;
    let ts = TARGET_STAY as i32;

    s += (state.lane_counts[p0 as usize * LANES + lo] as i32 + 2 - tl).pow(2);
    s += (state.stay_counts[p0 as usize] as i32 + 1 - ts).pow(2);

    s += (state.lane_counts[p1 as usize * LANES + lo] as i32 + 1 - tl).pow(2);
    s += (state.lane_counts[p1 as usize * LANES + lo + 1] as i32 + 1 - tl).pow(2);

    s += (state.lane_counts[p2 as usize * LANES + lo + 1] as i32 + 2 - tl).pow(2);
    s += (state.stay_counts[p2 as usize] as i32 + 1 - ts).pow(2);

    s += (state.lane_counts[p3 as usize * LANES + lo + 1] as i32 + 1 - tl).pow(2);
    s += (state.lane_counts[p3 as usize * LANES + lo] as i32 + 1 - tl).pow(2);

    s
}

fn top_n_variants(state: &CspState, variants: &[[u8; 4]; 8], q_idx: usize) -> [[u8; 4]; 6] {
    let mut scored: [([u8; 4], i32); 8] = std::array::from_fn(|i| {
        (variants[i], lane_variant_score(state, &variants[i], q_idx))
    });
    scored.sort_by_key(|&(_, s)| s);
    [scored[0].0, scored[1].0, scored[2].0, scored[3].0, scored[4].0, scored[5].0]
}

const DEPTH_TIMEOUT_SECS: [u64; 13] = [
    100,      // week 0
    100,      // week 1
    100,      // week 2
    200,      // week 3
    300,      // week 4
    3600,     // week 5   (1 hr)
    14400,    // week 6   (4 hr)
    43200,    // week 7   (12 hr)
    86400,    // week 8   (24 hr)
    172800,   // week 9   (48 hr)
    259200,   // week 10  (72 hr)
    604800,   // week 11  (1 week)
    604800,   // week 12  (1 week)
];

fn solve_csp(
    halves: &[([u8; 8], [u8; 8]); WEEKS],
    pair_futures: &[[bool; WEEKS]; TEAMS * TEAMS],
    state: &mut CspState,
    week: usize,
    rng: &mut SmallRng,
    attempt_start: &Instant,
    max_week: &AtomicUsize,
    current_week: &AtomicUsize,
) -> bool {
    current_week.store(week, Ordering::Relaxed);
    let prev_max = max_week.fetch_max(week, Ordering::Relaxed);
    let high_water = prev_max.max(week);
    let timeout = std::time::Duration::from_secs(DEPTH_TIMEOUT_SECS[high_water]);
    if attempt_start.elapsed() > timeout {
        return false;
    }

    if week == WEEKS {
        return true;
    }

    // Hard gate: all 120 pairs must be covered by COVERAGE_WEEKS
    if week == COVERAGE_WEEKS {
        for i in 0..TEAMS {
            for j in (i + 1)..TEAMS {
                if state.matchup_counts[i * TEAMS + j] == 0 {
                    return false;
                }
            }
        }
    }

    let (early, late) = &halves[week];
    let mut early_choices = enumerate_half_choices(early, state, week);
    shuffle(&mut early_choices, rng);

    for ec in &early_choices {
        let mut applied_e = Vec::with_capacity(8);
        for &(lo, hi) in &ec.matchups {
            let idx = pair_idx(lo, hi);
            state.matchup_counts[idx] += 1;
            applied_e.push(idx);
        }

        let mut late_choices = enumerate_half_choices(late, state, week);
        shuffle(&mut late_choices, rng);

        let ea_vars = lane_variants(quad_positions(&ec.qa_teams, ec.qa_mc));
        let eb_vars = lane_variants(quad_positions(&ec.qb_teams, ec.qb_mc));

        for lc in &late_choices {
            let mut m_ok = true;
            for &(lo, hi) in &lc.matchups {
                if state.matchup_counts[pair_idx(lo, hi)] >= 2 {
                    m_ok = false;
                    break;
                }
            }
            if !m_ok { continue; }

            let mut applied_l = Vec::with_capacity(8);
            for &(lo, hi) in &lc.matchups {
                let idx = pair_idx(lo, hi);
                state.matchup_counts[idx] += 1;
                applied_l.push(idx);
            }

            let saved_last = state.last_week_matchups;
            state.last_week_matchups = [false; TEAMS * TEAMS];
            for &(lo, hi) in ec.matchups.iter().chain(lc.matchups.iter()) {
                state.last_week_matchups[pair_idx(lo, hi)] = true;
            }

            let la_vars = lane_variants(quad_positions(&lc.qa_teams, lc.qa_mc));
            let lb_vars = lane_variants(quad_positions(&lc.qb_teams, lc.qb_mc));

            let ea_top = top_n_variants(state, &ea_vars, 0);
            for &ea in &ea_top {
                apply_quad_lanes(state, &ea, 0);
                if !quad_lanes_ok(state, &ea, 0, week) {
                    undo_quad_lanes(state, &ea, 0);
                    continue;
                }

                let eb_top = top_n_variants(state, &eb_vars, 1);
                for &eb in &eb_top {
                    apply_quad_lanes(state, &eb, 1);
                    if !quad_lanes_ok(state, &eb, 1, week) {
                        undo_quad_lanes(state, &eb, 1);
                        continue;
                    }

                    let la_top = top_n_variants(state, &la_vars, 2);
                    for &la in &la_top {
                        apply_quad_lanes(state, &la, 2);
                        if !quad_lanes_ok(state, &la, 2, week) {
                            undo_quad_lanes(state, &la, 2);
                            continue;
                        }

                        let lb_top = top_n_variants(state, &lb_vars, 3);
                        for &lb in &lb_top {
                            apply_quad_lanes(state, &lb, 3);
                            if !quad_lanes_ok(state, &lb, 3, week) {
                                undo_quad_lanes(state, &lb, 3);
                                continue;
                            }

                            state.assignment[week] = [ea, eb, la, lb];

                            let next = week + 1;
                            let feasible = next == WEEKS
                                || forward_check(state, week, pair_futures);
                            if feasible
                                && solve_csp(halves, pair_futures, state, next, rng, attempt_start, max_week, current_week)
                            {
                                return true;
                            }

                            undo_quad_lanes(state, &lb, 3);
                        }
                        undo_quad_lanes(state, &la, 2);
                    }
                    undo_quad_lanes(state, &eb, 1);
                }
                undo_quad_lanes(state, &ea, 0);
            }

            state.last_week_matchups = saved_last;
            for &idx in &applied_l {
                state.matchup_counts[idx] -= 1;
            }
        }

        for &idx in &applied_e {
            state.matchup_counts[idx] -= 1;
        }
    }
    false
}

// ═══════════════════════════════════════════════════════════════════════════
// Status tracking
// ═══════════════════════════════════════════════════════════════════════════

struct CoreStatus {
    phase: AtomicU8,
    p1_seed: AtomicU64,
    p2_attempt: AtomicU64,
    p2_successes: AtomicU64,
    max_week: AtomicUsize,
    current_week: AtomicUsize,
    best_cost: AtomicU32,
}

impl CoreStatus {
    fn new() -> Self {
        CoreStatus {
            phase: AtomicU8::new(0),
            p1_seed: AtomicU64::new(0),
            p2_attempt: AtomicU64::new(0),
            p2_successes: AtomicU64::new(0),
            max_week: AtomicUsize::new(0),
            current_week: AtomicUsize::new(0),
            best_cost: AtomicU32::new(u32::MAX),
        }
    }
}

fn dump_status(statuses: &[Arc<CoreStatus>], global_best: &AtomicU32) {
    eprintln!("=== Status [{}] ===", now_iso());
    for (i, s) in statuses.iter().enumerate() {
        let phase = match s.phase.load(Ordering::Relaxed) {
            1 => "Phase1",
            2 => "Phase2",
            _ => "idle  ",
        };
        let seed = s.p1_seed.load(Ordering::Relaxed);
        let att = s.p2_attempt.load(Ordering::Relaxed);
        let ok = s.p2_successes.load(Ordering::Relaxed);
        let mw = s.max_week.load(Ordering::Relaxed);
        let cw = s.current_week.load(Ordering::Relaxed);
        let bc = s.best_cost.load(Ordering::Relaxed);
        let bc_str = if bc == u32::MAX { "---".to_string() } else { bc.to_string() };
        eprintln!(
            "  core {:>2}: {} p1={:<4} att={:<4} ok={:<3} now={:>2} peak={:>2}/12 best={}",
            i, phase, seed, att, ok, cw, mw, bc_str,
        );
    }
    let gb = global_best.load(Ordering::Relaxed);
    let gb_str = if gb == u32::MAX { "---".to_string() } else { gb.to_string() };
    eprintln!("  Global best: {}", gb_str);
    eprintln!("==============");
}

// ═══════════════════════════════════════════════════════════════════════════
// Main: multi-threaded Phase 1 → Phase 2 (lane-aware) → save
// ═══════════════════════════════════════════════════════════════════════════

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let mut weights_path = "../weights.json";
    let mut cores_override: Option<usize> = None;
    let mut i = 1;
    while i < args.len() {
        match args[i].as_str() {
            "--cores" => {
                i += 1;
                cores_override = Some(args[i].parse().expect("--cores requires a number"));
            }
            other => weights_path = Box::leak(other.to_string().into_boxed_str()),
        }
        i += 1;
    }
    let weights_str = fs::read_to_string(weights_path)
        .unwrap_or_else(|e| panic!("Failed to read {}: {}", weights_path, e));
    let w8: Weights = serde_json::from_str(&weights_str)
        .unwrap_or_else(|e| panic!("Failed to parse {}: {}", weights_path, e));

    let patterns = enumerate_valid_patterns();
    eprintln!("[{}] Valid early/late patterns: {}", now_iso(), patterns.len());

    let run_ts = chrono::Local::now().format("%Y-%m-%dT%H%M%S").to_string();
    let results_dir = format!("results/construct-{}", run_ts);
    fs::create_dir_all(&results_dir).expect("Failed to create results directory");
    eprintln!("[{}] Results dir: {}", now_iso(), results_dir);

    let shutdown = Arc::new(AtomicBool::new(false));
    {
        let shutdown = Arc::clone(&shutdown);
        ctrlc::set_handler(move || {
            if shutdown.load(Ordering::SeqCst) {
                eprintln!("\n[{}] Force exit.", now_iso());
                std::process::exit(1);
            }
            shutdown.store(true, Ordering::SeqCst);
            eprintln!("\n[{}] Ctrl+C received, finishing current attempt... (press again to force)", now_iso());
        }).expect("Failed to set Ctrl+C handler");
    }

    let global_best = Arc::new(AtomicU32::new(u32::MAX));
    let w8 = Arc::new(w8);
    let patterns = Arc::new(patterns);
    let results_dir = Arc::new(results_dir);

    let available = thread::available_parallelism()
        .map(|n| n.get())
        .unwrap_or(4);
    let num_cores = cores_override.unwrap_or(available);

    let statuses: Vec<Arc<CoreStatus>> = (0..num_cores)
        .map(|_| Arc::new(CoreStatus::new()))
        .collect();

    eprintln!(
        "[{}] Starting construct: {} cores, Phase1 → Phase2 (lane-aware CSP). Ctrl+C to stop, Enter for status.",
        now_iso(), num_cores,
    );

    {
        let statuses = statuses.clone();
        let global_best = Arc::clone(&global_best);
        let shutdown = Arc::clone(&shutdown);
        thread::spawn(move || {
            let stdin = io::stdin();
            for line in stdin.lock().lines() {
                if shutdown.load(Ordering::Relaxed) { break; }
                if line.is_ok() {
                    dump_status(&statuses, &global_best);
                }
            }
        });
    }

    let handles: Vec<_> = (0..num_cores)
        .map(|core_id| {
            let shutdown = Arc::clone(&shutdown);
            let global_best = Arc::clone(&global_best);
            let w8 = Arc::clone(&w8);
            let patterns = Arc::clone(&patterns);
            let results_dir = Arc::clone(&results_dir);
            let status = Arc::clone(&statuses[core_id]);

            thread::spawn(move || {
                let mut phase1_seed: u64 = core_id as u64;

                loop {
                    if shutdown.load(Ordering::Relaxed) { return; }

                    // ── Phase 1 ──
                    status.phase.store(1, Ordering::Relaxed);
                    let seed = phase1_seed;
                    phase1_seed += num_cores as u64;
                    status.p1_seed.store(seed, Ordering::Relaxed);

                    let mut halves = None;
                    {
                        let mut r = SmallRng::seed_from_u64(seed);
                        let mut csp = EarlyLateCsp::new((*patterns).clone(), &mut r);
                        if csp.solve(0) {
                            halves = Some(csp.build_halves());
                        }
                    }
                    let halves = match halves {
                        Some(h) => h,
                        None => continue,
                    };
                    eprintln!(
                        "[{}] core {} Phase1 seed {} → solved",
                        now_iso(), core_id, seed,
                    );

                    if shutdown.load(Ordering::Relaxed) { return; }

                    // ── Phase 2: lane-aware CSP (30s budget) ──
                    status.phase.store(2, Ordering::Relaxed);
                    status.p2_attempt.store(0, Ordering::Relaxed);
                    status.p2_successes.store(0, Ordering::Relaxed);
                    status.max_week.store(0, Ordering::Relaxed);
                    status.best_cost.store(u32::MAX, Ordering::Relaxed);

                    let pair_futures = precompute_pair_futures(&halves);
                    let mut local_best_cost = u32::MAX;
                    let mut attempt = 0u64;
                    let mut successes = 0u64;

                    loop {
                        if shutdown.load(Ordering::Relaxed) { return; }

                        let mut state = CspState {
                            matchup_counts: [0; TEAMS * TEAMS],
                            last_week_matchups: [false; TEAMS * TEAMS],
                            lane_counts: [0; TEAMS * LANES],
                            stay_counts: [0; TEAMS],
                            assignment: [[[0u8; POS]; QUADS]; WEEKS],
                        };
                        let mut r = SmallRng::seed_from_u64(seed.wrapping_mul(10007).wrapping_add(attempt));
                        status.max_week.store(0, Ordering::Relaxed);
                        status.current_week.store(0, Ordering::Relaxed);
                        attempt += 1;
                        status.p2_attempt.store(attempt, Ordering::Relaxed);

                        let attempt_start = Instant::now();

                        if solve_csp(&halves, &pair_futures, &mut state, 0, &mut r, &attempt_start, &status.max_week, &status.current_week) {
                            let cost = evaluate(&state.assignment, &w8);
                            successes += 1;
                            status.p2_successes.store(successes, Ordering::Relaxed);

                            let prev_global = global_best.fetch_min(cost.total, Ordering::Relaxed);
                            let marker = if cost.total < prev_global { " ★" } else { "" };
                            eprintln!(
                                "[{}] core {} p1={} #{}: {}{}",
                                now_iso(), core_id, seed, successes, cost_label(&cost), marker,
                            );

                            if cost.total < local_best_cost {
                                local_best_cost = cost.total;
                                status.best_cost.store(local_best_cost, Ordering::Relaxed);
                                let label = format!("c{}-p1s{}-n{}", core_id, seed, successes);
                                let filename = format!(
                                    "{}/{:04}-{}.tsv", results_dir, cost.total, label,
                                );
                                let _ = fs::write(&filename, assignment_to_tsv(&state.assignment));
                            }
                        }
                    }

                }
            })
        })
        .collect();

    for h in handles {
        h.join().unwrap();
    }
    eprintln!("[{}] Construct finished.", now_iso());
    std::process::exit(0);
}
