use rand::rngs::SmallRng;
use rand::{Rng, SeedableRng};
use std::fs;
use std::sync::atomic::{AtomicBool, AtomicU32, Ordering};
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
                for w in 0..WEEKS {
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
// Phase 2: Backtracking CSP for matchup assignment
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
    assignment: Assignment,
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

fn forward_check(
    state: &CspState,
    week: usize,
    pair_futures: &[[bool; WEEKS]; TEAMS * TEAMS],
) -> bool {
    for i in 0..TEAMS {
        for j in (i + 1)..TEAMS {
            let idx = i * TEAMS + j;
            if state.matchup_counts[idx] == 0 {
                let mut reachable = false;
                for w in (week + 1)..WEEKS {
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
    true
}

struct HalfChoice {
    qa_teams: [u8; 4],
    qa_mc: u8,
    qb_teams: [u8; 4],
    qb_mc: u8,
    matchups: [(u8, u8); 8],
}

fn enumerate_half_choices(group: &[u8; 8], state: &CspState) -> Vec<HalfChoice> {
    let partitions = generate_partitions(group);
    let mut choices = Vec::new();

    for (qa, qb) in &partitions {
        for mc_a in 0..3u8 {
            let m_a = compute_matchups(qa, mc_a);
            let mut ok = true;
            for &(lo, hi) in &m_a {
                let idx = pair_idx(lo, hi);
                if state.matchup_counts[idx] >= 2 || state.last_week_matchups[idx] {
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
                    if state.matchup_counts[idx] >= 2 || state.last_week_matchups[idx] {
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

fn shuffle<T>(v: &mut [T], rng: &mut SmallRng) {
    for i in (1..v.len()).rev() {
        let j = rng.random_range(0..=i);
        v.swap(i, j);
    }
}

fn solve_csp(
    halves: &[([u8; 8], [u8; 8]); WEEKS],
    pair_futures: &[[bool; WEEKS]; TEAMS * TEAMS],
    state: &mut CspState,
    week: usize,
    rng: &mut SmallRng,
    deadline: &Instant,
) -> bool {
    if Instant::now() > *deadline {
        return false;
    }

    if week == WEEKS {
        for i in 0..TEAMS {
            for j in (i + 1)..TEAMS {
                if state.matchup_counts[i * TEAMS + j] == 0 {
                    return false;
                }
            }
        }
        return true;
    }

    let (early, late) = &halves[week];
    let mut early_choices = enumerate_half_choices(early, state);
    shuffle(&mut early_choices, rng);

    for ec in &early_choices {
        let mut applied_e = Vec::with_capacity(8);
        for &(lo, hi) in &ec.matchups {
            let idx = pair_idx(lo, hi);
            state.matchup_counts[idx] += 1;
            applied_e.push(idx);
        }

        let mut late_choices = enumerate_half_choices(late, state);
        shuffle(&mut late_choices, rng);

        for lc in &late_choices {
            let mut ok = true;
            for &(lo, hi) in &lc.matchups {
                if state.matchup_counts[pair_idx(lo, hi)] >= 2 {
                    ok = false;
                    break;
                }
            }
            if !ok { continue; }

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

            state.assignment[week] = [
                quad_positions(&ec.qa_teams, ec.qa_mc),
                quad_positions(&ec.qb_teams, ec.qb_mc),
                quad_positions(&lc.qa_teams, lc.qa_mc),
                quad_positions(&lc.qb_teams, lc.qb_mc),
            ];

            let next = week + 1;
            let feasible = next == WEEKS || forward_check(state, week, pair_futures);
            if feasible && solve_csp(halves, pair_futures, state, next, rng, deadline) {
                return true;
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
// Phase 3: Lane + switch optimization via SA (3B iterations)
// ═══════════════════════════════════════════════════════════════════════════

fn toggle_orientation(quad: &[u8; 4]) -> [u8; 4] {
    let [p0, p1, p2, p3] = *quad;
    [p1, p0, p3, p2]
}

const SA_ITERATIONS: u64 = 3_000_000_000;
const CHECKPOINT_INTERVAL: u64 = 10_000_000;

struct Phase3Context<'a> {
    results_dir: &'a str,
    label: String,
    global_best: &'a AtomicU32,
}

fn phase_3(
    assignment: &mut Assignment,
    w8: &Weights,
    seed: u64,
    shutdown: &AtomicBool,
    ctx: &Phase3Context,
) {
    let mut rng = SmallRng::seed_from_u64(seed);
    let mut best = *assignment;
    let mut best_cost = evaluate(assignment, w8).total;
    let mut current_cost = best_cost;
    let mut temp = 10.0f64;
    let mut last_saved_cost = u32::MAX;

    for iter in 0..SA_ITERATIONS {
        if iter % CHECKPOINT_INTERVAL == 0 {
            if shutdown.load(Ordering::Relaxed) {
                break;
            }
            if iter > 0 {
                let cost_bd = evaluate(&best, w8);
                let prev = ctx.global_best.fetch_min(best_cost, Ordering::Relaxed);
                let marker = if best_cost < prev { " ★" } else { "" };
                eprintln!(
                    "[{}] {} @{}M: {}{}", now_iso(), ctx.label,
                    iter / 1_000_000, cost_label(&cost_bd), marker,
                );
                if best_cost < last_saved_cost {
                    last_saved_cost = best_cost;
                    let filename = format!(
                        "{}/{:04}-{}.tsv", ctx.results_dir, best_cost, ctx.label,
                    );
                    let _ = fs::write(&filename, assignment_to_tsv(&best));
                }
            }
        }

        let move_type = rng.random_range(0..10u8);
        let w = rng.random_range(0..WEEKS);

        match move_type {
            0..=3 => {
                let q = rng.random_range(0..QUADS);
                let saved = assignment[w][q];
                let [p0, p1, p2, p3] = saved;
                let flip = rng.random_range(0..3u8);
                assignment[w][q] = match flip {
                    0 => [p2, p1, p0, p3],
                    1 => [p0, p3, p2, p1],
                    _ => [p2, p3, p0, p1],
                };
                let nc = evaluate(assignment, w8).total;
                let delta = nc as f64 - current_cost as f64;
                if delta < 0.0 || rng.random::<f64>() < (-delta / temp).exp() {
                    current_cost = nc;
                    if nc < best_cost { best_cost = nc; best = *assignment; }
                } else {
                    assignment[w][q] = saved;
                }
            }
            4..=6 => {
                let q = rng.random_range(0..QUADS);
                let saved = assignment[w][q];
                assignment[w][q] = toggle_orientation(&saved);
                let nc = evaluate(assignment, w8).total;
                let delta = nc as f64 - current_cost as f64;
                if delta < 0.0 || rng.random::<f64>() < (-delta / temp).exp() {
                    current_cost = nc;
                    if nc < best_cost { best_cost = nc; best = *assignment; }
                } else {
                    assignment[w][q] = saved;
                }
            }
            _ => {
                let half = rng.random_range(0..2usize);
                let q1 = half * 2;
                let q2 = q1 + 1;
                assignment[w].swap(q1, q2);
                let nc = evaluate(assignment, w8).total;
                let delta = nc as f64 - current_cost as f64;
                if delta < 0.0 || rng.random::<f64>() < (-delta / temp).exp() {
                    current_cost = nc;
                    if nc < best_cost { best_cost = nc; best = *assignment; }
                } else {
                    assignment[w].swap(q1, q2);
                }
            }
        }

        temp *= 0.999_999;
        if iter % 100_000_000 == 0 && iter > 0 {
            temp = 10.0;
        }
    }

    *assignment = best;
}

// ═══════════════════════════════════════════════════════════════════════════
// Main: multi-threaded loop over Phase 1 → Phase 2 → Phase 3
// ═══════════════════════════════════════════════════════════════════════════

const PHASE2_ATTEMPTS: u64 = 100;
const TOP_N_PHASE2: usize = 3;
const SA_RUNS_PER_PHASE2: u64 = 2;

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
            eprintln!("\n[{}] Ctrl+C received, finishing current phase... (press again to force)", now_iso());
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

    eprintln!(
        "[{}] Starting construct loop: {} cores, {}×Phase2, top-{} → {}×SA({}M each). Ctrl+C to stop.",
        now_iso(), num_cores, PHASE2_ATTEMPTS, TOP_N_PHASE2,
        SA_RUNS_PER_PHASE2, SA_ITERATIONS / 1_000_000,
    );

    let handles: Vec<_> = (0..num_cores)
        .map(|core_id| {
            let shutdown = Arc::clone(&shutdown);
            let global_best = Arc::clone(&global_best);
            let w8 = Arc::clone(&w8);
            let patterns = Arc::clone(&patterns);
            let results_dir = Arc::clone(&results_dir);

            thread::spawn(move || {
                let mut rng = SmallRng::from_os_rng();
                let mut phase1_seed: u64 = core_id as u64;

                loop {
                    if shutdown.load(Ordering::Relaxed) { return; }

                    // ── Phase 1 ──
                    let mut halves = None;
                    let seed = phase1_seed;
                    phase1_seed += num_cores as u64;
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

                    // ── Phase 2: collect top N (10s budget) ──
                    let pair_futures = precompute_pair_futures(&halves);
                    let mut phase2_results: Vec<(Assignment, u32)> = Vec::new();
                    let phase2_deadline = Instant::now() + std::time::Duration::from_secs(10);
                    let mut attempt = 0u64;

                    while Instant::now() < phase2_deadline {
                        if shutdown.load(Ordering::Relaxed) { return; }

                        let mut state = CspState {
                            matchup_counts: [0; TEAMS * TEAMS],
                            last_week_matchups: [false; TEAMS * TEAMS],
                            assignment: [[[0u8; POS]; QUADS]; WEEKS],
                        };
                        let mut r = SmallRng::seed_from_u64(seed.wrapping_mul(10007).wrapping_add(attempt));
                        attempt += 1;

                        let attempt_deadline = Instant::now() + std::time::Duration::from_millis(500);
                        let deadline = attempt_deadline.min(phase2_deadline);
                        if solve_csp(&halves, &pair_futures, &mut state, 0, &mut r, &deadline) {
                            let cost = evaluate(&state.assignment, &w8);
                            phase2_results.push((state.assignment, cost.total));
                        }
                    }

                    if phase2_results.is_empty() {
                        eprintln!(
                            "[{}] core {} Phase2 p1={} → 0/{} succeeded, skipping",
                            now_iso(), core_id, seed, attempt,
                        );
                        continue;
                    }

                    phase2_results.sort_by_key(|&(_, c)| c);
                    phase2_results.dedup_by(|a, b| a.0 == b.0);
                    phase2_results.truncate(TOP_N_PHASE2);

                    eprintln!(
                        "[{}] core {} Phase2 p1={} → {}/{} succeeded, top {}: [{}]",
                        now_iso(), core_id, seed,
                        phase2_results.len(), attempt,
                        phase2_results.len().min(TOP_N_PHASE2),
                        phase2_results.iter().map(|(_, c)| c.to_string()).collect::<Vec<_>>().join(", "),
                    );

                    // ── Phase 3: SA on each top result ──
                    for (rank, (base_assignment, pre_cost)) in phase2_results.iter().enumerate() {
                        for sa_run in 0..SA_RUNS_PER_PHASE2 {
                            if shutdown.load(Ordering::Relaxed) { return; }

                            let sa_seed = rng.random::<u64>();
                            let label = format!("c{}-p1s{}-r{}-sa{}", core_id, seed, rank, sa_run);
                            let ctx = Phase3Context {
                                results_dir: &results_dir,
                                label: label.clone(),
                                global_best: &global_best,
                            };
                            let mut a = *base_assignment;

                            eprintln!(
                                "[{}] {} starting (pre: {})",
                                now_iso(), label, pre_cost,
                            );

                            phase_3(&mut a, &w8, sa_seed, &shutdown, &ctx);

                            let final_cost = evaluate(&a, &w8);
                            let prev_best = global_best.fetch_min(final_cost.total, Ordering::Relaxed);
                            let is_new_best = final_cost.total < prev_best;

                            eprintln!(
                                "[{}] {} done | {} {}",
                                now_iso(), label, cost_label(&final_cost),
                                if is_new_best { "★ NEW BEST" } else { "" },
                            );
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
}
