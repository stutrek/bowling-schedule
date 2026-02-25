use rand::rngs::SmallRng;
use rand::{Rng, SeedableRng};
use serde::Deserialize;
use std::fs;
use std::io::{self, BufRead};
use std::sync::atomic::{AtomicBool, AtomicU32, AtomicU64, AtomicU8, AtomicUsize, Ordering};
use std::sync::Arc;
use std::thread;
use std::time::Instant;

use solver_native::*;

// ═══════════════════════════════════════════════════════════════════════════
// Partial solution input (matches first_half output)
// ═══════════════════════════════════════════════════════════════════════════

#[derive(Deserialize)]
struct PartialSolution {
    halves: Vec<([u8; 8], [u8; 8])>,
    assignment: [[[u8; POS]; QUADS]; WEEKS],
    matchup_counts: Vec<u8>,
    lane_counts: Vec<i16>,
    stay_counts: Vec<i16>,
}

/// First-half solved weeks 0-5 fully + early half of week 6.
const RESUME_WEEK: usize = 6;

const LANE_SLACK: i16 = 6;
const STAY_SLACK: i16 = 6;
const TARGET_LANE: i16 = (WEEKS as i16 * 2) / LANES as i16;
const TARGET_STAY: i16 = WEEKS as i16 / 2;

// ═══════════════════════════════════════════════════════════════════════════
// CSP helpers (tuned for second half)
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

fn quad_lanes_ok(state: &CspState, pos: &[u8; 4], q_idx: usize) -> bool {
    let lo = (q_idx % 2) * 2;
    for &p in pos {
        let t = p as usize;
        if state.lane_counts[t * LANES + lo] > TARGET_LANE + LANE_SLACK { return false; }
        if state.lane_counts[t * LANES + lo + 1] > TARGET_LANE + LANE_SLACK { return false; }
        if state.stay_counts[t] > TARGET_STAY + STAY_SLACK { return false; }
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
                if state.matchup_counts[pair_idx(lo, hi)] >= 2 {
                    ok = false;
                    break;
                }
            }
            if !ok { continue; }

            for mc_b in 0..3u8 {
                let m_b = compute_matchups(qb, mc_b);
                let mut ok = true;
                for &(lo, hi) in &m_b {
                    if state.matchup_counts[pair_idx(lo, hi)] >= 2 {
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

fn forward_check(state: &CspState, week: usize) -> bool {
    let remaining = (WEEKS - week - 1) as i16;
    for t in 0..TEAMS {
        for l in 0..LANES {
            let c = state.lane_counts[t * LANES + l];
            if c > TARGET_LANE + LANE_SLACK { return false; }
            if c + remaining * 2 < TARGET_LANE - LANE_SLACK { return false; }
        }
        let s = state.stay_counts[t];
        if s > TARGET_STAY + STAY_SLACK { return false; }
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

const DEPTH_TIMEOUT_SECS: [u64; 7] = [
    3600,     // week 6 late  (1 hr)
    3600,     // week 7       (1 hr)
    7200,     // week 8       (2 hr)
    14400,    // week 9       (4 hr)
    43200,    // week 10      (12 hr)
    86400,    // week 11      (24 hr)
    86400,    // terminal
];

/// Solve the late half of week RESUME_WEEK, then full weeks RESUME_WEEK+1..WEEKS-1.
fn solve_stage2(
    halves: &[([u8; 8], [u8; 8]); WEEKS],
    state: &mut CspState,
    week: usize,
    early_done: bool,
    rng: &mut SmallRng,
    attempt_start: &Instant,
    max_week: &AtomicUsize,
    current_week: &AtomicUsize,
    shutdown: &AtomicBool,
) -> bool {
    if shutdown.load(Ordering::Relaxed) { return false; }
    current_week.store(week, Ordering::Relaxed);
    let prev_max = max_week.fetch_max(week, Ordering::Relaxed);
    let high_water = prev_max.max(week);
    let timeout_idx = (high_water - RESUME_WEEK).min(DEPTH_TIMEOUT_SECS.len() - 1);
    let timeout = std::time::Duration::from_secs(DEPTH_TIMEOUT_SECS[timeout_idx]);
    if attempt_start.elapsed() > timeout {
        return false;
    }

    if week == WEEKS {
        return true;
    }

    // Week RESUME_WEEK: early half already done, just solve late half
    if week == RESUME_WEEK && early_done {
        let (_early, late) = &halves[week];
        let mut late_choices = enumerate_half_choices(late, state);
        shuffle(&mut late_choices, rng);

        for lc in &late_choices {
            let mut applied_l = Vec::with_capacity(8);
            for &(lo, hi) in &lc.matchups {
                let idx = pair_idx(lo, hi);
                state.matchup_counts[idx] += 1;
                applied_l.push(idx);
            }

            let la_vars = lane_variants(quad_positions(&lc.qa_teams, lc.qa_mc));
            let lb_vars = lane_variants(quad_positions(&lc.qb_teams, lc.qb_mc));

            let la_top = top_n_variants(state, &la_vars, 2);
            for &la in &la_top {
                apply_quad_lanes(state, &la, 2);
                if !quad_lanes_ok(state, &la, 2) {
                    undo_quad_lanes(state, &la, 2);
                    continue;
                }

                let lb_top = top_n_variants(state, &lb_vars, 3);
                for &lb in &lb_top {
                    apply_quad_lanes(state, &lb, 3);
                    if !quad_lanes_ok(state, &lb, 3) {
                        undo_quad_lanes(state, &lb, 3);
                        continue;
                    }

                    state.assignment[week][2] = la;
                    state.assignment[week][3] = lb;

                    if forward_check(state, week)
                        && solve_stage2(halves, state, week + 1, false, rng, attempt_start, max_week, current_week, shutdown)
                    {
                        return true;
                    }

                    undo_quad_lanes(state, &lb, 3);
                }
                undo_quad_lanes(state, &la, 2);
            }

            for &idx in &applied_l {
                state.matchup_counts[idx] -= 1;
            }
        }
        return false;
    }

    // Full weeks (7-11)
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

        let ea_vars = lane_variants(quad_positions(&ec.qa_teams, ec.qa_mc));
        let eb_vars = lane_variants(quad_positions(&ec.qb_teams, ec.qb_mc));

        for lc in &late_choices {
            let mut applied_l = Vec::with_capacity(8);
            for &(lo, hi) in &lc.matchups {
                let idx = pair_idx(lo, hi);
                state.matchup_counts[idx] += 1;
                applied_l.push(idx);
            }

            let la_vars = lane_variants(quad_positions(&lc.qa_teams, lc.qa_mc));
            let lb_vars = lane_variants(quad_positions(&lc.qb_teams, lc.qb_mc));

            let ea_top = top_n_variants(state, &ea_vars, 0);
            for &ea in &ea_top {
                apply_quad_lanes(state, &ea, 0);
                if !quad_lanes_ok(state, &ea, 0) {
                    undo_quad_lanes(state, &ea, 0);
                    continue;
                }

                let eb_top = top_n_variants(state, &eb_vars, 1);
                for &eb in &eb_top {
                    apply_quad_lanes(state, &eb, 1);
                    if !quad_lanes_ok(state, &eb, 1) {
                        undo_quad_lanes(state, &eb, 1);
                        continue;
                    }

                    let la_top = top_n_variants(state, &la_vars, 2);
                    for &la in &la_top {
                        apply_quad_lanes(state, &la, 2);
                        if !quad_lanes_ok(state, &la, 2) {
                            undo_quad_lanes(state, &la, 2);
                            continue;
                        }

                        let lb_top = top_n_variants(state, &lb_vars, 3);
                        for &lb in &lb_top {
                            apply_quad_lanes(state, &lb, 3);
                            if !quad_lanes_ok(state, &lb, 3) {
                                undo_quad_lanes(state, &lb, 3);
                                continue;
                            }

                            state.assignment[week] = [ea, eb, la, lb];

                            let feasible = week + 1 == WEEKS
                                || forward_check(state, week);
                            if feasible
                                && solve_stage2(halves, state, week + 1, false, rng, attempt_start, max_week, current_week, shutdown)
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
    file_idx: AtomicU64,
    attempt: AtomicU64,
    successes: AtomicU64,
    max_week: AtomicUsize,
    current_week: AtomicUsize,
    best_cost: AtomicU32,
}

impl CoreStatus {
    fn new() -> Self {
        CoreStatus {
            phase: AtomicU8::new(0),
            file_idx: AtomicU64::new(0),
            attempt: AtomicU64::new(0),
            successes: AtomicU64::new(0),
            max_week: AtomicUsize::new(0),
            current_week: AtomicUsize::new(0),
            best_cost: AtomicU32::new(u32::MAX),
        }
    }
}

fn dump_status(statuses: &[Arc<CoreStatus>], global_best: &AtomicU32, files: &[String]) {
    eprintln!("=== Status [{}] ===", now_iso());
    for (i, s) in statuses.iter().enumerate() {
        let phase = s.phase.load(Ordering::Relaxed);
        if phase == 0 {
            eprintln!("  core {:>2}: idle", i);
            continue;
        }
        let fi = s.file_idx.load(Ordering::Relaxed) as usize;
        let att = s.attempt.load(Ordering::Relaxed);
        let ok = s.successes.load(Ordering::Relaxed);
        let mw = s.max_week.load(Ordering::Relaxed);
        let cw = s.current_week.load(Ordering::Relaxed);
        let bc = s.best_cost.load(Ordering::Relaxed);
        let bc_str = if bc == u32::MAX { "---".to_string() } else { bc.to_string() };
        let fname = if fi < files.len() { &files[fi] } else { "?" };
        eprintln!(
            "  core {:>2}: file={} att={:<4} ok={:<3} now={:>2} peak={:>2}/12 best={}",
            i, fname, att, ok, cw, mw, bc_str,
        );
    }
    let gb = global_best.load(Ordering::Relaxed);
    let gb_str = if gb == u32::MAX { "---".to_string() } else { gb.to_string() };
    eprintln!("  Global best: {}", gb_str);
    eprintln!("==============");
}

// ═══════════════════════════════════════════════════════════════════════════
// Main
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
            "--weights" => {
                i += 1;
                weights_path = Box::leak(args[i].clone().into_boxed_str());
            }
            _ => {}
        }
        i += 1;
    }

    let weights_str = fs::read_to_string(weights_path)
        .unwrap_or_else(|e| panic!("Failed to read {}: {}", weights_path, e));
    let w8: Weights = serde_json::from_str(&weights_str)
        .unwrap_or_else(|e| panic!("Failed to parse {}: {}", weights_path, e));

    let input_dir = "results/first-half";
    let output_dir = "results/complete";
    fs::create_dir_all(output_dir).expect("Failed to create output directory");

    let mut files: Vec<String> = Vec::new();
    if let Ok(entries) = fs::read_dir(input_dir) {
        for entry in entries.flatten() {
            let path = entry.path();
            if path.extension().and_then(|e| e.to_str()) == Some("json") {
                files.push(path.to_string_lossy().to_string());
            }
        }
    }
    files.sort();

    if files.is_empty() {
        eprintln!("[{}] No .json files found in {}", now_iso(), input_dir);
        std::process::exit(1);
    }
    eprintln!("[{}] Found {} partial solutions in {}", now_iso(), files.len(), input_dir);

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
    let files = Arc::new(files);
    let next_file = Arc::new(AtomicUsize::new(0));

    let available = thread::available_parallelism()
        .map(|n| n.get())
        .unwrap_or(4);
    let num_cores = cores_override.unwrap_or(available);

    let statuses: Vec<Arc<CoreStatus>> = (0..num_cores)
        .map(|_| Arc::new(CoreStatus::new()))
        .collect();

    eprintln!(
        "[{}] Starting complete: {} cores. Ctrl+C to stop, Enter for status.",
        now_iso(), num_cores,
    );

    {
        let statuses = statuses.clone();
        let global_best = Arc::clone(&global_best);
        let shutdown = Arc::clone(&shutdown);
        let files = Arc::clone(&files);
        thread::spawn(move || {
            let stdin = io::stdin();
            for line in stdin.lock().lines() {
                if shutdown.load(Ordering::Relaxed) { break; }
                if line.is_ok() {
                    dump_status(&statuses, &global_best, &files);
                }
            }
        });
    }

    let handles: Vec<_> = (0..num_cores)
        .map(|core_id| {
            let shutdown = Arc::clone(&shutdown);
            let global_best = Arc::clone(&global_best);
            let w8 = Arc::clone(&w8);
            let files = Arc::clone(&files);
            let next_file = Arc::clone(&next_file);
            let status = Arc::clone(&statuses[core_id]);

            thread::spawn(move || {
                loop {
                    if shutdown.load(Ordering::Relaxed) { return; }

                    let fi = next_file.fetch_add(1, Ordering::Relaxed);
                    if fi >= files.len() {
                        // Wrap around to keep trying
                        next_file.store(0, Ordering::Relaxed);
                        let fi = next_file.fetch_add(1, Ordering::Relaxed);
                        if fi >= files.len() { return; }
                        status.file_idx.store(fi as u64, Ordering::Relaxed);
                    } else {
                        status.file_idx.store(fi as u64, Ordering::Relaxed);
                    }

                    let fi = status.file_idx.load(Ordering::Relaxed) as usize;
                    let file_path = &files[fi];

                    let content = match fs::read_to_string(file_path) {
                        Ok(c) => c,
                        Err(e) => {
                            eprintln!("[{}] core {} Failed to read {}: {}", now_iso(), core_id, file_path, e);
                            continue;
                        }
                    };
                    let partial: PartialSolution = match serde_json::from_str(&content) {
                        Ok(p) => p,
                        Err(e) => {
                            eprintln!("[{}] core {} Failed to parse {}: {}", now_iso(), core_id, file_path, e);
                            continue;
                        }
                    };

                    let mut halves = [([0u8; 8], [0u8; 8]); WEEKS];
                    for (w, h) in partial.halves.iter().enumerate() {
                        if w < WEEKS { halves[w] = *h; }
                    }

                    eprintln!(
                        "[{}] core {} → processing {}",
                        now_iso(), core_id, file_path,
                    );

                    status.phase.store(1, Ordering::Relaxed);
                    status.attempt.store(0, Ordering::Relaxed);
                    status.successes.store(0, Ordering::Relaxed);
                    status.max_week.store(RESUME_WEEK, Ordering::Relaxed);
                    status.best_cost.store(u32::MAX, Ordering::Relaxed);

                    let mut attempt = 0u64;
                    let mut successes = 0u64;
                    let mut local_best_cost = u32::MAX;
                    let mut all_time_peak: usize = RESUME_WEEK;
                    let mut last_peak_improvement = Instant::now();
                        const STAGNATION_SECS: u64 = 3600;

                    loop {
                        if shutdown.load(Ordering::Relaxed) { return; }

                        if last_peak_improvement.elapsed().as_secs() > STAGNATION_SECS {
                            eprintln!(
                                "[{}] core {} {} → stagnant (peak {} after {}s, {} attempts), next file",
                                now_iso(), core_id, file_path, all_time_peak,
                                last_peak_improvement.elapsed().as_secs(), attempt,
                            );
                            break;
                        }

                        let mut state = CspState {
                            matchup_counts: [0; TEAMS * TEAMS],
                            lane_counts: [0; TEAMS * LANES],
                            stay_counts: [0; TEAMS],
                            assignment: partial.assignment,
                        };
                        for (i, &v) in partial.matchup_counts.iter().enumerate() {
                            if i < TEAMS * TEAMS { state.matchup_counts[i] = v; }
                        }
                        for (i, &v) in partial.lane_counts.iter().enumerate() {
                            if i < TEAMS * LANES { state.lane_counts[i] = v; }
                        }
                        for (i, &v) in partial.stay_counts.iter().enumerate() {
                            if i < TEAMS { state.stay_counts[i] = v; }
                        }

                        let mut r = SmallRng::seed_from_u64((fi as u64).wrapping_mul(10007).wrapping_add(attempt));
                        status.max_week.store(RESUME_WEEK, Ordering::Relaxed);
                        status.current_week.store(RESUME_WEEK, Ordering::Relaxed);
                        attempt += 1;
                        status.attempt.store(attempt, Ordering::Relaxed);

                        let attempt_start = Instant::now();

                        let solved = solve_stage2(&halves, &mut state, RESUME_WEEK, true, &mut r, &attempt_start, &status.max_week, &status.current_week, &shutdown);
                        let attempt_peak = status.max_week.load(Ordering::Relaxed);
                        if attempt_peak > all_time_peak {
                            all_time_peak = attempt_peak;
                            last_peak_improvement = Instant::now();
                        }
                        if solved {
                            let cost = evaluate(&state.assignment, &w8);
                            successes += 1;
                            status.successes.store(successes, Ordering::Relaxed);

                            let prev_global = global_best.fetch_min(cost.total, Ordering::Relaxed);
                            let marker = if cost.total < prev_global { " ★" } else { "" };
                            eprintln!(
                                "[{}] core {} #{}: {}{}",
                                now_iso(), core_id, successes, cost_label(&cost), marker,
                            );

                            if cost.total < local_best_cost {
                                local_best_cost = cost.total;
                                status.best_cost.store(local_best_cost, Ordering::Relaxed);
                                let ts = chrono::Local::now().format("%Y%m%d-%H%M%S%z");
                                let filename = format!(
                                    "{}/{:04}-c{}-{}.tsv", output_dir, cost.total, core_id, ts,
                                );
                                let _ = fs::write(&filename, assignment_to_tsv(&state.assignment));
                                eprintln!("[{}] Saved {}", now_iso(), filename);
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
    eprintln!("[{}] Complete finished.", now_iso());
    std::process::exit(0);
}
