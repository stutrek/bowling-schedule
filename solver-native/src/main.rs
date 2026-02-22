use chrono::Local;
use rand::rngs::SmallRng;
use rand::{Rng, SeedableRng};
use serde::Deserialize;
use std::fs;
use std::sync::{Arc, Condvar, Mutex};
use std::thread;
use std::time::Duration;

const TEAMS: usize = 16;
const LANES: usize = 4;
const WEEKS: usize = 12;
const QUADS: usize = 4;
const POS: usize = 4;

type Assignment = [[[u8; POS]; QUADS]; WEEKS];

#[derive(Deserialize)]
struct Config {
    weights: Weights,
    solver: SolverParams,
}

#[derive(Deserialize, Clone)]
struct Weights {
    matchup_zero: u32,
    matchup_triple: u32,
    consecutive_opponents: u32,
    early_late_balance: f64,
    early_late_alternation: u32,
    lane_balance: f64,
    lane_switch: f64,
}

#[derive(Deserialize)]
struct SolverParams {
    t0: f64,
    max_iterations: u64,
    progress_interval: u64,
    sync_interval: u64,
    restart_interval: u64,
    stagnation_threshold: u32,
}

#[derive(Clone)]
struct CostBreakdown {
    matchup_balance: u32,
    consecutive_opponents: u32,
    early_late_balance: u32,
    early_late_alternation: u32,
    lane_balance: u32,
    lane_switch_balance: u32,
    total: u32,
}

struct BestResults {
    overall: Option<(Assignment, CostBreakdown)>,
    best_matchup: Option<(Assignment, CostBreakdown)>,
    overall_dirty: bool,
    matchup_dirty: bool,
}

struct CoreSlot {
    best_cost: u32,
    best_assignment: Assignment,
    reset_to: Option<Assignment>,
    reset_temp: Option<f64>,
}

struct SyncState {
    slots: Vec<CoreSlot>,
    checked_in: usize,
    epoch: u64,
    global_best_cost: u32,
    global_best_assignment: Assignment,
    stagnation_epochs: u32,
    prev_global_best_cost: u32,
    generation_dir: String,
}

fn random_assignment(rng: &mut SmallRng) -> Assignment {
    let mut a = [[[0u8; POS]; QUADS]; WEEKS];
    for w in 0..WEEKS {
        let mut teams: [u8; TEAMS] = std::array::from_fn(|i| i as u8);
        for i in (1..TEAMS).rev() {
            let j = rng.random_range(0..=i);
            teams.swap(i, j);
        }
        for q in 0..QUADS {
            for p in 0..POS {
                a[w][q][p] = teams[q * POS + p];
            }
        }
    }
    a
}

fn evaluate(a: &Assignment, w8: &Weights) -> CostBreakdown {
    let mut matchups = [0i32; TEAMS * TEAMS];
    let mut week_matchup = [0u8; WEEKS * TEAMS * TEAMS];
    let mut lane_counts = [0i32; TEAMS * LANES];
    let mut stay_count = [0i32; TEAMS];
    let mut early_count = [0i32; TEAMS];
    let mut early_late = [0u8; TEAMS * WEEKS];

    for w in 0..WEEKS {
        for q in 0..QUADS {
            let [pa, pb, pc, pd] = a[w][q];
            let early: u8 = if q < 2 { 1 } else { 0 };
            let lane_off = (q % 2) * 2;

            let pairs: [(u8, u8); 4] = [(pa, pb), (pc, pd), (pa, pd), (pc, pb)];
            for &(t1, t2) in &pairs {
                let lo = t1.min(t2) as usize;
                let hi = t1.max(t2) as usize;
                matchups[lo * TEAMS + hi] += 1;
                week_matchup[w * TEAMS * TEAMS + lo * TEAMS + hi] = 1;
            }

            lane_counts[pa as usize * LANES + lane_off] += 2;
            lane_counts[pb as usize * LANES + lane_off] += 1;
            lane_counts[pb as usize * LANES + lane_off + 1] += 1;
            lane_counts[pc as usize * LANES + lane_off + 1] += 2;
            lane_counts[pd as usize * LANES + lane_off + 1] += 1;
            lane_counts[pd as usize * LANES + lane_off] += 1;

            stay_count[pa as usize] += 1;
            stay_count[pc as usize] += 1;

            for &t in &[pa, pb, pc, pd] {
                early_late[t as usize * WEEKS + w] = early;
                if early == 1 {
                    early_count[t as usize] += 1;
                }
            }
        }
    }

    let mut matchup_balance: u32 = 0;
    for i in 0..TEAMS {
        for j in (i + 1)..TEAMS {
            let c = matchups[i * TEAMS + j];
            if c == 0 {
                matchup_balance += w8.matchup_zero;
            } else if c >= 3 {
                matchup_balance += (c - 2) as u32 * w8.matchup_triple;
            }
        }
    }

    let mut consecutive_opponents: u32 = 0;
    for w in 0..(WEEKS - 1) {
        let b1 = w * TEAMS * TEAMS;
        let b2 = (w + 1) * TEAMS * TEAMS;
        for i in 0..TEAMS {
            for j in (i + 1)..TEAMS {
                let idx = i * TEAMS + j;
                if week_matchup[b1 + idx] != 0 && week_matchup[b2 + idx] != 0 {
                    consecutive_opponents += w8.consecutive_opponents;
                }
            }
        }
    }

    let mut early_late_balance: u32 = 0;
    let target_e: f64 = WEEKS as f64 / 2.0;
    for t in 0..TEAMS {
        let dev = (early_count[t] as f64 - target_e).abs();
        early_late_balance += (dev * dev * w8.early_late_balance) as u32;
    }

    let mut early_late_alternation: u32 = 0;
    for t in 0..TEAMS {
        for w in 0..(WEEKS - 2) {
            let base = t * WEEKS;
            if early_late[base + w] == early_late[base + w + 1]
                && early_late[base + w + 1] == early_late[base + w + 2]
            {
                early_late_alternation += w8.early_late_alternation;
            }
        }
    }

    let mut lane_balance: u32 = 0;
    let target_l: f64 = (WEEKS as f64 * 2.0) / LANES as f64;
    for t in 0..TEAMS {
        for l in 0..LANES {
            lane_balance +=
                ((lane_counts[t * LANES + l] as f64 - target_l).abs() * w8.lane_balance) as u32;
        }
    }

    let mut lane_switch_balance: u32 = 0;
    let target_stay: f64 = WEEKS as f64 / 2.0;
    for t in 0..TEAMS {
        let dev = (stay_count[t] as f64 - target_stay).abs();
        lane_switch_balance += (dev * w8.lane_switch) as u32;
    }

    let total = matchup_balance
        + consecutive_opponents
        + early_late_balance
        + early_late_alternation
        + lane_balance
        + lane_switch_balance;

    CostBreakdown {
        matchup_balance,
        consecutive_opponents,
        early_late_balance,
        early_late_alternation,
        lane_balance,
        lane_switch_balance,
        total,
    }
}

fn perturb(a: &mut Assignment, rng: &mut SmallRng, n: usize) {
    for _ in 0..n {
        let w = rng.random_range(0..WEEKS);
        let q1 = rng.random_range(0..QUADS);
        let mut q2 = rng.random_range(0..(QUADS - 1));
        if q2 >= q1 {
            q2 += 1;
        }
        let p1 = rng.random_range(0..POS);
        let p2 = rng.random_range(0..POS);
        let tmp = a[w][q1][p1];
        a[w][q1][p1] = a[w][q2][p2];
        a[w][q2][p2] = tmp;
    }
}

fn assignment_to_tsv(a: &Assignment) -> String {
    let slot_names = ["Early 1", "Early 2", "Late 1", "Late 2"];
    let mut lines = vec![String::from("Week\tSlot\tLane 1\tLane 2\tLane 3\tLane 4")];

    for w in 0..WEEKS {
        let mut slots: [[String; LANES]; 4] = Default::default();

        for q in 0..QUADS {
            let [pa, pb, pc, pd] = a[w][q];
            let slot_base = if q < 2 { 0 } else { 2 };
            let lane_base = (q % 2) * 2;

            slots[slot_base][lane_base] = format!("{} v {}", pa + 1, pb + 1);
            slots[slot_base][lane_base + 1] = format!("{} v {}", pc + 1, pd + 1);
            slots[slot_base + 1][lane_base] = format!("{} v {}", pa + 1, pd + 1);
            slots[slot_base + 1][lane_base + 1] = format!("{} v {}", pc + 1, pb + 1);
        }

        for (s, slot_row) in slots.iter().enumerate() {
            lines.push(format!(
                "{}\t{}\t{}\t{}\t{}\t{}",
                w + 1,
                slot_names[s],
                slot_row[0],
                slot_row[1],
                slot_row[2],
                slot_row[3]
            ));
        }
    }

    lines.join("\n")
}

fn cost_label(c: &CostBreakdown) -> String {
    format!(
        "total: {:>4} matchup: {:>3} consec: {:>3} el_bal: {:>3} el_alt: {:>3} lane: {:>3} switch: {:>3}",
        c.total, c.matchup_balance, c.consecutive_opponents,
        c.early_late_balance, c.early_late_alternation, c.lane_balance,
        c.lane_switch_balance,
    )
}

fn now_iso() -> String {
    Local::now().format("%Y-%m-%dT%H:%M:%S%:z").to_string()
}

fn new_generation_dir(base: &str) -> String {
    let ts = Local::now().format("%Y-%m-%dT%H%M%S%.3f");
    let dir = format!("{}/gen-{}", base, ts);
    fs::create_dir_all(&dir).expect("Failed to create generation directory");
    dir
}

fn complete_dir(base: &str) -> String {
    let dir = format!("{}/complete", base);
    fs::create_dir_all(&dir).expect("Failed to create complete directory");
    dir
}

fn save_assignment(
    results_dir: &str,
    prefix: &str,
    key: u32,
    a: &Assignment,
    c: &CostBreakdown,
    last_saved: &mut Option<Assignment>,
) -> bool {
    if last_saved.as_ref() == Some(a) {
        return false;
    }
    let ts = Local::now().format("%Y-%m-%dT%H%M%S%z");
    let filename = format!("{}/{}-{}-{}.tsv", results_dir, prefix, key, ts);
    let tsv = assignment_to_tsv(a);
    let _ = fs::write(&filename, &tsv);
    eprintln!("[{}] Saved {}: {} ({})", now_iso(), prefix, filename, cost_label(c));
    *last_saved = Some(*a);
    true
}

fn main() {
    let config_path = std::env::args().nth(1).unwrap_or_else(|| "config.toml".to_string());
    let config_str = fs::read_to_string(&config_path)
        .unwrap_or_else(|e| panic!("Failed to read {}: {}", config_path, e));
    let config: Config = toml::from_str(&config_str)
        .unwrap_or_else(|e| panic!("Failed to parse {}: {}", config_path, e));

    let results_dir = "results";
    fs::create_dir_all(results_dir).expect("Failed to create results directory");

    let num_cores = thread::available_parallelism()
        .map(|n| n.get())
        .unwrap_or(4);
    let max_iterations = config.solver.max_iterations;
    let progress_interval = config.solver.progress_interval;
    let sync_interval = config.solver.sync_interval;
    let restart_interval = config.solver.restart_interval;
    let stagnation_threshold = config.solver.stagnation_threshold;
    let t0 = config.solver.t0;
    let weights = Arc::new(config.weights);

    let dummy_assignment: Assignment = [[[0u8; POS]; QUADS]; WEEKS];
    let initial_gen_dir = new_generation_dir(results_dir);
    eprintln!("[{}] Generation output: {}", now_iso(), initial_gen_dir);

    let sync_pair = Arc::new((
        Mutex::new(SyncState {
            slots: (0..num_cores)
                .map(|_| CoreSlot {
                    best_cost: u32::MAX,
                    best_assignment: dummy_assignment,
                    reset_to: None,
                    reset_temp: None,
                })
                .collect(),
            checked_in: 0,
            epoch: 0,
            global_best_cost: u32::MAX,
            global_best_assignment: dummy_assignment,
            stagnation_epochs: 0,
            prev_global_best_cost: u32::MAX,
            generation_dir: initial_gen_dir,
        }),
        Condvar::new(),
    ));

    let best_results = Arc::new(Mutex::new(BestResults {
        overall: None,
        best_matchup: None,
        overall_dirty: false,
        matchup_dirty: false,
    }));

    eprintln!(
        "[{}] Starting solver: {} cores, {:.1}B iterations/run, sync every {}M, stagnation threshold {}. Ctrl+C to stop.",
        now_iso(), num_cores, max_iterations as f64 / 1e9, sync_interval / 1_000_000, stagnation_threshold,
    );
    eprintln!(
        "[{}] Weights: matchup_zero={} matchup_triple={} consec={} el_bal={} el_alt={} lane={} switch={}",
        now_iso(), weights.matchup_zero, weights.matchup_triple, weights.consecutive_opponents,
        weights.early_late_balance, weights.early_late_alternation, weights.lane_balance, weights.lane_switch,
    );

    let saver_best = Arc::clone(&best_results);
    let saver_sync = Arc::clone(&sync_pair);
    thread::spawn(move || {
        let mut last_overall: Option<Assignment> = None;
        let mut last_matchup: Option<Assignment> = None;
        loop {
            thread::sleep(Duration::from_secs(1));
            let gen_dir = {
                let sync = saver_sync.0.lock().unwrap();
                sync.generation_dir.clone()
            };
            let mut state = saver_best.lock().unwrap();

            if state.overall_dirty {
                if let Some((ref a, ref c)) = state.overall {
                    save_assignment(&gen_dir, "best-overall", c.total, a, c, &mut last_overall);
                }
                state.overall_dirty = false;
            }

            if state.matchup_dirty {
                if let Some((ref a, ref c)) = state.best_matchup {
                    save_assignment(&gen_dir, "best-matchup", c.matchup_balance, a, c, &mut last_matchup);
                }
                state.matchup_dirty = false;
            }
        }
    });

    let handles: Vec<_> = (0..num_cores)
        .map(|core_id| {
            let sync_pair = Arc::clone(&sync_pair);
            let best_results = Arc::clone(&best_results);
            let base_dir = results_dir.to_string();
            let w8 = Arc::clone(&weights);
            thread::spawn(move || {
                let mut rng = SmallRng::from_os_rng();
                let mut last_perfect: Option<Assignment> = None;
                let mut last_sync: Option<Assignment> = None;
                let mut last_gen_best: Option<Assignment> = None;

                loop {
                    let cool_rate: f64 = (0.005_f64 / t0).ln() / max_iterations as f64;
                    let mut a = random_assignment(&mut rng);
                    let mut cost = evaluate(&a, &w8);
                    let mut best_a = a;
                    let mut best_cost = cost.total;
                    let mut temp: f64 = t0;
                    let mut cool_offset: u64 = 0;

                    for i in 0..max_iterations {
                        if best_cost == 0 {
                            let final_cost = evaluate(&best_a, &w8);
                            let gen_dir = sync_pair.0.lock().unwrap().generation_dir.clone();
                            save_assignment(&gen_dir, "perfect", 0, &best_a, &final_cost, &mut last_perfect);
                            eprintln!(
                                "[{}] core {} found PERFECT solution! ({})",
                                now_iso(), core_id, cost_label(&final_cost),
                            );
                            {
                                let mut br = best_results.lock().unwrap();
                                let is_new = br.overall.as_ref()
                                    .map_or(true, |(_, c)| final_cost.total < c.total);
                                if is_new {
                                    br.overall = Some((best_a, final_cost.clone()));
                                    br.overall_dirty = true;
                                }
                                let is_new_matchup = br.best_matchup.as_ref()
                                    .map_or(true, |(_, c)| final_cost.matchup_balance < c.matchup_balance);
                                if is_new_matchup {
                                    br.best_matchup = Some((best_a, final_cost));
                                    br.matchup_dirty = true;
                                }
                            }
                            a = random_assignment(&mut rng);
                            cost = evaluate(&a, &w8);
                            best_a = a;
                            best_cost = cost.total;
                            temp = t0;
                            cool_offset = i;
                        }

                        let rand_val: f64 = rng.random();

                        if rand_val < 0.4 {
                            let w = rng.random_range(0..WEEKS);
                            let q1 = rng.random_range(0..QUADS);
                            let mut q2 = rng.random_range(0..(QUADS - 1));
                            if q2 >= q1 { q2 += 1; }
                            let p1 = rng.random_range(0..POS);
                            let p2 = rng.random_range(0..POS);
                            let tmp = a[w][q1][p1];
                            a[w][q1][p1] = a[w][q2][p2];
                            a[w][q2][p2] = tmp;

                            let new_cost = evaluate(&a, &w8);
                            let delta = new_cost.total as i64 - cost.total as i64;
                            if delta <= 0 || rng.random::<f64>() < (-delta as f64 / temp).exp() {
                                cost = new_cost;
                                if cost.total < best_cost { best_cost = cost.total; best_a = a; }
                            } else {
                                a[w][q2][p2] = a[w][q1][p1];
                                a[w][q1][p1] = tmp;
                            }
                        } else if rand_val < 0.65 {
                            let w = rng.random_range(0..WEEKS);
                            let q = rng.random_range(0..QUADS);
                            let p1 = rng.random_range(0..POS);
                            let mut p2 = rng.random_range(0..(POS - 1));
                            if p2 >= p1 { p2 += 1; }
                            let tmp = a[w][q][p1];
                            a[w][q][p1] = a[w][q][p2];
                            a[w][q][p2] = tmp;

                            let new_cost = evaluate(&a, &w8);
                            let delta = new_cost.total as i64 - cost.total as i64;
                            if delta <= 0 || rng.random::<f64>() < (-delta as f64 / temp).exp() {
                                cost = new_cost;
                                if cost.total < best_cost { best_cost = cost.total; best_a = a; }
                            } else {
                                a[w][q][p2] = a[w][q][p1];
                                a[w][q][p1] = tmp;
                            }
                        } else if rand_val < 0.85 {
                            let team = rng.random_range(0..TEAMS) as u8;
                            let w1 = rng.random_range(0..WEEKS);
                            let mut w2 = rng.random_range(0..(WEEKS - 1));
                            if w2 >= w1 { w2 += 1; }

                            let mut qi1: Option<usize> = None;
                            let mut pi1: Option<usize> = None;
                            let mut qi2: Option<usize> = None;
                            let mut pi2: Option<usize> = None;

                            for q in 0..QUADS {
                                for p in 0..POS {
                                    if a[w1][q][p] == team && qi1.is_none() { qi1 = Some(q); pi1 = Some(p); }
                                    if a[w2][q][p] == team && qi2.is_none() { qi2 = Some(q); pi2 = Some(p); }
                                }
                            }

                            if let (Some(qi1), Some(pi1), Some(qi2), Some(pi2)) = (qi1, pi1, qi2, pi2) {
                                let other1 = a[w2][qi1][pi1];
                                let other2 = a[w1][qi2][pi2];
                                let save = (a[w1][qi1][pi1], a[w1][qi2][pi2], a[w2][qi1][pi1], a[w2][qi2][pi2]);

                                a[w1][qi1][pi1] = other2;
                                a[w1][qi2][pi2] = team;
                                a[w2][qi2][pi2] = other1;
                                a[w2][qi1][pi1] = team;

                                let new_cost = evaluate(&a, &w8);
                                let delta = new_cost.total as i64 - cost.total as i64;
                                if delta <= 0 || rng.random::<f64>() < (-delta as f64 / temp).exp() {
                                    cost = new_cost;
                                    if cost.total < best_cost { best_cost = cost.total; best_a = a; }
                                } else {
                                    a[w1][qi1][pi1] = save.0;
                                    a[w1][qi2][pi2] = save.1;
                                    a[w2][qi1][pi1] = save.2;
                                    a[w2][qi2][pi2] = save.3;
                                }
                            }
                        } else {
                            let w = rng.random_range(0..WEEKS);
                            let q1 = rng.random_range(0..QUADS);
                            let mut q2 = rng.random_range(0..(QUADS - 1));
                            if q2 >= q1 { q2 += 1; }
                            let tmp = a[w][q1];
                            a[w][q1] = a[w][q2];
                            a[w][q2] = tmp;

                            let new_cost = evaluate(&a, &w8);
                            let delta = new_cost.total as i64 - cost.total as i64;
                            if delta <= 0 || rng.random::<f64>() < (-delta as f64 / temp).exp() {
                                cost = new_cost;
                                if cost.total < best_cost { best_cost = cost.total; best_a = a; }
                            } else {
                                a[w][q2] = a[w][q1];
                                a[w][q1] = tmp;
                            }
                        }

                        if i > 0 && i % restart_interval == 0 && cost.total > best_cost {
                            if i % (restart_interval * 2) == 0 {
                                a = random_assignment(&mut rng);
                            } else {
                                a = best_a;
                                perturb(&mut a, &mut rng, 20);
                            }
                            cost = evaluate(&a, &w8);
                        }

                        temp = t0 * (cool_rate * (i - cool_offset) as f64).exp();

                        if i > 0 && i % progress_interval == 0 {
                            let label = if i >= 1_000_000_000 {
                                format!("{:.2}B", i as f64 / 1_000_000_000.0)
                            } else {
                                format!("{}M", i / 1_000_000)
                            };
                            let best_breakdown = evaluate(&best_a, &w8);
                            eprintln!(
                                "[{}] core {} @ {} | best: {} | temp: {:.2}",
                                now_iso(), core_id, label, cost_label(&best_breakdown), temp,
                            );
                        }

                        if i > 0 && i % sync_interval == 0 {
                            let (lock, cvar) = &*sync_pair;
                            let mut sync = lock.lock().unwrap();

                            sync.slots[core_id].best_cost = best_cost;
                            sync.slots[core_id].best_assignment = best_a;
                            sync.slots[core_id].reset_to = None;
                            sync.slots[core_id].reset_temp = None;
                            sync.checked_in += 1;

                            if sync.checked_in == num_cores {
                                let mut gb_cost = sync.global_best_cost;
                                let mut gb_assign = sync.global_best_assignment;
                                for slot in &sync.slots {
                                    if slot.best_cost < gb_cost {
                                        gb_cost = slot.best_cost;
                                        gb_assign = slot.best_assignment;
                                    }
                                }
                                sync.global_best_cost = gb_cost;
                                sync.global_best_assignment = gb_assign;

                                let improved = gb_cost < sync.prev_global_best_cost;
                                if improved {
                                    sync.stagnation_epochs = 0;
                                    sync.prev_global_best_cost = gb_cost;
                                } else {
                                    sync.stagnation_epochs += 1;
                                }

                                let generational_restart = sync.stagnation_epochs >= stagnation_threshold;

                                let global_cost = evaluate(&sync.global_best_assignment, &w8);
                                {
                                    let mut br = best_results.lock().unwrap();
                                    let is_new = br.overall.as_ref()
                                        .map_or(true, |(_, c)| global_cost.total < c.total);
                                    if is_new {
                                        br.overall = Some((sync.global_best_assignment, global_cost.clone()));
                                        br.overall_dirty = true;
                                    }
                                    let is_new_matchup = br.best_matchup.as_ref()
                                        .map_or(true, |(_, c)| global_cost.matchup_balance < c.matchup_balance);
                                    if is_new_matchup {
                                        br.best_matchup = Some((sync.global_best_assignment, global_cost.clone()));
                                        br.matchup_dirty = true;
                                    }
                                }

                                save_assignment(
                                    &sync.generation_dir.clone(), "sync",
                                    global_cost.total,
                                    &sync.global_best_assignment,
                                    &global_cost,
                                    &mut last_sync,
                                );

                                if generational_restart {
                                    let mut gen_rng = SmallRng::seed_from_u64(sync.epoch.wrapping_mul(97));
                                    for slot in &mut sync.slots {
                                        slot.reset_to = Some(random_assignment(&mut gen_rng));
                                        slot.reset_temp = Some(t0);
                                    }

                                    eprintln!(
                                        "[{}] GENERATION COMPLETE epoch {} | best: {} ({}) | stagnant {} epochs | restarting all cores",
                                        now_iso(),
                                        sync.epoch,
                                        sync.global_best_cost,
                                        cost_label(&global_cost),
                                        sync.stagnation_epochs,
                                    );

                                    let cdir = complete_dir(&base_dir);
                                    save_assignment(
                                        &cdir, "gen-best",
                                        global_cost.total,
                                        &sync.global_best_assignment,
                                        &global_cost,
                                        &mut last_gen_best,
                                    );

                                    sync.generation_dir = new_generation_dir(&base_dir);
                                    eprintln!("[{}] New generation output: {}", now_iso(), sync.generation_dir);
                                    last_sync = None;

                                    sync.stagnation_epochs = 0;
                                    sync.global_best_cost = u32::MAX;
                                    sync.prev_global_best_cost = u32::MAX;
                                    sync.global_best_assignment = [[[0u8; POS]; QUADS]; WEEKS];
                                } else {
                                    let mut dedup_resets: Vec<usize> = Vec::new();
                                    for ci in 0..num_cores {
                                        if dedup_resets.contains(&ci) { continue; }
                                        for cj in (ci + 1)..num_cores {
                                            if dedup_resets.contains(&cj) { continue; }
                                            if sync.slots[ci].best_assignment == sync.slots[cj].best_assignment {
                                                dedup_resets.push(cj);
                                            }
                                        }
                                    }

                                    let mut ranked: Vec<(usize, u32)> = sync.slots.iter()
                                        .enumerate()
                                        .map(|(id, s)| (id, s.best_cost))
                                        .collect();
                                    ranked.sort_by(|a, b| b.1.cmp(&a.1));

                                    let num_reset = num_cores / 2;
                                    let mut reset_set: Vec<usize> = Vec::new();
                                    let mut reset_info: Vec<(usize, usize, &str)> = Vec::new();

                                    for (rank, &(cid, _)) in ranked.iter().enumerate() {
                                        if rank >= num_reset { break; }
                                        if !reset_set.contains(&cid) {
                                            reset_set.push(cid);
                                        }
                                    }

                                    for &cid in &dedup_resets {
                                        if !reset_set.contains(&cid) {
                                            reset_set.push(cid);
                                        }
                                    }

                                    // Escalate temperature with stagnation: t0 at low stagnation, up to 2*t0 near threshold
                                    let stag_ratio = sync.stagnation_epochs as f64 / stagnation_threshold as f64;
                                    let reset_temp_val = t0 * (1.0 + stag_ratio);

                                    let max_perturb = if reset_set.len() <= 1 { 0 } else { reset_set.len() - 1 };
                                    for (rank, &cid) in reset_set.iter().enumerate() {
                                        let perturbations = if max_perturb == 0 {
                                            0
                                        } else {
                                            rank * 100 / max_perturb
                                        };
                                        let mut new_a = sync.global_best_assignment;
                                        if perturbations > 0 {
                                            let seed = (cid as u64).wrapping_mul(31) ^ sync.epoch;
                                            let mut prng = SmallRng::seed_from_u64(seed);
                                            perturb(&mut new_a, &mut prng, perturbations);
                                        }
                                        sync.slots[cid].reset_to = Some(new_a);
                                        sync.slots[cid].reset_temp = Some(reset_temp_val);
                                        let reason = if dedup_resets.contains(&cid) { "dedup" } else { "worst50" };
                                        reset_info.push((cid, perturbations, reason));
                                    }

                                    let reset_desc: Vec<String> = reset_info.iter()
                                        .map(|(cid, p, reason)| format!("core{}({}perturb,{})", cid, p, reason))
                                        .collect();
                                    eprintln!(
                                        "[{}] SYNC epoch {} | global best: {} ({}) | stagnation: {}/{} | reset_temp: {:.1} | reset: [{}]",
                                        now_iso(),
                                        sync.epoch,
                                        sync.global_best_cost,
                                        cost_label(&global_cost),
                                        sync.stagnation_epochs,
                                        stagnation_threshold,
                                        reset_temp_val,
                                        reset_desc.join(", "),
                                    );
                                }

                                sync.epoch += 1;
                                sync.checked_in = 0;
                                cvar.notify_all();
                            } else {
                                let epoch_before = sync.epoch;
                                while sync.epoch == epoch_before {
                                    sync = cvar.wait(sync).unwrap();
                                }
                            }

                            let my_reset = sync.slots[core_id].reset_to;
                            let my_reset_temp = sync.slots[core_id].reset_temp;
                            drop(sync);

                            if let Some(new_a) = my_reset {
                                a = new_a;
                                cost = evaluate(&a, &w8);
                                best_a = new_a;
                                best_cost = cost.total;
                                temp = my_reset_temp.unwrap_or(t0);
                                cool_offset = i;
                            }
                        }
                    }

                    let final_cost = evaluate(&best_a, &w8);
                    {
                        let mut br = best_results.lock().unwrap();
                        let is_new_overall = br.overall.as_ref()
                            .map_or(true, |(_, c)| final_cost.total < c.total);
                        if is_new_overall {
                            eprintln!(
                                "[{}] core {} new best overall: {}",
                                now_iso(), core_id, cost_label(&final_cost),
                            );
                            br.overall = Some((best_a, final_cost.clone()));
                            br.overall_dirty = true;
                        }

                        let is_new_matchup = br.best_matchup.as_ref()
                            .map_or(true, |(_, c)| final_cost.matchup_balance < c.matchup_balance);
                        if is_new_matchup {
                            eprintln!(
                                "[{}] core {} new best matchup: {}",
                                now_iso(), core_id, cost_label(&final_cost),
                            );
                            br.best_matchup = Some((best_a, final_cost));
                            br.matchup_dirty = true;
                        }
                    }
                }
            })
        })
        .collect();

    for h in handles {
        h.join().unwrap();
    }
}
