use chrono::Local;
use rand::rngs::SmallRng;
use rand::{Rng, SeedableRng};
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

#[derive(Clone)]
struct CostBreakdown {
    matchup_balance: u32,
    consecutive_opponents: u32,
    early_late_balance: u32,
    early_late_alternation: u32,
    lane_balance: u32,
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
    /// Written by sync logic when this core should be reset
    reset_to: Option<Assignment>,
}

struct SyncState {
    slots: Vec<CoreSlot>,
    checked_in: usize,
    epoch: u64,
    global_best_cost: u32,
    global_best_assignment: Assignment,
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

fn evaluate(a: &Assignment) -> CostBreakdown {
    let mut matchups = [0i32; TEAMS * TEAMS];
    let mut week_matchup = [0u8; WEEKS * TEAMS * TEAMS];
    let mut lane_counts = [0i32; TEAMS * LANES];
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
                matchup_balance += 30;
            } else if c >= 3 {
                matchup_balance += (c - 2) as u32 * 40;
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
                    consecutive_opponents += 10;
                }
            }
        }
    }

    let mut early_late_balance: u32 = 0;
    let target_e: f64 = WEEKS as f64 / 2.0;
    for t in 0..TEAMS {
        let dev = (early_count[t] as f64 - target_e).abs();
        early_late_balance += (dev * dev * 20.0) as u32;
    }

    let mut early_late_alternation: u32 = 0;
    for t in 0..TEAMS {
        for w in 0..(WEEKS - 2) {
            let base = t * WEEKS;
            if early_late[base + w] == early_late[base + w + 1]
                && early_late[base + w + 1] == early_late[base + w + 2]
            {
                early_late_alternation += 25;
            }
        }
    }

    let mut lane_balance: u32 = 0;
    let target_l: f64 = (WEEKS as f64 * 2.0) / LANES as f64;
    for t in 0..TEAMS {
        for l in 0..LANES {
            lane_balance +=
                ((lane_counts[t * LANES + l] as f64 - target_l).abs() * 15.0) as u32;
        }
    }

    let total = matchup_balance
        + consecutive_opponents
        + early_late_balance
        + early_late_alternation
        + lane_balance;

    CostBreakdown {
        matchup_balance,
        consecutive_opponents,
        early_late_balance,
        early_late_alternation,
        lane_balance,
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
        "total:{} matchup:{} consec:{} el_bal:{} el_alt:{} lane:{}",
        c.total, c.matchup_balance, c.consecutive_opponents,
        c.early_late_balance, c.early_late_alternation, c.lane_balance,
    )
}

fn now_iso() -> String {
    Local::now().format("%Y-%m-%dT%H:%M:%S%:z").to_string()
}

fn save_assignment(results_dir: &str, prefix: &str, key: u32, a: &Assignment, c: &CostBreakdown) {
    let ts = Local::now().format("%Y-%m-%dT%H%M%S%z");
    let filename = format!("{}/{}-{}-{}.tsv", results_dir, prefix, key, ts);
    let tsv = assignment_to_tsv(a);
    let _ = fs::write(&filename, &tsv);
    eprintln!("[{}] Saved {}: {} ({})", now_iso(), prefix, filename, cost_label(c));
}

fn main() {
    let results_dir = "results";
    fs::create_dir_all(results_dir).expect("Failed to create results directory");

    let num_cores = thread::available_parallelism()
        .map(|n| n.get())
        .unwrap_or(4);
    let max_iterations: u64 = 6_000_000_000;
    let progress_interval: u64 = 10_000_000;
    let sync_interval: u64 = 50_000_000;
    let restart_interval: u64 = 100_000;

    let dummy_assignment: Assignment = [[[0u8; POS]; QUADS]; WEEKS];

    let sync_pair = Arc::new((
        Mutex::new(SyncState {
            slots: (0..num_cores)
                .map(|_| CoreSlot {
                    best_cost: u32::MAX,
                    best_assignment: dummy_assignment,
                    reset_to: None,
                })
                .collect(),
            checked_in: 0,
            epoch: 0,
            global_best_cost: u32::MAX,
            global_best_assignment: dummy_assignment,
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
        "[{}] Starting solver: {} cores, {}B iterations/run, sync every {}M. Ctrl+C to stop.",
        now_iso(), num_cores, max_iterations / 1_000_000_000, sync_interval / 1_000_000,
    );

    // Saver thread: writes dirty results to disk every second
    let saver_best = Arc::clone(&best_results);
    let saver_dir = results_dir.to_string();
    thread::spawn(move || {
        loop {
            thread::sleep(Duration::from_secs(1));
            let mut state = saver_best.lock().unwrap();

            if state.overall_dirty {
                if let Some((ref a, ref c)) = state.overall {
                    save_assignment(&saver_dir, "best-overall", c.total, a, c);
                }
                state.overall_dirty = false;
            }

            if state.matchup_dirty {
                if let Some((ref a, ref c)) = state.best_matchup {
                    save_assignment(&saver_dir, "best-matchup", c.matchup_balance, a, c);
                }
                state.matchup_dirty = false;
            }
        }
    });

    let handles: Vec<_> = (0..num_cores)
        .map(|core_id| {
            let sync_pair = Arc::clone(&sync_pair);
            let best_results = Arc::clone(&best_results);
            let results_dir = results_dir.to_string();
            thread::spawn(move || {
                let mut rng = SmallRng::from_os_rng();
                let t0: f64 = 30.0;

                loop {
                    let cool_rate: f64 = (0.005_f64 / t0).ln() / max_iterations as f64;
                    let mut a = random_assignment(&mut rng);
                    let mut cost = evaluate(&a);
                    let mut best_a = a;
                    let mut best_cost = cost.total;
                    let mut temp: f64 = t0;

                    for i in 0..max_iterations {
                        if best_cost == 0 {
                            let final_cost = evaluate(&best_a);
                            save_assignment(&results_dir, "perfect", 0, &best_a, &final_cost);
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
                            cost = evaluate(&a);
                            best_a = a;
                            best_cost = cost.total;
                            temp = t0;
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

                            let new_cost = evaluate(&a);
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

                            let new_cost = evaluate(&a);
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

                                let new_cost = evaluate(&a);
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

                            let new_cost = evaluate(&a);
                            let delta = new_cost.total as i64 - cost.total as i64;
                            if delta <= 0 || rng.random::<f64>() < (-delta as f64 / temp).exp() {
                                cost = new_cost;
                                if cost.total < best_cost { best_cost = cost.total; best_a = a; }
                            } else {
                                a[w][q2] = a[w][q1];
                                a[w][q1] = tmp;
                            }
                        }

                        temp = t0 * (cool_rate * i as f64).exp();

                        if i > 0 && i % restart_interval == 0 && cost.total > best_cost {
                            if i % (restart_interval * 2) == 0 {
                                a = random_assignment(&mut rng);
                            } else {
                                a = best_a;
                                perturb(&mut a, &mut rng, 20);
                            }
                            cost = evaluate(&a);
                            temp = t0 * 0.3;
                        }

                        if i > 0 && i % progress_interval == 0 {
                            let label = if i >= 1_000_000_000 {
                                format!("{:.2}B", i as f64 / 1_000_000_000.0)
                            } else {
                                format!("{}M", i / 1_000_000)
                            };
                            eprintln!(
                                "[{}] core {} @ {} iter, run best: {}",
                                now_iso(), core_id, label, best_cost,
                            );
                        }

                        // Soft sync checkpoint
                        if i > 0 && i % sync_interval == 0 {
                            let (lock, cvar) = &*sync_pair;
                            let mut sync = lock.lock().unwrap();

                            // Write our state into our slot
                            sync.slots[core_id].best_cost = best_cost;
                            sync.slots[core_id].best_assignment = best_a;
                            sync.slots[core_id].reset_to = None;
                            sync.checked_in += 1;

                            if sync.checked_in == num_cores {
                                // Last core to arrive — perform the sync

                                // Update global best from all cores
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

                                // Sort cores by cost (worst first)
                                let mut ranked: Vec<(usize, u32)> = sync.slots.iter()
                                    .enumerate()
                                    .map(|(id, s)| (id, s.best_cost))
                                    .collect();
                                ranked.sort_by(|a, b| b.1.cmp(&a.1));

                                let num_reset = num_cores / 2;
                                let mut reset_info: Vec<(usize, usize)> = Vec::new();

                                for (rank, &(cid, _)) in ranked.iter().enumerate() {
                                    if rank >= num_reset {
                                        break;
                                    }
                                    // Graduated perturbations: worst gets 0 (exact copy),
                                    // scaling up to 100 for the least-worst of the reset group
                                    let perturbations = if num_reset <= 1 {
                                        0
                                    } else {
                                        rank * 100 / (num_reset - 1)
                                    };
                                    let mut new_a = sync.global_best_assignment;
                                    if perturbations > 0 {
                                        // Use a quick deterministic-ish rng seeded from core id + epoch
                                        let seed = (cid as u64).wrapping_mul(31) ^ sync.epoch;
                                        let mut prng = SmallRng::seed_from_u64(seed);
                                        perturb(&mut new_a, &mut prng, perturbations);
                                    }
                                    sync.slots[cid].reset_to = Some(new_a);
                                    reset_info.push((cid, perturbations));
                                }

                                // Save global best at sync time
                                let global_cost = evaluate(&sync.global_best_assignment);
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

                                // Also save a sync-point snapshot directly
                                save_assignment(
                                    &results_dir, "sync",
                                    global_cost.total,
                                    &sync.global_best_assignment,
                                    &global_cost,
                                );

                                // Log the sync
                                let reset_desc: Vec<String> = reset_info.iter()
                                    .map(|(cid, p)| format!("core{}({}perturb)", cid, p))
                                    .collect();
                                eprintln!(
                                    "[{}] SYNC epoch {} | global best: {} | reset: [{}]",
                                    now_iso(),
                                    sync.epoch,
                                    sync.global_best_cost,
                                    reset_desc.join(", "),
                                );

                                sync.epoch += 1;
                                sync.checked_in = 0;
                                cvar.notify_all();
                            } else {
                                // Wait for all cores to check in
                                let epoch_before = sync.epoch;
                                while sync.epoch == epoch_before {
                                    sync = cvar.wait(sync).unwrap();
                                }
                            }

                            // Read reset before releasing the lock
                            let my_reset = sync.slots[core_id].reset_to;
                            drop(sync);

                            if let Some(new_a) = my_reset {
                                a = new_a;
                                cost = evaluate(&a);
                                best_a = new_a;
                                best_cost = cost.total;
                                temp = t0 * 0.3;
                            }
                        }
                    }

                    // Run complete — update shared bests
                    let final_cost = evaluate(&best_a);
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
