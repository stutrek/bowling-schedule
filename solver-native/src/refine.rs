use rand::rngs::SmallRng;
use rand::{Rng, SeedableRng};
use std::fs;
use std::sync::Arc;
use std::thread;

use solver_native::*;

fn main() {
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 3 {
        eprintln!("Usage: refine <weights.json> <file1.tsv> [file2.tsv ...]");
        std::process::exit(1);
    }

    let weights_path = &args[1];
    let weights_str = fs::read_to_string(weights_path)
        .unwrap_or_else(|e| panic!("Failed to read {}: {}", weights_path, e));
    let loaded_weights: Weights = serde_json::from_str(&weights_str)
        .unwrap_or_else(|e| panic!("Failed to parse {}: {}", weights_path, e));
    let w8 = Arc::new(loaded_weights);

    let out_dir = "results/refined";
    fs::create_dir_all(out_dir).expect("Failed to create results/refined directory");

    let mut inputs: Vec<(String, Assignment)> = Vec::new();
    for path in &args[2..] {
        let content = match fs::read_to_string(path) {
            Ok(c) => c,
            Err(e) => { eprintln!("Warning: could not read {}: {}", path, e); continue; }
        };
        let a = match parse_tsv(&content) {
            Some(a) => a,
            None => { eprintln!("Warning: could not parse {}", path); continue; }
        };
        let cost = evaluate(&a, &w8);
        eprintln!("[{}] Loaded: {} ({})", now_iso(), path, cost_label(&cost));
        inputs.push((path.clone(), a));
    }

    let handles: Vec<_> = inputs.into_iter().enumerate().map(|(idx, (path, a))| {
        let w8 = Arc::clone(&w8);
        let original_cost = evaluate(&a, &w8).total;
        let tag = format!("core{} (start: {})", idx, original_cost);
        thread::spawn(move || {
            let refined = refine(a, &w8, &tag);
            let final_cost = evaluate(&refined, &w8);
            eprintln!("[{}] {} | Refined: {} -> ({})", now_iso(), tag, path, cost_label(&final_cost));
            let mut dummy: Option<Assignment> = None;
            let prefix = format!("refined-core{}", idx);
            save_assignment("results/refined", &prefix, final_cost.total, &refined, &final_cost, &mut dummy);
        })
    }).collect();

    for h in handles {
        h.join().expect("Refine thread panicked");
    }
}

fn refine(mut a: Assignment, w8: &Weights, tag: &str) -> Assignment {
    let mut cost = evaluate(&a, w8);
    eprintln!("[{}] {} | Starting refinement: {}", now_iso(), tag, cost_label(&cost));

    // Phase 1: Exhaustive orthogonal hill-climb
    let mut improved = true;
    let mut pass = 0u32;
    while improved {
        pass += 1;
        improved = false;
        let before = cost.total;

        // Try all week swaps (Move A)
        for w1 in 0..WEEKS {
            for w2 in (w1 + 1)..WEEKS {
                let tmp = a[w1];
                a[w1] = a[w2];
                a[w2] = tmp;
                let nc = evaluate(&a, w8);
                if nc.total < cost.total {
                    cost = nc;
                    improved = true;
                } else {
                    a[w2] = a[w1];
                    a[w1] = tmp;
                }
            }
        }

        // Try all early/late flips (Move B)
        for w in 0..WEEKS {
            let tmp0 = a[w][0]; let tmp1 = a[w][1];
            a[w][0] = a[w][2]; a[w][2] = tmp0;
            a[w][1] = a[w][3]; a[w][3] = tmp1;
            let nc = evaluate(&a, w8);
            if nc.total < cost.total {
                cost = nc;
                improved = true;
            } else {
                let tmp0 = a[w][0]; let tmp1 = a[w][1];
                a[w][0] = a[w][2]; a[w][2] = tmp0;
                a[w][1] = a[w][3]; a[w][3] = tmp1;
            }
        }

        // Try all lane pair swaps (Move C)
        for w in 0..WEEKS {
            let tmp0 = a[w][0]; let tmp2 = a[w][2];
            a[w][0] = a[w][1]; a[w][1] = tmp0;
            a[w][2] = a[w][3]; a[w][3] = tmp2;
            let nc = evaluate(&a, w8);
            if nc.total < cost.total {
                cost = nc;
                improved = true;
            } else {
                let tmp1 = a[w][1]; let tmp3 = a[w][3];
                a[w][1] = a[w][0]; a[w][0] = tmp1;
                a[w][3] = a[w][2]; a[w][2] = tmp3;
            }
        }

        // Try all stay/switch rotations (Move D)
        for w in 0..WEEKS {
            for q in 0..QUADS {
                a[w][q].swap(0, 1);
                a[w][q].swap(2, 3);
                let nc = evaluate(&a, w8);
                if nc.total < cost.total {
                    cost = nc;
                    improved = true;
                } else {
                    a[w][q].swap(0, 1);
                    a[w][q].swap(2, 3);
                }
            }
        }

        eprintln!(
            "[{}] {} | Orthogonal pass {} | {} -> {} ({})",
            now_iso(), tag, pass, before, cost.total, cost_label(&cost),
        );
    }

    // Phase 2: Exhaustive two-week pair-swap search (all 66 week-pairs)
    if cost.matchup_balance > 0 {
        improved = true;
        let mut epass = 0u32;
        while improved {
            epass += 1;
            improved = false;
            let before = cost.total;

            for w1 in 0..WEEKS {
                for w2 in (w1 + 1)..WEEKS {
                    let mut best_delta = 0i64;
                    let mut best_swap: Option<(usize, usize, usize, usize)> = None;
                    for q1 in 0..QUADS {
                        for p1 in 0..POS {
                            for q2 in 0..QUADS {
                                for p2 in 0..POS {
                                    let t1 = a[w1][q1][p1];
                                    let t2 = a[w2][q2][p2];
                                    if t1 == t2 { continue; }
                                    a[w1][q1][p1] = t2;
                                    a[w2][q2][p2] = t1;
                                    let nc = evaluate(&a, w8);
                                    let d = nc.total as i64 - cost.total as i64;
                                    if d < best_delta {
                                        best_delta = d;
                                        best_swap = Some((q1, p1, q2, p2));
                                    }
                                    a[w1][q1][p1] = t1;
                                    a[w2][q2][p2] = t2;
                                }
                            }
                        }
                    }
                    if let Some((q1, p1, q2, p2)) = best_swap {
                        let tmp = a[w1][q1][p1];
                        a[w1][q1][p1] = a[w2][q2][p2];
                        a[w2][q2][p2] = tmp;
                        cost = evaluate(&a, w8);
                        improved = true;
                    }
                }
            }

            eprintln!(
                "[{}] {} | Exhaustive pass {} | {} -> {} ({})",
                now_iso(), tag, epass, before, cost.total, cost_label(&cost),
            );
        }
    }

    // Phase 3: Mini-SA with temperature floor for escaping shallow local minima
    let mut rng = SmallRng::from_os_rng();
    let t0: f64 = 5.0;
    let temp_floor: f64 = 0.5;
    let iterations: u64 = 100_000_000;
    let cool_rate: f64 = (0.005_f64 / t0).ln() / iterations as f64;
    let mut best_a = a;
    let mut best_cost = cost.total;
    eprintln!("[{}] {} | Starting mini-SA: {}M iterations, temp floor {}", now_iso(), tag, iterations / 1_000_000, temp_floor);

    for i in 0..iterations {
        let temp = (t0 * (cool_rate * i as f64).exp()).max(temp_floor);

        let rand_val: f64 = rng.random();

        if rand_val < 0.25 {
            // Inter-quad player swap
            let w = rng.random_range(0..WEEKS);
            let q1 = rng.random_range(0..QUADS);
            let mut q2 = rng.random_range(0..(QUADS - 1));
            if q2 >= q1 { q2 += 1; }
            let p1 = rng.random_range(0..POS);
            let p2 = rng.random_range(0..POS);
            let tmp = a[w][q1][p1];
            a[w][q1][p1] = a[w][q2][p2];
            a[w][q2][p2] = tmp;

            let nc = evaluate(&a, w8);
            let delta = nc.total as i64 - cost.total as i64;
            if delta <= 0 || rng.random::<f64>() < (-delta as f64 / temp).exp() {
                cost = nc;
                if cost.total < best_cost { best_cost = cost.total; best_a = a; }
            } else {
                a[w][q2][p2] = a[w][q1][p1];
                a[w][q1][p1] = tmp;
            }
        } else if rand_val < 0.40 {
            // Week swap (A)
            let w1 = rng.random_range(0..WEEKS);
            let mut w2 = rng.random_range(0..(WEEKS - 1));
            if w2 >= w1 { w2 += 1; }
            let tmp = a[w1];
            a[w1] = a[w2];
            a[w2] = tmp;

            let nc = evaluate(&a, w8);
            let delta = nc.total as i64 - cost.total as i64;
            if delta <= 0 || rng.random::<f64>() < (-delta as f64 / temp).exp() {
                cost = nc;
                if cost.total < best_cost { best_cost = cost.total; best_a = a; }
            } else {
                a[w2] = a[w1];
                a[w1] = tmp;
            }
        } else if rand_val < 0.55 {
            // Early/late flip (B)
            let w = rng.random_range(0..WEEKS);
            let tmp0 = a[w][0]; let tmp1 = a[w][1];
            a[w][0] = a[w][2]; a[w][2] = tmp0;
            a[w][1] = a[w][3]; a[w][3] = tmp1;

            let nc = evaluate(&a, w8);
            let delta = nc.total as i64 - cost.total as i64;
            if delta <= 0 || rng.random::<f64>() < (-delta as f64 / temp).exp() {
                cost = nc;
                if cost.total < best_cost { best_cost = cost.total; best_a = a; }
            } else {
                let tmp0 = a[w][0]; let tmp1 = a[w][1];
                a[w][0] = a[w][2]; a[w][2] = tmp0;
                a[w][1] = a[w][3]; a[w][3] = tmp1;
            }
        } else if rand_val < 0.70 {
            // Stay/switch rotation (D)
            let w = rng.random_range(0..WEEKS);
            let q = rng.random_range(0..QUADS);
            a[w][q].swap(0, 1);
            a[w][q].swap(2, 3);

            let nc = evaluate(&a, w8);
            let delta = nc.total as i64 - cost.total as i64;
            if delta <= 0 || rng.random::<f64>() < (-delta as f64 / temp).exp() {
                cost = nc;
                if cost.total < best_cost { best_cost = cost.total; best_a = a; }
            } else {
                a[w][q].swap(0, 1);
                a[w][q].swap(2, 3);
            }
        } else if rand_val < 0.80 {
            // Lane pair swap (C)
            let w = rng.random_range(0..WEEKS);
            let tmp0 = a[w][0]; let tmp2 = a[w][2];
            a[w][0] = a[w][1]; a[w][1] = tmp0;
            a[w][2] = a[w][3]; a[w][3] = tmp2;

            let nc = evaluate(&a, w8);
            let delta = nc.total as i64 - cost.total as i64;
            if delta <= 0 || rng.random::<f64>() < (-delta as f64 / temp).exp() {
                cost = nc;
                if cost.total < best_cost { best_cost = cost.total; best_a = a; }
            } else {
                let tmp1 = a[w][1]; let tmp3 = a[w][3];
                a[w][1] = a[w][0]; a[w][0] = tmp1;
                a[w][3] = a[w][2]; a[w][2] = tmp3;
            }
        } else {
            // Compound move (2-4 swaps)
            let saved = a;
            let num_swaps = rng.random_range(2..=4u32);
            for _ in 0..num_swaps {
                let w = rng.random_range(0..WEEKS);
                let q1 = rng.random_range(0..QUADS);
                let mut q2 = rng.random_range(0..(QUADS - 1));
                if q2 >= q1 { q2 += 1; }
                let p1 = rng.random_range(0..POS);
                let p2 = rng.random_range(0..POS);
                let tmp = a[w][q1][p1];
                a[w][q1][p1] = a[w][q2][p2];
                a[w][q2][p2] = tmp;
            }
            let nc = evaluate(&a, w8);
            let delta = nc.total as i64 - cost.total as i64;
            if delta <= 0 || rng.random::<f64>() < (-delta as f64 / temp).exp() {
                cost = nc;
                if cost.total < best_cost { best_cost = cost.total; best_a = a; }
            } else {
                a = saved;
            }
        }

        if i > 0 && i % 10_000_000 == 0 {
            let label = if i >= 1_000_000_000 {
                format!("{:.2}B", i as f64 / 1e9)
            } else {
                format!("{}M", i / 1_000_000)
            };
            eprintln!(
                "[{}] {} | mini-SA @ {} | best: {} | current: {} | temp: {:.2}",
                now_iso(), tag, label, cost_label(&evaluate(&best_a, w8)), cost.total, temp,
            );
        }
    }

    // Final orthogonal hill-climb on the SA result
    a = best_a;
    cost = evaluate(&a, w8);
    improved = true;
    while improved {
        improved = false;

        for w1 in 0..WEEKS {
            for w2 in (w1 + 1)..WEEKS {
                let tmp = a[w1]; a[w1] = a[w2]; a[w2] = tmp;
                let nc = evaluate(&a, w8);
                if nc.total < cost.total { cost = nc; improved = true; }
                else { a[w2] = a[w1]; a[w1] = tmp; }
            }
        }
        for w in 0..WEEKS {
            let tmp0 = a[w][0]; let tmp1 = a[w][1];
            a[w][0] = a[w][2]; a[w][2] = tmp0;
            a[w][1] = a[w][3]; a[w][3] = tmp1;
            let nc = evaluate(&a, w8);
            if nc.total < cost.total { cost = nc; improved = true; }
            else {
                let tmp0 = a[w][0]; let tmp1 = a[w][1];
                a[w][0] = a[w][2]; a[w][2] = tmp0;
                a[w][1] = a[w][3]; a[w][3] = tmp1;
            }
        }
        for w in 0..WEEKS {
            let tmp0 = a[w][0]; let tmp2 = a[w][2];
            a[w][0] = a[w][1]; a[w][1] = tmp0;
            a[w][2] = a[w][3]; a[w][3] = tmp2;
            let nc = evaluate(&a, w8);
            if nc.total < cost.total { cost = nc; improved = true; }
            else {
                let tmp1 = a[w][1]; let tmp3 = a[w][3];
                a[w][1] = a[w][0]; a[w][0] = tmp1;
                a[w][3] = a[w][2]; a[w][2] = tmp3;
            }
        }
        for w in 0..WEEKS {
            for q in 0..QUADS {
                a[w][q].swap(0, 1); a[w][q].swap(2, 3);
                let nc = evaluate(&a, w8);
                if nc.total < cost.total { cost = nc; improved = true; }
                else { a[w][q].swap(0, 1); a[w][q].swap(2, 3); }
            }
        }
    }
    eprintln!("[{}] {} | Final polish: {}", now_iso(), tag, cost_label(&cost));

    a
}
