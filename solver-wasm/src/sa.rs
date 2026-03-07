use rand::rngs::SmallRng;
use rand::Rng;

use solver_core::winter::*;

const NUM_MOVES: usize = 11;
const BASE_WEIGHTS: [f64; NUM_MOVES] = [
    0.25, 0.15, 0.10, 0.08, 0.06, 0.06, 0.05, 0.06, 0.06, 0.06, 0.07,
];
const STATS_RECOMPUTE: u64 = 10_000;

struct MoveStats {
    attempts: [u64; NUM_MOVES],
    accepts: [u64; NUM_MOVES],
    cumulative: [f64; NUM_MOVES],
}

impl MoveStats {
    fn new() -> Self {
        let mut s = MoveStats {
            attempts: [0; NUM_MOVES],
            accepts: [0; NUM_MOVES],
            cumulative: [0.0; NUM_MOVES],
        };
        s.recompute();
        s
    }

    fn recompute(&mut self) {
        let mut weights = [0.0f64; NUM_MOVES];
        for m in 0..NUM_MOVES {
            let rate = if self.attempts[m] > 0 {
                self.accepts[m] as f64 / self.attempts[m] as f64
            } else {
                0.5
            };
            weights[m] = BASE_WEIGHTS[m] * (0.1 + rate);
        }
        let sum: f64 = weights.iter().sum();
        let mut cum = 0.0;
        for m in 0..NUM_MOVES {
            cum += weights[m] / sum;
            self.cumulative[m] = cum;
        }
        self.cumulative[NUM_MOVES - 1] = 1.0;
        self.attempts = [0; NUM_MOVES];
        self.accepts = [0; NUM_MOVES];
    }

    fn select(&self, rand_val: f64) -> usize {
        for m in 0..NUM_MOVES {
            if rand_val < self.cumulative[m] { return m; }
        }
        NUM_MOVES - 1
    }
}

// Guided move helpers

fn find_team_in_week(a: &Assignment, w: usize, team: u8) -> Option<(usize, usize)> {
    for q in 0..QUADS {
        for p in 0..POS {
            if a[w][q][p] == team { return Some((q, p)); }
        }
    }
    None
}

fn same_half(q1: usize, q2: usize) -> bool {
    (q1 < 2 && q2 < 2) || (q1 >= 2 && q2 >= 2)
}

fn guided_matchup(a: &mut Assignment, rng: &mut SmallRng) -> bool {
    let mut matchups = [false; TEAMS * TEAMS];
    for w in 0..WEEKS {
        for q in 0..QUADS {
            let [pa, pb, pc, pd] = a[w][q];
            for &(t1, t2) in &[(pa, pb), (pc, pd), (pa, pd), (pc, pb)] {
                matchups[t1.min(t2) as usize * TEAMS + t1.max(t2) as usize] = true;
            }
        }
    }

    let start = rng.random_range(0..TEAMS);
    let mut ta = 0u8;
    let mut tb = 0u8;
    let mut found = false;
    'outer: for off_i in 0..TEAMS {
        let i = (start + off_i) % TEAMS;
        for j in (i + 1)..TEAMS {
            if !matchups[i * TEAMS + j] {
                ta = i as u8;
                tb = j as u8;
                found = true;
                break 'outer;
            }
        }
    }
    if !found { return false; }

    let week_start = rng.random_range(0..WEEKS);
    for off in 0..WEEKS {
        let w = (week_start + off) % WEEKS;
        let pos_a = find_team_in_week(a, w, ta);
        let pos_b = find_team_in_week(a, w, tb);
        let (qa, _pa) = match pos_a { Some(x) => x, None => continue };
        let (qb, pb) = match pos_b { Some(x) => x, None => continue };

        if !same_half(qa, qb) || qa == qb { continue; }

        let candidates: Vec<usize> = (0..POS)
            .filter(|&p| a[w][qa][p] != ta)
            .collect();
        if candidates.is_empty() { continue; }
        let ci = rng.random_range(0..candidates.len());
        let swap_pos = candidates[ci];

        let tmp = a[w][qa][swap_pos];
        a[w][qa][swap_pos] = a[w][qb][pb];
        a[w][qb][pb] = tmp;
        return true;
    }
    false
}

fn guided_lane(a: &mut Assignment, w8: &Weights, rng: &mut SmallRng) -> bool {
    let mut lane_counts = [0i32; TEAMS * LANES];
    for w in 0..WEEKS {
        for q in 0..QUADS {
            let [pa, pb, pc, pd] = a[w][q];
            let lo = (q % 2) * 2;
            lane_counts[pa as usize * LANES + lo] += 2;
            lane_counts[pb as usize * LANES + lo] += 1;
            lane_counts[pb as usize * LANES + lo + 1] += 1;
            lane_counts[pc as usize * LANES + lo + 1] += 2;
            lane_counts[pd as usize * LANES + lo + 1] += 1;
            lane_counts[pd as usize * LANES + lo] += 1;
        }
    }

    let target_l = (WEEKS as f64 * 2.0) / LANES as f64;
    let mut worst_team = 0usize;
    let mut worst_dev = 0.0f64;
    for t in 0..TEAMS {
        for l in 0..LANES {
            let dev = (lane_counts[t * LANES + l] as f64 - target_l).abs();
            if dev > worst_dev { worst_dev = dev; worst_team = t; }
        }
    }
    if worst_dev < 1.0 { return false; }

    let start = rng.random_range(0..WEEKS * QUADS);
    for off in 0..(WEEKS * QUADS) {
        let idx = (start + off) % (WEEKS * QUADS);
        let w = idx / QUADS;
        let q = idx % QUADS;
        let mut team_pos = None;
        for p in 0..POS {
            if a[w][q][p] == worst_team as u8 { team_pos = Some(p); break; }
        }
        let tp = match team_pos { Some(p) => p, None => continue };

        let _ = w8; // weights used for target calc above
        let mut swap_pos = rng.random_range(0..(POS - 1));
        if swap_pos >= tp { swap_pos += 1; }
        a[w][q].swap(tp, swap_pos);
        return true;
    }
    false
}

fn guided_early_late(a: &mut Assignment, rng: &mut SmallRng) -> bool {
    let mut early_count = [0i32; TEAMS];
    for w in 0..WEEKS {
        for q in 0..2 {
            for p in 0..POS { early_count[a[w][q][p] as usize] += 1; }
        }
    }

    let target_e = WEEKS as f64 / 2.0;
    let mut worst_team = 0usize;
    let mut worst_dev = 0.0f64;
    let mut too_many_early = false;
    for t in 0..TEAMS {
        let dev = early_count[t] as f64 - target_e;
        if dev.abs() > worst_dev {
            worst_dev = dev.abs();
            worst_team = t;
            too_many_early = dev > 0.0;
        }
    }
    if worst_dev < 1.0 { return false; }

    let start = rng.random_range(0..WEEKS);
    for off in 0..WEEKS {
        let w = (start + off) % WEEKS;
        let team = worst_team as u8;
        let in_early = (0..2).any(|q| (0..POS).any(|p| a[w][q][p] == team));

        if (too_many_early && in_early) || (!too_many_early && !in_early) {
            let tmp0 = a[w][0]; let tmp1 = a[w][1];
            a[w][0] = a[w][2]; a[w][2] = tmp0;
            a[w][1] = a[w][3]; a[w][3] = tmp1;
            return true;
        }
    }
    false
}

/// Single-threaded SA solver with the full move set from split_sa.
/// Caller controls temperature and restarts; this just runs iterations.
pub struct SASolver {
    pub a: Assignment,
    pub cost: CostBreakdown,
    pub best_a: Assignment,
    pub best_cost: u32,
    pub iteration: u64,
    pub temp: f64,
    weights: Weights,
    stats: MoveStats,
    rng: SmallRng,
}

impl SASolver {
    pub fn new(a: Assignment, weights: Weights, temp: f64, rng: SmallRng) -> Self {
        let cost = evaluate(&a, &weights);
        let best_cost = cost.total;
        SASolver {
            best_a: a,
            best_cost,
            a,
            cost,
            iteration: 0,
            temp,
            weights,
            stats: MoveStats::new(),
            rng,
        }
    }

    pub fn set_assignment(&mut self, a: Assignment) {
        self.a = a;
        self.cost = evaluate(&a, &self.weights);
        self.best_a = a;
        self.best_cost = self.cost.total;
    }

    /// Run `n` SA iterations. Returns true if best_cost reached 0.
    pub fn step(&mut self, n: u64) -> bool {
        let end = self.iteration + n;

        while self.iteration < end {
            if self.best_cost == 0 { return true; }

            let i = self.iteration;

            if i > 0 && i % STATS_RECOMPUTE == 0 {
                self.stats.recompute();
            }

            // Periodic exhaustive single-week inter-quad swap search
            if i > 0 && i % 100_000 == 0 && self.cost.total > 0 {
                let w = self.rng.random_range(0..WEEKS);
                let q1 = self.rng.random_range(0..QUADS);
                let mut q2 = self.rng.random_range(0..(QUADS - 1));
                if q2 >= q1 { q2 += 1; }
                if q1 / 2 == q2 / 2 {
                    let mut best_delta = 0i64;
                    let mut best_swap: Option<(usize, usize)> = None;
                    for p1 in 0..POS {
                        for p2 in 0..POS {
                            let tmp = self.a[w][q1][p1];
                            self.a[w][q1][p1] = self.a[w][q2][p2];
                            self.a[w][q2][p2] = tmp;
                            let nc = evaluate(&self.a, &self.weights);
                            let d = nc.total as i64 - self.cost.total as i64;
                            if d < best_delta {
                                best_delta = d;
                                best_swap = Some((p1, p2));
                            }
                            self.a[w][q2][p2] = self.a[w][q1][p1];
                            self.a[w][q1][p1] = tmp;
                        }
                    }
                    if let Some((p1, p2)) = best_swap {
                        let tmp = self.a[w][q1][p1];
                        self.a[w][q1][p1] = self.a[w][q2][p2];
                        self.a[w][q2][p2] = tmp;
                        self.cost = evaluate(&self.a, &self.weights);
                        if self.cost.total < self.best_cost { self.best_cost = self.cost.total; self.best_a = self.a; }
                    }
                }
            }

            // Compound move (probability based on cost)
            let compound_prob = ((1000.0 - self.cost.total as f64) / 800.0).clamp(0.0, 0.5);
            if self.rng.random::<f64>() < compound_prob {
                let saved = self.a;
                let max_swaps = if self.cost.total < 200 { 12 } else if self.cost.total < 400 { 6 } else { 4 };
                let num_swaps = self.rng.random_range(2..=max_swaps);
                for _ in 0..num_swaps {
                    let w = self.rng.random_range(0..WEEKS);
                    let q1 = self.rng.random_range(0..QUADS);
                    let mut q2 = self.rng.random_range(0..(QUADS - 1));
                    if q2 >= q1 { q2 += 1; }
                    let p1 = self.rng.random_range(0..POS);
                    let p2 = self.rng.random_range(0..POS);
                    let tmp = self.a[w][q1][p1];
                    self.a[w][q1][p1] = self.a[w][q2][p2];
                    self.a[w][q2][p2] = tmp;
                }
                let new_cost = evaluate(&self.a, &self.weights);
                let delta = new_cost.total as i64 - self.cost.total as i64;
                if delta <= 0 || self.rng.random::<f64>() < (-delta as f64 / self.temp).exp() {
                    self.cost = new_cost;
                    if self.cost.total < self.best_cost { self.best_cost = self.cost.total; self.best_a = self.a; }
                } else {
                    self.a = saved;
                }
            } else {
                let move_id = self.stats.select(self.rng.random::<f64>());
                self.stats.attempts[move_id] += 1;
                let accepted = self.do_move(move_id);
                if accepted { self.stats.accepts[move_id] += 1; }
            }

            self.iteration += 1;
        }

        false
    }

    fn do_move(&mut self, move_id: usize) -> bool {
        match move_id {
            0 => self.move_inter_quad_swap(),
            1 => self.move_intra_quad_swap(),
            2 => self.move_cross_week_swap(),
            3 => self.move_quad_swap(),
            4 => self.move_week_swap(),
            5 => self.move_early_late_flip(),
            6 => self.move_lane_pair_swap(),
            7 => self.move_stay_switch(),
            8 => self.move_guided_matchup(),
            9 => self.move_guided_lane(),
            _ => self.move_guided_early_late(),
        }
    }

    fn accept(&mut self, new_cost: CostBreakdown, delta: i64) -> bool {
        if delta <= 0 || self.rng.random::<f64>() < (-delta as f64 / self.temp).exp() {
            self.cost = new_cost;
            if self.cost.total < self.best_cost { self.best_cost = self.cost.total; self.best_a = self.a; }
            true
        } else {
            false
        }
    }

    fn move_inter_quad_swap(&mut self) -> bool {
        let w = self.rng.random_range(0..WEEKS);
        let q1 = self.rng.random_range(0..QUADS);
        let mut q2 = self.rng.random_range(0..(QUADS - 1));
        if q2 >= q1 { q2 += 1; }
        let p1 = self.rng.random_range(0..POS);
        let p2 = self.rng.random_range(0..POS);
        let tmp = self.a[w][q1][p1];
        self.a[w][q1][p1] = self.a[w][q2][p2];
        self.a[w][q2][p2] = tmp;
        let nc = evaluate(&self.a, &self.weights);
        let delta = nc.total as i64 - self.cost.total as i64;
        if self.accept(nc, delta) { true }
        else { self.a[w][q2][p2] = self.a[w][q1][p1]; self.a[w][q1][p1] = tmp; false }
    }

    fn move_intra_quad_swap(&mut self) -> bool {
        let w = self.rng.random_range(0..WEEKS);
        let q = self.rng.random_range(0..QUADS);
        let p1 = self.rng.random_range(0..POS);
        let mut p2 = self.rng.random_range(0..(POS - 1));
        if p2 >= p1 { p2 += 1; }
        let tmp = self.a[w][q][p1];
        self.a[w][q][p1] = self.a[w][q][p2];
        self.a[w][q][p2] = tmp;
        let nc = evaluate(&self.a, &self.weights);
        let delta = nc.total as i64 - self.cost.total as i64;
        if self.accept(nc, delta) { true }
        else { self.a[w][q][p2] = self.a[w][q][p1]; self.a[w][q][p1] = tmp; false }
    }

    fn move_cross_week_swap(&mut self) -> bool {
        let team = self.rng.random_range(0..TEAMS) as u8;
        let w1 = self.rng.random_range(0..WEEKS);
        let mut w2 = self.rng.random_range(0..(WEEKS - 1));
        if w2 >= w1 { w2 += 1; }

        let mut qi1 = None; let mut pi1 = None;
        let mut qi2 = None; let mut pi2 = None;
        for q in 0..QUADS {
            for p in 0..POS {
                if self.a[w1][q][p] == team && qi1.is_none() { qi1 = Some(q); pi1 = Some(p); }
                if self.a[w2][q][p] == team && qi2.is_none() { qi2 = Some(q); pi2 = Some(p); }
            }
        }

        if let (Some(qi1), Some(pi1), Some(qi2), Some(pi2)) = (qi1, pi1, qi2, pi2) {
            let save = (self.a[w1][qi1][pi1], self.a[w1][qi2][pi2], self.a[w2][qi1][pi1], self.a[w2][qi2][pi2]);
            let other1 = self.a[w2][qi1][pi1];
            let other2 = self.a[w1][qi2][pi2];
            self.a[w1][qi1][pi1] = other2;
            self.a[w1][qi2][pi2] = team;
            self.a[w2][qi2][pi2] = other1;
            self.a[w2][qi1][pi1] = team;
            let nc = evaluate(&self.a, &self.weights);
            let delta = nc.total as i64 - self.cost.total as i64;
            if self.accept(nc, delta) { true }
            else {
                self.a[w1][qi1][pi1] = save.0; self.a[w1][qi2][pi2] = save.1;
                self.a[w2][qi1][pi1] = save.2; self.a[w2][qi2][pi2] = save.3;
                false
            }
        } else { false }
    }

    fn move_quad_swap(&mut self) -> bool {
        let w = self.rng.random_range(0..WEEKS);
        let q1 = self.rng.random_range(0..QUADS);
        let mut q2 = self.rng.random_range(0..(QUADS - 1));
        if q2 >= q1 { q2 += 1; }
        let tmp = self.a[w][q1];
        self.a[w][q1] = self.a[w][q2];
        self.a[w][q2] = tmp;
        let nc = evaluate(&self.a, &self.weights);
        let delta = nc.total as i64 - self.cost.total as i64;
        if self.accept(nc, delta) { true }
        else { self.a[w][q2] = self.a[w][q1]; self.a[w][q1] = tmp; false }
    }

    fn move_week_swap(&mut self) -> bool {
        let w1 = self.rng.random_range(0..WEEKS);
        let mut w2 = self.rng.random_range(0..(WEEKS - 1));
        if w2 >= w1 { w2 += 1; }
        let tmp = self.a[w1];
        self.a[w1] = self.a[w2];
        self.a[w2] = tmp;
        let nc = evaluate(&self.a, &self.weights);
        let delta = nc.total as i64 - self.cost.total as i64;
        if self.accept(nc, delta) { true }
        else { self.a[w2] = self.a[w1]; self.a[w1] = tmp; false }
    }

    fn move_early_late_flip(&mut self) -> bool {
        let w = self.rng.random_range(0..WEEKS);
        let tmp0 = self.a[w][0]; let tmp1 = self.a[w][1];
        self.a[w][0] = self.a[w][2]; self.a[w][2] = tmp0;
        self.a[w][1] = self.a[w][3]; self.a[w][3] = tmp1;
        let nc = evaluate(&self.a, &self.weights);
        let delta = nc.total as i64 - self.cost.total as i64;
        if self.accept(nc, delta) { true }
        else {
            let tmp0 = self.a[w][0]; let tmp1 = self.a[w][1];
            self.a[w][0] = self.a[w][2]; self.a[w][2] = tmp0;
            self.a[w][1] = self.a[w][3]; self.a[w][3] = tmp1;
            false
        }
    }

    fn move_lane_pair_swap(&mut self) -> bool {
        let w = self.rng.random_range(0..WEEKS);
        let tmp0 = self.a[w][0]; let tmp2 = self.a[w][2];
        self.a[w][0] = self.a[w][1]; self.a[w][1] = tmp0;
        self.a[w][2] = self.a[w][3]; self.a[w][3] = tmp2;
        let nc = evaluate(&self.a, &self.weights);
        let delta = nc.total as i64 - self.cost.total as i64;
        if self.accept(nc, delta) { true }
        else {
            let tmp1 = self.a[w][1]; let tmp3 = self.a[w][3];
            self.a[w][1] = self.a[w][0]; self.a[w][0] = tmp1;
            self.a[w][3] = self.a[w][2]; self.a[w][2] = tmp3;
            false
        }
    }

    fn move_stay_switch(&mut self) -> bool {
        let w = self.rng.random_range(0..WEEKS);
        let q = self.rng.random_range(0..QUADS);
        self.a[w][q].swap(0, 1);
        self.a[w][q].swap(2, 3);
        let nc = evaluate(&self.a, &self.weights);
        let delta = nc.total as i64 - self.cost.total as i64;
        if self.accept(nc, delta) { true }
        else { self.a[w][q].swap(0, 1); self.a[w][q].swap(2, 3); false }
    }

    fn move_guided_matchup(&mut self) -> bool {
        let saved = self.a;
        if guided_matchup(&mut self.a, &mut self.rng) {
            let nc = evaluate(&self.a, &self.weights);
            let delta = nc.total as i64 - self.cost.total as i64;
            if self.accept(nc, delta) { true }
            else { self.a = saved; false }
        } else { false }
    }

    fn move_guided_lane(&mut self) -> bool {
        let saved = self.a;
        if guided_lane(&mut self.a, &self.weights, &mut self.rng) {
            let nc = evaluate(&self.a, &self.weights);
            let delta = nc.total as i64 - self.cost.total as i64;
            if self.accept(nc, delta) { true }
            else { self.a = saved; false }
        } else { false }
    }

    fn move_guided_early_late(&mut self) -> bool {
        let saved = self.a;
        if guided_early_late(&mut self.a, &mut self.rng) {
            let nc = evaluate(&self.a, &self.weights);
            let delta = nc.total as i64 - self.cost.total as i64;
            if self.accept(nc, delta) { true }
            else { self.a = saved; false }
        } else { false }
    }
}
