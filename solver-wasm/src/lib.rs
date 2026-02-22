use rand::Rng;
use rand::rngs::SmallRng;
use rand::SeedableRng;
use wasm_bindgen::prelude::*;

const TEAMS: usize = 16;
const LANES: usize = 4;
const WEEKS: usize = 12;
const QUADS: usize = 4;
const POS: usize = 4;

type Assignment = [[[u8; POS]; QUADS]; WEEKS];

#[wasm_bindgen]
pub struct CostBreakdown {
    pub matchup_balance: u32,
    pub consecutive_opponents: u32,
    pub early_late_balance: u32,
    pub early_late_alternation: u32,
    pub lane_balance: u32,
    pub lane_switch_balance: u32,
    pub total: u32,
}

#[wasm_bindgen]
pub struct SolverResult {
    cost: CostBreakdown,
    assignment: Vec<u8>,
}

#[wasm_bindgen]
impl SolverResult {
    #[wasm_bindgen(getter)]
    pub fn cost(&self) -> CostBreakdown {
        CostBreakdown {
            matchup_balance: self.cost.matchup_balance,
            consecutive_opponents: self.cost.consecutive_opponents,
            early_late_balance: self.cost.early_late_balance,
            early_late_alternation: self.cost.early_late_alternation,
            lane_balance: self.cost.lane_balance,
            lane_switch_balance: self.cost.lane_switch_balance,
            total: self.cost.total,
        }
    }

    #[wasm_bindgen(getter)]
    pub fn assignment(&self) -> Vec<u8> {
        self.assignment.clone()
    }
}

#[wasm_bindgen]
pub struct Solver {
    rng: SmallRng,
    a: Assignment,
    cost: CostBreakdown,
    best_a: Assignment,
    best_cost: u32,
    temp: f64,
    t0: f64,
    cool_rate: f64,
    restart_interval: u32,
    iteration: u32,
    max_iterations: u32,
}

#[wasm_bindgen]
impl Solver {
    #[wasm_bindgen(constructor)]
    pub fn new(max_iterations: u32) -> Solver {
        let mut rng = SmallRng::from_os_rng();
        let t0: f64 = 30.0;
        let cool_rate: f64 = (0.005_f64 / t0).ln() / max_iterations as f64;
        let a = random_assignment(&mut rng);
        let cost = evaluate(&a);
        let best_cost = cost.total;
        Solver {
            rng,
            a,
            best_a: a,
            best_cost,
            temp: t0,
            t0,
            cool_rate,
            restart_interval: 100_000,
            iteration: 0,
            max_iterations,
            cost,
        }
    }

    /// Run `chunk_size` iterations. Returns true when fully done.
    pub fn step(&mut self, chunk_size: u32) -> bool {
        let end = (self.iteration + chunk_size).min(self.max_iterations);

        while self.iteration < end {
            if self.best_cost == 0 {
                self.iteration = self.max_iterations;
                return true;
            }

            let i = self.iteration;
            let rand_val: f64 = self.rng.random();

            if rand_val < 0.4 {
                let w = self.rng.random_range(0..WEEKS);
                let q1 = self.rng.random_range(0..QUADS);
                let mut q2 = self.rng.random_range(0..(QUADS - 1));
                if q2 >= q1 { q2 += 1; }
                let p1 = self.rng.random_range(0..POS);
                let p2 = self.rng.random_range(0..POS);
                let tmp = self.a[w][q1][p1];
                self.a[w][q1][p1] = self.a[w][q2][p2];
                self.a[w][q2][p2] = tmp;

                let new_cost = evaluate(&self.a);
                let delta = new_cost.total as i64 - self.cost.total as i64;
                if delta <= 0 || self.rng.random::<f64>() < (-delta as f64 / self.temp).exp() {
                    self.cost = new_cost;
                    if self.cost.total < self.best_cost { self.best_cost = self.cost.total; self.best_a = self.a; }
                } else {
                    self.a[w][q2][p2] = self.a[w][q1][p1];
                    self.a[w][q1][p1] = tmp;
                }
            } else if rand_val < 0.65 {
                let w = self.rng.random_range(0..WEEKS);
                let q = self.rng.random_range(0..QUADS);
                let p1 = self.rng.random_range(0..POS);
                let mut p2 = self.rng.random_range(0..(POS - 1));
                if p2 >= p1 { p2 += 1; }
                let tmp = self.a[w][q][p1];
                self.a[w][q][p1] = self.a[w][q][p2];
                self.a[w][q][p2] = tmp;

                let new_cost = evaluate(&self.a);
                let delta = new_cost.total as i64 - self.cost.total as i64;
                if delta <= 0 || self.rng.random::<f64>() < (-delta as f64 / self.temp).exp() {
                    self.cost = new_cost;
                    if self.cost.total < self.best_cost { self.best_cost = self.cost.total; self.best_a = self.a; }
                } else {
                    self.a[w][q][p2] = self.a[w][q][p1];
                    self.a[w][q][p1] = tmp;
                }
            } else if rand_val < 0.85 {
                let team = self.rng.random_range(0..TEAMS) as u8;
                let w1 = self.rng.random_range(0..WEEKS);
                let mut w2 = self.rng.random_range(0..(WEEKS - 1));
                if w2 >= w1 { w2 += 1; }

                let mut qi1: Option<usize> = None;
                let mut pi1: Option<usize> = None;
                let mut qi2: Option<usize> = None;
                let mut pi2: Option<usize> = None;

                for q in 0..QUADS {
                    for p in 0..POS {
                        if self.a[w1][q][p] == team && qi1.is_none() { qi1 = Some(q); pi1 = Some(p); }
                        if self.a[w2][q][p] == team && qi2.is_none() { qi2 = Some(q); pi2 = Some(p); }
                    }
                }

                if let (Some(qi1), Some(pi1), Some(qi2), Some(pi2)) = (qi1, pi1, qi2, pi2) {
                    let other1 = self.a[w2][qi1][pi1];
                    let other2 = self.a[w1][qi2][pi2];
                    let save = (self.a[w1][qi1][pi1], self.a[w1][qi2][pi2], self.a[w2][qi1][pi1], self.a[w2][qi2][pi2]);

                    self.a[w1][qi1][pi1] = other2;
                    self.a[w1][qi2][pi2] = team;
                    self.a[w2][qi2][pi2] = other1;
                    self.a[w2][qi1][pi1] = team;

                    let new_cost = evaluate(&self.a);
                    let delta = new_cost.total as i64 - self.cost.total as i64;
                    if delta <= 0 || self.rng.random::<f64>() < (-delta as f64 / self.temp).exp() {
                        self.cost = new_cost;
                        if self.cost.total < self.best_cost { self.best_cost = self.cost.total; self.best_a = self.a; }
                    } else {
                        self.a[w1][qi1][pi1] = save.0;
                        self.a[w1][qi2][pi2] = save.1;
                        self.a[w2][qi1][pi1] = save.2;
                        self.a[w2][qi2][pi2] = save.3;
                    }
                }
            } else {
                let w = self.rng.random_range(0..WEEKS);
                let q1 = self.rng.random_range(0..QUADS);
                let mut q2 = self.rng.random_range(0..(QUADS - 1));
                if q2 >= q1 { q2 += 1; }
                let tmp = self.a[w][q1];
                self.a[w][q1] = self.a[w][q2];
                self.a[w][q2] = tmp;

                let new_cost = evaluate(&self.a);
                let delta = new_cost.total as i64 - self.cost.total as i64;
                if delta <= 0 || self.rng.random::<f64>() < (-delta as f64 / self.temp).exp() {
                    self.cost = new_cost;
                    if self.cost.total < self.best_cost { self.best_cost = self.cost.total; self.best_a = self.a; }
                } else {
                    self.a[w][q2] = self.a[w][q1];
                    self.a[w][q1] = tmp;
                }
            }

            self.temp = self.t0 * (self.cool_rate * i as f64).exp();

            if i > 0 && i % self.restart_interval == 0 && self.cost.total > self.best_cost {
                if i % (self.restart_interval * 2) == 0 {
                    self.a = random_assignment(&mut self.rng);
                } else {
                    self.a = self.best_a;
                    perturb(&mut self.a, &mut self.rng, 20);
                }
                self.cost = evaluate(&self.a);
                self.temp = self.t0 * 0.3;
            }

            self.iteration += 1;
        }

        self.iteration >= self.max_iterations
    }

    pub fn current_iteration(&self) -> u32 {
        self.iteration
    }

    pub fn best_cost_total(&self) -> u32 {
        self.best_cost
    }

    pub fn result(self) -> SolverResult {
        let final_cost = evaluate(&self.best_a);
        let mut flat: Vec<u8> = Vec::with_capacity(WEEKS * QUADS * POS);
        for w in 0..WEEKS {
            for q in 0..QUADS {
                for p in 0..POS {
                    flat.push(self.best_a[w][q][p]);
                }
            }
        }
        SolverResult {
            cost: final_cost,
            assignment: flat,
        }
    }
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
                    consecutive_opponents += 15;
                }
            }
        }
    }

    let mut early_late_balance: u32 = 0;
    let target_e: f64 = WEEKS as f64 / 2.0;
    for t in 0..TEAMS {
        let dev = (early_count[t] as f64 - target_e).abs();
        early_late_balance += (dev * dev * 25.0) as u32;
    }

    let mut early_late_alternation: u32 = 0;
    for t in 0..TEAMS {
        for w in 0..(WEEKS - 2) {
            let base = t * WEEKS;
            if early_late[base + w] == early_late[base + w + 1]
                && early_late[base + w + 1] == early_late[base + w + 2]
            {
                early_late_alternation += 40;
            }
        }
    }

    let mut lane_balance: u32 = 0;
    let target_l: f64 = (WEEKS as f64 * 2.0) / LANES as f64;
    for t in 0..TEAMS {
        for l in 0..LANES {
            lane_balance += ((lane_counts[t * LANES + l] as f64 - target_l).abs() * 15.0) as u32;
        }
    }

    let mut lane_switch_balance: u32 = 0;
    let target_stay: f64 = WEEKS as f64 / 2.0;
    for t in 0..TEAMS {
        let dev = (stay_count[t] as f64 - target_stay).abs();
        lane_switch_balance += (dev * 5.0) as u32;
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
