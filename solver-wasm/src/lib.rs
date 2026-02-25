use rand::Rng;
use rand::rngs::SmallRng;
use rand::SeedableRng;
use wasm_bindgen::prelude::*;

use solver_core::{
    self as core, Assignment, Weights,
    TEAMS, WEEKS, QUADS, POS,
};

#[wasm_bindgen]
pub struct WasmCostBreakdown {
    pub matchup_balance: u32,
    pub consecutive_opponents: u32,
    pub early_late_balance: u32,
    pub early_late_alternation: u32,
    pub lane_balance: u32,
    pub lane_switch_balance: u32,
    pub late_lane_balance: u32,
    pub total: u32,
}

fn to_wasm_cost(c: &core::CostBreakdown) -> WasmCostBreakdown {
    WasmCostBreakdown {
        matchup_balance: c.matchup_balance,
        consecutive_opponents: c.consecutive_opponents,
        early_late_balance: c.early_late_balance,
        early_late_alternation: c.early_late_alternation,
        lane_balance: c.lane_balance,
        lane_switch_balance: c.lane_switch_balance,
        late_lane_balance: c.late_lane_balance,
        total: c.total,
    }
}

#[wasm_bindgen]
pub struct SolverResult {
    cost: WasmCostBreakdown,
    assignment: Vec<u8>,
}

#[wasm_bindgen]
impl SolverResult {
    #[wasm_bindgen(getter)]
    pub fn cost(&self) -> WasmCostBreakdown {
        WasmCostBreakdown {
            matchup_balance: self.cost.matchup_balance,
            consecutive_opponents: self.cost.consecutive_opponents,
            early_late_balance: self.cost.early_late_balance,
            early_late_alternation: self.cost.early_late_alternation,
            lane_balance: self.cost.lane_balance,
            lane_switch_balance: self.cost.lane_switch_balance,
            late_lane_balance: self.cost.late_lane_balance,
            total: self.cost.total,
        }
    }

    #[wasm_bindgen(getter)]
    pub fn assignment(&self) -> Vec<u8> {
        self.assignment.clone()
    }
}

/// Evaluate a flat assignment (WEEKS*QUADS*POS u8s) with given weights JSON.
/// Returns a WasmCostBreakdown.
#[wasm_bindgen]
pub fn evaluate_assignment(flat: &[u8], weights_json: &str) -> WasmCostBreakdown {
    let w8: Weights = serde_json::from_str(weights_json)
        .expect("Invalid weights JSON");
    let a = core::flat_to_assignment(flat);
    let c = core::evaluate(&a, &w8);
    to_wasm_cost(&c)
}

#[wasm_bindgen]
pub struct Solver {
    rng: SmallRng,
    a: Assignment,
    cost: core::CostBreakdown,
    best_a: Assignment,
    best_cost: u32,
    temp: f64,
    t0: f64,
    cool_rate: f64,
    restart_interval: u32,
    iteration: u32,
    max_iterations: u32,
    weights: Weights,
}

#[wasm_bindgen]
impl Solver {
    #[wasm_bindgen(constructor)]
    pub fn new(max_iterations: u32, weights_json: &str) -> Solver {
        let weights: Weights = serde_json::from_str(weights_json)
            .expect("Invalid weights JSON");
        let mut rng = SmallRng::from_os_rng();
        let t0: f64 = 30.0;
        let cool_rate: f64 = (0.005_f64 / t0).ln() / max_iterations as f64;
        let a = core::random_assignment(&mut rng);
        let cost = core::evaluate(&a, &weights);
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
            weights,
        }
    }

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

                let new_cost = core::evaluate(&self.a, &self.weights);
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

                let new_cost = core::evaluate(&self.a, &self.weights);
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

                    let new_cost = core::evaluate(&self.a, &self.weights);
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

                let new_cost = core::evaluate(&self.a, &self.weights);
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
                    self.a = core::random_assignment(&mut self.rng);
                } else {
                    self.a = self.best_a;
                    core::perturb(&mut self.a, &mut self.rng, 20);
                }
                self.cost = core::evaluate(&self.a, &self.weights);
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
        let final_cost = core::evaluate(&self.best_a, &self.weights);
        let flat = core::assignment_to_flat(&self.best_a);
        SolverResult {
            cost: to_wasm_cost(&final_cost),
            assignment: flat,
        }
    }
}
