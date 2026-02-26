use rand::rngs::SmallRng;
use rand::SeedableRng;
use wasm_bindgen::prelude::*;

use solver_core::{
    self as core, Weights,
    sa::SASolver,
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
    pub commissioner_overlap: u32,
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
        commissioner_overlap: c.commissioner_overlap,
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
            commissioner_overlap: self.cost.commissioner_overlap,
            total: self.cost.total,
        }
    }

    #[wasm_bindgen(getter)]
    pub fn assignment(&self) -> Vec<u8> {
        self.assignment.clone()
    }
}

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
    inner: SASolver,
    t0: f64,
    cool_rate: f64,
    temp_floor: f64,
    restart_interval: u64,
    chunk_iterations: u64,
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
        let a = core::random_assignment(&mut rng);
        let t0: f64 = 30.0;
        let temp_floor: f64 = 1.0;
        Solver {
            inner: SASolver::new(a, weights.clone(), t0, rng),
            t0,
            cool_rate: (0.005_f64 / t0).ln() / max_iterations as f64,
            temp_floor,
            restart_interval: 100_000,
            chunk_iterations: 0,
            max_iterations,
            weights,
        }
    }

    pub fn new_with_seed(max_iterations: u32, weights_json: &str, seed_flat: &[u8]) -> Solver {
        let weights: Weights = serde_json::from_str(weights_json)
            .expect("Invalid weights JSON");
        let rng = SmallRng::from_os_rng();
        let a = core::flat_to_assignment(seed_flat);
        let t0: f64 = 30.0;
        let temp_floor: f64 = 1.0;
        Solver {
            inner: SASolver::new(a, weights.clone(), t0, rng),
            t0,
            cool_rate: (0.005_f64 / t0).ln() / max_iterations as f64,
            temp_floor,
            restart_interval: 100_000,
            chunk_iterations: 0,
            max_iterations,
            weights,
        }
    }

    pub fn step(&mut self, chunk_size: u32) -> bool {
        let chunk_end = (self.chunk_iterations + chunk_size as u64).min(self.max_iterations as u64);
        let n = chunk_end - self.chunk_iterations;
        if n == 0 { return true; }

        self.inner.temp = (self.t0 * (self.cool_rate * self.chunk_iterations as f64).exp()).max(self.temp_floor);
        self.inner.step(n);

        // Restart logic
        if self.chunk_iterations > 0
            && self.chunk_iterations % self.restart_interval == 0
            && self.inner.cost.total > self.inner.best_cost
        {
            let mut rng = SmallRng::from_os_rng();
            if self.chunk_iterations % (self.restart_interval * 2) == 0 {
                let a = core::random_assignment(&mut rng);
                self.inner.a = a;
                self.inner.cost = core::evaluate(&a, &self.weights);
            } else {
                self.inner.a = self.inner.best_a;
                core::perturb(&mut self.inner.a, &mut rng, 20);
                self.inner.cost = core::evaluate(&self.inner.a, &self.weights);
            }
        }

        self.chunk_iterations = chunk_end;
        self.chunk_iterations >= self.max_iterations as u64
    }

    pub fn current_iteration(&self) -> u32 {
        self.chunk_iterations as u32
    }

    pub fn best_cost_total(&self) -> u32 {
        self.inner.best_cost
    }

    pub fn best_assignment(&self) -> Vec<u8> {
        let mut a = self.inner.best_a;
        core::reassign_commissioners(&mut a);
        core::assignment_to_flat(&a)
    }

    pub fn result(self) -> SolverResult {
        let mut a = self.inner.best_a;
        core::reassign_commissioners(&mut a);
        let final_cost = core::evaluate(&a, &self.weights);
        let flat = core::assignment_to_flat(&a);
        SolverResult {
            cost: to_wasm_cost(&final_cost),
            assignment: flat,
        }
    }
}
