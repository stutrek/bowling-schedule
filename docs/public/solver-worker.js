import init, { Solver } from './solver_wasm.js';

let cancelled = false;

self.onmessage = async (e) => {
    const { type, maxIterations, weightsJson, seedFlat } = e.data;

    if (type === 'cancel') {
        cancelled = true;
        return;
    }

    if (type === 'solve') {
        cancelled = false;

        try {
            await init({
                module_or_path: new URL(
                    './solver_wasm_bg.wasm',
                    self.location.href,
                ),
            });
        } catch {
            // Already initialized
        }

        const chunkSize = 100_000;

        try {
            const solver = seedFlat
                ? Solver.new_with_seed(
                      maxIterations,
                      weightsJson,
                      new Uint8Array(seedFlat),
                  )
                : new Solver(maxIterations, weightsJson);

            let prevBest = Number.POSITIVE_INFINITY;

            while (true) {
                const done = solver.step(chunkSize);
                const currentBest = solver.best_cost_total();
                const improved = currentBest < prevBest;

                const msg = {
                    type: 'progress',
                    iteration: solver.current_iteration(),
                    maxIterations,
                    bestCost: currentBest,
                };

                if (improved) {
                    msg.bestAssignment = Array.from(solver.best_assignment());
                    prevBest = currentBest;
                }

                self.postMessage(msg);

                if (done || cancelled) break;

                await new Promise((r) => setTimeout(r, 0));
            }

            if (cancelled) {
                solver.free();
                self.postMessage({ type: 'cancelled' });
                return;
            }

            const result = solver.result();
            const cost = result.cost;
            const msg = {
                type: 'done',
                cost: {
                    matchupBalance: cost.matchup_balance,
                    consecutiveOpponents: cost.consecutive_opponents,
                    earlyLateBalance: cost.early_late_balance,
                    earlyLateAlternation: cost.early_late_alternation,
                    laneBalance: cost.lane_balance,
                    laneSwitchBalance: cost.lane_switch_balance,
                    commissionerOverlap: cost.commissioner_overlap,
                    total: cost.total,
                },
                assignment: Array.from(result.assignment),
            };
            cost.free();
            result.free();
            self.postMessage(msg);
        } catch (err) {
            self.postMessage({ type: 'error', message: String(err) });
        }
    }
};
