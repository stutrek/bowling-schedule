import init, { Solver } from './solver_wasm.js';

let cancelled = false;

self.onmessage = async (e) => {
    const { type, maxIterations } = e.data;

    if (type === 'cancel') {
        cancelled = true;
        return;
    }

    if (type === 'solve') {
        cancelled = false;

        try {
            await init(new URL('./solver_wasm_bg.wasm', self.location.href));
        } catch {
            // Already initialized
        }

        const chunkSize = 1_000_000;

        try {
            const solver = new Solver(maxIterations);

            while (true) {
                const done = solver.step(chunkSize);

                self.postMessage({
                    type: 'progress',
                    iteration: solver.current_iteration(),
                    maxIterations,
                    bestCost: solver.best_cost_total(),
                });

                if (done || cancelled) break;

                // Yield so the main thread can process progress and we can receive cancel
                await new Promise((r) => setTimeout(r, 0));
            }

            if (cancelled) {
                solver.free();
                self.postMessage({ type: 'cancelled' });
                return;
            }

            const result = solver.result(); // consumes solver
            const cost = result.cost;
            const msg = {
                type: 'done',
                cost: {
                    matchupBalance: cost.matchup_balance,
                    consecutiveOpponents: cost.consecutive_opponents,
                    earlyLateBalance: cost.early_late_balance,
                    earlyLateAlternation: cost.early_late_alternation,
                    laneBalance: cost.lane_balance,
                    total: cost.total,
                },
                assignment: Array.from(result.assignment),
            };
            result.free();
            self.postMessage(msg);
        } catch (err) {
            self.postMessage({ type: 'error', message: String(err) });
        }
    }
};
