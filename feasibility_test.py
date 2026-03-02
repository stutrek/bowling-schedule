#!/usr/bin/env python3
"""Quick test: can CP-SAT find ANY schedule satisfying all tight + half-season constraints?"""

import json, time
from ortools.sat.python import cp_model
import cp_sat_solver as cs

with open('weights.json') as f:
    w8 = json.load(f)
for k in w8:
    w8[k] = int(w8[k])

print('Building tight model...')
model, components, total, x, derived = cs.build_model(w8, tight=True)
print('Done. Searching for any feasible solution...\n')

solver = cp_model.CpSolver()
solver.parameters.max_time_in_seconds = 600
solver.parameters.num_workers = 10
solver.parameters.log_search_progress = True


class FirstSolution(cp_model.CpSolverSolutionCallback):
    def __init__(self):
        super().__init__()
        self.found = False
        self.t0 = time.time()

    def on_solution_callback(self):
        if not self.found:
            self.found = True
            elapsed = time.time() - self.t0
            cost = self.value(total)
            parts = '  '.join(f'{k}={self.value(v)}' for k, v in components.items())
            print(f'\n*** FEASIBLE at {elapsed:.1f}s  cost={cost}  {parts} ***\n')
            self.stop_search()


cb = FirstSolution()
status = solver.solve(model, cb)

if not cb.found:
    labels = {
        cp_model.INFEASIBLE: 'INFEASIBLE (proven impossible!)',
        cp_model.UNKNOWN: 'UNKNOWN (ran out of time)',
        cp_model.MODEL_INVALID: 'MODEL_INVALID',
    }
    print(f'\nNo solution found. Status: {labels.get(status, status)}')
    print(f'Wall time: {solver.wall_time:.1f}s')
