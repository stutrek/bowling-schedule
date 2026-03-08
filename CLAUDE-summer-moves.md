# Summer Solver SA Moves

Reference for all perturbation moves used in the summer bowling schedule SA solver. Moves are implemented in both CPU (Rust) and GPU (WGSL) and must stay in sync.

## Files
- CPU moves: `solver-native/src/cpu_sa_summer.rs`
- GPU moves: `solver-native/src/summer_solver.wgsl` (bottom half)
- GPU exchange: `solver-native/src/summer_exchange.wgsl`

## Core Invariant

Every move must preserve validity: each team appears exactly 3 times per week, never twice in the same slot. All moves check this via `team_in_slot()` before swapping and reject if the constraint would be violated. Moves also reject if they would create a self-matchup (team playing itself).

## Move Taxonomy

There are two fundamental swap primitives that most moves build on:

### Primitive: Team Swap
Swap one team from each of two matchups in **different slots** of the same week. The two swapped teams exchange slots. Changes: slot assignments, matchups, and lanes for both teams.

### Primitive: Opponent Swap
Swap one team from each of two matchups in the **same slot**. Changes: matchups and lanes, but NOT slot assignments (both teams stay in the same slot).

## The 15 Moves

| ID | Name | Code | Type | What it does |
|----|------|------|------|-------------|
| 0 | `tm_swap` | `move_team_swap` | Random | Random team swap across two positions in different slots |
| 1 | `mtch_sw` | `move_matchup_swap` | Random | Swap two **entire** matchups between different slots (both teams move together) |
| 2 | `opp_sw` | `move_opponent_swap` | Random | Random opponent swap — two matchups in the same slot exchange one team each |
| 3 | `ln_week` | `move_lane_swap_week` | Random | Swap two entire lane columns across all slots in a week (bulk relabeling) |
| 4 | `slot_sw` | `move_slot_swap` | Random | Swap two full slots (games 1-4 only, not game 5) within a week |
| 5 | `g_match` | `move_guided_matchup` | Guided | Find under/over-matched pair, create/break a matchup via targeted team swap |
| 6 | `g_lane` | `move_guided_lane` | Guided | Find worst lane imbalance, opponent-swap within same slot to fix |
| 7 | `g_slot` | `move_guided_slot` | Guided | Find worst slot imbalance, team-swap into the underrepresented slot |
| 8 | `g_lnsw` | `move_guided_lane_switch` | Guided | Find team with worst lane-switch penalty, opponent-swap to align consecutive games to same lane |
| 9 | `pr_swap` | `move_pair_swap_in_slot` | Random | Swap two entire matchups between lane pairs within the same slot |
| 10 | `g_ln_xs` | `move_guided_lane_cross_slot` | Guided | Fix lane imbalance via team-swap across slots (unlike move 6 which stays in-slot) |
| 11 | `ln_chas` | `move_lane_chase` | Compound | Multi-week lane fix: iterates weeks, tries pair-swap in-slot then team-swap across slots |
| 12 | `g_brkfx` | `move_guided_break_fix` | Guided | Fix post-break lane switches by identifying the isolated game and opponent-swapping it to the consecutive pair's lane |
| 13 | `g_lnsaf` | `move_guided_lane_safe` | Guided | Lane balance fix that only swaps to lanes matching an adjacent game, avoiding new lane switches |
| 14 | `g_lnpr` | `move_guided_lane_pair` | Guided | Find two teams with complementary lane imbalances and opponent-swap them in the same slot |

## Move Categories

### Random moves (0-4, 9)
Pick random positions and swap. These provide broad exploration of the search space.

### Guided moves (5-8, 10, 12-14)
Analyze current state to find the worst violation, then perform a targeted swap to fix it. Pattern:
1. Compute the relevant statistic (matchup counts, lane counts, slot counts, etc.)
2. Find the team/pair with worst deviation from target
3. Search weeks for a valid swap that improves the imbalance
4. Apply via team_swap or opponent_swap primitive

### Compound moves (11)
`lane_chase` iterates across multiple weeks making swaps to aggressively fix the worst lane imbalance. Uses two strategies per week: first tries a pair-swap within the slot, falls back to team-swap across slots.

## Moves 12-14: Lane/Break Tension Moves

Moves 12-14 specifically address the tension between `lane_switch_post_break` (ln_brk) and `lane_balance` (lane) penalties:

### Move 12: `g_brkfx` — Guided break-lane alignment
Targets `ln_brk` directly. Unlike move 8 (`g_lnsw`) which picks a random game to fix, this move identifies the break structure (which games are consecutive, which is isolated) and always fixes the *isolated* game — the one across the gap. It opponent-swaps the isolated game to the lane pair used by the consecutive pair.

### Move 13: `g_lnsaf` — Lane-balance-safe fix
Targets `lane` without worsening `ln_brk`. Finds the worst lane imbalance, then only performs the swap if the target lane matches what an adjacent game already uses. This breaks the tension between the two penalties by ensuring lane balance fixes are lane-switch-neutral.

### Move 14: `g_lnpr` — Complementary pair lane swap
Targets `lane` more effectively. Finds two teams with opposite imbalances (team A over on lane X / under on lane Y, team B the reverse), then opponent-swaps them in a slot where both are present on their respective overrepresented lanes. Fixes two imbalances in one move.

## Special Mechanisms

### Compound move (CPU only)
At low cost, the CPU worker probabilistically chains 2-10 random team swaps into a single SA step (accept/reject as a batch). This helps escape local minima.

### Exhaustive local search (CPU only)
Every 100k iterations, the CPU tries all possible team swaps in a random week, accepting the single best-improving move. This is a greedy hill-climb step.

### Adaptive move selection (CPU + GPU)
The CPU tracks accept rates per move and reweights selection every 10k iterations. Moves with higher accept rates get selected more often. The GPU thresholds are updated from CPU stats each dispatch. Base weights:
```
tm_swap:0.12  mtch_sw:0.08  opp_sw:0.08  ln_week:0.04  slot_sw:0.04
g_match:0.08  g_lane:0.08   g_slot:0.06  g_lnsw:0.08   pr_swap:0.06
g_ln_xs:0.05  ln_chas:0.05  g_brkfx:0.08 g_lnsaf:0.06  g_lnpr:0.04
```

### GPU move selection
GPU uses cumulative thresholds passed via `move_thresh` storage buffer (array of 16 u32s, 0-100 scale). Thresholds are adaptively updated from CPU worker accept rates each dispatch cycle.

### GPU chain exchange
`summer_exchange.wgsl` swaps entire assignments between chains (replica exchange / parallel tempering). Pairs of chains swap their full state and costs, then update best if improved.

## Key Constraints When Adding/Modifying Moves

1. **Team-per-slot uniqueness**: A team can only appear once per slot. Always check `team_in_slot()` before swapping.
2. **No self-matchup**: Never place a team against itself. Check opponent after swap.
3. **Slot 4 structure**: Only lane pairs 2-3 are valid in slot 4 (game 5). Pairs 0-1 are always EMPTY. Moves like `slot_swap` exclude slot 4. `is_valid_position(slot, pair)` encodes this.
4. **CPU/GPU parity**: Both implementations must produce identical move logic. The GPU uses packed u32 encoding (`left | right << 8`) while CPU uses `(u8, u8)` tuples.
5. **Undo on reject**: CPU saves/restores the full assignment. GPU uses `save_all`/`restore_all`. Guided moves that fail validation early return false without modifying state.
6. **Move count sync**: `NUM_MOVES` (CPU), `GpuSummerMoveThresholds.t` array size, `move_thresh` WGSL array size, `BASE_WEIGHTS`, `MOVE_NAMES`, and the dispatch `if/else` chains in both CPU and GPU must all agree.
