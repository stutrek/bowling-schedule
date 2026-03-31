# Bowling Schedule Solver

Simulated annealing solver for generating bowling league schedules. Has both a regular season solver and a summer season solver with different constraints.

## Summer Schedule

The summer league has unusual constraints that differ from the regular season.

### Structure
- 12 teams, 10 weeks, 3 games per team per week
- 5 game slots per week, 4 lanes
- Games 1-4: all 4 lanes active (8 teams per slot, 4 matchups)
- Game 5: only lanes 3-4 active (4 teams, 2 matchups)
- This gives 18 matchup positions per week (4×4 + 1×2 = 18), and 36 team-slots = 12 teams × 3 games

### Ideal Game Spacing
Each team ideally plays two consecutive games and a third game with a one-game break. The break game can come before or after the consecutive pair:
- Example patterns: games 1,2,4 or games 1,3,4 or games 2,3,5
- Some teams must play three consecutive games or have a two-game break for the math to work out — this is minimized via the `time_gap_consecutive` and `time_gap_large` penalties.

### Commissioner Constraint
Two player-commissioners need coverage across the night. Ideally one has early games (game 1) and the other has late games (game 5), so they don't have to work double duty. The solver minimizes overlap of any team pair in games 1 and 5 (`commissioner_overlap`). In post-processing, the best pair gets reassigned to teams 1 and 2 (`reassign_summer_commissioners`).

### Cost Function (penalties to minimize)
Weights are in `summer_weights.json`. All penalties are flat per-violation unless noted.

| Penalty | What it penalizes |
|---|---|
| `matchup_balance` | Any team pair that doesn't play each other 2-3 times across the season |
| `lane_switch_consecutive` | Changing lanes between consecutive games |
| `lane_switch_post_break` | Changing lanes across a gap (lower weight since less disruptive) |
| `time_gap_large` | Gap of 2+ slots between consecutive games for a team |
| `time_gap_consecutive` | Three consecutive games (no break at all) |
| `lane_balance` | Any team not on each lane 7-8 times (target 30/4 = 7.5) |
| `commissioner_overlap` | Multiplied by min co-appearance count of any pair in slots 1 and 5 |
| `slot_balance` | Slots 1-4: each team should play 6-7 times; slot 5: 3-4 times |
| `repeat_matchup_same_night` | Same two teams matched up more than once in the same week |

### Key Data Structures
- `SummerAssignment`: `[week][slot][lane_pair] -> (team_a, team_b)`, 0-indexed teams, `EMPTY=0xFF` for unused positions
- Slot 4 (game 5) only uses lane pairs 2-3; pairs 0-1 are always `(EMPTY, EMPTY)`

### SA Moves
See [CLAUDE-summer-moves.md](CLAUDE-summer-moves.md) for detailed documentation of all 12 perturbation moves (random, guided, compound), the adaptive selection mechanism, and constraints for adding/modifying moves.

### Architecture
- `solver-core/src/summer.rs` — CPU evaluation, random generation, perturbation, TSV I/O, commissioner reassignment
- `solver-native/src/summer_main.rs` — CLI entry point for the SA solver
- `solver-native/src/summer_solver.wgsl` — GPU (WebGPU) cost evaluation shader
- `solver-native/src/cpu_sa_summer.rs` — CPU SA loop
- `solver-native/src/gpu_types_summer.rs` — GPU buffer types
- `solver-native/src/output_summer.rs` — Result output/saving

### Running
```
cargo run --release -p solver-native --bin summer_solver -- [options]
```
