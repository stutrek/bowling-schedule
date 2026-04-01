# Bowling Schedule Optimizer

Optimized bowling schedules for a league with two seasons -- a 16-team winter season and a 12-team summer season. A computer searches through millions of possible schedules to find ones that balance many competing constraints simultaneously.

**[See visualizations of the schedules →](https://stutrek.github.io/bowling-schedule/)**

---

## Winter Schedule

An optimized 12-week schedule for 16 teams across 4 lanes (5, 6, 7, 8), built with a quad-based structure that's better for bowlers and easier for computers to optimize.

### The Problem and the Big Improvement

The current schedule has 16 teams playing two games each week. But a team can end up on completely different lanes between their two games -- maybe lanes 5 and 7, or lanes 6 and 8. This causes real problems on bowling night.

#### Quads

The new schedule uses **quads**. A quad is a group of 4 teams assigned to one lane pair for one half of a night. They play two games back-to-back on the same two lanes. Between games, you either bowl on the same lane twice, or you switch to the lane next to you -- on the same ball return. Every team in your quad plays every other team in your quad across the two games. Each week has 4 quads (2 early, 2 late), and every team appears in exactly one quad per week.

#### Why quads are better

The current schedule doesn't use quads -- teams can end up on completely different lanes between their two games. Keeping teams on the same lane pair means:

- If a lane breaks down, the delay only affects the other lane in your pair, not the whole league
- One team in each game just bowled on that same lane -- no warm-up needed
- The ball return doesn't get clogged with house balls from the previous group
- Your jacket and bag stay right where you left them
- Teams aren't awkwardly standing around waiting for their next game with nowhere to sit

### The scheduling challenge

Quads are great for bowlers, but they make scheduling harder. On top of the quad structure, the schedule needs to be fair across nine dimensions -- and optimizing one often makes another worse. A single "cost" score (lower is better, 0 is perfect) measures how good a schedule is:

1. **Matchup balance** -- every pair of teams should play each other 1 or 2 times across the season (penalty for 0 or 3+)
2. **Consecutive opponents** -- no pair should play each other in back-to-back weeks
3. **Early/late balance** -- each team should play 6 weeks early and 6 weeks late
4. **Early/late alternation** -- no team should play early (or late) three weeks in a row
5. **Lane balance** -- each team should play on each of the 4 lanes roughly equally (6 times each)
6. **Lane switches** -- each team should stay on their lane about half the time and switch about half the time
7. **Last game lane balance** -- each team's 6 last games of the night should be spread evenly across the 4 lanes (1-2 times each). This prevents a team from always ending their night on a problematic lane.
8. **Commissioner overlap** -- there are two commissioners in the league. If both have early games the same week (or both late), one person has to cover double duty. The schedule minimizes the number of weeks the commissioners share the same time slot.
9. **Half-season repeat** -- no pair of teams should play each other more than once in each half of the season. This spreads matchups across the full 12 weeks instead of clustering them.

The relative importance of each constraint is configured in [weights.json](weights.json).

The position of a team within its quad determines three things at once: which lane they're on, who they play, and whether they stay or switch. This compact representation means a single swap of two players between quads changes matchups, lane balance, and stay/switch all at once -- which is why the optimizer can explore the search space so efficiently.

### The Algorithm

The solver starts with a random schedule and tries to improve it by making small random changes -- swapping two teams between groups, flipping who plays early vs late, etc. If a change makes the schedule better, it's kept. Otherwise it's rejected. To speed this up, the solver runs thousands of copies of this process at the same time on a GPU, with CPU workers running alongside for deeper optimization. The GPU and CPU sides share their best schedules with each other.

This approach can get stuck. The solver might find a pretty good schedule where every single change makes it worse, even though a much better schedule exists nearby -- it just takes a few bad steps to get there. To avoid this, some of the GPU copies are dedicated explorers that accept worse schedules freely, always searching for new starting points. When an explorer stumbles onto something promising, it gets passed to the strict copies for fine-tuning.

The core technique is called [simulated annealing](https://en.wikipedia.org/wiki/Simulated_annealing), the explorer/strict copy structure is [parallel tempering](https://en.wikipedia.org/wiki/Parallel_tempering), and solutions are exchanged between copies using the [Metropolis criterion](https://en.wikipedia.org/wiki/Metropolis%E2%80%93Hastings_algorithm).

#### Temperature ladder

The GPU runs chains organized into pods of 8, with temperatures spaced geometrically from 6.0 to 18.0. CPU workers run at temperatures from 12.0 to 15.0 (one per core). Cold chains are picky -- they mostly only accept swaps that improve the schedule. Hot chains are adventurous -- they'll accept worse schedules to explore new territory.

#### Move types

Each iteration, the solver picks one of 11 types of random changes to try. The percentage is the base probability before adaptive adjustment.

- **Inter-quad player swap (25%)** -- Pick two quads in the same week, swap one player between them. The workhorse move -- it directly changes matchups, lane assignments, and stay/switch in one step.
- **Intra-quad player swap (10%)** -- Swap two players within the same quad. Changes lane and stay/switch assignments without affecting who plays whom.
- **Team cross-week swap (10%)** -- Pick a team, find where it sits in two different weeks, and swap its position with the displaced teams in both weeks. Helps spread out repeat matchups and balance early/late across the season.
- **Quad swap (6%)** -- Trade two entire quads within a week. Moves all four players to different lanes at once.
- **Week swap (6%)** -- Swap two entire weeks. Reorders the season without changing any matchups -- just changes when they happen, which helps avoid back-to-back repeat opponents.
- **Early/late flip (5%)** -- Take everyone who was playing early in a week and make them late, and vice versa.
- **Lane pair swap (4%)** -- Swap the two early quads with each other and the two late quads with each other. Moves teams between lane pairs (5-6 and 7-8) while preserving who plays early vs late.
- **Stay/switch rotation (6%)** -- Within a quad, swap who stays on their lane vs who switches between games.
- **Targeted: fix a missing matchup (5%)** -- Find two teams that haven't played each other, find a week where they're in neighboring quads, and swap one into the other's quad.
- **Targeted: fix lane imbalance (15%)** -- Find the team with the most lopsided lane distribution, find a quad containing that team, and swap their position within the quad to change their lane.
- **Targeted: fix early/late imbalance (8%)** -- Find the team that's played early (or late) too many times and flip a week's early/late assignment to correct it.

#### Adaptive move selection

The solver tracks which move types are producing accepted changes and gives them higher probability. Move weights are recomputed every 10,000 iterations based on each move's recent acceptance rate.

#### Compound moves

Near the end of optimization (when cost is already low), the solver bundles multiple inter-quad swaps together as one big move -- up to 12 swaps at once. This helps escape dead ends that single swaps can't get out of.

#### Exhaustive search

Every 100,000 iterations, the solver picks two quads in the same week and brute-forces every possible player swap between them (only 16 possibilities). It keeps the best one. A periodic sanity check that finds improvements the random moves might miss.

### Winter results

The best practical results score around 520. Schedules with lower numerical scores exist, but they achieve those numbers by over-optimizing less important metrics (like half-season repeat matchups) at the expense of attributes that matter more to bowlers, such as lane balance and early/late distribution.

### Running the winter solver

```bash
# Run the solver (auto-detects GPU size and CPU cores)
cargo run --release --bin solver

# Run without CPU workers (GPU only)
cargo run --release --bin solver -- --no-cpu

# Run without seeding from previous results
cargo run --release --bin solver -- --no-seed
```

Results are saved to `solver-native/results/gpu/`.

---

## Summer Schedule

A 10-week schedule for 12 teams with 3 games per team per week. The summer league has a different structure and different constraints than the winter season.

### The structure

Each week has 5 game slots and 4 lanes. Games 1-4 use all 4 lanes (8 teams playing, 4 matchups per slot). Game 5 only uses lanes 3-4 (4 teams playing, 2 matchups). This gives 18 matchups per week and exactly 3 games per team.

The template maximizes the number of teams that play three consecutive games each week. With 12 teams and 5 game slots, 8 positions get three consecutive games and 4 positions have a one-game break between two of their games. The solver balances break positions evenly across teams over the season.

### The fixed template approach

The solver uses a **fixed day template** -- an ideal arrangement of 12 positions across the 5 game slots that guarantees good game spacing by construction. The only variables are:

- **Which team fills which position** each week (a permutation of 12 teams)
- **Whether to swap lanes 1-2** each week (a boolean per week)
- **Whether to swap lanes 3-4** each week (a boolean per week)

This reduces the search space dramatically compared to treating every position as independent. The template is the same every week; only the team assignments and lane swaps change.

The template guarantees these properties by construction:

- Consecutive games are always on the same lane -- no switching between back-to-back games
- For games 1-4, teams that switch lanes are always on the same lane pair (lanes 5-6 or 7-8), so they stay on the same ball return
- All teams in game 5 have a break before their slot, so they're ready to start at any moment
- No team ever has a two-game gap -- it's either three consecutive games or two consecutive with a one-game break

### Constraints

Seven penalties, weighted in [summer_fixed_weights.json](summer_fixed_weights.json):

1. **Matchup balance** -- every pair of teams should play each other 2 or 3 times across the season. Penalty scales with distance from this range.
2. **Slot balance** -- each team should play in each game slot a balanced number of times. Slots 1-4: 6-7 times each. Slot 5: 3-4 times.
3. **Lane balance** -- each team should play on each lane a balanced number of times. Lanes 1-2: 6-7 times each (they only get traffic from games 1-4). Lanes 3-4: 8-9 times each (they get traffic from all 5 games).
4. **Game 5 lane balance** -- within a team's game 5 appearances, they should be on lane 3 and lane 4 roughly equally.
5. **Commissioner overlap** -- two player-commissioners need coverage across the night. The schedule minimizes how often any pair of teams both appears in game 1 and game 5. In post-processing, the best pair gets assigned to teams 1 and 2.
6. **Matchup spacing** -- pairs that play each other multiple times should be spread across the season. Pairs with 2 matchups need at least 4 weeks apart; pairs with 3 need at least 2 weeks apart.
7. **Break balance** -- the 4 break positions (teams with a one-game gap instead of three consecutive) should be distributed evenly across teams over the season.

### The algorithm

The same parallel tempering approach as the winter solver -- GPU+CPU hybrid with replica exchange, adaptive move selection, and partition-based stagnation detection. The key difference is the move set, which is much simpler because of the fixed template structure:

- **Team swap (15%)** -- Swap two teams' positions in a random week.
- **Toggle lanes 1-2 (13%)** -- Flip the lane 1-2 swap flag for a random week.
- **Toggle lanes 3-4 (13%)** -- Flip the lane 3-4 swap flag for a random week.
- **Week swap (13%)** -- Swap all data between two weeks.
- **Guided matchup (16%)** -- Find the most over- or under-matched pair and make a targeted swap.
- **Guided slot (15%)** -- Find the worst slot imbalance and swap teams to fix it.
- **Guided lane (15%)** -- Find the worst lane imbalance and toggle a lane swap or move a team.

All four basic moves are self-inverse (applying the same move again undoes it). The three guided moves identify the worst cost component and make targeted changes to improve it.

### Summer results

Results are saved to `solver-native/results/summer-fixed/`.

### Running the summer solver

```bash
# Run the summer-fixed solver
cargo run --release --bin solver -- --league summer-fixed

# Run without CPU workers (GPU only)
cargo run --release --bin solver -- --league summer-fixed --no-cpu

# Run without seeding from previous results
cargo run --release --bin solver -- --league summer-fixed --no-seed
```

---

## Shared Infrastructure

Both solvers share the same parallel tempering infrastructure:

- **Replica exchange** -- Adjacent-temperature chains within each pod may swap schedules every dispatch using the Metropolis criterion. Even/odd parity alternation ensures every pair gets a chance. Good solutions found by hot chains flow down to cold chains for refinement.
- **GPU-CPU feedback** -- Every 10 dispatches, CPU workers' best solutions reseed GPU partition chains (with perturbation). GPU chains that beat their CPU partition owner send their solution back. This bidirectional flow combines massive GPU exploration with deeper CPU optimization.
- **Anti-stagnation** -- Partitions that stop improving get extra reseeding. Prolonged stagnation triggers an escalated shakeup that resets the entire partition.
- **Adaptive thresholds** -- GPU move selection probabilities are updated based on CPU workers' acceptance rates.

## Architecture

- `solver-core/` -- Shared Rust library with data structures, evaluation functions, and TSV I/O for both seasons
  - `winter.rs` -- Winter schedule types, evaluation, and moves
  - `summer_fixed.rs` -- Summer fixed-template types, evaluation, 7 moves, TSV parsing
- `solver-native/` -- Command-line solver binary with GPU+CPU hybrid solver for both seasons
  - `gpu_solver.rs` -- Main binary entry point, dispatches to winter or summer solver
  - `gpu_setup.rs` -- wgpu device/buffer/pipeline creation (shared)
  - `gpu_types.rs` -- GPU parameters, temperature constants, chain count detection (shared)
  - Winter: `winter_main.rs`, `cpu_sa_winter.rs`, `solver.wgsl`, `exchange.wgsl`, `gpu_types_winter.rs`, `output_winter.rs`
  - Summer: `summer_fixed_main.rs`, `cpu_sa_summer_fixed.rs`, `summer_fixed_solver.wgsl`, `summer_fixed_exchange.wgsl`, `gpu_types_summer_fixed.rs`, `output_summer_fixed.rs`
- `solver-wasm/` -- Browser-compatible WASM build of the evaluation function and a single-threaded SA solver, used by the web UI
- `docs/` -- Next.js web app with WASM-powered scoring, an interactive schedule editor, and TSV import/export
- `src/` -- Original TypeScript prototypes (pinsetter1-6 and early SA attempts)

## What Didn't Work

### Winter

**Manual scheduling.** The original schedule was created by hand. It was decent, but the goal was to do better with software.

**Rule-based construction (pinsetter1-6).** Six different TypeScript algorithms that try to build a schedule step by step using mathematical patterns -- rotations, interleaving, remappings. Pinsetter1 came closest: perfect matchups, early/late, alternation, and consecutive opponents. But it couldn't get lane balance right. The fundamental issue is that you can't build a schedule with simple rules and have all six constraints come out balanced.

**Locking early/late and optimizing the rest.** Restricted the solver to only make changes that preserve the early/late assignments. Early/late metrics were perfect, but matchup balance was catastrophic (5120 penalty). The constraint was too tight -- teams that always shared the same time slot could never be paired against each other.

**Hybrid: pinsetter1 + simulated annealing.** Used pinsetter1's schedule as a starting point, then ran the optimizer with limited moves to fix lane balance. Failed because converting between data formats lost information about player positions, corrupting the schedule.

**Prioritizing "hard" constraints.** Tried making some constraints 1000x more important than others ([lexicographic optimization](https://en.wikipedia.org/wiki/Lexicographic_optimization)). Made the scoring landscape too steep for the solver to navigate -- it could never satisfy the hard constraints because accepting any move that helped a soft constraint was nearly impossible.

**Build early/late first, then matchups.** Constructed a perfect early/late assignment matrix, then tried to optimize matchups within that framework. Got good matchups, but introduced early/late alternation violations -- fixing one dimension broke another.

**Constraint satisfaction / backtracking.** Tried building the schedule by placing teams one at a time, backtracking when constraints were violated -- like solving a Sudoku (`construct.rs`, `first_half.rs`). Could find partial solutions but didn't scale to the full 12-week schedule with all the soft constraints.

### Summer

**Building the schedule from scratch.** The original summer solver treated every position as an independent variable -- 200 logical positions across 10 weeks, with 15 different SA moves and a GPU+CPU hybrid. It could get close but struggled to balance all constraints simultaneously because the search space was too large and the constraints were too intertwined. Lane switching penalties and game spacing constraints fought with matchup balance and lane balance, making it hard for the solver to make progress on one without regressing on another.

**The fix: a fixed day template.** Instead of optimizing every position, the solver now uses a single ideal day layout as a fixed template and only varies which team fills which role each week. This guarantees good game spacing by construction and reduces the search space from 200 independent positions to 12-team permutations plus lane swap flags -- a dramatically simpler problem that the solver handles easily.

## Experimental: Island-based Elite Solver (`elite-experiments` branch)

The standard solver converges all 8 CPU partitions to the same local minimum -- typically around cost 520. Longer runs (12-24 hours) don't improve beyond this. The `elite-experiments` branch explores a different architecture designed to maintain diversity across the search while still deeply refining each solution.

### The idea

Instead of 8 fixed CPU partitions, the GPU's 65,536 chains are divided into independent islands. By default there are 2 islands per CPU core (e.g. 16 islands with 8 CPUs), each with ~4,096 chains. Replica exchange only happens within islands, never across them. Each island converges to its own local minimum independently. CPU workers cycle between islands, spending ~24 seconds deeply refining each one before moving on.

To prevent islands from all converging to the same basin (which is what happens with the standard solver), the system uses **team-normalized similarity detection** (enabled with `--dedup`). Since team labels are arbitrary (swapping all occurrences of team 3 and team 7 produces an equivalent real-world schedule), island bests are compared after normalizing team labels. If two islands converge to the same solution, the worse one is reset with fresh random schedules.

### Status

The architecture runs but may still have bugs related to cross-contamination between islands. A generation-tagging fix has been implemented to prevent stale CPU reports from leaking solutions between islands. The branch should be considered experimental.

### Running it

```bash
git checkout elite-experiments
cargo run --release -p solver-native --bin solver -- --league winter-elite

# With deduplication (reset islands that converge to the same solution)
cargo run --release -p solver-native --bin solver -- --league winter-elite --dedup

# Change the ratio of islands to CPUs (default: 2, so 8 CPUs = 16 islands)
cargo run --release -p solver-native --bin solver -- --league winter-elite --island-ratio 4
```

## Running the web viewer

```bash
cd docs
npm install
npm run dev
```

## Re-evaluating results with updated weights

```bash
cargo run --release --bin rescore -- ../weights.json results/gpu
```
