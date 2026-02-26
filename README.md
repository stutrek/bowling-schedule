# Bowling Schedule Optimizer

An optimized 12-week bowling schedule for 16 teams across 4 lanes (5, 6, 7, 8), built with a quad-based structure that's better for bowlers and easier for computers to optimize.

**[Try the interactive web UI →](https://stutrek.github.io/bowling-schedule/)**

## The Problem and the Big Improvement

The current schedule has 16 teams playing two games each week. But a team can end up on completely different lanes between their two games -- maybe lanes 5 and 7, or lanes 6 and 8. This causes real problems on bowling night.

### Quads

The new schedule uses **quads**. A quad is a group of 4 teams assigned to one lane pair for one half of a night. They play two games back-to-back on the same two lanes. Between games, you either bowl on the same lane twice, or you switch to the lane next to you -- on the same ball return. Every team in your quad plays every other team in your quad across the two games. Each week has 4 quads (2 early, 2 late), and every team appears in exactly one quad per week.

### Why quads are better

The current schedule doesn't use quads -- teams can end up on completely different lanes between their two games. Keeping teams on the same lane pair means:

- If a lane breaks down, the delay only affects the other lane in your pair, not the whole league
- One team in each game just bowled on that same lane -- no warm-up needed
- The ball return doesn't get clogged with house balls from the previous group
- Your jacket and bag stay right where you left them
- Teams aren't awkwardly standing around waiting for their next game with nowhere to sit

### The scheduling challenge

Quads are great for bowlers, but they make scheduling harder. On top of the quad structure, the schedule needs to be fair across six dimensions -- and optimizing one often makes another worse. A single "cost" score (lower is better, 0 is perfect) measures how good a schedule is:

1. **Matchup balance** -- every pair of teams should play each other 1 or 2 times across the season (penalty for 0 or 3+)
2. **Consecutive opponents** -- no pair should play each other in back-to-back weeks
3. **Early/late balance** -- each team should play 6 weeks early and 6 weeks late
4. **Early/late alternation** -- no team should play early (or late) three weeks in a row
5. **Lane balance** -- each team should play on each of the 4 lanes roughly equally (6 times each)
6. **Lane switches** -- each team should stay on their lane about half the time and switch about half the time
7. **Last game lane balance** -- each team's 6 last games of the night should be spread evenly across the 4 lanes (1-2 times each). This prevents a team from always ending their night on a problematic lane.

The relative importance of each constraint is configured in [weights.json](weights.json).

The position of a team within its quad determines three things at once: which lane they're on, who they play, and whether they stay or switch. This compact representation means a single swap of two players between quads changes matchups, lane balance, and stay/switch all at once -- which is why the optimizer can explore the search space so efficiently.

## The Algorithm

The solver uses a technique called [parallel tempering](https://en.wikipedia.org/wiki/Parallel_tempering), which is an advanced form of [simulated annealing](https://en.wikipedia.org/wiki/Simulated_annealing). The core idea: imagine trying to solve a jigsaw puzzle by randomly swapping pieces. If a swap makes things better, keep it. If it makes things worse, sometimes keep it anyway -- to avoid getting stuck in a dead end. Run many copies of this process at different levels of "willingness to accept bad swaps," and let them share good solutions with each other.

The implementation lives in [`solver-native/src/split_sa.rs`](solver-native/src/split_sa.rs).

### Temperature ladder

Each CPU core runs at a different "temperature." Cold cores are picky -- they mostly only accept swaps that improve the schedule. Hot cores are adventurous -- they'll accept worse schedules to explore new territory. The temperatures are spread in a geometric progression across all available cores.

### Move types

Each iteration, the solver picks one of 11 types of random changes to try. The percentage is the base probability before adaptive adjustment.

- **Inter-quad player swap (25%)** -- Pick two quads in the same week, swap one player between them. The workhorse move -- it directly changes matchups, lane assignments, and stay/switch in one step.
- **Intra-quad player swap (15%)** -- Swap two players within the same quad. Changes lane and stay/switch assignments without affecting who plays whom.
- **Team cross-week swap (10%)** -- Pick a team, find where it sits in two different weeks, and swap its position with the displaced teams in both weeks. Helps spread out repeat matchups and balance early/late across the season.
- **Quad swap (8%)** -- Trade two entire quads within a week. Moves all four players to different lanes at once.
- **Week swap (6%)** -- Swap two entire weeks. Reorders the season without changing any matchups -- just changes when they happen, which helps avoid back-to-back repeat opponents.
- **Early/late flip (6%)** -- Take everyone who was playing early in a week and make them late, and vice versa.
- **Lane pair swap (5%)** -- Swap the two early quads with each other and the two late quads with each other. Moves teams between lane pairs (5-6 and 7-8) while preserving who plays early vs late.
- **Stay/switch rotation (6%)** -- Within a quad, swap who stays on their lane vs who switches between games.
- **Targeted: fix a missing matchup (6%)** -- Find two teams that haven't played each other, find a week where they're in neighboring quads, and swap one into the other's quad.
- **Targeted: fix lane imbalance (6%)** -- Find the team with the most lopsided lane distribution, find a quad containing that team, and swap their position within the quad to change their lane.
- **Targeted: fix early/late imbalance (7%)** -- Find the team that's played early (or late) too many times and flip a week's early/late assignment to correct it.

### Adaptive move selection

The solver tracks which move types are producing accepted changes and gives them higher probability. Move weights are recomputed every 10,000 iterations based on each move's recent acceptance rate.

### Compound moves

Near the end of optimization (when cost is already low), the solver bundles multiple inter-quad swaps together as one big move -- up to 12 swaps at once. This helps escape dead ends that single swaps can't get out of.

### Exhaustive search

Every 100,000 iterations, the solver picks two quads in the same week and brute-forces every possible player swap between them (only 16 possibilities). It keeps the best one. A periodic sanity check that finds improvements the random moves might miss.

### Sharing between cores

Every 100,000 iterations, all cores pause and adjacent-temperature pairs may swap their schedules. This is called [replica exchange](https://en.wikipedia.org/wiki/Parallel_tempering#Replica_exchange). It lets good solutions found by adventurous (hot) cores flow down to the picky (cold) cores for refinement.

### Anti-stagnation

If the best known score hasn't improved in a while, the solver shakes things up. Cores that converged to the same solution get re-seeded from a scrambled version of the global best. The hottest cores (top third of the temperature ladder) also get re-seeded. The amount of scrambling increases the longer stagnation persists.

### Auto-save and restart

The solver saves the schedule to a TSV file whenever cost drops below 100. On reaching cost 0 (a perfect schedule), it saves immediately and starts a fresh search for another solution.

### Phase modes

The solver can run in three modes:

- **`--full`** -- Solve the entire 12-week schedule at once. All weeks and quads are unlocked.
- **Phase 1** (default) -- Solve the first 7.5 weeks. Useful for breaking the problem in half.
- **Phase 2** (`--phase2`) -- Solve the second half, seeded from Phase 1 results.

## Results

Best known result: **cost 85** (matchup 0, consecutive 0, early/late balance 0, alternation 60, lane 20, switch 5). This can be loaded in the web viewer via the "Load Best" button.

## Architecture

- `solver-core/` -- Shared Rust library with data structures, the evaluation function, and TSV I/O
- `solver-native/` -- Command-line solver binaries. `split-sa` is the primary solver; other binaries (`construct`, `first-half`, `complete`, `refine`) are earlier experiments
- `solver-wasm/` -- Browser-compatible build of the evaluation function, used by the web UI
- `docs/` -- Next.js web app with WASM-powered scoring, an interactive schedule editor, and TSV import/export
- `src/` -- Original TypeScript prototypes (pinsetter1-6 and early SA attempts)

## What Didn't Work

### Manual scheduling

The original schedule was created by hand. It was decent, but the goal was to do better with software.

### Rule-based construction (pinsetter1-6)

Six different TypeScript algorithms that try to build a schedule step by step using mathematical patterns -- rotations, interleaving, remappings. Pinsetter1 came closest: perfect matchups, early/late, alternation, and consecutive opponents. But it couldn't get lane balance right. The fundamental issue is that you can't build a schedule with simple rules and have all six constraints come out balanced.

### Locking early/late and optimizing the rest

Restricted the solver to only make changes that preserve the early/late assignments. Early/late metrics were perfect, but matchup balance was catastrophic (5120 penalty). The constraint was too tight -- teams that always shared the same time slot could never be paired against each other.

### Hybrid: pinsetter1 + simulated annealing

Used pinsetter1's schedule as a starting point, then ran the optimizer with limited moves to fix lane balance. Failed because converting between data formats lost information about player positions, corrupting the schedule.

### Prioritizing "hard" constraints

Tried making some constraints 1000x more important than others ([lexicographic optimization](https://en.wikipedia.org/wiki/Lexicographic_optimization)). Made the scoring landscape too steep for the solver to navigate -- it could never satisfy the hard constraints because accepting any move that helped a soft constraint was nearly impossible.

### Build early/late first, then matchups

Constructed a perfect early/late assignment matrix, then tried to optimize matchups within that framework. Got good matchups, but introduced early/late alternation violations -- fixing one dimension broke another.

### Constraint satisfaction / backtracking

Tried building the schedule by placing teams one at a time, backtracking when constraints were violated -- like solving a Sudoku (`construct.rs`, `first_half.rs`). Could find partial solutions but didn't scale to the full 12-week schedule with all six soft constraints.

## Running It

```bash
# Solve the full schedule using all available cores
cargo run --release --bin split-sa -- --full

# Specify number of cores
cargo run --release --bin split-sa -- --full --cores 10

# Seed from an existing schedule
cargo run --release --bin split-sa -- --full --cores 10 path/to/seed.tsv
```

Results are saved to `solver-native/results/split-sa/full/`.

To run the web viewer:

```bash
cd docs
npm install
npm run dev
```
