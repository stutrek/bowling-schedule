import type { Schedule } from './Schedule';
import type { Config } from './types';

export type Assignment = number[][][];

export interface CostBreakdown {
    matchupBalance: number;
    consecutiveOpponents: number;
    earlyLateBalance: number;
    earlyLateAlternation: number;
    laneBalance: number;
    laneSwitchBalance: number;
    total: number;
}

function cloneAssignment(a: Assignment): Assignment {
    return a.map((w) => w.map((g) => [...g]));
}

/**
 * Quadrant layout per week:
 *   q=0: early, lanes 0-1    q=1: early, lanes 2-3
 *   q=2: late,  lanes 0-1    q=3: late,  lanes 2-3
 *
 * fillGroup([A,B,C,D]) produces:
 *   A vs B (slot+0, lane+0)   C vs D (slot+0, lane+1)
 *   A vs D (slot+1, lane+0)   C vs B (slot+1, lane+1)
 */

function randomAssignment(config: Config): Assignment {
    const { teams: T, days: W } = config;
    const a: Assignment = [];
    for (let w = 0; w < W; w++) {
        const teams = Array.from({ length: T }, (_, i) => i);
        for (let i = T - 1; i > 0; i--) {
            const j = Math.floor(Math.random() * (i + 1));
            [teams[i], teams[j]] = [teams[j], teams[i]];
        }
        a.push([
            teams.slice(0, 4),
            teams.slice(4, 8),
            teams.slice(8, 12),
            teams.slice(12, 16),
        ]);
    }
    return a;
}

function evaluateAssignment(a: Assignment, config: Config): CostBreakdown {
    const { teams: T, days: W, lanes: L } = config;
    const matchups = new Int32Array(T * T);
    const weekMatchup = new Uint8Array(W * T * T);
    const laneCounts = new Int32Array(T * L);
    const stayCount = new Int32Array(T);
    const earlyCount = new Int32Array(T);
    const earlyLate = new Uint8Array(T * W);

    for (let w = 0; w < W; w++) {
        for (let q = 0; q < 4; q++) {
            const [pa, pb, pc, pd] = a[w][q];
            const early = q < 2 ? 1 : 0;
            const laneOff = (q % 2) * 2;

            const pairs: [number, number][] = [
                [pa, pb],
                [pc, pd],
                [pa, pd],
                [pc, pb],
            ];
            for (const [t1, t2] of pairs) {
                const lo = t1 < t2 ? t1 : t2;
                const hi = t1 < t2 ? t2 : t1;
                matchups[lo * T + hi]++;
                weekMatchup[w * T * T + lo * T + hi] = 1;
            }

            laneCounts[pa * L + laneOff] += 2;
            laneCounts[pb * L + laneOff]++;
            laneCounts[pb * L + laneOff + 1]++;
            laneCounts[pc * L + laneOff + 1] += 2;
            laneCounts[pd * L + laneOff + 1]++;
            laneCounts[pd * L + laneOff]++;

            // pa and pc stay on the same lane both games; pb and pd switch
            stayCount[pa]++;
            stayCount[pc]++;

            for (const t of [pa, pb, pc, pd]) {
                earlyLate[t * W + w] = early;
                if (early) earlyCount[t]++;
            }
        }
    }

    for (let t = 0; t < T; t++) earlyCount[t] = Math.round(earlyCount[t] / 2);

    let matchupBalance = 0;
    for (let i = 0; i < T; i++) {
        for (let j = i + 1; j < T; j++) {
            const c = matchups[i * T + j];
            if (c === 0) matchupBalance += 30;
            else if (c >= 3) matchupBalance += (c - 2) * 40;
        }
    }

    let consecutiveOpponents = 0;
    for (let w = 0; w < W - 1; w++) {
        const b1 = w * T * T;
        const b2 = (w + 1) * T * T;
        for (let i = 0; i < T; i++) {
            for (let j = i + 1; j < T; j++) {
                const idx = i * T + j;
                if (weekMatchup[b1 + idx] && weekMatchup[b2 + idx]) {
                    consecutiveOpponents += 15;
                }
            }
        }
    }

    let earlyLateBalance = 0;
    const targetE = W / 2;
    for (let t = 0; t < T; t++) {
        const dev = Math.abs(earlyCount[t] - targetE);
        earlyLateBalance += dev * dev * 25;
    }

    let earlyLateAlternation = 0;
    for (let t = 0; t < T; t++) {
        for (let w = 0; w < W - 2; w++) {
            const base = t * W;
            if (
                earlyLate[base + w] === earlyLate[base + w + 1] &&
                earlyLate[base + w + 1] === earlyLate[base + w + 2]
            ) {
                earlyLateAlternation += 40;
            }
        }
    }

    let laneBalance = 0;
    const targetL = (W * 2) / L;
    for (let t = 0; t < T; t++) {
        for (let l = 0; l < L; l++) {
            laneBalance += Math.abs(laneCounts[t * L + l] - targetL) * 15;
        }
    }

    let laneSwitchBalance = 0;
    const targetStay = W / 2;
    for (let t = 0; t < T; t++) {
        laneSwitchBalance += Math.abs(stayCount[t] - targetStay) * 5;
    }

    const total =
        matchupBalance +
        consecutiveOpponents +
        earlyLateBalance +
        earlyLateAlternation +
        laneBalance +
        laneSwitchBalance;
    return {
        matchupBalance,
        consecutiveOpponents,
        earlyLateBalance,
        earlyLateAlternation,
        laneBalance,
        laneSwitchBalance,
        total,
    };
}

function applyToSchedule(assignment: Assignment, schedule: Schedule): void {
    schedule.schedule = [];
    schedule.createSchedule();
    for (let w = 0; w < assignment.length; w++) {
        for (let q = 0; q < 4; q++) {
            const [a, b, c, d] = assignment[w][q];
            const slot = q < 2 ? 0 : 2;
            const lane = (q % 2) * 2;
            schedule.setGame(a, b, w, slot, lane);
            schedule.setGame(c, d, w, slot, lane + 1);
            schedule.setGame(a, d, w, slot + 1, lane);
            schedule.setGame(c, b, w, slot + 1, lane + 1);
        }
    }
}

export function solveSchedule(
    schedule: Schedule,
    { maxIterations = 1_000_000, runs = 1, verbose = false } = {},
): CostBreakdown {
    const config = schedule.config;
    const { days: W } = config;

    let globalBest: Assignment | null = null;
    let globalBestCost = Number.POSITIVE_INFINITY;

    for (let run = 0; run < runs; run++) {
        let a = randomAssignment(config);
        let cost = evaluateAssignment(a, config);
        let bestA = cloneAssignment(a);
        let bestCost = cost.total;

        const T0 = 30.0;
        const coolRate = Math.exp(Math.log(0.005 / T0) / maxIterations);
        let temp = T0;
        const restartInterval = 100_000;

        for (let i = 0; i < maxIterations; i++) {
            if (bestCost === 0) break;

            const rand = Math.random();
            let revertFn: (() => void) | null = null;

            if (rand < 0.4) {
                // Inter-group swap: swap two teams between different quadrants in the same week
                const w = Math.floor(Math.random() * W);
                const q1 = Math.floor(Math.random() * 4);
                let q2 = Math.floor(Math.random() * 3);
                if (q2 >= q1) q2++;
                const p1 = Math.floor(Math.random() * 4);
                const p2 = Math.floor(Math.random() * 4);
                const tmp = a[w][q1][p1];
                a[w][q1][p1] = a[w][q2][p2];
                a[w][q2][p2] = tmp;
                revertFn = () => {
                    a[w][q2][p2] = a[w][q1][p1];
                    a[w][q1][p1] = tmp;
                };
            } else if (rand < 0.65) {
                // Intra-group reorder: swap two positions within a quadrant
                const w = Math.floor(Math.random() * W);
                const q = Math.floor(Math.random() * 4);
                const p1 = Math.floor(Math.random() * 4);
                let p2 = Math.floor(Math.random() * 3);
                if (p2 >= p1) p2++;
                const tmp = a[w][q][p1];
                a[w][q][p1] = a[w][q][p2];
                a[w][q][p2] = tmp;
                revertFn = () => {
                    a[w][q][p2] = a[w][q][p1];
                    a[w][q][p1] = tmp;
                };
            } else if (rand < 0.85) {
                // Cross-week team swap: find a team in two weeks, swap its quadrant placement
                const team = Math.floor(Math.random() * config.teams);
                const w1 = Math.floor(Math.random() * W);
                let w2 = Math.floor(Math.random() * (W - 1));
                if (w2 >= w1) w2++;
                let qi1 = -1;
                let pi1 = -1;
                let qi2 = -1;
                let pi2 = -1;
                for (let q = 0; q < 4; q++) {
                    const idx = a[w1][q].indexOf(team);
                    if (idx !== -1) {
                        qi1 = q;
                        pi1 = idx;
                        break;
                    }
                }
                for (let q = 0; q < 4; q++) {
                    const idx = a[w2][q].indexOf(team);
                    if (idx !== -1) {
                        qi2 = q;
                        pi2 = idx;
                        break;
                    }
                }
                if (qi1 === -1 || qi2 === -1) continue;
                const other1 = a[w2][qi1][pi1];
                const other2 = a[w1][qi2][pi2];
                a[w1][qi1][pi1] = other2;
                a[w1][qi2][pi2] = team;
                a[w2][qi2][pi2] = other1;
                a[w2][qi1][pi1] = team;
                // Actually this gets complicated — simpler: swap the team with whoever is at its mirror position
                // Revert: restore original values
                revertFn = () => {
                    a[w1][qi1][pi1] = team;
                    a[w1][qi2][pi2] = other2;
                    a[w2][qi2][pi2] = team;
                    a[w2][qi1][pi1] = other1;
                };
            } else {
                // Quadrant swap: swap two entire quadrants in a week
                const w = Math.floor(Math.random() * W);
                const q1 = Math.floor(Math.random() * 4);
                let q2 = Math.floor(Math.random() * 3);
                if (q2 >= q1) q2++;
                const tmp = a[w][q1];
                a[w][q1] = a[w][q2];
                a[w][q2] = tmp;
                revertFn = () => {
                    a[w][q2] = a[w][q1];
                    a[w][q1] = tmp;
                };
            }

            const newCost = evaluateAssignment(a, config);
            const delta = newCost.total - cost.total;

            if (delta <= 0 || Math.random() < Math.exp(-delta / temp)) {
                cost = newCost;
                if (newCost.total < bestCost) {
                    bestCost = newCost.total;
                    bestA = cloneAssignment(a);
                    if (verbose && bestCost % 50 === 0) {
                        console.log(
                            `[run ${run}, iter ${i}] cost=${bestCost}`,
                            newCost,
                        );
                    }
                }
            } else if (revertFn) {
                revertFn();
            }

            temp *= coolRate;

            if (i > 0 && i % restartInterval === 0 && cost.total > bestCost) {
                if (i % (restartInterval * 2) === 0) {
                    a = randomAssignment(config);
                } else {
                    a = cloneAssignment(bestA);
                    // Perturb the best solution
                    for (let p = 0; p < 20; p++) {
                        const pw = Math.floor(Math.random() * W);
                        const pq1 = Math.floor(Math.random() * 4);
                        let pq2 = Math.floor(Math.random() * 3);
                        if (pq2 >= pq1) pq2++;
                        const pp1 = Math.floor(Math.random() * 4);
                        const pp2 = Math.floor(Math.random() * 4);
                        const tmp = a[pw][pq1][pp1];
                        a[pw][pq1][pp1] = a[pw][pq2][pp2];
                        a[pw][pq2][pp2] = tmp;
                    }
                }
                cost = evaluateAssignment(a, config);
                temp = T0 * 0.3;
            }
        }

        if (bestCost < globalBestCost) {
            globalBestCost = bestCost;
            globalBest = bestA;
        }

        if (verbose) {
            console.log(
                `Run ${run}: best cost=${bestCost}`,
                evaluateAssignment(bestA, config),
            );
        }
    }

    if (globalBest) applyToSchedule(globalBest, schedule);
    return evaluateSchedule(schedule);
}

/** Score an already-filled Schedule using the constraint metrics */
export function evaluateSchedule(schedule: Schedule): CostBreakdown {
    const { teams: T, days: W, lanes: L } = schedule.config;
    const matchups = new Int32Array(T * T);
    const weekMatchup = new Uint8Array(W * T * T);
    const laneCounts = new Int32Array(T * L);
    const earlyCount = new Int32Array(T);
    const earlyLate = new Uint8Array(T * W);
    const teamWeekLane = new Int32Array(T * W).fill(-1);

    for (const g of schedule.schedule) {
        if (g.teams[0] === -1 || g.teams[1] === -1) continue;
        const [a, b] = g.teams;
        const lo = a < b ? a : b;
        const hi = a < b ? b : a;
        matchups[lo * T + hi]++;
        weekMatchup[g.day * T * T + lo * T + hi] = 1;
        laneCounts[a * L + g.lane]++;
        laneCounts[b * L + g.lane]++;
        for (const t of [a, b]) {
            const idx = t * W + g.day;
            if (teamWeekLane[idx] === -1) teamWeekLane[idx] = g.lane;
        }
        const early = g.timeSlot < 2 ? 1 : 0;
        earlyLate[a * W + g.day] = early;
        earlyLate[b * W + g.day] = early;
        if (early) {
            earlyCount[a]++;
            earlyCount[b]++;
        }
    }

    for (let t = 0; t < T; t++) earlyCount[t] = Math.round(earlyCount[t] / 2);

    let matchupBalance = 0;
    for (let i = 0; i < T; i++) {
        for (let j = i + 1; j < T; j++) {
            const c = matchups[i * T + j];
            if (c === 0) matchupBalance += 30;
            else if (c >= 3) matchupBalance += (c - 2) * 40;
        }
    }

    let consecutiveOpponents = 0;
    for (let w = 0; w < W - 1; w++) {
        const b1 = w * T * T;
        const b2 = (w + 1) * T * T;
        for (let i = 0; i < T; i++) {
            for (let j = i + 1; j < T; j++) {
                const idx = i * T + j;
                if (weekMatchup[b1 + idx] && weekMatchup[b2 + idx]) {
                    consecutiveOpponents += 15;
                }
            }
        }
    }

    let earlyLateBalance = 0;
    const targetE = W / 2;
    for (let t = 0; t < T; t++) {
        const dev = Math.abs(earlyCount[t] - targetE);
        earlyLateBalance += dev * dev * 25;
    }

    let earlyLateAlternation = 0;
    for (let t = 0; t < T; t++) {
        for (let w = 0; w < W - 2; w++) {
            const base = t * W;
            if (
                earlyLate[base + w] === earlyLate[base + w + 1] &&
                earlyLate[base + w + 1] === earlyLate[base + w + 2]
            ) {
                earlyLateAlternation += 40;
            }
        }
    }

    let laneBalance = 0;
    const targetL = (W * 2) / L;
    for (let t = 0; t < T; t++) {
        for (let l = 0; l < L; l++) {
            laneBalance += Math.abs(laneCounts[t * L + l] - targetL) * 15;
        }
    }

    const stayCount = new Int32Array(T);
    for (const g of schedule.schedule) {
        if (g.teams[0] === -1 || g.teams[1] === -1) continue;
        for (const t of g.teams) {
            const idx = t * W + g.day;
            if (teamWeekLane[idx] === g.lane) stayCount[t]++;
        }
    }
    // Each team sees 2 games/week; stayCount counted once per game on its first-seen lane
    // so stayCount[t] = stays + switches = 2*W but we only want the stays.
    // Actually: teamWeekLane records the first lane seen. Each game where lane == teamWeekLane
    // is a "same lane" game. Over a week with 2 games, if both are same lane: count += 2.
    // If they switch: count += 1 (only the first game matches).
    // So stays = stayCount[t] - W (subtract the guaranteed first-game matches).
    let laneSwitchBalance = 0;
    const targetStay = W / 2;
    for (let t = 0; t < T; t++) {
        const stays = stayCount[t] - W;
        laneSwitchBalance += Math.abs(stays - targetStay) * 5;
    }

    const total =
        matchupBalance +
        consecutiveOpponents +
        earlyLateBalance +
        earlyLateAlternation +
        laneBalance +
        laneSwitchBalance;
    return {
        matchupBalance,
        consecutiveOpponents,
        earlyLateBalance,
        earlyLateAlternation,
        laneBalance,
        laneSwitchBalance,
        total,
    };
}
