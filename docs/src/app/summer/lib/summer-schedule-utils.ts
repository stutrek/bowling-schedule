import summerWeights from '../../../../../summer_fixed_weights.json';

export const S_TEAMS = 12;
export const S_WEEKS = 10;
export const S_SLOTS = 5;
export const S_PAIRS = 4;
export const S_LANES = 4;

export const slotNames = ['Game 1', 'Game 2', 'Game 3', 'Game 4', 'Game 5'];

export interface SummerMatchup {
    teamA: number;
    teamB: number;
}

/** [week][slot][lanePair] — null means empty (slot 4, pairs 0-1) */
export type SummerSchedule = (SummerMatchup | null)[][][];

export interface SummerCostBreakdown {
    matchupBalance: number;
    slotBalance: number;
    laneBalance: number;
    game5LaneBalance: number;
    commissionerOverlap: number;
    matchupSpacing: number;
    breakBalance: number;
    total: number;
}

export interface SummerViolations {
    spacingPairs: Set<string>; // "week-lo-hi" for matchups too close together
}

export interface SummerAnalysis {
    matchups: number[][]; // [teamA][teamB] count
    laneCounts: number[][]; // [lane][team] count
    slotCounts: number[][]; // [slot][team] count
    teamWeekSlots: number[][][]; // [team][week] -> sorted slot indices
    laneSwitchCounts: { consecutive: number; postBreak: number }[]; // per team
    breakCounts: number[]; // per team: weeks where team has a break (non-consecutive slots)
}

export function isValidPosition(slot: number, pair: number): boolean {
    return slot < 4 ? pair < S_PAIRS : pair >= 2 && pair < S_PAIRS;
}

export function createEmptySchedule(): SummerSchedule {
    return Array.from({ length: S_WEEKS }, () =>
        Array.from({ length: S_SLOTS }, (_, slot) =>
            Array.from({ length: S_PAIRS }, (_, pair) =>
                isValidPosition(slot, pair) ? null : null,
            ),
        ),
    );
}

export function parseSummerTSV(tsv: string): SummerSchedule | null {
    const lines = tsv.trim().split('\n');
    if (lines.length < 2) return null;

    const schedule = createEmptySchedule();
    let foundAny = false;

    for (let i = 1; i < lines.length; i++) {
        const cols = lines[i].split('\t');
        if (cols.length < 6) continue;

        const week = Number.parseInt(cols[0].trim(), 10) - 1;
        const slot = Number.parseInt(cols[1].trim(), 10) - 1;

        if (Number.isNaN(week) || Number.isNaN(slot)) continue;
        if (week < 0 || week >= S_WEEKS || slot < 0 || slot >= S_SLOTS)
            continue;

        for (let pair = 0; pair < S_PAIRS; pair++) {
            const cell = cols[pair + 2]?.trim();
            if (!cell || cell === '-') continue;

            const match = cell.match(/(\d+)\s*v\s*(\d+)/);
            if (!match) continue;

            const teamA = Number.parseInt(match[1], 10) - 1;
            const teamB = Number.parseInt(match[2], 10) - 1;
            schedule[week][slot][pair] = { teamA, teamB };
            foundAny = true;
        }
    }

    return foundAny ? schedule : null;
}

export function summerScheduleToTSV(schedule: SummerSchedule): string {
    const rows: string[] = [
        ['Week', 'Slot', 'Lane 1', 'Lane 2', 'Lane 3', 'Lane 4'].join('\t'),
    ];

    for (let week = 0; week < S_WEEKS; week++) {
        for (let slot = 0; slot < S_SLOTS; slot++) {
            const cells = [String(week + 1), String(slot + 1)];
            for (let pair = 0; pair < S_PAIRS; pair++) {
                const m = schedule[week][slot][pair];
                if (m) {
                    cells.push(`${m.teamA + 1} v ${m.teamB + 1}`);
                } else {
                    cells.push('-');
                }
            }
            rows.push(cells.join('\t'));
        }
    }

    return rows.join('\n');
}

export function analyzeSummerSchedule(
    schedule: SummerSchedule,
): SummerAnalysis {
    const matchups = Array.from({ length: S_TEAMS }, () =>
        new Array(S_TEAMS).fill(0),
    );
    const laneCounts = Array.from({ length: S_LANES }, () =>
        new Array(S_TEAMS).fill(0),
    );
    const slotCounts = Array.from({ length: S_SLOTS }, () =>
        new Array(S_TEAMS).fill(0),
    );
    // [team][week] -> sorted slot indices
    const teamWeekSlots: number[][][] = Array.from({ length: S_TEAMS }, () =>
        Array.from({ length: S_WEEKS }, () => []),
    );

    for (let w = 0; w < S_WEEKS; w++) {
        for (let s = 0; s < S_SLOTS; s++) {
            for (let p = 0; p < S_PAIRS; p++) {
                const m = schedule[w][s][p];
                if (!m) continue;

                const { teamA, teamB } = m;
                matchups[teamA][teamB]++;
                matchups[teamB][teamA]++;

                laneCounts[p][teamA]++;
                laneCounts[p][teamB]++;

                slotCounts[s][teamA]++;
                slotCounts[s][teamB]++;

                teamWeekSlots[teamA][w].push(s);
                teamWeekSlots[teamB][w].push(s);
            }
        }
    }

    // Sort slots per team per week
    for (let t = 0; t < S_TEAMS; t++) {
        for (let w = 0; w < S_WEEKS; w++) {
            teamWeekSlots[t][w].sort((a, b) => a - b);
        }
    }

    // Lane switches
    const laneSwitchCounts = Array.from({ length: S_TEAMS }, () => ({
        consecutive: 0,
        postBreak: 0,
    }));

    for (let w = 0; w < S_WEEKS; w++) {
        for (let t = 0; t < S_TEAMS; t++) {
            const games: { slot: number; pair: number }[] = [];
            for (let s = 0; s < S_SLOTS; s++) {
                for (let p = 0; p < S_PAIRS; p++) {
                    const m = schedule[w][s][p];
                    if (!m) continue;
                    if (m.teamA === t || m.teamB === t) {
                        games.push({ slot: s, pair: p });
                    }
                }
            }
            games.sort((a, b) => a.slot - b.slot);

            for (let i = 0; i < games.length - 1; i++) {
                const gap = games[i + 1].slot - games[i].slot - 1;
                if (games[i].pair !== games[i + 1].pair) {
                    if (gap === 0) {
                        laneSwitchCounts[t].consecutive++;
                    } else {
                        laneSwitchCounts[t].postBreak++;
                    }
                }
            }
        }
    }

    // Break counts: weeks where team has non-consecutive slots (a break game)
    const breakCounts = new Array(S_TEAMS).fill(0);
    for (let t = 0; t < S_TEAMS; t++) {
        for (let w = 0; w < S_WEEKS; w++) {
            const slots = teamWeekSlots[t][w];
            if (slots.length !== 3) continue;
            const isConsecutive =
                slots[1] === slots[0] + 1 && slots[2] === slots[1] + 1;
            if (!isConsecutive) {
                breakCounts[t]++;
            }
        }
    }

    return {
        matchups,
        laneCounts,
        slotCounts,
        teamWeekSlots,
        laneSwitchCounts,
        breakCounts,
    };
}

export function evaluateSummerCost(
    schedule: SummerSchedule,
): SummerCostBreakdown {
    const w8 = summerWeights;

    const matchupCounts = new Array(S_TEAMS * S_TEAMS).fill(0);
    const laneCnts = new Array(S_TEAMS * S_LANES).fill(0);
    const slotCnts = new Array(S_TEAMS * S_SLOTS).fill(0);
    // game 5 (slot 4) lane counts per team
    const game5Lane2 = new Array(S_TEAMS).fill(0);
    const game5Lane3 = new Array(S_TEAMS).fill(0);
    // Track which slots each team plays in per week (for commissioner + break balance)
    const teamWeekSlots: number[][][] = Array.from({ length: S_WEEKS }, () =>
        Array.from({ length: S_TEAMS }, () => []),
    );

    for (let w = 0; w < S_WEEKS; w++) {
        for (let s = 0; s < S_SLOTS; s++) {
            for (let p = 0; p < S_PAIRS; p++) {
                const m = schedule[w][s][p];
                if (!m) continue;

                const t1 = m.teamA;
                const t2 = m.teamB;
                const lo = Math.min(t1, t2);
                const hi = Math.max(t1, t2);

                matchupCounts[lo * S_TEAMS + hi]++;
                laneCnts[t1 * S_LANES + p]++;
                laneCnts[t2 * S_LANES + p]++;
                slotCnts[t1 * S_SLOTS + s]++;
                slotCnts[t2 * S_SLOTS + s]++;

                teamWeekSlots[w][t1].push(s);
                if (t2 !== t1) {
                    teamWeekSlots[w][t2].push(s);
                }

                if (s === 4) {
                    // Game 5 lane balance
                    if (p === 2) {
                        game5Lane2[t1]++;
                        game5Lane2[t2]++;
                    } else if (p === 3) {
                        game5Lane3[t1]++;
                        game5Lane3[t2]++;
                    }
                }
            }
        }
    }

    // 1. Matchup balance: penalize pairs outside [2, 3], scaled by distance
    let matchupBalance = 0;
    for (let i = 0; i < S_TEAMS; i++) {
        for (let j = i + 1; j < S_TEAMS; j++) {
            const c = matchupCounts[i * S_TEAMS + j];
            if (c < 2) {
                matchupBalance += w8.matchup_balance * (2 - c);
            } else if (c > 3) {
                matchupBalance += w8.matchup_balance * (c - 3);
            }
        }
    }

    // 2. Slot balance: slots 0-3 target [6,7], slot 4 target [3,4], scaled by distance
    let slotBalance = 0;
    for (let t = 0; t < S_TEAMS; t++) {
        for (let s = 0; s < S_SLOTS; s++) {
            const c = slotCnts[t * S_SLOTS + s];
            if (s < 4) {
                if (c < 6) slotBalance += w8.slot_balance * (6 - c);
                else if (c > 7) slotBalance += w8.slot_balance * (c - 7);
            } else {
                if (c < 3) slotBalance += w8.slot_balance * (3 - c);
                else if (c > 4) slotBalance += w8.slot_balance * (c - 4);
            }
        }
    }

    // 3. Lane balance: lanes 0-1 target [6,7], lanes 2-3 target [8,9]
    let laneBalance = 0;
    for (let t = 0; t < S_TEAMS; t++) {
        for (let l = 0; l < S_LANES; l++) {
            const c = laneCnts[t * S_LANES + l];
            const [lo, hi] = l < 2 ? [6, 7] : [8, 9];
            if (c < lo) laneBalance += w8.lane_balance * (lo - c);
            else if (c > hi) laneBalance += w8.lane_balance * (c - hi);
        }
    }

    // 4. Game 5 lane balance: per team, penalize if lane 2 vs lane 3 diff > 1
    let game5LaneBalance = 0;
    for (let t = 0; t < S_TEAMS; t++) {
        const diff = Math.abs(game5Lane2[t] - game5Lane3[t]);
        if (diff > 1) {
            game5LaneBalance += w8.game5_lane_balance * (diff - 1);
        }
    }

    // 5. Commissioner overlap: minimize co-appearance in slot 0 and slot 4
    const teamSlot0 = Array.from({ length: S_TEAMS }, () =>
        new Array(S_WEEKS).fill(false),
    );
    const teamSlot4 = Array.from({ length: S_TEAMS }, () =>
        new Array(S_WEEKS).fill(false),
    );
    for (let w = 0; w < S_WEEKS; w++) {
        for (let t = 0; t < S_TEAMS; t++) {
            for (const s of teamWeekSlots[w][t]) {
                if (s === 0) teamSlot0[t][w] = true;
                if (s === 4) teamSlot4[t][w] = true;
            }
        }
    }

    let minCo = 20;
    for (let i = 0; i < S_TEAMS; i++) {
        for (let j = i + 1; j < S_TEAMS; j++) {
            let co = 0;
            for (let w = 0; w < S_WEEKS; w++) {
                if (teamSlot0[i][w] && teamSlot0[j][w]) co++;
                if (teamSlot4[i][w] && teamSlot4[j][w]) co++;
            }
            if (co < minCo) minCo = co;
        }
    }
    const commissionerOverlap = w8.commissioner_overlap * minCo;

    // 7. Matchup spacing: penalize pairs playing too close together
    // 2 total matchups → need 4+ weeks apart; 3 total → need 2+ weeks apart
    const pairWeeks: number[][] = Array.from(
        { length: S_TEAMS * S_TEAMS },
        () => [],
    );
    for (let w = 0; w < S_WEEKS; w++) {
        for (let s = 0; s < S_SLOTS; s++) {
            for (let p = 0; p < S_PAIRS; p++) {
                const m = schedule[w][s][p];
                if (!m) continue;
                const lo = Math.min(m.teamA, m.teamB);
                const hi = Math.max(m.teamA, m.teamB);
                pairWeeks[lo * S_TEAMS + hi].push(w);
            }
        }
    }
    let matchupSpacing = 0;
    for (let i = 0; i < S_TEAMS; i++) {
        for (let j = i + 1; j < S_TEAMS; j++) {
            const weeks = pairWeeks[i * S_TEAMS + j];
            if (weeks.length < 2) continue;
            weeks.sort((a, b) => a - b);
            const minGap = weeks.length === 2 ? 4 : 2;
            for (let k = 1; k < weeks.length; k++) {
                if (weeks[k] - weeks[k - 1] < minGap) {
                    matchupSpacing += w8.matchup_spacing;
                }
            }
        }
    }

    // 8. Break balance: each team should have break weeks ~1/3 of the time
    // Count break positions per week by checking all teams' slot patterns
    let breakPositionsPerWeek = 0;
    // Check week 0 to count how many teams have non-consecutive slots
    for (let t = 0; t < S_TEAMS; t++) {
        const slots = teamWeekSlots[0][t];
        if (slots.length === 3) {
            const isConsecutive =
                slots[1] === slots[0] + 1 && slots[2] === slots[1] + 1;
            if (!isConsecutive) breakPositionsPerWeek++;
        }
    }
    const totalBreakSlots = breakPositionsPerWeek * S_WEEKS;
    const targetLo = Math.floor(totalBreakSlots / S_TEAMS);
    const targetHi = Math.ceil(totalBreakSlots / S_TEAMS);
    const breakCounts = new Array(S_TEAMS).fill(0);
    for (let w = 0; w < S_WEEKS; w++) {
        for (let t = 0; t < S_TEAMS; t++) {
            const slots = teamWeekSlots[w][t];
            if (slots.length !== 3) continue;
            const isConsecutive =
                slots[1] === slots[0] + 1 && slots[2] === slots[1] + 1;
            if (!isConsecutive) breakCounts[t]++;
        }
    }
    let breakBalance = 0;
    for (let t = 0; t < S_TEAMS; t++) {
        const c = breakCounts[t];
        if (c < targetLo) breakBalance += w8.break_balance * (targetLo - c);
        else if (c > targetHi)
            breakBalance += w8.break_balance * (c - targetHi);
    }

    const total =
        matchupBalance +
        slotBalance +
        laneBalance +
        game5LaneBalance +
        commissionerOverlap +
        matchupSpacing +
        breakBalance;

    return {
        matchupBalance,
        slotBalance,
        laneBalance,
        game5LaneBalance,
        commissionerOverlap,
        matchupSpacing,
        breakBalance,
        total,
    };
}

export function computeSummerViolations(
    schedule: SummerSchedule,
): SummerViolations {
    // Build per-pair week lists
    const pairWeeks: number[][] = Array.from(
        { length: S_TEAMS * S_TEAMS },
        () => [],
    );
    for (let w = 0; w < S_WEEKS; w++) {
        for (let s = 0; s < S_SLOTS; s++) {
            for (let p = 0; p < S_PAIRS; p++) {
                const m = schedule[w][s][p];
                if (!m) continue;
                const lo = Math.min(m.teamA, m.teamB);
                const hi = Math.max(m.teamA, m.teamB);
                pairWeeks[lo * S_TEAMS + hi].push(w);
            }
        }
    }

    const spacingPairs = new Set<string>();
    for (let i = 0; i < S_TEAMS; i++) {
        for (let j = i + 1; j < S_TEAMS; j++) {
            const weeks = pairWeeks[i * S_TEAMS + j];
            if (weeks.length < 2) continue;
            weeks.sort((a, b) => a - b);
            const minGap = weeks.length === 2 ? 4 : 2;
            for (let k = 1; k < weeks.length; k++) {
                if (weeks[k] - weeks[k - 1] < minGap) {
                    spacingPairs.add(`${weeks[k - 1]}-${i}-${j}`);
                    spacingPairs.add(`${weeks[k]}-${i}-${j}`);
                }
            }
        }
    }

    return { spacingPairs };
}
