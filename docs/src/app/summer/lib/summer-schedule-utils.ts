import summerWeights from '../../../../../summer_weights.json';

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
    laneSwitches: number;
    laneSwitchBreak: number;
    timeGaps: number;
    laneBalance: number;
    commissionerOverlap: number;
    repeatMatchupSameNight: number;
    slotBalance: number;
    total: number;
}

export interface SummerAnalysis {
    matchups: number[][]; // [teamA][teamB] count
    laneCounts: number[][]; // [lane][team] count
    slotCounts: number[][]; // [slot][team] count
    teamWeekSlots: number[][][]; // [team][week] -> sorted slot indices
    laneSwitchCounts: { consecutive: number; postBreak: number }[]; // per team
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

    return {
        matchups,
        laneCounts,
        slotCounts,
        teamWeekSlots,
        laneSwitchCounts,
    };
}

export function evaluateSummerCost(
    schedule: SummerSchedule,
): SummerCostBreakdown {
    const w8 = summerWeights;

    const matchupCounts = Array.from({ length: S_TEAMS * S_TEAMS }, () => 0);
    const weekMatchupCounts = Array.from(
        { length: S_WEEKS * S_TEAMS * S_TEAMS },
        () => 0,
    );
    const laneCnts = Array.from({ length: S_TEAMS * S_LANES }, () => 0);
    const slotCnts = Array.from({ length: S_TEAMS * S_SLOTS }, () => 0);
    const teamWeekSlots: number[][][] = Array.from({ length: S_WEEKS }, () =>
        Array.from({ length: S_TEAMS }, () => []),
    );

    // First pass: collect data
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
                weekMatchupCounts[w * S_TEAMS * S_TEAMS + lo * S_TEAMS + hi]++;

                laneCnts[t1 * S_LANES + p]++;
                laneCnts[t2 * S_LANES + p]++;

                slotCnts[t1 * S_SLOTS + s]++;
                slotCnts[t2 * S_SLOTS + s]++;

                teamWeekSlots[w][t1].push(s);
                if (t2 !== t1) {
                    teamWeekSlots[w][t2].push(s);
                }
            }
        }
        for (let t = 0; t < S_TEAMS; t++) {
            teamWeekSlots[w][t].sort((a, b) => a - b);
        }
    }

    // 1. Matchup balance: penalize pairs outside [2, 3]
    let matchupBalance = 0;
    for (let i = 0; i < S_TEAMS; i++) {
        for (let j = i + 1; j < S_TEAMS; j++) {
            const c = matchupCounts[i * S_TEAMS + j];
            if (c < 2 || c > 3) {
                matchupBalance += w8.matchup_balance;
            }
        }
    }

    // 2. Lane switches
    let laneSwitches = 0;
    let laneSwitchBreak = 0;
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
                        laneSwitches += w8.lane_switch_consecutive;
                    } else {
                        laneSwitchBreak += w8.lane_switch_post_break;
                    }
                }
            }
        }
    }

    // 3. Time gaps
    let timeGaps = 0;
    for (let w = 0; w < S_WEEKS; w++) {
        for (let t = 0; t < S_TEAMS; t++) {
            const slots = teamWeekSlots[w][t];
            if (slots.length < 2) continue;

            // 3 consecutive (all gaps = 0)
            if (
                slots.length === 3 &&
                slots[1] === slots[0] + 1 &&
                slots[2] === slots[1] + 1
            ) {
                timeGaps += w8.time_gap_consecutive;
            }

            for (let i = 0; i < slots.length - 1; i++) {
                const gap = slots[i + 1] - slots[i] - 1;
                if (gap >= 2) {
                    timeGaps += w8.time_gap_large;
                }
            }
        }
    }

    // 4. Lane balance — penalty for counts outside [7, 8]
    let laneBalance = 0;
    for (let t = 0; t < S_TEAMS; t++) {
        for (let l = 0; l < S_LANES; l++) {
            const c = laneCnts[t * S_LANES + l];
            if (c < 7 || c > 8) {
                laneBalance += w8.lane_balance;
            }
        }
    }

    // 5. Commissioner overlap
    const teamWeekSlotSet: boolean[][] = Array.from(
        { length: S_TEAMS * S_WEEKS },
        () => Array.from({ length: S_SLOTS }, () => false),
    );
    for (let w = 0; w < S_WEEKS; w++) {
        for (let t = 0; t < S_TEAMS; t++) {
            for (const s of teamWeekSlots[w][t]) {
                teamWeekSlotSet[t * S_WEEKS + w][s] = true;
            }
        }
    }

    let minCo = Number.POSITIVE_INFINITY;
    for (let i = 0; i < S_TEAMS; i++) {
        for (let j = i + 1; j < S_TEAMS; j++) {
            let co = 0;
            for (let w = 0; w < S_WEEKS; w++) {
                if (
                    teamWeekSlotSet[i * S_WEEKS + w][0] &&
                    teamWeekSlotSet[j * S_WEEKS + w][0]
                ) {
                    co++;
                }
                if (
                    teamWeekSlotSet[i * S_WEEKS + w][4] &&
                    teamWeekSlotSet[j * S_WEEKS + w][4]
                ) {
                    co++;
                }
            }
            if (co < minCo) minCo = co;
        }
    }
    const commissionerOverlap = w8.commissioner_overlap * minCo;

    // 6. Repeat matchups same night
    let repeatMatchupSameNight = 0;
    for (let w = 0; w < S_WEEKS; w++) {
        for (let i = 0; i < S_TEAMS; i++) {
            for (let j = i + 1; j < S_TEAMS; j++) {
                const c =
                    weekMatchupCounts[w * S_TEAMS * S_TEAMS + i * S_TEAMS + j];
                if (c > 1) {
                    repeatMatchupSameNight +=
                        (c - 1) * w8.repeat_matchup_same_night;
                }
            }
        }
    }

    // 7. Slot balance
    let slotBalance = 0;
    for (let t = 0; t < S_TEAMS; t++) {
        for (let s = 0; s < S_SLOTS; s++) {
            const c = slotCnts[t * S_SLOTS + s];
            const ok = s < 4 ? c === 6 || c === 7 : c === 3 || c === 4;
            if (!ok) {
                slotBalance += w8.slot_balance;
            }
        }
    }

    const total =
        matchupBalance +
        laneSwitches +
        laneSwitchBreak +
        timeGaps +
        laneBalance +
        commissionerOverlap +
        repeatMatchupSameNight +
        slotBalance;

    return {
        matchupBalance,
        laneSwitches,
        laneSwitchBreak,
        timeGaps,
        laneBalance,
        commissionerOverlap,
        repeatMatchupSameNight,
        slotBalance,
        total,
    };
}
