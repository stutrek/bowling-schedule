import { Schedule } from '../../../../src/Schedule';
import type { Config, Game } from '../../../../src/types';

export const config: Config = {
    teams: 16,
    timeSlots: 4,
    lanes: 4,
    days: 12,
};

export const slotNames = ['Early 1', 'Early 2', 'Late 1', 'Late 2'];

export interface Analysis {
    matchups: number[][];
    laneCounts: number[][];
    lastGameLaneCounts: number[][];
    slotCounts: number[][];
    groups: number[][];
    laneSwitchCounts: number[];
}

export interface Violations {
    consecutivePairs: Set<string>;
    earlyLateStreaks: Set<string>;
}

export interface CostBreakdown {
    matchupBalance: number;
    consecutiveOpponents: number;
    earlyLateBalance: number;
    earlyLateAlternation: number;
    laneBalance: number;
    laneSwitchBalance: number;
    lateLaneBalance: number;
    total: number;
}

export function findGame(
    games: Game[],
    day: number,
    slot: number,
    lane: number,
): Game | undefined {
    return games.find(
        (g) => g.day === day && g.timeSlot === slot && g.lane === lane,
    );
}

export function scheduleToTSV(schedule: Schedule): string {
    const { days, timeSlots, lanes } = schedule.config;
    const rows: string[] = [
        ['Week', 'Time', 'Lane 1', 'Lane 2', 'Lane 3', 'Lane 4'].join('\t'),
    ];
    for (let day = 0; day < days; day++) {
        for (let slot = 0; slot < timeSlots; slot++) {
            const cells = [String(day + 1), slotNames[slot]];
            for (let lane = 0; lane < lanes; lane++) {
                const g = findGame(schedule.schedule, day, slot, lane);
                cells.push(
                    g && g.teams[0] !== -1
                        ? `${g.teams[0] + 1} v ${g.teams[1] + 1}`
                        : '',
                );
            }
            rows.push(cells.join('\t'));
        }
    }
    return rows.join('\n');
}

export function parseTSV(tsv: string): Schedule | null {
    const lines = tsv.trim().split('\n');
    if (lines.length < 2) return null;

    const slotMap: Record<string, number> = {
        'early 1': 0,
        'early 2': 1,
        'late 1': 2,
        'late 2': 3,
    };

    const s = new Schedule(config);
    s.createSchedule();

    for (let i = 1; i < lines.length; i++) {
        const cols = lines[i].split('\t');
        if (cols.length < 6) continue;
        const week = Number.parseInt(cols[0].trim(), 10) - 1;
        const slot = slotMap[cols[1].trim().toLowerCase()];
        if (Number.isNaN(week) || slot === undefined) continue;

        for (let lane = 0; lane < 4; lane++) {
            const match = cols[lane + 2]?.trim().match(/(\d+)\s*v\s*(\d+)/);
            if (!match) continue;
            const t1 = Number.parseInt(match[1], 10) - 1;
            const t2 = Number.parseInt(match[2], 10) - 1;
            s.setGame(t1, t2, week, slot, lane);
        }
    }

    return s;
}

export function analyzeSchedule(schedule: Schedule): Analysis {
    const { teams, lanes, timeSlots, days } = schedule.config;
    const matchups = Array.from({ length: teams }, () =>
        new Array(teams).fill(0),
    );
    const laneCounts = Array.from({ length: lanes }, () =>
        new Array(teams).fill(0),
    );
    const lastGameLaneCounts = Array.from({ length: lanes }, () =>
        new Array(teams).fill(0),
    );
    const slotCounts = Array.from({ length: timeSlots }, () =>
        new Array(teams).fill(0),
    );
    const groups = Array.from({ length: teams }, () => new Array(days).fill(0));
    const teamWeekLane = Array.from({ length: teams }, () =>
        new Array(days).fill(-1),
    );

    for (const game of schedule.schedule) {
        if (game.teams[0] === -1 || game.teams[1] === -1) continue;
        matchups[game.teams[0]][game.teams[1]]++;
        matchups[game.teams[1]][game.teams[0]]++;
        laneCounts[game.lane][game.teams[0]]++;
        laneCounts[game.lane][game.teams[1]]++;
        if (game.timeSlot >= 2) {
            lastGameLaneCounts[game.lane][game.teams[0]]++;
            lastGameLaneCounts[game.lane][game.teams[1]]++;
        }
        slotCounts[game.timeSlot][game.teams[0]]++;
        slotCounts[game.timeSlot][game.teams[1]]++;
        const group = (game.timeSlot < 2 ? 0 : 2) + (game.lane < 2 ? 1 : 2);
        groups[game.teams[0]][game.day] = group;
        groups[game.teams[1]][game.day] = group;
        for (const t of game.teams) {
            if (teamWeekLane[t][game.day] === -1)
                teamWeekLane[t][game.day] = game.lane;
        }
    }

    const laneSwitchCounts: number[] = new Array(teams).fill(0);
    for (const game of schedule.schedule) {
        if (game.teams[0] === -1 || game.teams[1] === -1) continue;
        for (const t of game.teams) {
            if (teamWeekLane[t][game.day] !== game.lane) {
                laneSwitchCounts[t]++;
            }
        }
    }

    return {
        matchups,
        laneCounts,
        lastGameLaneCounts,
        slotCounts,
        groups,
        laneSwitchCounts,
    };
}

export function computeViolations(schedule: Schedule): Violations {
    const { teams: T, days: W } = schedule.config;

    const weekMatchups: Map<number, Set<string>> = new Map();
    for (let w = 0; w < W; w++) weekMatchups.set(w, new Set());

    for (const g of schedule.schedule) {
        if (g.teams[0] === -1) continue;
        const lo = Math.min(g.teams[0], g.teams[1]);
        const hi = Math.max(g.teams[0], g.teams[1]);
        weekMatchups.get(g.day)?.add(`${lo}-${hi}`);
    }

    const consecutivePairs = new Set<string>();
    for (let w = 0; w < W - 1; w++) {
        const set1 = weekMatchups.get(w);
        const set2 = weekMatchups.get(w + 1);
        if (!set1 || !set2) continue;
        set1.forEach((pair) => {
            if (set2.has(pair)) {
                consecutivePairs.add(`${w}-${pair}`);
                consecutivePairs.add(`${w + 1}-${pair}`);
            }
        });
    }

    const earlyLate = new Uint8Array(T * W);
    for (const g of schedule.schedule) {
        if (g.teams[0] === -1) continue;
        const val = g.timeSlot < 2 ? 1 : 2;
        earlyLate[g.teams[0] * W + g.day] = val;
        earlyLate[g.teams[1] * W + g.day] = val;
    }

    const earlyLateStreaks = new Set<string>();
    for (let t = 0; t < T; t++) {
        for (let w = 0; w < W - 2; w++) {
            const base = t * W;
            const v0 = earlyLate[base + w];
            const v1 = earlyLate[base + w + 1];
            const v2 = earlyLate[base + w + 2];
            if (v0 > 0 && v0 === v1 && v1 === v2) {
                earlyLateStreaks.add(`${t}-${w}`);
                earlyLateStreaks.add(`${t}-${w + 1}`);
                earlyLateStreaks.add(`${t}-${w + 2}`);
            }
        }
    }

    return { consecutivePairs, earlyLateStreaks };
}

export function cloneGames(games: Game[]): Game[] {
    return games.map((g) => ({
        ...g,
        teams: [...g.teams] as [number, number],
    }));
}

export function swapTeamsInGames(
    games: Game[],
    team1: number,
    team2: number,
    days: number[],
): Game[] {
    return games.map((g) => {
        if (!days.includes(g.day)) return g;
        let t0 = g.teams[0];
        let t1 = g.teams[1];
        if (t0 === team1) t0 = team2;
        else if (t0 === team2) t0 = team1;
        if (t1 === team1) t1 = team2;
        else if (t1 === team2) t1 = team1;
        const lo = Math.min(t0, t1);
        const hi = Math.max(t0, t1);
        return { ...g, teams: [lo, hi] as [number, number] };
    });
}

export function rebuildSchedule(games: Game[]): Schedule {
    const s = new Schedule(config);
    s.schedule = games;
    return s;
}

export function applyAssignmentToSchedule(flat: number[]): Schedule {
    const s = new Schedule(config);
    s.schedule = [];
    s.createSchedule();
    let idx = 0;
    for (let w = 0; w < config.days; w++) {
        for (let q = 0; q < 4; q++) {
            const a = flat[idx++];
            const b = flat[idx++];
            const c = flat[idx++];
            const d = flat[idx++];
            const slot = q < 2 ? 0 : 2;
            const lane = (q % 2) * 2;
            s.setGame(a, b, w, slot, lane);
            s.setGame(c, d, w, slot, lane + 1);
            s.setGame(a, d, w, slot + 1, lane);
            s.setGame(c, b, w, slot + 1, lane + 1);
        }
    }
    return s;
}

/** Convert Schedule back to flat assignment [WEEKS * 4 quads * 4 positions]. */
export function scheduleToFlat(schedule: Schedule): number[] {
    const flat: number[] = new Array(config.days * 4 * 4);
    for (let w = 0; w < config.days; w++) {
        for (let q = 0; q < 4; q++) {
            const slotBase = q < 2 ? 0 : 2;
            const laneBase = (q % 2) * 2;
            const g1 = findGame(schedule.schedule, w, slotBase, laneBase);
            const g2 = findGame(schedule.schedule, w, slotBase + 1, laneBase);
            const g3 = findGame(schedule.schedule, w, slotBase, laneBase + 1);
            if (!g1 || !g2 || !g3) continue;
            const g4 = findGame(
                schedule.schedule,
                w,
                slotBase + 1,
                laneBase + 1,
            );
            if (!g4) continue;
            const pa =
                g1.teams.find((t) => g2.teams.includes(t)) ?? g1.teams[0];
            const pb = g1.teams[0] === pa ? g1.teams[1] : g1.teams[0];
            const pc =
                g3.teams.find((t) => g4.teams.includes(t)) ?? g3.teams[0];
            const pd = g3.teams[0] === pc ? g3.teams[1] : g3.teams[0];
            const idx = w * 16 + q * 4;
            flat[idx] = pa;
            flat[idx + 1] = pb;
            flat[idx + 2] = pc;
            flat[idx + 3] = pd;
        }
    }
    return flat;
}

export function gameHasConsecutiveViolation(
    violations: Violations,
    game: Game,
): boolean {
    if (game.teams[0] === -1) return false;
    const lo = Math.min(game.teams[0], game.teams[1]);
    const hi = Math.max(game.teams[0], game.teams[1]);
    return violations.consecutivePairs.has(`${game.day}-${lo}-${hi}`);
}
