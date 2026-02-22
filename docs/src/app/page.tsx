'use client';

import { useState, useRef, useEffect } from 'react';
import { Schedule } from '../../../src/Schedule';
import { evaluateSchedule } from '../../../src/solver';
import type { CostBreakdown } from '../../../src/solver';
import type { Config, Game } from '../../../src/types';
import { pinsetter as pinsetter1 } from '../../../src/pinsetter1';
import { pinsetter as pinsetter2 } from '../../../src/pinsetter2';
import { pinsetter as pinsetter3 } from '../../../src/pinsetter3';
import { pinsetter as pinsetter4 } from '../../../src/pinsetter4';
import { pinsetter as pinsetter5 } from '../../../src/pinsetter5';
import { pinsetter as pinsetter6 } from '../../../src/pinsetter6';

type AlgorithmId =
    | 'solver'
    | 'pinsetter1'
    | 'pinsetter2'
    | 'pinsetter3'
    | 'pinsetter4'
    | 'pinsetter5'
    | 'pinsetter6';

const algorithms: { id: AlgorithmId; label: string }[] = [
    { id: 'solver', label: 'SA Solver' },
    { id: 'pinsetter1', label: 'Pinsetter 1 — highs vs lows' },
    { id: 'pinsetter2', label: 'Pinsetter 2 — rotations' },
    { id: 'pinsetter3', label: 'Pinsetter 3 — mixed transforms' },
    { id: 'pinsetter4', label: 'Pinsetter 4 — remaps' },
    { id: 'pinsetter5', label: 'Pinsetter 5 — reverseTwos' },
    { id: 'pinsetter6', label: 'Pinsetter 6 — rotate pairs' },
];

const pinsetterFns: Record<string, (s: Schedule) => Schedule> = {
    pinsetter1,
    pinsetter2,
    pinsetter3,
    pinsetter4,
    pinsetter5,
    pinsetter6,
};

const config: Config = {
    teams: 16,
    timeSlots: 4,
    lanes: 4,
    days: 12,
};

const slotNames = ['Early 1', 'Early 2', 'Late 1', 'Late 2'];

const gamesColors = ['bg-red-500', 'bg-lime-300', 'bg-lime-700', 'bg-red-300'];
const lanesColors = [
    'bg-red-300',
    'bg-red-300',
    'bg-red-300',
    'bg-red-300',
    'bg-red-300',
    'bg-lime-300',
    'bg-lime-700',
    'bg-lime-300',
    'bg-red-300',
    'bg-red-300',
    'bg-red-300',
    'bg-red-300',
    'bg-red-300',
    'bg-red-300',
    'bg-red-300',
    'bg-red-300',
];

function findGame(
    games: Game[],
    day: number,
    slot: number,
    lane: number,
): Game | undefined {
    return games.find(
        (g) => g.day === day && g.timeSlot === slot && g.lane === lane,
    );
}

function scheduleToTSV(schedule: Schedule): string {
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

function parseTSV(tsv: string): Schedule | null {
    const lines = tsv.trim().split('\n');
    if (lines.length < 2) return null;

    const slotMap: Record<string, number> = {
        'early 1': 0, 'early 2': 1, 'late 1': 2, 'late 2': 3,
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

interface Analysis {
    matchups: number[][];
    laneCounts: number[][];
    slotCounts: number[][];
    groups: number[][];
    laneSwitchCounts: number[];
}

function analyzeSchedule(schedule: Schedule): Analysis {
    const { teams, lanes, timeSlots, days } = schedule.config;
    const matchups = Array.from({ length: teams }, () =>
        new Array(teams).fill(0),
    );
    const laneCounts = Array.from({ length: lanes }, () =>
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

    // stays = weeks where both games were on the same lane, switches = different lane
    const laneSwitchCounts: number[] = new Array(teams).fill(0);
    for (const game of schedule.schedule) {
        if (game.teams[0] === -1 || game.teams[1] === -1) continue;
        for (const t of game.teams) {
            if (teamWeekLane[t][game.day] !== game.lane) {
                laneSwitchCounts[t]++;
            }
        }
    }

    return { matchups, laneCounts, slotCounts, groups, laneSwitchCounts };
}

interface Violations {
    consecutivePairs: Set<string>;
    earlyLateStreaks: Set<string>;
}

function computeViolations(schedule: Schedule): Violations {
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

function cloneGames(games: Game[]): Game[] {
    return games.map((g) => ({
        ...g,
        teams: [...g.teams] as [number, number],
    }));
}

function swapTeamsInGames(
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

function rebuildSchedule(games: Game[]): Schedule {
    const s = new Schedule(config);
    s.schedule = games;
    return s;
}

function applyAssignmentToSchedule(flat: number[]): Schedule {
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

interface WorkerProgress {
    iteration: number;
    maxIterations: number;
    bestCost: number;
    done: boolean;
}

interface SolverProgress {
    workers: WorkerProgress[];
    globalBestCost: number | null;
}

function gameHasConsecutiveViolation(
    violations: Violations,
    game: Game,
): boolean {
    if (game.teams[0] === -1) return false;
    const lo = Math.min(game.teams[0], game.teams[1]);
    const hi = Math.max(game.teams[0], game.teams[1]);
    return violations.consecutivePairs.has(`${game.day}-${lo}-${hi}`);
}

export default function Page() {
    const [algorithm, setAlgorithm] = useState<AlgorithmId>('solver');
    const [schedule, setSchedule] = useState<Schedule | null>(null);
    const [cost, setCost] = useState<CostBreakdown | null>(null);
    const [analysis, setAnalysis] = useState<Analysis | null>(null);
    const [violations, setViolations] = useState<Violations | null>(null);
    const [generating, setGenerating] = useState(false);
    const [copied, setCopied] = useState(false);
    const copiedTimer = useRef<ReturnType<typeof setTimeout>>();

    const [selectedTeam, setSelectedTeam] = useState<{
        team: number;
        day: number;
    } | null>(null);
    const [editHistory, setEditHistory] = useState<Game[][]>([]);
    const [highlightInput, setHighlightInput] = useState('');
    const [solverProgress, setSolverProgress] = useState<SolverProgress | null>(
        null,
    );
    const workersRef = useRef<Worker[]>([]);

    const highlightedTeams = new Set(
        highlightInput
            .split(',')
            .map((s) => Number.parseInt(s.trim(), 10) - 1)
            .filter((n) => !Number.isNaN(n) && n >= 0 && n < config.teams),
    );

    useEffect(() => {
        const handler = (e: KeyboardEvent) => {
            if (e.key === 'Escape') setSelectedTeam(null);
        };
        document.addEventListener('keydown', handler);
        return () => document.removeEventListener('keydown', handler);
    }, []);

    function generate() {
        setGenerating(true);
        setSelectedTeam(null);
        setEditHistory([]);
        setSolverProgress(null);

        if (algorithm === 'solver') {
            for (const w of workersRef.current) w.terminate();
            workersRef.current = [];

            const numWorkers = navigator.hardwareConcurrency || 4;
            const iterationsPerWorker = 600_000_000;
            const basePath =
                typeof window !== 'undefined'
                    ? window.location.pathname.replace(/\/+$/, '')
                    : '';

            let bestCost = Number.POSITIVE_INFINITY;
            let bestResult: {
                cost: CostBreakdown;
                assignment: number[];
            } | null = null;
            const workerStates: WorkerProgress[] = Array.from(
                { length: numWorkers },
                () => ({
                    iteration: 0,
                    maxIterations: iterationsPerWorker,
                    bestCost: Number.POSITIVE_INFINITY,
                    done: false,
                }),
            );
            let doneCount = 0;

            setSolverProgress({
                workers: [...workerStates],
                globalBestCost: null,
            });

            for (let i = 0; i < numWorkers; i++) {
                const worker = new Worker(`${basePath}/solver-worker.js`, {
                    type: 'module',
                });
                workersRef.current.push(worker);
                const workerIdx = i;

                worker.onmessage = (e: MessageEvent) => {
                    const msg = e.data;
                    if (msg.type === 'progress') {
                        workerStates[workerIdx] = {
                            iteration: msg.iteration,
                            maxIterations: msg.maxIterations,
                            bestCost: msg.bestCost,
                            done: false,
                        };
                        const globalBest = Math.min(
                            ...workerStates.map((w) => w.bestCost),
                        );
                        setSolverProgress({
                            workers: [...workerStates],
                            globalBestCost:
                                globalBest === Number.POSITIVE_INFINITY
                                    ? null
                                    : globalBest,
                        });
                    } else if (msg.type === 'done') {
                        workerStates[workerIdx].done = true;
                        workerStates[workerIdx].iteration = iterationsPerWorker;
                        doneCount++;
                        if (msg.cost.total < bestCost) {
                            bestCost = msg.cost.total;
                            bestResult = {
                                cost: msg.cost,
                                assignment: msg.assignment,
                            };
                        }
                        const globalBest = Math.min(
                            ...workerStates.map((w) => w.bestCost),
                        );
                        setSolverProgress({
                            workers: [...workerStates],
                            globalBestCost:
                                globalBest === Number.POSITIVE_INFINITY
                                    ? null
                                    : globalBest,
                        });

                        if (doneCount === numWorkers) {
                            if (bestResult) {
                                const s = applyAssignmentToSchedule(
                                    bestResult.assignment,
                                );
                                setSchedule(s);
                                setCost(bestResult.cost);
                                setAnalysis(analyzeSchedule(s));
                                setViolations(computeViolations(s));
                            }
                            setGenerating(false);
                            setSolverProgress(null);
                            for (const w of workersRef.current) w.terminate();
                            workersRef.current = [];
                        }
                    } else if (msg.type === 'error') {
                        console.error('Solver worker error:', msg.message);
                        workerStates[workerIdx].done = true;
                        doneCount++;
                        if (doneCount === numWorkers) {
                            if (bestResult) {
                                const s = applyAssignmentToSchedule(
                                    bestResult.assignment,
                                );
                                setSchedule(s);
                                setCost(bestResult.cost);
                                setAnalysis(analyzeSchedule(s));
                                setViolations(computeViolations(s));
                            }
                            setGenerating(false);
                            setSolverProgress(null);
                            for (const w of workersRef.current) w.terminate();
                            workersRef.current = [];
                        }
                    }
                };

                worker.postMessage({
                    type: 'solve',
                    maxIterations: iterationsPerWorker,
                });
            }
        } else {
            setTimeout(() => {
                const s = new Schedule(config);
                s.createSchedule();
                pinsetterFns[algorithm](s);
                const c = evaluateSchedule(s);
                setSchedule(s);
                setCost(c);
                setAnalysis(analyzeSchedule(s));
                setViolations(computeViolations(s));
                setGenerating(false);
            }, 50);
        }
    }

    function cancelSolve() {
        for (const w of workersRef.current) {
            w.postMessage({ type: 'cancel' });
            w.terminate();
        }
        workersRef.current = [];
        setGenerating(false);
        setSolverProgress(null);
    }

    function handleTeamClick(team: number, day: number) {
        if (!schedule) return;

        if (!selectedTeam) {
            setSelectedTeam({ team, day });
            return;
        }

        if (selectedTeam.team === team && selectedTeam.day === day) {
            setSelectedTeam(null);
            return;
        }

        const days = selectedTeam.day === day ? [day] : [selectedTeam.day, day];

        setEditHistory((h) => [...h, cloneGames(schedule.schedule)]);

        const newGames = swapTeamsInGames(
            schedule.schedule,
            selectedTeam.team,
            team,
            days,
        );
        const newSchedule = rebuildSchedule(newGames);
        setSchedule(newSchedule);
        setCost(evaluateSchedule(newSchedule));
        setAnalysis(analyzeSchedule(newSchedule));
        setViolations(computeViolations(newSchedule));
        setSelectedTeam(null);
    }

    function undo() {
        if (!editHistory.length) return;
        const prev = editHistory[editHistory.length - 1];
        const newSchedule = rebuildSchedule(prev);
        setSchedule(newSchedule);
        setCost(evaluateSchedule(newSchedule));
        setAnalysis(analyzeSchedule(newSchedule));
        setViolations(computeViolations(newSchedule));
        setEditHistory((h) => h.slice(0, -1));
        setSelectedTeam(null);
    }

    function copyTSV() {
        if (!schedule) return;
        navigator.clipboard.writeText(scheduleToTSV(schedule));
        setCopied(true);
        clearTimeout(copiedTimer.current);
        copiedTimer.current = setTimeout(() => setCopied(false), 2000);
    }

    async function pasteTSV() {
        try {
            const text = await navigator.clipboard.readText();
            const s = parseTSV(text);
            if (!s) {
                alert('Could not parse TSV from clipboard');
                return;
            }
            setSchedule(s);
            setCost(evaluateSchedule(s));
            setAnalysis(analyzeSchedule(s));
            setViolations(computeViolations(s));
            setEditHistory([]);
            setSelectedTeam(null);
        } catch {
            alert('Could not read clipboard. Try pasting into the text area instead.');
        }
    }

    function teamButtonClass(team: number, day: number): string {
        const base =
            'px-1 py-0.5 rounded text-xs font-mono cursor-pointer transition-colors';
        const hl = highlightedTeams.has(team)
            ? ' ring-2 ring-amber-400 bg-amber-50'
            : '';
        if (
            selectedTeam &&
            selectedTeam.team === team &&
            selectedTeam.day === day
        ) {
            return `${base} bg-blue-600 text-white ring-2 ring-blue-400`;
        }
        if (selectedTeam) {
            return `${base} hover:bg-blue-100 text-blue-700 underline decoration-dotted${hl}`;
        }
        return `${base} hover:bg-gray-200${hl}`;
    }

    const hasSchedule = schedule && cost && analysis && violations;

    return (
        <div className="flex min-h-screen">
            {/* Main content — scrolls normally */}
            <div
                className={`flex-1 p-4 overflow-y-auto ${hasSchedule ? 'pr-2' : ''}`}
            >
                <div className="prose max-w-none">
                    <h1>Bowling Schedule Generator</h1>
                    <p>
                        Generates an optimized 12-week schedule for 16 teams
                        across 4 lanes. Teams stay on their lane pair (1-2 or
                        3-4), play every other team, and alternate early/late as
                        much as possible.
                    </p>

                    <div className="not-prose flex items-center gap-4 flex-wrap">
                        <select
                            value={algorithm}
                            onChange={(e) =>
                                setAlgorithm(e.target.value as AlgorithmId)
                            }
                            className="px-3 py-3 rounded-lg border border-gray-300 bg-white text-sm"
                            disabled={generating}
                        >
                            {algorithms.map((a) => (
                                <option key={a.id} value={a.id}>
                                    {a.label}
                                </option>
                            ))}
                        </select>
                        <button
                            type="button"
                            onClick={generate}
                            disabled={generating}
                            className="px-6 py-3 rounded-lg font-semibold text-white bg-blue-600 hover:bg-blue-700 disabled:opacity-50 disabled:cursor-wait transition-colors"
                        >
                            {generating
                                ? algorithm === 'solver'
                                    ? 'Solving (WASM)...'
                                    : 'Generating...'
                                : 'Generate'}
                        </button>
                        {generating && algorithm === 'solver' && (
                            <button
                                type="button"
                                onClick={cancelSolve}
                                className="px-4 py-3 rounded-lg font-semibold text-white bg-red-500 hover:bg-red-600 transition-colors"
                            >
                                Cancel
                            </button>
                        )}
                        <button
                            type="button"
                            onClick={pasteTSV}
                            disabled={generating}
                            className="px-4 py-3 rounded-lg font-medium border border-gray-300 hover:bg-gray-100 disabled:opacity-50 transition-colors text-sm"
                        >
                            Paste TSV
                        </button>
                    </div>

                    {solverProgress &&
                        (() => {
                            const { workers, globalBestCost } = solverProgress;
                            const totalIter = workers.reduce(
                                (s, w) => s + w.iteration,
                                0,
                            );
                            const totalMax = workers.reduce(
                                (s, w) => s + w.maxIterations,
                                0,
                            );
                            const pct =
                                totalMax > 0 ? (totalIter / totalMax) * 100 : 0;
                            const doneCount = workers.filter(
                                (w) => w.done,
                            ).length;
                            return (
                                <div className="not-prose mt-3 p-3 rounded-lg bg-blue-50 border border-blue-200 text-sm font-mono space-y-2">
                                    <div className="flex justify-between">
                                        <span>
                                            {workers.length} cores &middot;{' '}
                                            {doneCount} finished &middot;{' '}
                                            {(totalIter / 1_000_000).toFixed(0)}
                                            M /{' '}
                                            {(totalMax / 1_000_000).toFixed(0)}M
                                            iter
                                        </span>
                                        <span className="font-bold">
                                            {globalBestCost != null
                                                ? `Best cost: ${globalBestCost}`
                                                : 'Starting...'}
                                        </span>
                                    </div>
                                    <div className="w-full bg-blue-200 rounded-full h-2">
                                        <div
                                            className="bg-blue-600 h-2 rounded-full transition-all duration-300"
                                            style={{ width: `${pct}%` }}
                                        />
                                    </div>
                                    <div
                                        className="grid gap-1"
                                        style={{
                                            gridTemplateColumns: `repeat(${Math.min(workers.length, 12)}, 1fr)`,
                                        }}
                                    >
                                        {workers.map((w, i) => {
                                            const wp =
                                                w.maxIterations > 0
                                                    ? (w.iteration /
                                                          w.maxIterations) *
                                                      100
                                                    : 0;
                                            return (
                                                // biome-ignore lint/suspicious/noArrayIndexKey: stable worker indices
                                                <div
                                                    key={i}
                                                    title={`Core ${i + 1}: ${(w.iteration / 1_000_000).toFixed(0)}M iter, best ${w.bestCost === Number.POSITIVE_INFINITY ? '–' : w.bestCost}`}
                                                >
                                                    <div className="bg-blue-200 rounded-full h-1.5">
                                                        <div
                                                            className={`h-1.5 rounded-full transition-all duration-300 ${w.done ? 'bg-green-500' : 'bg-blue-500'}`}
                                                            style={{
                                                                width: `${wp}%`,
                                                            }}
                                                        />
                                                    </div>
                                                </div>
                                            );
                                        })}
                                    </div>
                                </div>
                            );
                        })()}

                    {hasSchedule && (
                        <>
                            <h2>Constraint Scores</h2>
                            <p className="text-sm text-gray-600">
                                Lower is better. Zero means the constraint is
                                perfectly satisfied.
                            </p>
                            <table className="text-sm">
                                <tbody>
                                    {[
                                        {
                                            label: 'Matchup balance',
                                            value: cost.matchupBalance,
                                            desc: 'Every pair plays exactly 1 or 2 times',
                                        },
                                        {
                                            label: 'Consecutive opponents',
                                            value: cost.consecutiveOpponents,
                                            desc: 'No repeat matchups in adjacent weeks',
                                        },
                                        {
                                            label: 'Early/late balance',
                                            value: cost.earlyLateBalance,
                                            desc: '6 early + 6 late per team',
                                        },
                                        {
                                            label: 'Early/late alternation',
                                            value: cost.earlyLateAlternation,
                                            desc: 'No three early or late in a row',
                                        },
                                        {
                                            label: 'Lane balance',
                                            value: cost.laneBalance,
                                            desc: 'Each team on each lane 6 times',
                                        },
                                        {
                                            label: 'Lane switches',
                                            value: cost.laneSwitchBalance,
                                            desc: 'Equal stay vs switch between games',
                                        },
                                    ].map((row) => (
                                        <tr key={row.label}>
                                            <td>{row.label}</td>
                                            <td
                                                className={
                                                    row.value === 0
                                                        ? 'text-green-700 font-bold'
                                                        : 'text-red-700 font-bold'
                                                }
                                            >
                                                {row.value}
                                            </td>
                                            <td className="text-gray-500">
                                                {row.desc}
                                            </td>
                                        </tr>
                                    ))}
                                    <tr className="border-t-2">
                                        <td className="font-bold">Total</td>
                                        <td
                                            className={`font-bold ${cost.total === 0 ? 'text-green-700' : ''}`}
                                        >
                                            {cost.total}
                                        </td>
                                        <td />
                                    </tr>
                                </tbody>
                            </table>

                            <h2>Matchups</h2>
                            <div className="overflow-x-auto">
                                <table className="text-sm">
                                    <thead>
                                        <tr>
                                            <th>Team</th>
                                            {Array.from(
                                                { length: config.teams },
                                                (_, i) => (
                                                    // biome-ignore lint/suspicious/noArrayIndexKey: sequential
                                                    <th key={i}>{i + 1}</th>
                                                ),
                                            )}
                                        </tr>
                                    </thead>
                                    <tbody>
                                        {analysis.matchups.map((row, i) => (
                                            // biome-ignore lint/suspicious/noArrayIndexKey: sequential
                                            <tr key={i}>
                                                <td>{i + 1}</td>
                                                {row.map((matchup, j) => (
                                                    <td
                                                        // biome-ignore lint/suspicious/noArrayIndexKey: sequential
                                                        key={j}
                                                        className={
                                                            i === j
                                                                ? ''
                                                                : gamesColors[
                                                                      matchup
                                                                  ]
                                                        }
                                                        title={
                                                            i !== j
                                                                ? `Teams ${i + 1} & ${j + 1}: ${matchup} matchup${matchup !== 1 ? 's' : ''} (expect 1–2)`
                                                                : undefined
                                                        }
                                                    >
                                                        {i === j
                                                            ? ' '
                                                            : matchup}
                                                    </td>
                                                ))}
                                            </tr>
                                        ))}
                                    </tbody>
                                </table>
                            </div>

                            <h2>Lane Counts</h2>
                            <div className="overflow-x-auto">
                                <table className="text-sm">
                                    <thead>
                                        <tr>
                                            <th> </th>
                                            {Array.from(
                                                { length: config.teams },
                                                (_, i) => (
                                                    // biome-ignore lint/suspicious/noArrayIndexKey: sequential
                                                    <th key={i}>{i + 1}</th>
                                                ),
                                            )}
                                        </tr>
                                    </thead>
                                    <tbody>
                                        {analysis.laneCounts.map((row, i) => (
                                            // biome-ignore lint/suspicious/noArrayIndexKey: sequential
                                            <tr key={i}>
                                                <td>Lane {i + 1}</td>
                                                {row.map((count, j) => (
                                                    <td
                                                        // biome-ignore lint/suspicious/noArrayIndexKey: sequential
                                                        key={j}
                                                        className={
                                                            lanesColors[count]
                                                        }
                                                        title={`Team ${j + 1} on lane ${i + 1}: ${count}× (expect 6)`}
                                                    >
                                                        {count}
                                                    </td>
                                                ))}
                                            </tr>
                                        ))}
                                    </tbody>
                                </table>
                            </div>

                            <h2>Lane Switches</h2>
                            <div className="overflow-x-auto">
                                <table className="text-sm">
                                    <thead>
                                        <tr>
                                            <th> </th>
                                            {Array.from(
                                                { length: config.teams },
                                                (_, i) => (
                                                    // biome-ignore lint/suspicious/noArrayIndexKey: sequential
                                                    <th key={i}>{i + 1}</th>
                                                ),
                                            )}
                                        </tr>
                                    </thead>
                                    <tbody>
                                        <tr>
                                            <td>Stay</td>
                                            {analysis.laneSwitchCounts.map(
                                                (sw, i) => (
                                                    <td
                                                        // biome-ignore lint/suspicious/noArrayIndexKey: sequential
                                                        key={`stay-${i}`}
                                                        className={
                                                            Math.abs(config.days - sw - config.days / 2) === 0
                                                                ? 'bg-green-100'
                                                                : Math.abs(config.days - sw - config.days / 2) <= 1
                                                                  ? 'bg-yellow-100'
                                                                  : 'bg-red-100'
                                                        }
                                                        title={`Team ${i + 1}: ${config.days - sw} stay, ${sw} switch (expect ${config.days / 2} each)`}
                                                    >
                                                        {config.days - sw}
                                                    </td>
                                                ),
                                            )}
                                        </tr>
                                        <tr>
                                            <td>Switch</td>
                                            {analysis.laneSwitchCounts.map(
                                                (sw, i) => (
                                                    <td
                                                        // biome-ignore lint/suspicious/noArrayIndexKey: sequential
                                                        key={`switch-${i}`}
                                                        className={
                                                            Math.abs(sw - config.days / 2) === 0
                                                                ? 'bg-green-100'
                                                                : Math.abs(sw - config.days / 2) <= 1
                                                                  ? 'bg-yellow-100'
                                                                  : 'bg-red-100'
                                                        }
                                                        title={`Team ${i + 1}: ${config.days - sw} stay, ${sw} switch (expect ${config.days / 2} each)`}
                                                    >
                                                        {sw}
                                                    </td>
                                                ),
                                            )}
                                        </tr>
                                    </tbody>
                                </table>
                            </div>

                            <h2>Slot Counts</h2>
                            <div className="overflow-x-auto">
                                <table className="text-sm">
                                    <thead>
                                        <tr>
                                            <th> </th>
                                            {Array.from(
                                                { length: config.teams },
                                                (_, i) => (
                                                    // biome-ignore lint/suspicious/noArrayIndexKey: sequential
                                                    <th key={i}>{i + 1}</th>
                                                ),
                                            )}
                                        </tr>
                                    </thead>
                                    <tbody>
                                        {analysis.slotCounts.map((row, i) => (
                                            // biome-ignore lint/suspicious/noArrayIndexKey: sequential
                                            <tr key={i}>
                                                <td>Slot {i + 1}</td>
                                                {row.map((count, j) => (
                                                    <td
                                                        // biome-ignore lint/suspicious/noArrayIndexKey: sequential
                                                        key={j}
                                                        className={
                                                            lanesColors[count]
                                                        }
                                                        title={`Team ${j + 1} in ${slotNames[i]}: ${count}× (expect 6)`}
                                                    >
                                                        {count}
                                                    </td>
                                                ))}
                                            </tr>
                                        ))}
                                    </tbody>
                                </table>
                            </div>

                            <h2>Early Or Late</h2>
                            <div className="overflow-x-auto">
                                <table className="text-sm">
                                    <thead>
                                        <tr>
                                            <th> </th>
                                            {Array.from(
                                                { length: config.days },
                                                (_, i) => (
                                                    // biome-ignore lint/suspicious/noArrayIndexKey: sequential
                                                    <th key={i}>{i + 1}</th>
                                                ),
                                            )}
                                        </tr>
                                    </thead>
                                    <tbody>
                                        {analysis.groups.map((row, i) => (
                                            // biome-ignore lint/suspicious/noArrayIndexKey: sequential
                                            <tr key={i}>
                                                <td>Team {i + 1}</td>
                                                {row.map((group, j) => {
                                                    const isViolation =
                                                        violations.earlyLateStreaks.has(
                                                            `${i}-${j}`,
                                                        );
                                                    let bg =
                                                        group < 3
                                                            ? 'bg-white'
                                                            : 'bg-black';
                                                    if (isViolation)
                                                        bg =
                                                            group < 3
                                                                ? 'bg-red-200'
                                                                : 'bg-red-800';
                                                    return (
                                                        <td
                                                            // biome-ignore lint/suspicious/noArrayIndexKey: sequential
                                                            key={j}
                                                            className={bg}
                                                            title={
                                                                isViolation
                                                                    ? `Team ${i + 1}: 3+ consecutive ${group < 3 ? 'early' : 'late'} weeks`
                                                                    : undefined
                                                            }
                                                        >
                                                            {group}
                                                        </td>
                                                    );
                                                })}
                                            </tr>
                                        ))}
                                    </tbody>
                                </table>
                            </div>

                            <div className="mt-4 flex gap-2">
                                <button
                                    type="button"
                                    onClick={copyTSV}
                                    className="not-prose px-4 py-2 rounded text-sm font-medium border border-gray-300 hover:bg-gray-100 transition-colors"
                                >
                                    {copied ? 'Copied!' : 'Copy for Sheets'}
                                </button>
                                <button
                                    type="button"
                                    onClick={pasteTSV}
                                    className="not-prose px-4 py-2 rounded text-sm font-medium border border-gray-300 hover:bg-gray-100 transition-colors"
                                >
                                    Paste from TSV
                                </button>
                            </div>
                        </>
                    )}
                </div>
            </div>

            {/* Sidebar — sticky schedule editor */}
            {hasSchedule && (
                <aside
                    className="sticky top-0 h-screen overflow-y-auto border-l border-gray-200 bg-gray-50 flex-shrink-0 p-3 flex flex-col gap-3"
                    style={{ width: '480px' }}
                >
                    <div className="flex items-center gap-2 flex-wrap">
                        <button
                            type="button"
                            onClick={undo}
                            disabled={editHistory.length === 0}
                            className="px-3 py-1.5 rounded font-medium border border-gray-300 hover:bg-gray-100 disabled:opacity-40 disabled:cursor-default disabled:hover:bg-white transition-colors text-sm bg-white"
                        >
                            Undo
                            {editHistory.length > 0
                                ? ` (${editHistory.length})`
                                : ''}
                        </button>
                        <span className="text-xs text-gray-500 font-bold">
                            Total: {cost.total}
                        </span>
                    </div>

                    <div>
                        <label
                            htmlFor="highlight-teams"
                            className="text-xs text-gray-500 block mb-1"
                        >
                            Highlight teams (comma-separated)
                        </label>
                        <input
                            id="highlight-teams"
                            type="text"
                            value={highlightInput}
                            onChange={(e) => setHighlightInput(e.target.value)}
                            placeholder="e.g. 1, 5, 12"
                            className="w-full px-2 py-1.5 rounded border border-gray-300 text-sm bg-white"
                        />
                    </div>

                    {selectedTeam && (
                        <p className="text-xs text-blue-600 m-0">
                            Team {selectedTeam.team + 1} selected (week{' '}
                            {selectedTeam.day + 1}). Click another team to swap.
                            Escape to cancel.
                        </p>
                    )}

                    <div className="overflow-y-auto flex-1 -mx-1 px-1">
                        <table className="text-xs w-full border-collapse">
                            <thead className="sticky top-0 bg-gray-50 z-10">
                                <tr>
                                    <th className="text-left px-1">Wk</th>
                                    <th className="text-left px-1">Time</th>
                                    <th className="text-center px-1">Lane 1</th>
                                    <th className="text-center px-1">Lane 2</th>
                                    <th className="text-center px-1">Lane 3</th>
                                    <th className="text-center px-1">Lane 4</th>
                                </tr>
                            </thead>
                            <tbody>
                                {Array.from({ length: config.days }, (_, day) =>
                                    Array.from(
                                        { length: config.timeSlots },
                                        (_, slot) => (
                                            <tr
                                                key={`${day}-${slot}`}
                                                className={
                                                    slot === 0
                                                        ? 'border-t border-gray-300'
                                                        : ''
                                                }
                                            >
                                                {slot === 0 && (
                                                    <td
                                                        rowSpan={
                                                            config.timeSlots
                                                        }
                                                        className="font-bold align-middle text-center px-1"
                                                    >
                                                        {day + 1}
                                                    </td>
                                                )}
                                                <td
                                                    className={`px-1 whitespace-nowrap ${
                                                        slot < 2
                                                            ? 'bg-sky-50'
                                                            : 'bg-amber-50'
                                                    }`}
                                                >
                                                    {slotNames[slot]}
                                                </td>
                                                {Array.from(
                                                    {
                                                        length: config.lanes,
                                                    },
                                                    (_, lane) => {
                                                        const game = findGame(
                                                            schedule.schedule,
                                                            day,
                                                            slot,
                                                            lane,
                                                        );
                                                        const hasViolation =
                                                            game
                                                                ? gameHasConsecutiveViolation(
                                                                      violations,
                                                                      game,
                                                                  )
                                                                : false;
                                                        return (
                                                            <td
                                                                key={`${day}-${slot}-${lane}`}
                                                                className={`text-center whitespace-nowrap px-0.5 ${hasViolation ? 'bg-red-100 border-l-2 border-red-500' : ''}`}
                                                                title={
                                                                    hasViolation &&
                                                                    game
                                                                        ? `Teams ${game.teams[0] + 1} & ${game.teams[1] + 1} also play in week ${day > 0 && violations.consecutivePairs.has(`${day - 1}-${Math.min(...game.teams)}-${Math.max(...game.teams)}`) ? day : day + 2}`
                                                                        : undefined
                                                                }
                                                            >
                                                                {game &&
                                                                game
                                                                    .teams[0] !==
                                                                    -1 ? (
                                                                    <>
                                                                        <button
                                                                            type="button"
                                                                            onClick={() =>
                                                                                handleTeamClick(
                                                                                    game
                                                                                        .teams[0],
                                                                                    day,
                                                                                )
                                                                            }
                                                                            className={teamButtonClass(
                                                                                game
                                                                                    .teams[0],
                                                                                day,
                                                                            )}
                                                                        >
                                                                            {game
                                                                                .teams[0] +
                                                                                1}
                                                                        </button>
                                                                        <span className="text-gray-400">
                                                                            v
                                                                        </span>
                                                                        <button
                                                                            type="button"
                                                                            onClick={() =>
                                                                                handleTeamClick(
                                                                                    game
                                                                                        .teams[1],
                                                                                    day,
                                                                                )
                                                                            }
                                                                            className={teamButtonClass(
                                                                                game
                                                                                    .teams[1],
                                                                                day,
                                                                            )}
                                                                        >
                                                                            {game
                                                                                .teams[1] +
                                                                                1}
                                                                        </button>
                                                                    </>
                                                                ) : null}
                                                            </td>
                                                        );
                                                    },
                                                )}
                                            </tr>
                                        ),
                                    ),
                                )}
                            </tbody>
                        </table>
                    </div>
                </aside>
            )}
        </div>
    );
}
