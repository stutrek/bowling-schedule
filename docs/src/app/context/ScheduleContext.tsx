'use client';

import {
    createContext,
    useContext,
    useState,
    useRef,
    useEffect,
    type ReactNode,
} from 'react';
import type { Schedule } from '../../../../src/Schedule';
import type { Game } from '../../../../src/types';
import weights from '../../../../weights.json';
import {
    config,
    analyzeSchedule,
    computeViolations,
    cloneGames,
    swapTeamsInGames,
    rebuildSchedule,
    applyAssignmentToSchedule,
    parseTSV,
    scheduleToFlat,
} from '../lib/schedule-utils';
import type {
    Analysis,
    CostBreakdown,
    Violations,
} from '../lib/schedule-utils';

const weightsJson = JSON.stringify(weights);

type EvaluateFn = (
    flat: Uint8Array,
    weightsJson: string,
) => {
    matchup_balance: number;
    consecutive_opponents: number;
    early_late_balance: number;
    early_late_alternation: number;
    early_late_consecutive: number;
    lane_balance: number;
    lane_switch_balance: number;
    late_lane_balance: number;
    commissioner_overlap: number;
    total: number;
    free: () => void;
};

interface ScheduleContextValue {
    schedule: Schedule | null;
    cost: CostBreakdown | null;
    analysis: Analysis | null;
    violations: Violations | null;
    editHistory: Game[][];
    selectedTeam: { team: number; day: number } | null;
    highlightInput: string;
    generating: boolean;
    solverProgress: SolverProgress | null;
    wasmReady: boolean;
    weightsJson: string;
    setScheduleFromAssignment: (flat: number[]) => void;
    setScheduleFromPinsetter: (s: Schedule) => void;
    importTSV: (text: string) => boolean;
    handleTeamClick: (team: number, day: number) => void;
    undo: () => void;
    setSelectedTeam: (val: { team: number; day: number } | null) => void;
    setHighlightInput: (val: string) => void;
    setGenerating: (val: boolean) => void;
    setSolverProgress: (val: SolverProgress | null) => void;
}

export interface WorkerProgress {
    iteration: number;
    maxIterations: number;
    bestCost: number;
    done: boolean;
}

export interface SolverProgress {
    workers: WorkerProgress[];
    globalBestCost: number | null;
}

const ScheduleContext = createContext<ScheduleContextValue | null>(null);

export function useSchedule() {
    const ctx = useContext(ScheduleContext);
    if (!ctx)
        throw new Error('useSchedule must be used within ScheduleProvider');
    return ctx;
}

export function ScheduleProvider({ children }: { children: ReactNode }) {
    const [schedule, setSchedule] = useState<Schedule | null>(null);
    const [cost, setCost] = useState<CostBreakdown | null>(null);
    const [analysis, setAnalysis] = useState<Analysis | null>(null);
    const [violations, setViolations] = useState<Violations | null>(null);
    const [editHistory, setEditHistory] = useState<Game[][]>([]);
    const [selectedTeam, setSelectedTeam] = useState<{
        team: number;
        day: number;
    } | null>(null);
    const [highlightInput, setHighlightInput] = useState('');
    const [generating, setGenerating] = useState(false);
    const [solverProgress, setSolverProgress] = useState<SolverProgress | null>(
        null,
    );
    const [wasmReady, setWasmReady] = useState(false);

    const evaluateRef = useRef<EvaluateFn | null>(null);
    const scheduleRef = useRef<Schedule | null>(null);
    scheduleRef.current = schedule;

    useEffect(() => {
        (async () => {
            try {
                const basePath = window.location.pathname.replace(/\/+$/, '');
                const wasm = await import(
                    /* webpackIgnore: true */ `${basePath}/solver_wasm.js`
                );
                await wasm.default({
                    module_or_path: new URL(
                        `${basePath}/solver_wasm_bg.wasm`,
                        window.location.origin,
                    ),
                });
                evaluateRef.current = wasm.evaluate_assignment;
                setWasmReady(true);
            } catch (err) {
                console.error('Failed to load WASM:', err);
            }
        })();
    }, []);

    useEffect(() => {
        if (wasmReady && scheduleRef.current) {
            const c = scoreSchedule(scheduleRef.current);
            if (c) setCost(c);
        }
    }, [wasmReady]);

    function scoreSchedule(s: Schedule): CostBreakdown | null {
        const evaluate = evaluateRef.current;
        if (!evaluate) return null;
        const flat = scheduleToFlat(s);
        const c = evaluate(new Uint8Array(flat), weightsJson);
        const result = {
            matchupBalance: c.matchup_balance,
            consecutiveOpponents: c.consecutive_opponents,
            earlyLateBalance: c.early_late_balance,
            earlyLateAlternation: c.early_late_alternation,
            earlyLateConsecutive: c.early_late_consecutive,
            laneBalance: c.lane_balance,
            laneSwitchBalance: c.lane_switch_balance,
            lateLaneBalance: c.late_lane_balance,
            commissionerOverlap: c.commissioner_overlap,
            total: c.total,
        };
        c.free();
        return result;
    }

    function updateAll(s: Schedule) {
        setSchedule(s);
        const c = scoreSchedule(s);
        if (c) setCost(c);
        setAnalysis(analyzeSchedule(s));
        setViolations(computeViolations(s));
    }

    function setScheduleFromAssignment(flat: number[]) {
        const s = applyAssignmentToSchedule(flat);
        setEditHistory([]);
        setSelectedTeam(null);
        updateAll(s);
    }

    function setScheduleFromPinsetter(s: Schedule) {
        setEditHistory([]);
        setSelectedTeam(null);
        updateAll(s);
    }

    function importTSV(text: string): boolean {
        const s = parseTSV(text);
        if (!s) return false;
        setEditHistory([]);
        setSelectedTeam(null);
        updateAll(s);
        return true;
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
        updateAll(newSchedule);
        setSelectedTeam(null);
    }

    function undo() {
        if (!editHistory.length) return;
        const prev = editHistory[editHistory.length - 1];
        const newSchedule = rebuildSchedule(prev);
        updateAll(newSchedule);
        setEditHistory((h) => h.slice(0, -1));
        setSelectedTeam(null);
    }

    useEffect(() => {
        const handler = (e: KeyboardEvent) => {
            if (e.key === 'Escape') setSelectedTeam(null);
        };
        document.addEventListener('keydown', handler);
        return () => document.removeEventListener('keydown', handler);
    }, []);

    return (
        <ScheduleContext.Provider
            value={{
                schedule,
                cost,
                analysis,
                violations,
                editHistory,
                selectedTeam,
                highlightInput,
                generating,
                solverProgress,
                wasmReady,
                weightsJson,
                setScheduleFromAssignment,
                setScheduleFromPinsetter,
                importTSV,
                handleTeamClick,
                undo,
                setSelectedTeam,
                setHighlightInput,
                setGenerating,
                setSolverProgress,
            }}
        >
            {children}
        </ScheduleContext.Provider>
    );
}
