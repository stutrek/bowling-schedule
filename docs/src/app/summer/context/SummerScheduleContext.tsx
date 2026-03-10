'use client';

import {
    createContext,
    useContext,
    useState,
    useEffect,
    type ReactNode,
} from 'react';
import type {
    SummerSchedule,
    SummerCostBreakdown,
    SummerAnalysis,
    SummerViolations,
} from '../lib/summer-schedule-utils';
import {
    parseSummerTSV,
    analyzeSummerSchedule,
    evaluateSummerCost,
    computeSummerViolations,
    S_SLOTS,
    S_PAIRS,
} from '../lib/summer-schedule-utils';

export interface SummerSelection {
    team: number;
    week: number;
    slot: number;
    pair: number;
    side: 'A' | 'B';
}

interface SummerScheduleContextValue {
    schedule: SummerSchedule | null;
    cost: SummerCostBreakdown | null;
    analysis: SummerAnalysis | null;
    violations: SummerViolations | null;
    highlightInput: string;
    resultFiles: string[];
    selectedTeam: SummerSelection | null;
    editHistory: SummerSchedule[];
    importTSV: (text: string) => boolean;
    setHighlightInput: (val: string) => void;
    loadResultFile: (filename: string) => Promise<void>;
    handleTeamClick: (
        team: number,
        week: number,
        slot: number,
        pair: number,
        side: 'A' | 'B',
    ) => void;
    undo: () => void;
    setSelectedTeam: (val: SummerSelection | null) => void;
}

const SummerScheduleContext = createContext<SummerScheduleContextValue | null>(
    null,
);

export function useSummerSchedule() {
    const ctx = useContext(SummerScheduleContext);
    if (!ctx)
        throw new Error(
            'useSummerSchedule must be used within SummerScheduleProvider',
        );
    return ctx;
}

function cloneSchedule(schedule: SummerSchedule): SummerSchedule {
    return schedule.map((week) =>
        week.map((slot) =>
            slot.map((m) => (m ? { teamA: m.teamA, teamB: m.teamB } : null)),
        ),
    );
}

export function SummerScheduleProvider({ children }: { children: ReactNode }) {
    const [schedule, setSchedule] = useState<SummerSchedule | null>(null);
    const [cost, setCost] = useState<SummerCostBreakdown | null>(null);
    const [analysis, setAnalysis] = useState<SummerAnalysis | null>(null);
    const [violations, setViolations] = useState<SummerViolations | null>(null);
    const [highlightInput, setHighlightInput] = useState('');
    const [resultFiles, setResultFiles] = useState<string[]>([]);
    const [selectedTeam, setSelectedTeam] = useState<SummerSelection | null>(
        null,
    );
    const [editHistory, setEditHistory] = useState<SummerSchedule[]>([]);

    function updateAll(s: SummerSchedule) {
        setSchedule(s);
        setCost(evaluateSummerCost(s));
        setAnalysis(analyzeSummerSchedule(s));
        setViolations(computeSummerViolations(s));
    }

    function importTSV(text: string): boolean {
        const s = parseSummerTSV(text);
        if (!s) return false;
        setEditHistory([]);
        setSelectedTeam(null);
        updateAll(s);
        return true;
    }

    function handleTeamClick(
        team: number,
        week: number,
        slot: number,
        pair: number,
        side: 'A' | 'B',
    ) {
        if (!schedule) return;

        // Nothing selected → select this team
        if (!selectedTeam) {
            setSelectedTeam({ team, week, slot, pair, side });
            return;
        }

        // Same position clicked → deselect
        if (
            selectedTeam.week === week &&
            selectedTeam.slot === slot &&
            selectedTeam.pair === pair &&
            selectedTeam.side === side
        ) {
            setSelectedTeam(null);
            return;
        }

        // Different week → deselect
        if (selectedTeam.week !== week) {
            setSelectedTeam(null);
            return;
        }

        // Check: selected team must NOT already be in the target game
        const targetMatchup = schedule[week][slot][pair];
        if (
            targetMatchup &&
            (targetMatchup.teamA === selectedTeam.team ||
                targetMatchup.teamB === selectedTeam.team)
        ) {
            setSelectedTeam(null);
            return;
        }

        // Perform the swap
        const newSchedule = cloneSchedule(schedule);

        // Replace the clicked team with the selected team in the target position
        const tgt = newSchedule[week][slot][pair];
        const clickedTeam = team;
        if (!tgt) return;
        if (side === 'A') {
            tgt.teamA = selectedTeam.team;
        } else {
            tgt.teamB = selectedTeam.team;
        }

        // Replace the selected team with the clicked team in the original position
        const src = newSchedule[week][selectedTeam.slot][selectedTeam.pair];
        if (!src) return;
        if (selectedTeam.side === 'A') {
            src.teamA = clickedTeam;
        } else {
            src.teamB = clickedTeam;
        }

        setEditHistory((h) => [...h, cloneSchedule(schedule)]);
        setSelectedTeam(null);
        updateAll(newSchedule);
    }

    function undo() {
        if (!editHistory.length) return;
        const prev = editHistory[editHistory.length - 1];
        updateAll(prev);
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

    async function loadResultFile(filename: string) {
        try {
            const basePath =
                typeof window !== 'undefined'
                    ? window.location.pathname
                          .replace(/\/summer\/?$/, '')
                          .replace(/\/+$/, '')
                    : '';
            const res = await fetch(`${basePath}/summer-results/${filename}`);
            if (!res.ok) throw new Error(`HTTP ${res.status}`);
            const text = await res.text();
            if (!importTSV(text)) {
                alert(`Could not parse ${filename}`);
            }
        } catch (e) {
            alert(`Failed to load ${filename}: ${e}`);
        }
    }

    // biome-ignore lint/correctness/useExhaustiveDependencies: run once on mount
    useEffect(() => {
        const basePath =
            typeof window !== 'undefined'
                ? window.location.pathname
                      .replace(/\/summer\/?$/, '')
                      .replace(/\/+$/, '')
                : '';
        fetch(`${basePath}/summer-results/manifest.json`)
            .then((r) => (r.ok ? r.json() : []))
            .then((files: string[]) => {
                setResultFiles(files);
                if (files.length > 0) {
                    loadResultFile(files[0]);
                }
            })
            .catch(() => {});
    }, []);

    return (
        <SummerScheduleContext.Provider
            value={{
                schedule,
                cost,
                analysis,
                violations,
                highlightInput,
                resultFiles,
                selectedTeam,
                editHistory,
                importTSV,
                setHighlightInput,
                loadResultFile,
                handleTeamClick,
                undo,
                setSelectedTeam,
            }}
        >
            {children}
        </SummerScheduleContext.Provider>
    );
}
