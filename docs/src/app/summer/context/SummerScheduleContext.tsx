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
} from '../lib/summer-schedule-utils';
import {
    parseSummerTSV,
    analyzeSummerSchedule,
    evaluateSummerCost,
} from '../lib/summer-schedule-utils';

interface SummerScheduleContextValue {
    schedule: SummerSchedule | null;
    cost: SummerCostBreakdown | null;
    analysis: SummerAnalysis | null;
    highlightInput: string;
    resultFiles: string[];
    importTSV: (text: string) => boolean;
    setHighlightInput: (val: string) => void;
    loadResultFile: (filename: string) => Promise<void>;
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

export function SummerScheduleProvider({ children }: { children: ReactNode }) {
    const [schedule, setSchedule] = useState<SummerSchedule | null>(null);
    const [cost, setCost] = useState<SummerCostBreakdown | null>(null);
    const [analysis, setAnalysis] = useState<SummerAnalysis | null>(null);
    const [highlightInput, setHighlightInput] = useState('');
    const [resultFiles, setResultFiles] = useState<string[]>([]);

    function updateAll(s: SummerSchedule) {
        setSchedule(s);
        setCost(evaluateSummerCost(s));
        setAnalysis(analyzeSummerSchedule(s));
    }

    function importTSV(text: string): boolean {
        const s = parseSummerTSV(text);
        if (!s) return false;
        updateAll(s);
        return true;
    }

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
                highlightInput,
                resultFiles,
                importTSV,
                setHighlightInput,
                loadResultFile,
            }}
        >
            {children}
        </SummerScheduleContext.Provider>
    );
}
