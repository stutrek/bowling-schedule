'use client';

import { useRef, useState, useEffect } from 'react';
import { Schedule } from '../../../../src/Schedule';
import { pinsetter as pinsetter1 } from '../../../../src/pinsetter1';
import {
    useSchedule,
    type SolverProgress,
    type WorkerProgress,
} from '../context/ScheduleContext';
import { config, scheduleToFlat, scheduleToTSV } from '../lib/schedule-utils';

type AlgorithmId = 'solver' | 'pinsetter1';

const algorithms: { id: AlgorithmId; label: string }[] = [
    { id: 'solver', label: 'SA Solver' },
];

const pinsetterFns: Record<string, (s: Schedule) => Schedule> = {
    pinsetter1,
};

export default function ControlBar() {
    const {
        schedule,
        generating,
        setGenerating,
        setSolverProgress,
        setScheduleFromAssignment,
        setScheduleFromPinsetter,
        importTSV,
        weightsJson,
    } = useSchedule();

    const algorithmRef = useRef<AlgorithmId>('solver');
    const workersRef = useRef<Worker[]>([]);
    const [resultFiles, setResultFiles] = useState<string[]>([]);
    const [useSeed, setUseSeed] = useState(false);
    const [iterations, setIterations] = useState(200_000_000);

    function generate() {
        const algorithm = algorithmRef.current;
        setGenerating(true);
        setSolverProgress(null);

        if (algorithm === 'solver') {
            for (const w of workersRef.current) w.terminate();
            workersRef.current = [];

            const numWorkers = navigator.hardwareConcurrency || 4;
            const iterationsPerWorker = iterations;
            const basePath =
                typeof window !== 'undefined'
                    ? window.location.pathname.replace(/\/+$/, '')
                    : '';

            let bestCost = Number.POSITIVE_INFINITY;
            let bestResult: {
                cost: {
                    matchupBalance: number;
                    consecutiveOpponents: number;
                    earlyLateBalance: number;
                    earlyLateAlternation: number;
                    laneBalance: number;
                    laneSwitchBalance: number;
                    total: number;
                };
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
                        if (msg.bestAssignment && msg.bestCost < bestCost) {
                            bestCost = msg.bestCost;
                            bestResult = {
                                cost: msg.cost,
                                assignment: msg.bestAssignment,
                            };
                            setScheduleFromAssignment(msg.bestAssignment);
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
                                setScheduleFromAssignment(
                                    bestResult.assignment,
                                );
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
                                setScheduleFromAssignment(
                                    bestResult.assignment,
                                );
                            }
                            setGenerating(false);
                            setSolverProgress(null);
                            for (const w of workersRef.current) w.terminate();
                            workersRef.current = [];
                        }
                    }
                };

                const seedFlat =
                    useSeed && schedule ? scheduleToFlat(schedule) : null;
                worker.postMessage({
                    type: 'solve',
                    maxIterations: iterationsPerWorker,
                    weightsJson,
                    seedFlat,
                });
            }
        } else {
            setTimeout(() => {
                const s = new Schedule(config);
                s.createSchedule();
                pinsetterFns[algorithm](s);
                setScheduleFromPinsetter(s);
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

    useEffect(() => {
        const basePath = window.location.pathname.replace(/\/+$/, '');
        fetch(`${basePath}/results/manifest.json`)
            .then((r) => (r.ok ? r.json() : []))
            .then((files: string[]) => {
                setResultFiles(files);
                if (files.length > 0) {
                    loadResultFile(files[0]);
                }
            })
            .catch(() => {});
    }, []);

    async function loadResultFile(filename: string) {
        try {
            const basePath = window.location.pathname.replace(/\/+$/, '');
            const res = await fetch(`${basePath}/results/${filename}`);
            if (!res.ok) throw new Error(`HTTP ${res.status}`);
            const text = await res.text();
            if (!importTSV(text)) {
                alert(`Could not parse ${filename}`);
            }
        } catch (e) {
            alert(`Failed to load ${filename}: ${e}`);
        }
    }

    async function pasteTSV() {
        try {
            const text = await navigator.clipboard.readText();
            if (!importTSV(text)) {
                alert('Could not parse TSV from clipboard');
            }
        } catch {
            alert(
                'Could not read clipboard. Try pasting into the text area instead.',
            );
        }
    }

    return (
        <div className="not-prose flex flex-col gap-3">
            <div className="flex items-center gap-4 flex-wrap">
                <select
                    defaultValue=""
                    onChange={(e) => {
                        const val = e.target.value;
                        if (val === '__pinsetter1') {
                            const s = new Schedule(config);
                            s.createSchedule();
                            pinsetter1(s);
                            setScheduleFromPinsetter(s);
                        } else if (val) {
                            loadResultFile(val);
                        }
                    }}
                    className="pl-3 pr-9 py-3 rounded-lg border border-gray-300 bg-white text-sm"
                    disabled={generating}
                >
                    <option value="" disabled>
                        Load a schedule...
                    </option>
                    {resultFiles.map((f) => (
                        <option key={f} value={f}>
                            {f.replace('.tsv', '')}
                        </option>
                    ))}
                    <option value="__pinsetter1">Manual attempt</option>
                </select>
                <button
                    type="button"
                    onClick={pasteTSV}
                    disabled={generating}
                    className="px-4 py-3 rounded-lg font-medium border border-gray-300 hover:bg-gray-100 disabled:opacity-50 transition-colors text-sm"
                >
                    Paste TSV
                </button>
                {schedule && (
                    <button
                        type="button"
                        onClick={() => {
                            navigator.clipboard.writeText(
                                scheduleToTSV(schedule),
                            );
                        }}
                        className="px-4 py-3 rounded-lg font-medium border border-gray-300 hover:bg-gray-100 transition-colors text-sm"
                    >
                        Copy TSV
                    </button>
                )}
            </div>
            <div className="flex items-center gap-4 flex-wrap">
                <select
                    defaultValue="solver"
                    onChange={(e) => {
                        algorithmRef.current = e.target.value as AlgorithmId;
                    }}
                    className="pl-3 pr-9 py-3 rounded-lg border border-gray-300 bg-white text-sm"
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
                        ? algorithmRef.current === 'solver'
                            ? 'Solving (WASM)...'
                            : 'Generating...'
                        : 'Generate new schedule'}
                </button>
                <select
                    value={iterations}
                    onChange={(e) => setIterations(Number(e.target.value))}
                    className="pl-3 pr-9 py-3 rounded-lg border border-gray-300 bg-white text-sm"
                    disabled={generating}
                >
                    <option value={100_000_000}>100M iter</option>
                    <option value={200_000_000}>200M iter</option>
                    <option value={400_000_000}>400M iter</option>
                    <option value={600_000_000}>600M iter</option>
                    <option value={1_000_000_000}>1B iter</option>
                </select>
                {schedule && (
                    <label className="flex items-center gap-1.5 text-sm cursor-pointer select-none">
                        <input
                            type="checkbox"
                            checked={useSeed}
                            onChange={(e) => setUseSeed(e.target.checked)}
                            disabled={generating}
                            className="w-4 h-4"
                        />
                        Seed from current
                    </label>
                )}
                {generating && algorithmRef.current === 'solver' && (
                    <button
                        type="button"
                        onClick={cancelSolve}
                        className="px-4 py-3 rounded-lg font-semibold text-white bg-red-500 hover:bg-red-600 transition-colors"
                    >
                        Cancel
                    </button>
                )}
            </div>
        </div>
    );
}
