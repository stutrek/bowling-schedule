'use client';

import { useRef } from 'react';
import { Schedule } from '../../../../src/Schedule';
import { pinsetter as pinsetter1 } from '../../../../src/pinsetter1';
import { pinsetter as pinsetter2 } from '../../../../src/pinsetter2';
import { pinsetter as pinsetter3 } from '../../../../src/pinsetter3';
import { pinsetter as pinsetter4 } from '../../../../src/pinsetter4';
import { pinsetter as pinsetter5 } from '../../../../src/pinsetter5';
import { pinsetter as pinsetter6 } from '../../../../src/pinsetter6';
import {
    useSchedule,
    type SolverProgress,
    type WorkerProgress,
} from '../context/ScheduleContext';
import { config } from '../lib/schedule-utils';

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

export default function ControlBar() {
    const {
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

    function generate() {
        const algorithm = algorithmRef.current;
        setGenerating(true);
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

                worker.postMessage({
                    type: 'solve',
                    maxIterations: iterationsPerWorker,
                    weightsJson,
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
        <div className="not-prose flex items-center gap-4 flex-wrap">
            <select
                defaultValue="solver"
                onChange={(e) => {
                    algorithmRef.current = e.target.value as AlgorithmId;
                }}
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
                    ? algorithmRef.current === 'solver'
                        ? 'Solving (WASM)...'
                        : 'Generating...'
                    : 'Generate'}
            </button>
            {generating && algorithmRef.current === 'solver' && (
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
    );
}
