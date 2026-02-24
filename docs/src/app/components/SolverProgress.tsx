'use client';

import { useSchedule } from '../context/ScheduleContext';

export default function SolverProgress() {
    const { solverProgress } = useSchedule();
    if (!solverProgress) return null;

    const { workers, globalBestCost } = solverProgress;
    const totalIter = workers.reduce((s, w) => s + w.iteration, 0);
    const totalMax = workers.reduce((s, w) => s + w.maxIterations, 0);
    const pct = totalMax > 0 ? (totalIter / totalMax) * 100 : 0;
    const doneCount = workers.filter((w) => w.done).length;

    return (
        <div className="not-prose mt-3 p-3 rounded-lg bg-blue-50 border border-blue-200 text-sm font-mono space-y-2">
            <div className="flex justify-between">
                <span>
                    {workers.length} cores &middot; {doneCount} finished
                    &middot; {(totalIter / 1_000_000).toFixed(0)}M /{' '}
                    {(totalMax / 1_000_000).toFixed(0)}M iter
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
                            ? (w.iteration / w.maxIterations) * 100
                            : 0;
                    return (
                        <div // biome-ignore lint/suspicious/noArrayIndexKey: stable worker indices
                            key={i}
                            title={`Core ${i + 1}: ${(w.iteration / 1_000_000).toFixed(0)}M iter, best ${w.bestCost === Number.POSITIVE_INFINITY ? '–' : w.bestCost}`}
                        >
                            <div className="bg-blue-200 rounded-full h-1.5">
                                <div
                                    className={`h-1.5 rounded-full transition-all duration-300 ${w.done ? 'bg-green-500' : 'bg-blue-500'}`}
                                    style={{ width: `${wp}%` }}
                                />
                            </div>
                        </div>
                    );
                })}
            </div>
        </div>
    );
}
