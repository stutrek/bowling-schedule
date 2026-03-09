'use client';

import { useSchedule } from '../context/ScheduleContext';
import { config } from '../lib/schedule-utils';

const colors = [
    'bg-red-300',
    'bg-lime-300',
    'bg-lime-300',
    'bg-lime-700',
    'bg-lime-300',
    'bg-lime-300',
    'bg-red-300',
];

export default function LastGameLaneTable() {
    const { analysis } = useSchedule();
    if (!analysis) return null;

    return (
        <>
            <h2>Last Game Lane Balance</h2>
            <div className="overflow-x-auto">
                <table className="text-sm">
                    <thead>
                        <tr>
                            <th> </th>
                            {Array.from({ length: config.teams }, (_, i) => (
                                // biome-ignore lint/suspicious/noArrayIndexKey: sequential
                                <th key={i}>{i + 1}</th>
                            ))}
                        </tr>
                    </thead>
                    <tbody>
                        {analysis.lastGameLaneCounts.map((row, i) => (
                            // biome-ignore lint/suspicious/noArrayIndexKey: sequential
                            <tr key={i}>
                                <td>Lane {i + 1}</td>
                                {row.map((count, j) => (
                                    <td
                                        // biome-ignore lint/suspicious/noArrayIndexKey: sequential
                                        key={j}
                                        className={
                                            colors[count] ?? 'bg-red-300'
                                        }
                                        title={`Team ${j + 1} last game on lane ${i + 1}: ${count}× (expect 3)`}
                                    >
                                        {count}
                                    </td>
                                ))}
                            </tr>
                        ))}
                    </tbody>
                </table>
            </div>
        </>
    );
}
