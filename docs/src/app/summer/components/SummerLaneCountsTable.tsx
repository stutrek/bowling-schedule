'use client';

import { useSummerSchedule } from '../context/SummerScheduleContext';
import { S_TEAMS } from '../lib/summer-schedule-utils';

function laneColor(count: number): string {
    if (count === 7 || count === 8) return 'bg-lime-300';
    return 'bg-red-300';
}

export default function SummerLaneCountsTable() {
    const { analysis } = useSummerSchedule();
    if (!analysis) return null;

    return (
        <>
            <h2>Lane Counts</h2>
            <p className="text-sm text-gray-600">
                Each team should be on each lane 7-8 times (30 games / 4 lanes).
            </p>
            <div className="overflow-x-auto">
                <table className="text-sm">
                    <thead>
                        <tr>
                            <th> </th>
                            {Array.from({ length: S_TEAMS }, (_, i) => (
                                // biome-ignore lint/suspicious/noArrayIndexKey: sequential
                                <th key={i}>{i + 1}</th>
                            ))}
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
                                        className={laneColor(count)}
                                        title={`Team ${j + 1} on lane ${i + 1}: ${count}x (expect 7-8)`}
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
