'use client';

import { useSummerSchedule } from '../context/SummerScheduleContext';
import { S_TEAMS } from '../lib/summer-schedule-utils';

function laneColor(lane: number, count: number): string {
    const target = lane < 2 ? 7 : 8;
    if (count === target) return 'bg-lime-300';
    return 'bg-red-300';
}

export default function SummerLaneCountsTable() {
    const { analysis } = useSummerSchedule();
    if (!analysis) return null;

    return (
        <>
            <h2>Lane Balance</h2>
            <p className="text-sm text-gray-600">
                Lanes 1-2: 7 times per team. Lanes 3-4: 8 times per team.
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
                                {row.map((count, j) => {
                                    const target = i < 2 ? 7 : 8;
                                    return (
                                        <td
                                            // biome-ignore lint/suspicious/noArrayIndexKey: sequential
                                            key={j}
                                            className={laneColor(i, count)}
                                            title={`Team ${j + 1} on lane ${i + 1}: ${count}x (target ${target})`}
                                        >
                                            {count}
                                        </td>
                                    );
                                })}
                            </tr>
                        ))}
                    </tbody>
                </table>
            </div>
        </>
    );
}
