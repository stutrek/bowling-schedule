'use client';

import { useSummerSchedule } from '../context/SummerScheduleContext';
import { S_TEAMS } from '../lib/summer-schedule-utils';

function breakColor(count: number): string {
    if (count >= 3 && count <= 4) return 'bg-lime-300';
    return 'bg-red-300';
}

export default function SummerSameLaneTable() {
    const { analysis } = useSummerSchedule();
    if (!analysis) return null;

    return (
        <>
            <h2>Break Balance</h2>
            <p className="text-sm text-gray-600">
                Weeks where a team has a break game (non-consecutive slots).
                Target: 3-4 per team.
            </p>
            <div className="overflow-x-auto">
                <table className="text-sm">
                    <thead>
                        <tr>
                            {Array.from({ length: S_TEAMS }, (_, i) => (
                                // biome-ignore lint/suspicious/noArrayIndexKey: sequential
                                <th key={i}>Team {i + 1}</th>
                            ))}
                        </tr>
                    </thead>
                    <tbody>
                        <tr>
                            {analysis.breakCounts.map((count, i) => (
                                <td
                                    // biome-ignore lint/suspicious/noArrayIndexKey: sequential
                                    key={i}
                                    className={breakColor(count)}
                                    title={`Team ${i + 1}: ${count} break weeks (target 3-4)`}
                                >
                                    {count}
                                </td>
                            ))}
                        </tr>
                    </tbody>
                </table>
            </div>
        </>
    );
}
