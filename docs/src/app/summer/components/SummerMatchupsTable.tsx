'use client';

import { useSummerSchedule } from '../context/SummerScheduleContext';
import { S_TEAMS } from '../lib/summer-schedule-utils';

function matchupColor(count: number): string {
    if (count === 0) return 'bg-red-500';
    if (count === 1) return 'bg-red-300';
    if (count === 2) return 'bg-lime-300';
    if (count === 3) return 'bg-lime-700';
    return 'bg-red-300';
}

export default function SummerMatchupsTable() {
    const { analysis } = useSummerSchedule();
    if (!analysis) return null;

    return (
        <>
            <h2>Matchups</h2>
            <p className="text-sm text-gray-600">
                Each pair should play 2-3 times. Green = good, red = bad.
            </p>
            <div className="overflow-x-auto">
                <table className="text-sm">
                    <thead>
                        <tr>
                            <th>Team</th>
                            {Array.from({ length: S_TEAMS }, (_, i) => (
                                // biome-ignore lint/suspicious/noArrayIndexKey: sequential
                                <th key={i}>{i + 1}</th>
                            ))}
                        </tr>
                    </thead>
                    <tbody>
                        {analysis.matchups.map((row, i) => (
                            // biome-ignore lint/suspicious/noArrayIndexKey: sequential
                            <tr key={i}>
                                <td>{i + 1}</td>
                                {row.map((count, j) => (
                                    <td
                                        // biome-ignore lint/suspicious/noArrayIndexKey: sequential
                                        key={j}
                                        className={
                                            i === j ? '' : matchupColor(count)
                                        }
                                        title={
                                            i !== j
                                                ? `Teams ${i + 1} & ${j + 1}: ${count} matchup${count !== 1 ? 's' : ''} (expect 2-3)`
                                                : undefined
                                        }
                                    >
                                        {i === j ? ' ' : count}
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
