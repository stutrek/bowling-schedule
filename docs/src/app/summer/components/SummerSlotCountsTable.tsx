'use client';

import { useSummerSchedule } from '../context/SummerScheduleContext';
import { S_TEAMS, slotNames } from '../lib/summer-schedule-utils';

function slotColor(slot: number, count: number): string {
    if (slot < 4) {
        return count === 6 || count === 7 ? 'bg-lime-300' : 'bg-red-300';
    }
    return count === 3 || count === 4 ? 'bg-lime-300' : 'bg-red-300';
}

export default function SummerSlotCountsTable() {
    const { analysis } = useSummerSchedule();
    if (!analysis) return null;

    return (
        <>
            <h2>Slot Counts</h2>
            <p className="text-sm text-gray-600">
                Games 1-4: each team 6-7 times. Game 5: each team 3-4 times.
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
                        {analysis.slotCounts.map((row, i) => (
                            // biome-ignore lint/suspicious/noArrayIndexKey: sequential
                            <tr key={i}>
                                <td>{slotNames[i]}</td>
                                {row.map((count, j) => (
                                    <td
                                        // biome-ignore lint/suspicious/noArrayIndexKey: sequential
                                        key={j}
                                        className={slotColor(i, count)}
                                        title={`Team ${j + 1} in ${slotNames[i]}: ${count}x (expect ${i < 4 ? '6-7' : '3-4'})`}
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
