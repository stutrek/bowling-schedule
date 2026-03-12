'use client';

import { useSchedule } from '../context/ScheduleContext';
import { config } from '../lib/schedule-utils';

export default function EarlyLateTable() {
    const { analysis, violations } = useSchedule();
    if (!analysis || !violations) return null;

    return (
        <>
            <h2>Early Or Late</h2>
            <div className="overflow-x-auto">
                <table className="text-sm">
                    <thead>
                        <tr>
                            <th> </th>
                            {Array.from({ length: config.days }, (_, i) => (
                                // biome-ignore lint/suspicious/noArrayIndexKey: sequential
                                <th key={i}>{i + 1}</th>
                            ))}
                        </tr>
                    </thead>
                    <tbody>
                        {analysis.groups.map((row, i) => (
                            // biome-ignore lint/suspicious/noArrayIndexKey: sequential
                            <tr key={i}>
                                <td>Team {i + 1}</td>
                                {row.map((group, j) => {
                                    const isStreak =
                                        violations.earlyLateStreaks.has(
                                            `${i}-${j}`,
                                        );
                                    const isConsecutive =
                                        !isStreak &&
                                        violations.earlyLateConsecutive.has(
                                            `${i}-${j}`,
                                        );
                                    let bg =
                                        group < 3 ? 'bg-white' : 'bg-black';
                                    if (isStreak)
                                        bg =
                                            group < 3
                                                ? 'bg-red-200'
                                                : 'bg-red-800';
                                    else if (isConsecutive)
                                        bg =
                                            group < 3
                                                ? 'bg-yellow-100'
                                                : 'bg-yellow-900';
                                    return (
                                        <td
                                            // biome-ignore lint/suspicious/noArrayIndexKey: sequential
                                            key={j}
                                            className={bg}
                                            title={
                                                isStreak
                                                    ? `Team ${i + 1}: 3+ consecutive ${group < 3 ? 'early' : 'late'} weeks`
                                                    : isConsecutive
                                                      ? `Team ${i + 1}: 2 consecutive ${group < 3 ? 'early' : 'late'} weeks`
                                                      : undefined
                                            }
                                        >
                                            {group}
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
