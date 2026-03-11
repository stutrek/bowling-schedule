'use client';

import { useSummerSchedule } from '../context/SummerScheduleContext';
import { S_TEAMS, S_WEEKS } from '../lib/summer-schedule-utils';

function isConsecutive(slots: number[]): boolean {
    return (
        slots.length === 3 &&
        slots[1] === slots[0] + 1 &&
        slots[2] === slots[1] + 1
    );
}

export default function SummerTimeGapsTable() {
    const { analysis } = useSummerSchedule();
    if (!analysis) return null;

    const teamStats = Array.from({ length: S_TEAMS }, (_, t) => {
        let consecutive = 0;
        for (let w = 0; w < S_WEEKS; w++) {
            if (isConsecutive(analysis.teamWeekSlots[t][w])) consecutive++;
        }
        return { consecutive, breakWeeks: S_WEEKS - consecutive };
    });

    return (
        <>
            <h2>Game Spacing</h2>
            <p className="text-sm text-gray-600">
                How many weeks each team plays three consecutive games vs. weeks
                with a break game. Ideally 3–4 break weeks and 6–7 consecutive
                weeks per team.
            </p>
            <div className="overflow-x-auto">
                <table className="text-sm">
                    <thead>
                        <tr>
                            <th />
                            {teamStats.map((_, t) => (
                                // biome-ignore lint/suspicious/noArrayIndexKey: sequential
                                <th key={t}>{t + 1}</th>
                            ))}
                        </tr>
                    </thead>
                    <tbody>
                        <tr>
                            <td>Break</td>
                            {teamStats.map((stats, t) => {
                                const off =
                                    stats.breakWeeks < 3 ||
                                    stats.breakWeeks > 4;
                                return (
                                    <td
                                        // biome-ignore lint/suspicious/noArrayIndexKey: sequential
                                        key={t}
                                        className={`text-center ${off ? 'bg-red-100' : ''}`}
                                    >
                                        {stats.breakWeeks}
                                    </td>
                                );
                            })}
                        </tr>
                        <tr>
                            <td>Consecutive</td>
                            {teamStats.map((stats, t) => {
                                const off =
                                    stats.consecutive < 6 ||
                                    stats.consecutive > 7;
                                return (
                                    <td
                                        // biome-ignore lint/suspicious/noArrayIndexKey: sequential
                                        key={t}
                                        className={`text-center ${off ? 'bg-red-100' : ''}`}
                                    >
                                        {stats.consecutive}
                                    </td>
                                );
                            })}
                        </tr>
                    </tbody>
                </table>
            </div>
        </>
    );
}
