'use client';

import { useSummerSchedule } from '../context/SummerScheduleContext';
import { S_TEAMS } from '../lib/summer-schedule-utils';

export default function SummerLaneSwitchesTable() {
    const { analysis } = useSummerSchedule();
    if (!analysis) return null;

    return (
        <>
            <h2>Lane Switches</h2>
            <p className="text-sm text-gray-600">
                How often each team changes lanes between games within a week.
            </p>
            <div className="overflow-x-auto">
                <table className="text-sm">
                    <thead>
                        <tr>
                            <th>Team</th>
                            <th>Consecutive switches</th>
                            <th>Post-break switches</th>
                        </tr>
                    </thead>
                    <tbody>
                        {Array.from({ length: S_TEAMS }, (_, t) => (
                            // biome-ignore lint/suspicious/noArrayIndexKey: sequential
                            <tr key={t}>
                                <td>{t + 1}</td>
                                <td className="text-center">
                                    {analysis.laneSwitchCounts[t].consecutive}
                                </td>
                                <td className="text-center">
                                    {analysis.laneSwitchCounts[t].postBreak}
                                </td>
                            </tr>
                        ))}
                    </tbody>
                </table>
            </div>
        </>
    );
}
