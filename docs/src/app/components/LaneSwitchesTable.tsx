'use client';

import { useSchedule } from '../context/ScheduleContext';
import { config } from '../lib/schedule-utils';

function deviationClass(dev: number) {
    if (dev === 0) return 'bg-green-100';
    if (dev <= 1) return 'bg-yellow-100';
    return 'bg-red-100';
}

export default function LaneSwitchesTable() {
    const { analysis } = useSchedule();
    if (!analysis) return null;

    const target = config.days / 2;

    return (
        <>
            <h2>Lane Switches</h2>
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
                        <tr>
                            <td>Stay</td>
                            {analysis.laneSwitchCounts.map((sw, i) => {
                                const stay = config.days - sw;
                                return (
                                    <td
                                        // biome-ignore lint/suspicious/noArrayIndexKey: sequential
                                        key={i}
                                        className={deviationClass(
                                            Math.abs(stay - target),
                                        )}
                                        title={`Team ${i + 1}: ${stay} stay, ${sw} switch (expect ${target} each)`}
                                    >
                                        {stay}
                                    </td>
                                );
                            })}
                        </tr>
                        <tr>
                            <td>Switch</td>
                            {analysis.laneSwitchCounts.map((sw, i) => (
                                <td
                                    // biome-ignore lint/suspicious/noArrayIndexKey: sequential
                                    key={i}
                                    className={deviationClass(
                                        Math.abs(sw - target),
                                    )}
                                    title={`Team ${i + 1}: ${config.days - sw} stay, ${sw} switch (expect ${target} each)`}
                                >
                                    {sw}
                                </td>
                            ))}
                        </tr>
                    </tbody>
                </table>
            </div>
        </>
    );
}
