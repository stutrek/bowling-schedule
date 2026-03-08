'use client';

import { useSummerSchedule } from '../context/SummerScheduleContext';
import { S_TEAMS, S_WEEKS } from '../lib/summer-schedule-utils';

function formatSlots(slots: number[]): string {
    return slots.map((s) => s + 1).join(', ');
}

function gapStatus(slots: number[]): 'good' | 'consecutive' | 'large_gap' {
    if (slots.length < 2) return 'good';
    if (
        slots.length === 3 &&
        slots[1] === slots[0] + 1 &&
        slots[2] === slots[1] + 1
    ) {
        return 'consecutive';
    }
    for (let i = 0; i < slots.length - 1; i++) {
        if (slots[i + 1] - slots[i] - 1 >= 2) return 'large_gap';
    }
    return 'good';
}

function statusColor(status: string): string {
    switch (status) {
        case 'consecutive':
            return 'bg-amber-200';
        case 'large_gap':
            return 'bg-red-200';
        default:
            return '';
    }
}

export default function SummerTimeGapsTable() {
    const { analysis } = useSummerSchedule();
    if (!analysis) return null;

    return (
        <>
            <h2>Game Spacing</h2>
            <p className="text-sm text-gray-600">
                Which game slots each team plays per week. Ideal: two
                consecutive + one with a 1-game break. Yellow = 3 consecutive,
                Red = 2+ game gap.
            </p>
            <div className="overflow-x-auto">
                <table className="text-sm">
                    <thead>
                        <tr>
                            <th>Team</th>
                            {Array.from({ length: S_WEEKS }, (_, i) => (
                                // biome-ignore lint/suspicious/noArrayIndexKey: sequential
                                <th key={i}>Wk {i + 1}</th>
                            ))}
                        </tr>
                    </thead>
                    <tbody>
                        {Array.from({ length: S_TEAMS }, (_, t) => (
                            // biome-ignore lint/suspicious/noArrayIndexKey: sequential
                            <tr key={t}>
                                <td>{t + 1}</td>
                                {Array.from({ length: S_WEEKS }, (_, w) => {
                                    const slots = analysis.teamWeekSlots[t][w];
                                    const status = gapStatus(slots);
                                    return (
                                        <td
                                            // biome-ignore lint/suspicious/noArrayIndexKey: sequential
                                            key={w}
                                            className={`text-center whitespace-nowrap ${statusColor(status)}`}
                                            title={
                                                status !== 'good'
                                                    ? status === 'consecutive'
                                                        ? 'Three consecutive games'
                                                        : 'Gap of 2+ slots'
                                                    : undefined
                                            }
                                        >
                                            {formatSlots(slots)}
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
