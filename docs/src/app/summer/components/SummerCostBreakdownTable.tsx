'use client';

import { useSummerSchedule } from '../context/SummerScheduleContext';

export default function SummerCostBreakdownTable() {
    const { cost } = useSummerSchedule();
    if (!cost) return null;

    const rows = [
        {
            label: 'Matchup balance',
            value: cost.matchupBalance,
            desc: 'Every pair plays 2-3 times across the season',
        },
        {
            label: 'Lane switches (consecutive)',
            value: cost.laneSwitches,
            desc: 'Changing lanes between back-to-back games',
        },
        {
            label: 'Lane switches (post-break)',
            value: cost.laneSwitchBreak,
            desc: 'Changing lanes across a gap',
        },
        {
            label: 'Time gaps',
            value: cost.timeGaps,
            desc: '3 consecutive games or 2+ slot gap between games',
        },
        {
            label: 'Lane balance',
            value: cost.laneBalance,
            desc: 'Each team on each lane 7-8 times',
        },
        {
            label: 'Commissioner overlap',
            value: cost.commissionerOverlap,
            desc: 'Min pair co-appearance in games 1 and 5',
        },
        {
            label: 'Repeat matchup same night',
            value: cost.repeatMatchupSameNight,
            desc: 'Same pair matched more than once per week',
        },
        {
            label: 'Slot balance',
            value: cost.slotBalance,
            desc: 'Games 1-4: 6-7 each; Game 5: 3-4 each',
        },
    ];

    return (
        <>
            <h2>Constraint Scores</h2>
            <p className="text-sm text-gray-600">
                Lower is better. Zero means the constraint is perfectly
                satisfied.
            </p>
            <table className="text-sm">
                <tbody>
                    {rows.map((row) => (
                        <tr key={row.label}>
                            <td>{row.label}</td>
                            <td
                                className={
                                    row.value === 0
                                        ? 'text-green-700 font-bold'
                                        : 'text-red-700 font-bold'
                                }
                            >
                                {row.value}
                            </td>
                            <td className="text-gray-500">{row.desc}</td>
                        </tr>
                    ))}
                    <tr className="border-t-2">
                        <td className="font-bold">Total</td>
                        <td
                            className={`font-bold ${cost.total === 0 ? 'text-green-700' : ''}`}
                        >
                            {cost.total}
                        </td>
                        <td />
                    </tr>
                </tbody>
            </table>
        </>
    );
}
