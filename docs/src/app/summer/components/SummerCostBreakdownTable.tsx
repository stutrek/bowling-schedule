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
            label: 'Slot balance',
            value: cost.slotBalance,
            desc: 'Games 1-4: 6-7 each; Game 5: 3-4 each',
        },
        {
            label: 'Lane balance',
            value: cost.laneBalance,
            desc: 'Lanes 1-2: 6-7 times; Lanes 3-4: 8-9 times',
        },
        {
            label: 'Game 5 lane balance',
            value: cost.game5LaneBalance,
            desc: 'Game 5 appearances split evenly between lanes 3 and 4',
        },
        {
            label: 'Same lane balance',
            value: cost.sameLaneBalance,
            desc: 'Same lane for all games 1-4: 3-4 times per team',
        },
        {
            label: 'Commissioner overlap',
            value: cost.commissionerOverlap,
            desc: 'Min pair co-appearance in games 1 and 5',
        },
        {
            label: 'Matchup spacing',
            value: cost.matchupSpacing,
            desc: '2 matchups: 4+ weeks apart; 3 matchups: 2+ weeks apart',
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
