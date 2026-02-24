'use client';

import { useSchedule } from '../context/ScheduleContext';

export default function CostBreakdownTable() {
    const { cost } = useSchedule();
    if (!cost) return null;

    const rows = [
        {
            label: 'Matchup balance',
            value: cost.matchupBalance,
            desc: 'Every pair plays exactly 1 or 2 times',
        },
        {
            label: 'Consecutive opponents',
            value: cost.consecutiveOpponents,
            desc: 'No repeat matchups in adjacent weeks',
        },
        {
            label: 'Early/late balance',
            value: cost.earlyLateBalance,
            desc: '6 early + 6 late per team',
        },
        {
            label: 'Early/late alternation',
            value: cost.earlyLateAlternation,
            desc: 'No three early or late in a row',
        },
        {
            label: 'Lane balance',
            value: cost.laneBalance,
            desc: 'Each team on each lane 6 times',
        },
        {
            label: 'Lane switches',
            value: cost.laneSwitchBalance,
            desc: 'Equal stay vs switch between games',
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
