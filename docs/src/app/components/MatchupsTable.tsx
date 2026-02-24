'use client';

import { useSchedule } from '../context/ScheduleContext';
import { config } from '../lib/schedule-utils';

const gamesColors = ['bg-red-500', 'bg-lime-300', 'bg-lime-700', 'bg-red-300'];

export default function MatchupsTable() {
    const { analysis } = useSchedule();
    if (!analysis) return null;

    return (
        <>
            <h2>Matchups</h2>
            <div className="overflow-x-auto">
                <table className="text-sm">
                    <thead>
                        <tr>
                            <th>Team</th>
                            {Array.from({ length: config.teams }, (_, i) => (
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
                                {row.map((matchup, j) => (
                                    <td
                                        // biome-ignore lint/suspicious/noArrayIndexKey: sequential
                                        key={j}
                                        className={
                                            i === j ? '' : gamesColors[matchup]
                                        }
                                        title={
                                            i !== j
                                                ? `Teams ${i + 1} & ${j + 1}: ${matchup} matchup${matchup !== 1 ? 's' : ''} (expect 1–2)`
                                                : undefined
                                        }
                                    >
                                        {i === j ? ' ' : matchup}
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
