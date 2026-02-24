'use client';

import { useSchedule } from '../context/ScheduleContext';
import { config, slotNames } from '../lib/schedule-utils';

const lanesColors = [
    'bg-red-300',
    'bg-red-300',
    'bg-red-300',
    'bg-red-300',
    'bg-red-300',
    'bg-lime-300',
    'bg-lime-700',
    'bg-lime-300',
    'bg-red-300',
    'bg-red-300',
    'bg-red-300',
    'bg-red-300',
    'bg-red-300',
    'bg-red-300',
    'bg-red-300',
    'bg-red-300',
];

export default function SlotCountsTable() {
    const { analysis } = useSchedule();
    if (!analysis) return null;

    return (
        <>
            <h2>Slot Counts</h2>
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
                        {analysis.slotCounts.map((row, i) => (
                            // biome-ignore lint/suspicious/noArrayIndexKey: sequential
                            <tr key={i}>
                                <td>Slot {i + 1}</td>
                                {row.map((count, j) => (
                                    <td
                                        // biome-ignore lint/suspicious/noArrayIndexKey: sequential
                                        key={j}
                                        className={lanesColors[count]}
                                        title={`Team ${j + 1} in ${slotNames[i]}: ${count}× (expect 6)`}
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
