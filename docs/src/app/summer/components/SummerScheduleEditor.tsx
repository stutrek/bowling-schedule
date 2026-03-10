'use client';

import { useRef, useState } from 'react';
import { useSummerSchedule } from '../context/SummerScheduleContext';
import {
    S_WEEKS,
    S_SLOTS,
    S_PAIRS,
    S_TEAMS,
    slotNames,
    summerScheduleToTSV,
} from '../lib/summer-schedule-utils';

export default function SummerScheduleEditor() {
    const {
        schedule,
        cost,
        violations,
        highlightInput,
        setHighlightInput,
        importTSV,
        selectedTeam,
        editHistory,
        handleTeamClick,
        undo,
    } = useSummerSchedule();

    const [copied, setCopied] = useState(false);
    const copiedTimer = useRef<ReturnType<typeof setTimeout>>();

    if (!schedule || !cost) return null;

    const highlightedTeams = new Set(
        highlightInput
            .split(',')
            .map((s) => Number.parseInt(s.trim(), 10) - 1)
            .filter((n) => !Number.isNaN(n) && n >= 0 && n < S_TEAMS),
    );

    function teamButtonClass(team: number, week: number): string {
        const base =
            'px-1 py-0.5 rounded text-xs font-mono cursor-pointer transition-colors';
        const hl = highlightedTeams.has(team)
            ? ' ring-2 ring-amber-400 bg-amber-50'
            : '';
        if (
            selectedTeam &&
            selectedTeam.team === team &&
            selectedTeam.week === week
        ) {
            return `${base} bg-blue-600 text-white ring-2 ring-blue-400`;
        }
        if (selectedTeam && selectedTeam.week === week) {
            return `${base} hover:bg-blue-100 text-blue-700 underline decoration-dotted${hl}`;
        }
        return `${base} hover:bg-gray-200${hl}`;
    }

    function hasSpacingViolation(
        week: number,
        teamA: number,
        teamB: number,
    ): boolean {
        if (!violations) return false;
        const lo = Math.min(teamA, teamB);
        const hi = Math.max(teamA, teamB);
        return violations.spacingPairs.has(`${week}-${lo}-${hi}`);
    }

    function copyTSV() {
        if (!schedule) return;
        navigator.clipboard.writeText(summerScheduleToTSV(schedule));
        setCopied(true);
        clearTimeout(copiedTimer.current);
        copiedTimer.current = setTimeout(() => setCopied(false), 2000);
    }

    async function pasteTSV() {
        try {
            const text = await navigator.clipboard.readText();
            if (!importTSV(text)) {
                alert('Could not parse summer TSV from clipboard');
            }
        } catch {
            alert('Could not read clipboard.');
        }
    }

    return (
        <aside className="w-full border-t border-gray-200 p-3 flex flex-col gap-3 bg-gray-50 lg:sticky lg:top-0 lg:h-screen lg:overflow-y-auto lg:border-t-0 lg:border-l lg:w-[480px] lg:flex-shrink-0">
            <div className="flex items-center gap-2 flex-wrap">
                <button
                    type="button"
                    onClick={undo}
                    disabled={editHistory.length === 0}
                    className="px-3 py-1.5 rounded font-medium border border-gray-300 hover:bg-gray-100 disabled:opacity-40 disabled:cursor-default disabled:hover:bg-white transition-colors text-sm bg-white"
                >
                    Undo
                    {editHistory.length > 0 ? ` (${editHistory.length})` : ''}
                </button>
                <span className="text-xs text-gray-500 font-bold">
                    Total: {cost.total}
                </span>
            </div>

            <div>
                <label
                    htmlFor="highlight-teams"
                    className="text-xs text-gray-500 block mb-1"
                >
                    Highlight teams (comma-separated)
                </label>
                <input
                    id="highlight-teams"
                    type="text"
                    value={highlightInput}
                    onChange={(e) => setHighlightInput(e.target.value)}
                    placeholder="e.g. 1, 5, 12"
                    className="w-full px-2 py-1.5 rounded border border-gray-300 text-sm bg-white"
                />
            </div>

            <p className="text-xs m-0">
                {selectedTeam ? (
                    <span className="text-blue-600">
                        Team {selectedTeam.team + 1} selected (wk{' '}
                        {selectedTeam.week + 1}). Click to swap, Esc to cancel.
                    </span>
                ) : (
                    <span className="text-gray-400">No selection</span>
                )}
            </p>

            <div className="overflow-y-auto max-h-[60vh] lg:max-h-none lg:flex-1 -mx-1 px-1">
                <table className="text-xs w-full border-collapse">
                    <thead className="sticky top-0 bg-gray-50 z-10">
                        <tr>
                            <th className="text-left px-1">Wk</th>
                            <th className="text-left px-1">Game</th>
                            <th className="text-center px-1">Lane 1</th>
                            <th className="text-center px-1">Lane 2</th>
                            <th className="text-center px-1">Lane 3</th>
                            <th className="text-center px-1">Lane 4</th>
                        </tr>
                    </thead>
                    <tbody>
                        {Array.from({ length: S_WEEKS }, (_, week) =>
                            Array.from({ length: S_SLOTS }, (_, slot) => (
                                <tr
                                    // biome-ignore lint/suspicious/noArrayIndexKey: stable grid
                                    key={`${week}-${slot}`}
                                    className={
                                        slot === 0
                                            ? 'border-t border-gray-300'
                                            : ''
                                    }
                                >
                                    {slot === 0 && (
                                        <td
                                            rowSpan={S_SLOTS}
                                            className="font-bold align-middle text-center px-1"
                                        >
                                            {week + 1}
                                        </td>
                                    )}
                                    <td className="px-1 whitespace-nowrap">
                                        {slotNames[slot]}
                                    </td>
                                    {Array.from(
                                        { length: S_PAIRS },
                                        (_, pair) => {
                                            const m =
                                                schedule[week][slot][pair];
                                            const violated = m
                                                ? hasSpacingViolation(
                                                      week,
                                                      m.teamA,
                                                      m.teamB,
                                                  )
                                                : false;
                                            return (
                                                <td
                                                    // biome-ignore lint/suspicious/noArrayIndexKey: stable grid
                                                    key={pair}
                                                    className={`text-center whitespace-nowrap px-0.5 ${violated ? 'bg-red-100 border-l-2 border-red-500' : ''}`}
                                                >
                                                    {m ? (
                                                        <>
                                                            <button
                                                                type="button"
                                                                onClick={() =>
                                                                    handleTeamClick(
                                                                        m.teamA,
                                                                        week,
                                                                        slot,
                                                                        pair,
                                                                        'A',
                                                                    )
                                                                }
                                                                className={teamButtonClass(
                                                                    m.teamA,
                                                                    week,
                                                                )}
                                                            >
                                                                {m.teamA + 1}
                                                            </button>
                                                            <span className="text-gray-400">
                                                                {' '}
                                                                v{' '}
                                                            </span>
                                                            <button
                                                                type="button"
                                                                onClick={() =>
                                                                    handleTeamClick(
                                                                        m.teamB,
                                                                        week,
                                                                        slot,
                                                                        pair,
                                                                        'B',
                                                                    )
                                                                }
                                                                className={teamButtonClass(
                                                                    m.teamB,
                                                                    week,
                                                                )}
                                                            >
                                                                {m.teamB + 1}
                                                            </button>
                                                        </>
                                                    ) : (
                                                        <span className="text-gray-300">
                                                            -
                                                        </span>
                                                    )}
                                                </td>
                                            );
                                        },
                                    )}
                                </tr>
                            )),
                        )}
                    </tbody>
                </table>
            </div>

            <div className="flex gap-2 pt-2 border-t border-gray-200">
                <button
                    type="button"
                    onClick={copyTSV}
                    className="px-4 py-2 rounded text-sm font-medium border border-gray-300 hover:bg-gray-100 transition-colors"
                >
                    {copied ? 'Copied!' : 'Copy for Sheets'}
                </button>
                <button
                    type="button"
                    onClick={pasteTSV}
                    className="px-4 py-2 rounded text-sm font-medium border border-gray-300 hover:bg-gray-100 transition-colors"
                >
                    Paste from TSV
                </button>
            </div>
        </aside>
    );
}
