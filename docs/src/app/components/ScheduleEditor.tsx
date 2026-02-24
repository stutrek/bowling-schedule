'use client';

import { useRef, useState } from 'react';
import { useSchedule } from '../context/ScheduleContext';
import {
    config,
    slotNames,
    findGame,
    gameHasConsecutiveViolation,
    scheduleToTSV,
} from '../lib/schedule-utils';

export default function ScheduleEditor() {
    const {
        schedule,
        cost,
        violations,
        editHistory,
        selectedTeam,
        highlightInput,
        handleTeamClick,
        undo,
        setHighlightInput,
        importTSV,
    } = useSchedule();

    const [copied, setCopied] = useState(false);
    const copiedTimer = useRef<ReturnType<typeof setTimeout>>();

    if (!schedule || !cost || !violations) return null;

    const highlightedTeams = new Set(
        highlightInput
            .split(',')
            .map((s) => Number.parseInt(s.trim(), 10) - 1)
            .filter((n) => !Number.isNaN(n) && n >= 0 && n < config.teams),
    );

    function teamButtonClass(team: number, day: number): string {
        const base =
            'px-1 py-0.5 rounded text-xs font-mono cursor-pointer transition-colors';
        const hl = highlightedTeams.has(team)
            ? ' ring-2 ring-amber-400 bg-amber-50'
            : '';
        if (
            selectedTeam &&
            selectedTeam.team === team &&
            selectedTeam.day === day
        ) {
            return `${base} bg-blue-600 text-white ring-2 ring-blue-400`;
        }
        if (selectedTeam) {
            return `${base} hover:bg-blue-100 text-blue-700 underline decoration-dotted${hl}`;
        }
        return `${base} hover:bg-gray-200${hl}`;
    }

    function copyTSV() {
        if (!schedule) return;
        navigator.clipboard.writeText(scheduleToTSV(schedule));
        setCopied(true);
        clearTimeout(copiedTimer.current);
        copiedTimer.current = setTimeout(() => setCopied(false), 2000);
    }

    async function pasteTSV() {
        try {
            const text = await navigator.clipboard.readText();
            if (!importTSV(text)) {
                alert('Could not parse TSV from clipboard');
            }
        } catch {
            alert('Could not read clipboard.');
        }
    }

    return (
        <aside
            className="sticky top-0 h-screen overflow-y-auto border-l border-gray-200 bg-gray-50 flex-shrink-0 p-3 flex flex-col gap-3"
            style={{ width: '480px' }}
        >
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

            {selectedTeam && (
                <p className="text-xs text-blue-600 m-0">
                    Team {selectedTeam.team + 1} selected (week{' '}
                    {selectedTeam.day + 1}). Click another team to swap. Escape
                    to cancel.
                </p>
            )}

            <div className="overflow-y-auto flex-1 -mx-1 px-1">
                <table className="text-xs w-full border-collapse">
                    <thead className="sticky top-0 bg-gray-50 z-10">
                        <tr>
                            <th className="text-left px-1">Wk</th>
                            <th className="text-left px-1">Time</th>
                            <th className="text-center px-1">Lane 1</th>
                            <th className="text-center px-1">Lane 2</th>
                            <th className="text-center px-1">Lane 3</th>
                            <th className="text-center px-1">Lane 4</th>
                        </tr>
                    </thead>
                    <tbody>
                        {Array.from({ length: config.days }, (_, day) =>
                            Array.from(
                                { length: config.timeSlots },
                                (_, slot) => (
                                    <tr // biome-ignore lint/suspicious/noArrayIndexKey: stable grid
                                        key={`${day}-${slot}`}
                                        className={
                                            slot === 0
                                                ? 'border-t border-gray-300'
                                                : ''
                                        }
                                    >
                                        {slot === 0 && (
                                            <td
                                                rowSpan={config.timeSlots}
                                                className="font-bold align-middle text-center px-1"
                                            >
                                                {day + 1}
                                            </td>
                                        )}
                                        <td
                                            className={`px-1 whitespace-nowrap ${slot < 2 ? 'bg-sky-50' : 'bg-amber-50'}`}
                                        >
                                            {slotNames[slot]}
                                        </td>
                                        {Array.from(
                                            { length: config.lanes },
                                            (_, lane) => {
                                                const game = findGame(
                                                    schedule.schedule,
                                                    day,
                                                    slot,
                                                    lane,
                                                );
                                                const hasViolation = game
                                                    ? gameHasConsecutiveViolation(
                                                          violations,
                                                          game,
                                                      )
                                                    : false;
                                                return (
                                                    <td // biome-ignore lint/suspicious/noArrayIndexKey: stable grid
                                                        key={`${day}-${slot}-${lane}`}
                                                        className={`text-center whitespace-nowrap px-0.5 ${hasViolation ? 'bg-red-100 border-l-2 border-red-500' : ''}`}
                                                        title={
                                                            hasViolation && game
                                                                ? `Teams ${game.teams[0] + 1} & ${game.teams[1] + 1} also play in week ${day > 0 && violations.consecutivePairs.has(`${day - 1}-${Math.min(...game.teams)}-${Math.max(...game.teams)}`) ? day : day + 2}`
                                                                : undefined
                                                        }
                                                    >
                                                        {game &&
                                                        game.teams[0] !== -1 ? (
                                                            <>
                                                                <button
                                                                    type="button"
                                                                    onClick={() =>
                                                                        handleTeamClick(
                                                                            game
                                                                                .teams[0],
                                                                            day,
                                                                        )
                                                                    }
                                                                    className={teamButtonClass(
                                                                        game
                                                                            .teams[0],
                                                                        day,
                                                                    )}
                                                                >
                                                                    {game
                                                                        .teams[0] +
                                                                        1}
                                                                </button>
                                                                <span className="text-gray-400">
                                                                    v
                                                                </span>
                                                                <button
                                                                    type="button"
                                                                    onClick={() =>
                                                                        handleTeamClick(
                                                                            game
                                                                                .teams[1],
                                                                            day,
                                                                        )
                                                                    }
                                                                    className={teamButtonClass(
                                                                        game
                                                                            .teams[1],
                                                                        day,
                                                                    )}
                                                                >
                                                                    {game
                                                                        .teams[1] +
                                                                        1}
                                                                </button>
                                                            </>
                                                        ) : null}
                                                    </td>
                                                );
                                            },
                                        )}
                                    </tr>
                                ),
                            ),
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
