import { useMemo } from 'react';
import Intro from './Intro.mdx';
import { Header } from './header';
import { Schedule } from '../../../src/schedule';
import { fillRandomly } from '../../../src/random';
import type { Config } from '../../../src/types';

const config: Config = {
    teams: 16,
    timeSlots: 4,
    lanes: 4,
    days: 12,
};

const gamesColors = ['bg-red-500', 'bg-lime-300', 'bg-lime-700', 'bg-red-300'];
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

export default function () {
    const schedule = useMemo(() => {
        const schedule = new Schedule(config);
        schedule.createSchedule();

        fillRandomly(schedule);
        console.log(schedule);
        return schedule;
    }, []);

    const { matchups, laneCounts, slotCounts } = useMemo(() => {
        const matchups: number[][] = new Array(schedule.config.teams)
            .fill([])
            .map(() => new Array(schedule.config.teams).fill(0));
        const laneCounts: number[][] = new Array(schedule.config.lanes)
            .fill([])
            .map(() => new Array(schedule.config.teams).fill(0));
        const slotCounts: number[][] = new Array(schedule.config.timeSlots)
            .fill([])
            .map(() => new Array(schedule.config.teams).fill(0));
        schedule.schedule.map((game) => {
            matchups[game.teams[0]][game.teams[1]]++;
            laneCounts[game.lane][game.teams[0]]++;
            laneCounts[game.lane][game.teams[1]]++;
            slotCounts[game.timeSlot][game.teams[0]]++;
            slotCounts[game.timeSlot][game.teams[1]]++;
        });

        return { matchups, laneCounts, slotCounts };
    }, [schedule]);

    return (
        <div className="container max-w-3xl mx-auto prose p-4">
            {/* <div className="text-center">
                Image?
                <Header showLogo={false} />
            </div> */}

            <h2>Matchups</h2>
            <table>
                <thead>
                    <tr>
                        <th>Team</th>
                        {Array.from({ length: config.teams }).map((_, i) => (
                            // biome-ignore lint/suspicious/noArrayIndexKey: <explanation>
                            <th key={i}>{16 - i}</th>
                        ))}
                    </tr>
                </thead>
                <tbody>
                    {matchups.map((row, i) => (
                        // biome-ignore lint/suspicious/noArrayIndexKey: <explanation>
                        <tr key={i}>
                            <td>{i + 1}</td>
                            {row.reverse().map(
                                (matchup, j) =>
                                    i < 15 - j && (
                                        <td
                                            // biome-ignore lint/suspicious/noArrayIndexKey: <explanation>
                                            key={j}
                                            className={gamesColors[matchup]}
                                        >
                                            {matchup}
                                        </td>
                                    ),
                            )}
                        </tr>
                    ))}
                </tbody>
            </table>

            <h2>Lane Counts</h2>
            <table>
                <thead>
                    <tr>
                        <th>Team</th>
                        {Array.from({ length: config.teams }).map((_, i) => (
                            // biome-ignore lint/suspicious/noArrayIndexKey: <explanation>
                            <th key={i}>{i + 1}</th>
                        ))}
                    </tr>
                </thead>
                <tbody>
                    {laneCounts.map((row, i) => (
                        // biome-ignore lint/suspicious/noArrayIndexKey: <explanation>
                        <tr key={i}>
                            <td>{i + 1}</td>
                            {row.map((_, j) => (
                                <td
                                    // biome-ignore lint/suspicious/noArrayIndexKey: <explanation>
                                    key={j}
                                    className={lanesColors[laneCounts[i][j]]}
                                >
                                    {laneCounts[i][j]}
                                </td>
                            ))}
                        </tr>
                    ))}
                </tbody>
            </table>

            <h2>Slot Counts</h2>
            <table>
                <thead>
                    <tr>
                        <th>Team</th>
                        {Array.from({ length: config.teams }).map((_, i) => (
                            // biome-ignore lint/suspicious/noArrayIndexKey: <explanation>
                            <th key={i}>{i + 1}</th>
                        ))}
                    </tr>
                </thead>
                <tbody>
                    {slotCounts.map((row, i) => (
                        // biome-ignore lint/suspicious/noArrayIndexKey: <explanation>
                        <tr key={i}>
                            <td>{i + 1}</td>
                            {row.map((_, j) => (
                                <td
                                    // biome-ignore lint/suspicious/noArrayIndexKey: <explanation>
                                    key={j}
                                    className={lanesColors[laneCounts[i][j]]}
                                >
                                    {laneCounts[i][j]}
                                </td>
                            ))}
                        </tr>
                    ))}
                </tbody>
            </table>

            <Intro />
        </div>
    );
}
