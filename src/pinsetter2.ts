import { groupBy } from 'lodash';
import type { Schedule } from './Schedule';

import chunk from 'lodash/chunk';
import sortBy from 'lodash/sortBy';
import zip from 'lodash/zip';

function shiftByTwo(arr: number[]) {
    const a = arr.shift() as number;
    const b = arr.shift() as number;
    arr.push(a, b);
}

class PinsetterAlgorithm {
    schedule: Schedule;
    daysFilled = 0;
    groupShares: number[][];

    constructor(schedule: Schedule) {
        this.schedule = schedule;
        this.groupShares = new Array(schedule.config.teams)
            .fill(0)
            .map(() => new Array(schedule.config.teams).fill(0));
    }

    fillGroup(teams: number[], day: number, slot: number, lane: number) {
        console.log('filling', teams);
        this.schedule.setGame(teams[0], teams[1], day, slot, lane);
        this.schedule.setGame(teams[2], teams[3], day, slot, lane + 1);
        this.schedule.setGame(teams[0], teams[3], day, slot + 1, lane);
        this.schedule.setGame(teams[2], teams[1], day, slot + 1, lane + 1);

        for (let i = 0; i < teams.length; i++) {
            for (let j = 0; j < teams.length; j++) {
                this.groupShares[teams[i]][teams[j]]++;
            }
        }
    }

    fillDay(groups: number[][]) {
        const day = this.daysFilled++;
        groups.forEach((group, i) => {
            const slot = i > 1 ? 2 : 0;
            const lane = (i % (this.schedule.config.lanes / 2)) * 2;
            this.fillGroup(group, day, slot, lane);
        });
    }

    selectGroup(
        cornerstoneTeam: number,
        alreadySelected: number[],
        slot: number,
        lane: number,
    ) {
        const teamsByFewestPlays = this.schedule
            .getMostCompatibleTeams(cornerstoneTeam, slot, lane)
            .filter((teamMeta) => !alreadySelected.includes(teamMeta.team));

        const opponentTeams = teamsByFewestPlays
            .splice(0, 2)
            .map((team) => team.team);

        const mostCompatibleWithFirstOpponent = this.schedule
            .getMostCompatibleTeams(opponentTeams[0], slot, lane + 1)
            .filter(
                (teamMeta) =>
                    !alreadySelected.includes(teamMeta.team) &&
                    teamMeta.team !== opponentTeams[1],
            );

        const mostCompatibleWithSecondOpponent = this.schedule
            .getMostCompatibleTeams(opponentTeams[0], slot, lane + 1)
            .filter(
                (teamMeta) =>
                    !alreadySelected.includes(teamMeta.team) &&
                    teamMeta.team !== opponentTeams[0],
            );

        let mostCompatibleWithOpponents: number = opponentTeams[3];
        let mostCompatibleScore = Number.POSITIVE_INFINITY;
        for (let i = 0; i < mostCompatibleWithFirstOpponent.length; i++) {
            for (let j = 0; j < mostCompatibleWithSecondOpponent.length; j++) {
                if (
                    mostCompatibleWithFirstOpponent[i].team ===
                    mostCompatibleWithSecondOpponent[j].team
                ) {
                    const compatibility = i + j;
                    if (compatibility < mostCompatibleScore) {
                        mostCompatibleScore = compatibility;
                        mostCompatibleWithOpponents =
                            mostCompatibleWithFirstOpponent[i].team;
                    }
                    break;
                }
            }
            if (i > mostCompatibleScore) {
                break;
            }
        }

        return [
            cornerstoneTeam,
            opponentTeams[0],
            mostCompatibleWithOpponents,
            opponentTeams[1],
        ];
    }

    fill() {
        for (let i = 0; i < this.schedule.config.days; i++) {
            const anchorTeams = [0, 4, 8, 12].map((n) => n + (i % 4));
            const teamsSoFar = [...anchorTeams];
            const groups: number[][] = [];
            for (let j = 0; j < anchorTeams.length; j++) {
                const team = anchorTeams[j];
                const group = this.selectGroup(
                    team,
                    teamsSoFar,
                    Math.floor(j / 2),
                    (j % 2) * 2,
                );
                teamsSoFar.push(...group);
                groups.push(group);
            }
            this.fillDay(groups);
        }
    }
}

export function pinsetter(schedule: Schedule) {
    const algorithm = new PinsetterAlgorithm(schedule);
    algorithm.fill();
    return schedule;
}
