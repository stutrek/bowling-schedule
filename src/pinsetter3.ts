import { get, groupBy } from 'lodash';
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
    constructor(schedule: Schedule) {
        this.schedule = schedule;
    }

    fillGroup(teams: number[], day: number, slot: number, lane: number) {
        this.schedule.setGame(teams[0], teams[1], day, slot, lane);
        this.schedule.setGame(teams[2], teams[3], day, slot, lane + 1);
        this.schedule.setGame(teams[0], teams[3], day, slot + 1, lane);
        this.schedule.setGame(teams[2], teams[1], day, slot + 1, lane + 1);
    }

    fillDay(teams: number[], swapSlots = false) {
        const day = this.daysFilled++;
        let groups = chunk(teams, 4);
        if (swapSlots) {
            groups.reverse();
            groups.forEach((group) => group.reverse());
        }

        if (day % 3 === 0) {
            groups = groups.map((group, i) => {
                const [a, b, c, d] = group;
                return [c, d, a, b];
            });
        }

        groups.forEach((group, i) => {
            const slot = i > 1 ? 2 : 0;
            const lane = (i % (this.schedule.config.lanes / 2)) * 2;
            this.fillGroup(group, day, slot, lane);
        });
    }

    getRotations(arr: number[][]) {
        const rotations = new Array(2)
            .fill(0)
            .map(() =>
                new Array(arr.length)
                    .fill(0)
                    .map(() => new Array(arr.length).fill(0)),
            );
        for (let i = 0; i < 2; i++) {
            for (let j = 0; j < arr.length; j++) {
                for (let k = 0; k < arr.length; k++) {
                    rotations[i][j][k] = arr[k][(k * i + j) % arr.length];
                }
            }
        }
        rotations.unshift(arr);
        return rotations;
    }

    highsVsLows(shiftAmount: number) {
        const teams = this.schedule.getNewTeamList();
        const chunks = chunk(teams, teams.length / 2);
        for (let i = 0; i < shiftAmount; i++) {
            shiftByTwo(chunks[1]);
        }
        return zip(...chunks).flat() as number[];
    }

    reverseSecondSet(arr: number[]) {
        return chunk(arr, 2).flatMap((chunk, i) =>
            i % 2 ? chunk.reverse() : chunk,
        );
    }

    interlaceHighAndLow(arr: number[]) {
        const chunks = chunk(arr, 4);
        const modified = chunks.map((chunk, i) => {
            const nextOne = i === chunks.length - 1 ? chunks[0] : chunks[i + 1];
            const [a, b, c, d] = chunk;
            return [a, nextOne[0], b, nextOne[1]];
        });
        return modified.flat();
    }

    rotateIndividualGroups(arr: number[]) {
        const chunks = chunk(arr, 4);
        const modified = chunks.map((chunk, i) => {
            // const nextOne = i === chunks.length - 1 ? chunks[0] : chunks[i + 1];
            const [a, b, c, d] = chunk;
            return [a, c, b, d];
        });
        return modified.flat();
    }

    rotateSecondTeams(arr: number[], backwards = false) {
        const chunks = chunk(arr, 2);
        const odds = chunks.filter((_, i) => i % 2);
        const evens = chunks.filter((_, i) => !(i % 2));
        if (backwards) {
            odds.unshift(odds.pop() as number[]);
        } else {
            odds.push(odds.shift() as number[]);
        }

        const modified = zip(evens, odds).flat(2) as number[];
        return modified;
    }
    rotateFirstTeams(arr: number[], backwards = false) {
        const chunks = chunk(arr, 2);
        const odds = chunks.filter((_, i) => i % 2);
        const evens = chunks.filter((_, i) => !(i % 2));
        if (backwards) {
            evens.unshift(evens.pop() as number[]);
        } else {
            evens.push(evens.shift() as number[]);
        }

        const modified = zip(evens, odds).flat(2) as number[];
        return modified;
    }

    swapEvensAndOdds(arr: number[]) {
        const chunks = chunk(arr, 2);
        chunks.push(chunks.shift() as number[]);
        return chunks.flat();
    }

    fill() {
        let teams: number[];
        // this.fillDay(this.schedule.getNewTeamList());
        // teams = this.rotateSecondTeams(
        //     this.rotateFirstTeams(this.schedule.getNewTeamList()),
        // );
        // teams = this.rotateIndividualGroups(teams);
        // this.fillDay(teams);

        // [0, 4], [1, 5], [0, 5], [1, 4]
        // const interlaced = this.interlaceHighAndLow(
        //     this.schedule.getNewTeamList(),
        // );
        // this.fillDay(interlaced);

        // [2,4], [3,5]
        teams = this.rotateSecondTeams(this.schedule.getNewTeamList());
        teams = this.rotateIndividualGroups(teams);
        this.fillDay(teams);

        teams = this.rotateFirstTeams(this.schedule.getNewTeamList());
        teams = this.rotateIndividualGroups(teams);
        this.fillDay(teams);

        // 2,6,3,7
        teams = this.rotateFirstTeams(
            this.rotateSecondTeams(this.schedule.getNewTeamList(), true),
        );
        teams = this.rotateIndividualGroups(teams);
        this.fillDay(teams);

        teams = this.rotateFirstTeams(
            this.rotateSecondTeams(this.schedule.getNewTeamList(), true),
            true,
        );
        teams = this.rotateIndividualGroups(teams);
        this.fillDay(teams);

        // 0,4,1,5

        teams = this.interlaceHighAndLow(
            this.swapEvensAndOdds(this.schedule.getNewTeamList()),
        );
        this.fillDay(teams);
        teams = this.interlaceHighAndLow(
            this.swapEvensAndOdds(this.schedule.getNewTeamList()),
        );
        this.fillDay(teams);
    }
}

export function pinsetter(schedule: Schedule) {
    const algorithm = new PinsetterAlgorithm(schedule);
    algorithm.fill();
    return schedule;
}
