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

    fill() {
        const perm1 = this.getRotations([
            [0, 1, 2, 3],
            [4, 5, 6, 7],
            [8, 9, 10, 11],
            [12, 13, 14, 15],
        ]);
        this.fillDay(perm1[0].flat());
        for (const permutation of perm1) {
            const flattened = permutation.flat();
            this.fillDay(flattened);
        }

        const perm2 = this.getRotations(perm1[1]);
        for (const permutation of perm2) {
            this.fillDay(permutation.flat());
        }

        const perm3 = this.getRotations(chunk(this.highsVsLows(1), 4));
        for (const permutation of perm3) {
            this.fillDay(permutation.flat());
        }

        const perm4 = this.getRotations(chunk(this.highsVsLows(3), 4));
        for (const permutation of perm4) {
            this.fillDay(permutation.flat());
        }
    }
}

export function pinsetter(schedule: Schedule) {
    const algorithm = new PinsetterAlgorithm(schedule);
    algorithm.fill();
    return schedule;
}
