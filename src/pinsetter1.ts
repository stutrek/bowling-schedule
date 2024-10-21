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
    reverse = false;
    constructor(schedule: Schedule) {
        this.schedule = schedule;
    }

    fillGroup(teams: number[], day: number, slot: number, lane: number) {
        this.schedule.setGame(teams[0], teams[1], day, slot, lane);
        this.schedule.setGame(teams[2], teams[3], day, slot, lane + 1);
        this.schedule.setGame(teams[0], teams[3], day, slot + 1, lane);
        this.schedule.setGame(teams[2], teams[1], day, slot + 1, lane + 1);
    }

    fillDay(teams: number[]) {
        if (this.reverse) {
            teams.reverse();
        }
        const day = this.daysFilled++;
        const groups = chunk(teams, 4);
        groups.forEach((group, i) => {
            const slot = i > 1 ? 2 : 0;
            const lane = (i % (this.schedule.config.lanes / 2)) * 2;
            this.fillGroup(group, day, slot, lane);
        });
    }

    highsVsLows(shiftAmount: number) {
        const teams = this.schedule.getNewTeamList();
        const chunks = chunk(teams, teams.length / 2);
        for (let i = 0; i < shiftAmount; i++) {
            shiftByTwo(chunks[1]);
        }
        this.fillDay(zip(...chunks).flat() as number[]);
    }

    fill() {
        this.highsVsLows(0);
        this.highsVsLows(1);
        this.highsVsLows(2);
        this.highsVsLows(3);

        this.fillDay([
            // hi
            0, 4, 1, 5, 2, 6, 3, 7, 8, 12, 9, 13, 10, 14, 11, 15,
        ]);
        this.fillDay([
            // hi
            0, 6, 1, 7, 2, 4, 3, 5, 8, 14, 9, 15, 10, 12, 11, 13,
        ]);

        this.fillDay(this.schedule.getNewTeamList());
        this.fillDay([0, 1, 3, 2, 5, 4, 6, 7, 8, 9, 11, 10, 12, 13, 15, 14]);

        // teams = this.schedule.getNewTeamList().sort((a, b) => {
        //     if (a % 2 === 0 && b % 2 === 1) {
        //         return -1;
        //     }
        //     if (a % 2 === 1 && b % 2 === 0) {
        //         return 1;
        //     }
        //     return a - b;
        // });
        // this.fillDay(teams);

        // const chunks = chunk(this.schedule.getNewTeamList(), teams.length / 2);
        // chunks[0].reverse();
        // this.fillDay(zip(...chunks).flat());

        // teams = sortBy(this.schedule.getNewTeamList(), (team) => team % 2);
        // this.fillDay(teams);

        // teams = sortBy(this.schedule.getNewTeamList(), (team) => team % 3);
        // this.fillDay(teams);
    }
}

export function pinsetter(schedule: Schedule) {
    const algorithm = new PinsetterAlgorithm(schedule);
    algorithm.fill();
    return schedule;
}
