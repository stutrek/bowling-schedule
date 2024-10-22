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

    highsVsLows(shiftAmount: number) {
        const teams = this.schedule.getNewTeamList();
        const chunks = chunk(teams, teams.length / 2);
        for (let i = 0; i < shiftAmount; i++) {
            shiftByTwo(chunks[1]);
        }
        return zip(...chunks).flat() as number[];
    }

    fill() {
        this.fillDay(this.highsVsLows(0), true);
        this.fillDay(this.highsVsLows(1));
        this.fillDay(this.highsVsLows(2), true);
        this.fillDay(this.highsVsLows(3));

        this.fillDay(
            [
                // hi
                0, 4, 1, 5, 2, 6, 3, 7, 8, 12, 9, 13, 10, 14, 11, 15,
            ],
            true,
        );
        this.fillDay([
            // hi
            0, 6, 1, 7, 2, 4, 3, 5, 8, 14, 9, 15, 10, 12, 11, 13,
        ]);

        this.fillDay(this.schedule.getNewTeamList(), true);
        this.fillDay([0, 1, 3, 2, 5, 4, 6, 7, 8, 9, 11, 10, 12, 13, 15, 14]);

        // fill remaining four weeks, after all teams have played each other once.
        this.fillDay(this.highsVsLows(3), true);
        this.fillDay(this.highsVsLows(2));
        this.fillDay(this.highsVsLows(1), true);
        this.fillDay(this.highsVsLows(0));
    }
}

export function pinsetter(schedule: Schedule) {
    const algorithm = new PinsetterAlgorithm(schedule);
    algorithm.fill();
    return schedule;
}
