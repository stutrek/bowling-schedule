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
        console.log('day', day, teams);
        this.schedule.setGame(teams[0], teams[2], day, slot, lane);
        this.schedule.setGame(teams[1], teams[3], day, slot, lane + 1);
        this.schedule.setGame(teams[0], teams[3], day, slot + 1, lane);
        this.schedule.setGame(teams[1], teams[2], day, slot + 1, lane + 1);
    }

    fillDay(teams: number[], swapSlots = false) {
        const day = this.daysFilled++;
        const groups = chunk(teams, 4);

        groups.forEach((group, i) => {
            const slot = i > 1 ? 2 : 0;
            const lane = (i % (this.schedule.config.lanes / 2)) * 2;
            this.fillGroup(group, day, slot, lane);
        });
    }

    rotate(teams: number[]) {
        const groups = chunk(teams, 2);
        groups.push(groups.shift() as number[]);
        const fours = chunk(groups.flat(), 4);
        const rotated = fours.map(([a, b, c, d]) => [a, c, b, d]);
        return rotated.flat();
    }

    rotateFirstSet(teams: number[]) {
        const groups = chunk(teams, 2);
        // const end = groups.pop();
        groups.push(groups.shift() as number[]);
        // groups.push(end as number[]);
        // const fours = chunk(groups.flat(), 4);
        // const rotated = fours.map(([a, b, c, d]) => [a, c, b, d]);
        return groups.flat();
    }

    rotateSecondSet(teams: number[]) {
        const groups = chunk(teams, 2);
        const end = groups.pop();
        groups.push(groups.shift() as number[]);
        groups.push(end as number[]);
        // const fours = chunk(groups.flat(), 4);
        // const rotated = fours.map(([a, b, c, d]) => [a, c, b, d]);
        return groups.flat();
    }

    fill() {
        let teams: number[];
        teams = this.schedule.getNewTeamList();
        this.fillDay(teams);
        teams = this.rotate(teams);
        this.fillDay(teams);
        teams = this.rotateFirstSet(teams);
        this.fillDay(teams);
        teams = this.rotate(teams);
        this.fillDay(teams);
    }
}

export function pinsetter(schedule: Schedule) {
    const algorithm = new PinsetterAlgorithm(schedule);
    algorithm.fill();
    console.log(schedule.schedule);
    return schedule;
}
