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

    fillDay(
        teams: number[],
        { swapSlots, swapLanes, swapGroups } = {
            swapGroups: false,
            swapLanes: false,
            swapSlots: false,
        },
    ) {
        const day = this.daysFilled++;

        let groups = chunk(teams, 4);
        if (swapSlots) {
            groups.reverse();
        }

        if (swapLanes) {
            // groups.forEach((group) => group.reverse());
            groups = groups.map((group, i) => {
                const [a, b, c, d] = group;
                return [c, d, a, b];
            });
        }

        if (swapGroups) {
            groups = [groups[1], groups[0], groups[3], groups[2]];
        }

        groups.forEach((group, i) => {
            const slot = i > 1 ? 2 : 0;
            const lane = (i % (this.schedule.config.lanes / 2)) * 2;
            this.fillGroup(group, day, slot, lane);
        });
    }

    rotateMovers(arr: number[]) {
        return [
            [arr[0], arr[5], arr[2], arr[7]],
            [arr[4], arr[9], arr[6], arr[11]],
            [arr[8], arr[13], arr[10], arr[15]],
            [arr[12], arr[1], arr[14], arr[3]],
        ].flat();
    }

    /*
        return [
            [arr[0], arr[1], arr[2], arr[3]],
            [arr[4], arr[5], arr[6], arr[7]],
            [arr[8], arr[9], arr[10], arr[11]],
            [arr[12], arr[13], arr[14], arr[15]],
        ].flat();
	*/

    remapOddsAndEvens(arr: number[]) {
        return [
            [arr[0], arr[2], arr[4], arr[6]],
            [arr[8], arr[10], arr[12], arr[14]],
            [arr[1], arr[3], arr[5], arr[7]],
            [arr[9], arr[11], arr[13], arr[15]],
        ].flat();
    }

    remap3(arr: number[]) {
        return [
            [arr[0], arr[8], arr[1], arr[9]],
            [arr[2], arr[10], arr[3], arr[11]],
            [arr[4], arr[12], arr[5], arr[13]],
            [arr[6], arr[14], arr[7], arr[15]],
        ].flat();
    }

    remap4(arr: number[]) {
        return [
            [arr[0], arr[10], arr[12], arr[2]],
            [arr[4], arr[14], arr[8], arr[6]],
            [arr[11], arr[1], arr[9], arr[3]],
            [arr[15], arr[5], arr[13], arr[7]],
        ].flat();
    }

    remap5(arr: number[]) {
        return [
            [arr[0], arr[14], arr[4], arr[10]],
            [arr[1], arr[15], arr[5], arr[11]],
            [arr[2], arr[8], arr[6], arr[12]],
            [arr[3], arr[13], arr[7], arr[9]],
        ].flat();
    }

    remap6(arr: number[]) {
        return [
            [arr[0], arr[1], arr[2], arr[3]],
            [arr[4], arr[5], arr[6], arr[7]],
            [arr[8], arr[9], arr[10], arr[11]],
            [arr[12], arr[13], arr[14], arr[15]],
        ].flat();
    }

    remap7(arr: number[]) {
        return [
            [arr[0], arr[4], arr[12], arr[6]],
            [arr[5], arr[1], arr[7], arr[3]],
            [arr[8], arr[5], arr[11], arr[12]],
            [arr[2], arr[13], arr[9], arr[15]],
        ].flat();
    }

    fill() {
        let teams = this.schedule.getNewTeamList();
        this.fillDay(teams);

        teams = this.rotateMovers(teams);
        this.fillDay(teams);
        teams = this.rotateMovers(teams);
        this.fillDay(teams);
        teams = this.rotateMovers(teams);
        this.fillDay(teams);

        teams = this.schedule.getNewTeamList();
        teams = this.remapOddsAndEvens(teams);
        this.fillDay(teams);
        teams = this.remapOddsAndEvens(teams);
        this.fillDay(teams);

        teams = this.schedule.getNewTeamList();
        teams = this.remap3(teams);
        this.fillDay(teams);

        teams = this.schedule.getNewTeamList();
        teams = this.remap4(teams);
        this.fillDay(teams);

        teams = this.remap5(this.schedule.getNewTeamList());
        this.fillDay(teams);

        // teams = this.rotateMovers(teams);
        // this.fillDay(teams);

        teams = this.schedule.getNewTeamList();
        this.fillDay(teams);
        teams = this.rotateMovers(teams);
        this.fillDay(teams);
        teams = this.schedule.getNewTeamList();

        this.fillDay(this.remap7(this.schedule.getNewTeamList()));
    }
}

export function pinsetter(schedule: Schedule) {
    const algorithm = new PinsetterAlgorithm(schedule);
    algorithm.fill();
    return schedule;
}
