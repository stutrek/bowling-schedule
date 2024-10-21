import { fillRandomly } from './random';
import { Schedule } from './Schedule';
import type { Config } from './types';

const config: Config = {
    teams: 16,
    timeSlots: 4,
    lanes: 4,
    days: 12,
};

const schedule = new Schedule(config);
schedule.createSchedule();

fillRandomly(schedule);

console.table(schedule.schedule);
