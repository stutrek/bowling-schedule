import { Schedule } from './Schedule';
import { solveSchedule } from './solver';
import type { Config } from './types';

const config: Config = {
    teams: 16,
    timeSlots: 4,
    lanes: 4,
    days: 12,
};

const schedule = new Schedule(config);
schedule.createSchedule();

console.time('solve');
const cost = solveSchedule(schedule, {
    maxIterations: 5_000_000,
    runs: 3,
    verbose: true,
});
console.timeEnd('solve');

console.log('\nConstraint scores:');
console.table(cost);
