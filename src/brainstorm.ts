import type { Config } from './types';
// const config: Config = {
//     teams: 16,
//     lanes: 4,
//     days: 12,
//     // gamesPerTeam: 2,
// };

function sortByCompatibility(team: number) {
    // sort by...
    // can be in this time slot
    // fewest times played this team
    // fewest times played next to this team
}

function selectGroup(foundingTeam: number, timeSlot: number) {
    // sort by...
    // foundingTeam (A)
    // can be in this time slot
    // fewest times played A
    // fewest times played next to A
    // take two (B, C)
    // get sort by compatibility for those two teams
    // find the team that is lowest in both (D)
    // return...
    // [
    //     [[A, B], [D, C]]
    //     [[A, C], [D, B]]
    // ]
}

export function calculate(config: Config) {
    const gamesPerTimeslot = config.teams / 2;
    const timeSlotsPerDay = Math.ceil(gamesPerTimeslot / config.lanes);

    // select two teams
    // for both
    // put the first one the first lane
    // select the two most appropriate opponents
    // select the most approp
}
