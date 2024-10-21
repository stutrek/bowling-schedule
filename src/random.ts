import shuffle from 'lodash/shuffle';
import chunk from 'lodash/chunk';

import type { Schedule } from './Schedule';

export function fillRandomly(schedule: Schedule) {
    const teams = new Array(schedule.config.teams).fill(0).map((_, i) => i);
    for (let i = 0; i < schedule.config.days; i++) {
        const totalGames = schedule.config.timeSlots * schedule.config.lanes;
        const gamesPerTeam = totalGames / (schedule.config.teams / 2);

        const shuffledTeams = shuffle(teams);
        const batches = chunk(shuffledTeams, teams.length / gamesPerTeam);
        for (let j = 0; j < batches.length; j++) {
            const batch = batches[j];
            for (let k = 0; k < gamesPerTeam; k++) {
                const shuffledBatch = shuffle(batch);
                for (let l = 0; l < shuffledBatch.length; l += 2) {
                    const team1 = shuffledBatch[l];
                    const team2 = shuffledBatch[l + 1];
                    schedule.setGame(team1, team2, i, k + j * 2, l / 2);
                }
            }
        }
    }
}
