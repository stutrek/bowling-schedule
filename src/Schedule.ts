import type { Config, Game } from './types';

export class Schedule {
    config: Config;
    schedule: Game[] = [];

    constructor(config: Config) {
        this.config = config;
    }

    getNewTeamList(): number[] {
        return new Array(this.config.teams).fill(0).map((_, i) => i);
    }

    createSchedule(): void {
        for (let day = 0; day < this.config.days; day++) {
            for (
                let timeSlot = 0;
                timeSlot < this.config.timeSlots;
                timeSlot++
            ) {
                for (let lane = 0; lane < this.config.lanes; lane++) {
                    const game: Game = {
                        teams: [-1, -1],
                        timeSlot,
                        lane,
                        day,
                    };

                    this.schedule.push(game);
                }
            }
        }
    }

    setGame(
        team1: number,
        team2: number,
        day: number,
        timeSlot: number,
        lane: number,
    ): void {
        const game = this.schedule.find(
            (game) =>
                game.timeSlot === timeSlot &&
                game.lane === lane &&
                game.day === day,
        );
        if (team1 > team2) {
            [team1, team2] = [team2, team1];
        }
        if (game) {
            game.teams = [team1, team2];
        }
    }
}
