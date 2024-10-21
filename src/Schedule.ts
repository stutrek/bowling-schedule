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

    getMostCompatibleTeams(team: number, slot: number, lane: number) {
        const playsByTeam = new Array(this.config.teams).fill(0);
        const slotsByTeam = new Array(this.config.teams).fill(0);
        const lanesByTeam = new Array(this.config.teams).fill(0);
        this.schedule.forEach((game) => {
            if (game.teams[0] === team) {
                playsByTeam[game.teams[1]]++;
                slotsByTeam[game.teams[1]]++;
                lanesByTeam[game.teams[1]]++;
            }
            if (game.teams[1] === team) {
                playsByTeam[game.teams[0]]++;
                slotsByTeam[game.teams[0]]++;
                lanesByTeam[game.teams[0]]++;
            }
        });
        const sorted = this.getNewTeamList()
            .filter((t) => t !== team)
            .sort((a, b) => {
                if (playsByTeam[a] !== playsByTeam[b]) {
                    return playsByTeam[a] - playsByTeam[b];
                }
                if (slotsByTeam[a] !== slotsByTeam[b]) {
                    return slotsByTeam[a] - slotsByTeam[b];
                }
                return lanesByTeam[a] - lanesByTeam[b];
            });
        return sorted.map((team) => ({
            team,
            plays: playsByTeam[team],
            slots: slotsByTeam[team],
            lanes: lanesByTeam[team],
        }));
    }

    getLeastPlayedTeams(team: number) {
        const playsByTeam = new Array(this.config.teams).fill(0);
        this.schedule.forEach((game) => {
            if (game.teams[0] === team) {
                playsByTeam[game.teams[1]]++;
            }
            if (game.teams[1] === team) {
                playsByTeam[game.teams[0]]++;
            }
        });
        const sorted = this.getNewTeamList()
            .filter((t) => t !== team)
            .sort((a, b) => playsByTeam[a] - playsByTeam[b]);
        return sorted.map((team) => ({
            team,
            plays: playsByTeam[team],
        }));
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
