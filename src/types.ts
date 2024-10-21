export type Config = {
    teams: number;
    lanes: number;
    timeSlots: number;
    days: number;
};

export type Game = {
    teams: [number, number];
    timeSlot: number;
    lane: number;
    day: number;
};
