#!/usr/bin/env node
/**
 * Converts the summer-dump.tsv (wide spreadsheet paste with team names)
 * into the standard summer schedule TSV format (Week/Slot/Lane columns with "N v M").
 *
 * Usage: node convert-summer-dump.js <input.tsv> <output.tsv>
 */

const fs = require('node:fs');

const inputPath = process.argv[2] || 'summer-dump.tsv';
const outputPath = process.argv[3] || 'real-summer-schedule.tsv';

const raw = fs.readFileSync(inputPath, 'utf-8');
const rows = raw.split('\n').map((line) => line.split('\t'));

// Discover team names from the data rows (rows with "Lane N" cells)
const teamSet = new Set();
for (const row of rows) {
    for (let c = 0; c < row.length; c++) {
        if (/^Lane\s+\d+$/.test(row[c].trim())) {
            const a = row[c + 1]?.trim();
            const b = row[c + 3]?.trim();
            if (a) teamSet.add(a);
            if (b) teamSet.add(b);
        }
    }
}

const teamNames = [...teamSet].sort();
if (teamNames.length !== 12) {
    console.error(`Expected 12 teams, found ${teamNames.length}:`, teamNames);
    process.exit(1);
}

const teamIndex = {};
teamNames.forEach((name, i) => {
    teamIndex[name] = i + 1;
});

console.error('Team mapping:');
teamNames.forEach((name, i) => {
    console.error(`  ${i + 1}: ${name}`);
});

// Each week occupies 6 columns: [lane_label, teamA, empty, teamB, empty, gap]
// 10 weeks -> columns 0-59
const COLS_PER_WEEK = 6;

// The rows are structured as:
// Row 0: date headers
// Row 1: time (7:00)
// Rows 2-5: Lane 5-8 for slot 1
// Row 6: blank
// Row 7: time (7:45)
// Rows 8-11: Lane 5-8 for slot 2
// Row 12: blank
// Row 13: time (8:30)
// Rows 14-17: Lane 5-8 for slot 3
// Row 18: blank
// Row 19: time (9:15)
// Rows 20-23: Lane 5-8 for slot 4
// Row 24: blank
// Row 25: time (10:00)
// Rows 26-27: Lane 7-8 for slot 5

// Slot definitions: [startRow, laneCount]
const slotDefs = [
    { startRow: 2, lanes: 4 }, // slot 1: rows 2-5
    { startRow: 8, lanes: 4 }, // slot 2: rows 8-11
    { startRow: 14, lanes: 4 }, // slot 3: rows 14-17
    { startRow: 20, lanes: 4 }, // slot 4: rows 20-23
    { startRow: 26, lanes: 2 }, // slot 5: rows 26-27 (only lanes 7-8)
];

const output = ['Week\tSlot\tLane 1\tLane 2\tLane 3\tLane 4'];

for (let week = 0; week < 10; week++) {
    const colBase = week * COLS_PER_WEEK;

    for (let slotIdx = 0; slotIdx < 5; slotIdx++) {
        const def = slotDefs[slotIdx];
        const laneCells = ['-', '-', '-', '-']; // 4 lane positions

        for (let laneOffset = 0; laneOffset < def.lanes; laneOffset++) {
            const row = rows[def.startRow + laneOffset];
            if (!row) continue;

            // Determine which lane this is from the lane label
            const laneLabel = row[colBase]?.trim();
            const laneMatch = laneLabel?.match(/Lane\s+(\d+)/);
            if (!laneMatch) continue;

            const physicalLane = Number.parseInt(laneMatch[1]); // 5, 6, 7, or 8
            const laneIdx = physicalLane - 5; // map to 0-3

            const teamA = row[colBase + 1]?.trim();
            const teamB = row[colBase + 3]?.trim();

            if (!teamA || !teamB) continue;

            const idxA = teamIndex[teamA];
            const idxB = teamIndex[teamB];

            if (!idxA || !idxB) {
                console.error(
                    `Unknown team: "${teamA}" or "${teamB}" at week ${week + 1}, slot ${slotIdx + 1}, lane ${physicalLane}`,
                );
                continue;
            }

            laneCells[laneIdx] = `${idxA} v ${idxB}`;
        }

        output.push(`${week + 1}\t${slotIdx + 1}\t${laneCells.join('\t')}`);
    }
}

fs.writeFileSync(outputPath, `${output.join('\n')}\n`);
console.error(`\nWrote ${outputPath}`);
