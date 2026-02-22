#!/usr/bin/env python3
"""Convert the real bowling schedule into the standard TSV format."""

TEAMS = {
    "The Milk Duds": 1,
    "Bowls on Parade": 2,
    "The Bowled and the Beautiful": 3,
    "WHO DO YOU THINK YOU ARE I AM": 4,
    "Glory Bowl": 5,
    "2 Legit 2 Split": 6,
    "Butt": 7,
    "Bite Legends": 8,
    "Deli Meats": 9,
    "Rambowls": 10,
    "Gutter Sluts": 11,
    "Tokyo Drifters": 12,
    "Blame It On The Lane": 13,
    "The Lane 5 Pole Dancers": 14,
    "Bowling": 15,
    "Pin Pals": 16,
}

def t(name):
    return TEAMS[name]

# Each week: [early1, early2, late1, late2]
# Each slot: [(lane1_a, lane1_b), (lane2_a, lane2_b), (lane3_a, lane3_b), (lane4_a, lane4_b)]
# Lanes map: Lane 5→1, Lane 6→2, Lane 7→3, Lane 8→4
# Slots map: 7:00→Early 1, 8:10→Early 2, 9:20→Late 1, 10:30→Late 2

WEEKS = [
    # Week 1 (21-Jan)
    [
        [("The Milk Duds","Bowls on Parade"),("The Bowled and the Beautiful","WHO DO YOU THINK YOU ARE I AM"),("Glory Bowl","2 Legit 2 Split"),("Butt","Bite Legends")],
        [("The Bowled and the Beautiful","Bite Legends"),("The Milk Duds","2 Legit 2 Split"),("Glory Bowl","Bowls on Parade"),("Butt","WHO DO YOU THINK YOU ARE I AM")],
        [("Deli Meats","Rambowls"),("Gutter Sluts","Tokyo Drifters"),("Blame It On The Lane","The Lane 5 Pole Dancers"),("Bowling","Pin Pals")],
        [("The Lane 5 Pole Dancers","Rambowls"),("Tokyo Drifters","Pin Pals"),("Gutter Sluts","Bowling"),("Deli Meats","Blame It On The Lane")],
    ],
    # Week 2 (28-Jan)
    [
        [("Gutter Sluts","Blame It On The Lane"),("Deli Meats","Bowling"),("The Lane 5 Pole Dancers","Tokyo Drifters"),("Rambowls","Pin Pals")],
        [("Deli Meats","Tokyo Drifters"),("The Lane 5 Pole Dancers","Bowling"),("Blame It On The Lane","Pin Pals"),("Gutter Sluts","Rambowls")],
        [("The Bowled and the Beautiful","2 Legit 2 Split"),("Glory Bowl","WHO DO YOU THINK YOU ARE I AM"),("The Milk Duds","Bite Legends"),("Butt","Bowls on Parade")],
        [("Butt","2 Legit 2 Split"),("The Milk Duds","WHO DO YOU THINK YOU ARE I AM"),("Glory Bowl","Bite Legends"),("The Bowled and the Beautiful","Bowls on Parade")],
    ],
    # Week 3 (4-Feb)
    [
        [("Deli Meats","Butt"),("Glory Bowl","Gutter Sluts"),("The Milk Duds","Pin Pals"),("The Bowled and the Beautiful","The Lane 5 Pole Dancers")],
        [("Deli Meats","The Bowled and the Beautiful"),("Gutter Sluts","The Milk Duds"),("The Lane 5 Pole Dancers","Butt"),("Glory Bowl","Pin Pals")],
        [("Rambowls","WHO DO YOU THINK YOU ARE I AM"),("Bowling","Bowls on Parade"),("2 Legit 2 Split","Tokyo Drifters"),("Blame It On The Lane","Bite Legends")],
        [("Blame It On The Lane","WHO DO YOU THINK YOU ARE I AM"),("Bowls on Parade","Tokyo Drifters"),("Bowling","2 Legit 2 Split"),("Rambowls","Bite Legends")],
    ],
    # Week 4 (11-Feb)
    [
        [("Blame It On The Lane","2 Legit 2 Split"),("Rambowls","Bowls on Parade"),("Bowling","WHO DO YOU THINK YOU ARE I AM"),("Bite Legends","Tokyo Drifters")],
        [("Rambowls","2 Legit 2 Split"),("Blame It On The Lane","Bowls on Parade"),("WHO DO YOU THINK YOU ARE I AM","Tokyo Drifters"),("Bowling","Bite Legends")],
        [("The Bowled and the Beautiful","Pin Pals"),("The Lane 5 Pole Dancers","The Milk Duds"),("Deli Meats","Glory Bowl"),("Gutter Sluts","Butt")],
        [("Butt","Pin Pals"),("Glory Bowl","The Lane 5 Pole Dancers"),("Deli Meats","The Milk Duds"),("Gutter Sluts","The Bowled and the Beautiful")],
    ],
    # Week 5 (18-Feb)
    [
        [("Glory Bowl","Bowling"),("Rambowls","Butt"),("The Bowled and the Beautiful","Blame It On The Lane"),("The Milk Duds","Tokyo Drifters")],
        [("The Milk Duds","Bowling"),("The Bowled and the Beautiful","Rambowls"),("Blame It On The Lane","Butt"),("Glory Bowl","Tokyo Drifters")],
        [("The Lane 5 Pole Dancers","Bite Legends"),("Gutter Sluts","Bowls on Parade"),("2 Legit 2 Split","Pin Pals"),("Deli Meats","WHO DO YOU THINK YOU ARE I AM")],
        [("Deli Meats","Bite Legends"),("The Lane 5 Pole Dancers","WHO DO YOU THINK YOU ARE I AM"),("Bowls on Parade","Pin Pals"),("Gutter Sluts","2 Legit 2 Split")],
    ],
    # Week 6 (25-Feb)
    [
        [("The Lane 5 Pole Dancers","2 Legit 2 Split"),("Bite Legends","Pin Pals"),("Gutter Sluts","WHO DO YOU THINK YOU ARE I AM"),("Deli Meats","Bowls on Parade")],
        [("Gutter Sluts","Bite Legends"),("WHO DO YOU THINK YOU ARE I AM","Pin Pals"),("The Lane 5 Pole Dancers","Bowls on Parade"),("Deli Meats","2 Legit 2 Split")],
        [("Glory Bowl","Rambowls"),("Butt","Bowling"),("The Bowled and the Beautiful","Tokyo Drifters"),("Blame It On The Lane","The Milk Duds")],
        [("Rambowls","The Milk Duds"),("Glory Bowl","Blame It On The Lane"),("Butt","Tokyo Drifters"),("The Bowled and the Beautiful","Bowling")],
    ],
    # Week 7 (4-Mar)
    [
        [("The Milk Duds","Bowls on Parade"),("Butt","Bite Legends"),("Gutter Sluts","Tokyo Drifters"),("Blame It On The Lane","The Lane 5 Pole Dancers")],
        [("The Lane 5 Pole Dancers","The Milk Duds"),("Gutter Sluts","Butt"),("Bite Legends","Tokyo Drifters"),("Blame It On The Lane","Bowls on Parade")],
        [("Bowling","Pin Pals"),("Glory Bowl","2 Legit 2 Split"),("Deli Meats","Rambowls"),("The Bowled and the Beautiful","WHO DO YOU THINK YOU ARE I AM")],
        [("The Bowled and the Beautiful","Pin Pals"),("Deli Meats","Glory Bowl"),("Rambowls","2 Legit 2 Split"),("Bowling","WHO DO YOU THINK YOU ARE I AM")],
    ],
    # Week 8 (11-Mar)
    [
        [("Rambowls","WHO DO YOU THINK YOU ARE I AM"),("Glory Bowl","Pin Pals"),("Deli Meats","The Bowled and the Beautiful"),("Bowling","2 Legit 2 Split")],
        [("Glory Bowl","WHO DO YOU THINK YOU ARE I AM"),("The Bowled and the Beautiful","2 Legit 2 Split"),("Deli Meats","Bowling"),("Rambowls","Pin Pals")],
        [("The Lane 5 Pole Dancers","Butt"),("Gutter Sluts","The Milk Duds"),("Blame It On The Lane","Bite Legends"),("Bowls on Parade","Tokyo Drifters")],
        [("Butt","Bowls on Parade"),("Gutter Sluts","Blame It On The Lane"),("The Milk Duds","Bite Legends"),("The Lane 5 Pole Dancers","Tokyo Drifters")],
    ],
    # Week 9 (18-Mar)
    [
        [("Blame It On The Lane","Bowling"),("Rambowls","Tokyo Drifters"),("Glory Bowl","Butt"),("The Bowled and the Beautiful","The Milk Duds")],
        [("The Milk Duds","Bowling"),("Rambowls","Butt"),("The Bowled and the Beautiful","Blame It On The Lane"),("Glory Bowl","Tokyo Drifters")],
        [("WHO DO YOU THINK YOU ARE I AM","Bowls on Parade"),("Bite Legends","2 Legit 2 Split"),("Deli Meats","Gutter Sluts"),("The Lane 5 Pole Dancers","Pin Pals")],
        [("Bowls on Parade","Pin Pals"),("Deli Meats","Bite Legends"),("Gutter Sluts","2 Legit 2 Split"),("The Lane 5 Pole Dancers","WHO DO YOU THINK YOU ARE I AM")],
    ],
    # Week 10 (25-Mar)
    [
        [("Gutter Sluts","Pin Pals"),("WHO DO YOU THINK YOU ARE I AM","Bite Legends"),("Bowls on Parade","2 Legit 2 Split"),("Deli Meats","The Lane 5 Pole Dancers")],
        [("Gutter Sluts","WHO DO YOU THINK YOU ARE I AM"),("Deli Meats","Bowls on Parade"),("Bite Legends","Pin Pals"),("The Lane 5 Pole Dancers","2 Legit 2 Split")],
        [("Glory Bowl","The Milk Duds"),("The Bowled and the Beautiful","Butt"),("Blame It On The Lane","Rambowls"),("Bowling","Tokyo Drifters")],
        [("The Bowled and the Beautiful","Tokyo Drifters"),("Glory Bowl","Blame It On The Lane"),("Butt","Bowling"),("Rambowls","The Milk Duds")],
    ],
    # Week 11 (1-Apr)
    [
        [("Bite Legends","Bowls on Parade"),("Rambowls","Bowling"),("Glory Bowl","The Bowled and the Beautiful"),("Gutter Sluts","The Lane 5 Pole Dancers")],
        [("Gutter Sluts","Bite Legends"),("The Bowled and the Beautiful","Rambowls"),("The Lane 5 Pole Dancers","Bowls on Parade"),("Glory Bowl","Bowling")],
        [("Deli Meats","Pin Pals"),("Blame It On The Lane","Tokyo Drifters"),("The Milk Duds","Butt"),("WHO DO YOU THINK YOU ARE I AM","2 Legit 2 Split")],
        [("Butt","Tokyo Drifters"),("Deli Meats","2 Legit 2 Split"),("Blame It On The Lane","The Milk Duds"),("WHO DO YOU THINK YOU ARE I AM","Pin Pals")],
    ],
    # Week 12 (8-Apr)
    [
        [("WHO DO YOU THINK YOU ARE I AM","2 Legit 2 Split"),("Deli Meats","Pin Pals"),("Blame It On The Lane","Butt"),("The Milk Duds","Tokyo Drifters")],
        [("Blame It On The Lane","Tokyo Drifters"),("2 Legit 2 Split","Pin Pals"),("Deli Meats","WHO DO YOU THINK YOU ARE I AM"),("The Milk Duds","Butt")],
        [("Glory Bowl","The Bowled and the Beautiful"),("Rambowls","Bowling"),("Gutter Sluts","The Lane 5 Pole Dancers"),("Bite Legends","Bowls on Parade")],
        [("Bowling","Bite Legends"),("Gutter Sluts","Bowls on Parade"),("The Bowled and the Beautiful","The Lane 5 Pole Dancers"),("Glory Bowl","Rambowls")],
    ],
]

SLOT_NAMES = ["Early 1", "Early 2", "Late 1", "Late 2"]

lines = ["Week\tSlot\tLane 1\tLane 2\tLane 3\tLane 4"]
for week_idx, week in enumerate(WEEKS):
    week_num = week_idx + 1
    for slot_idx, slot in enumerate(week):
        lanes = []
        for a, b in slot:
            lanes.append(f"{t(a)} v {t(b)}")
        lines.append(f"{week_num}\t{SLOT_NAMES[slot_idx]}\t" + "\t".join(lanes))

output = "\n".join(lines) + "\n"
print(output, end="")
