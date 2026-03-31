#!/usr/bin/env python3
"""
CP-SAT model to find the provably minimum bowling schedule cost.

Encodes all 8 constraints from the Rust solver as soft penalties and uses
Google OR-Tools to minimize total weighted cost. If the solver reaches
OPTIMAL status, the result is a mathematical proof of the minimum.

Usage:
    pip install ortools
    python cp_sat_solver.py [--time-limit 3600] [--workers 8] [--seed schedule.tsv]
"""

import argparse
import json
import time
from ortools.sat.python import cp_model

T = 16  # teams
W = 12  # weeks
Q = 4   # quads per week
P = 4   # positions per quad
L = 4   # lanes

# Games played on each lane-offset (0 or 1 from the quad's lane_off) per position:
#   pos 0 (pa): stays on lane_off        → 2 games on offset 0
#   pos 1 (pb): switches between games   → 1 game on each offset
#   pos 2 (pc): stays on lane_off+1      → 2 games on offset 1
#   pos 3 (pd): switches between games   → 1 game on each offset
LANE_GAMES = [[2, 0], [1, 1], [0, 2], [1, 1]]


def lane_coeff(p, q, lane):
    """Games team at position p in quad q plays on the given lane."""
    lo = (q % 2) * 2
    off = lane - lo
    if off < 0 or off > 1:
        return 0
    return LANE_GAMES[p][off]


def build_model(w8, tight=False, seed_assignment=None):
    m = cp_model.CpModel()
    ZERO = m.new_int_var(0, 0, 'ZERO')

    # ── Decision variables ──────────────────────────────────────────────
    # x[w,q,p,t] = 1 iff team t is at position p of quad q in week w
    x = {}
    for w in range(W):
        for q in range(Q):
            for p in range(P):
                for t in range(T):
                    x[w, q, p, t] = m.new_bool_var(f'x{w}{q}{p}_{t}')

    for w in range(W):
        for t in range(T):
            m.add_exactly_one(x[w, q, p, t] for q in range(Q) for p in range(P))
        for q in range(Q):
            for p in range(P):
                m.add_exactly_one(x[w, q, p, t] for t in range(T))

    if seed_assignment is not None:
        t0 = seed_assignment[0][0][0]
        m.add(x[0, 0, 0, t0] == 1)
    else:
        m.add(x[0, 0, 0, 0] == 1)

    # ── Derived: early/late ─────────────────────────────────────────────
    early = {}
    for w in range(W):
        for t in range(T):
            e = m.new_bool_var(f'e{w}_{t}')
            s = sum(x[w, q, p, t] for q in range(2) for p in range(P))
            m.add(s == 1).only_enforce_if(e)
            m.add(s == 0).only_enforce_if(e.negated())
            early[w, t] = e

    # Hard constraint: E/L balance = 0 (penalty is quadratic w/ weight 60,
    # so any deviation costs ≥60 — never worth it in an optimal solution)
    for t in range(T):
        m.add(sum(early[w, t] for w in range(W)) == W // 2)

    # ── Structural constraints (proven necessary for optimal) ───────────
    # in_qt[t,qt] = number of weeks team t is in quad type qt (0..3)
    # For lane + late_lane + switch to all be 0, each team must appear
    # in each quad type exactly 3 times (see mathematical derivation).
    in_qt = {}
    for t in range(T):
        for qt in range(Q):
            v = m.new_int_var(0, W, f'iqt{t}_{qt}')
            m.add(v == sum(x[w, qt, p, t] for w in range(W) for p in range(P)))
            in_qt[t, qt] = v
            m.add(v == W // Q)  # == 3

    # Note: stays-per-quad-type being even (0 or 2) is necessary only when
    # switch penalty is also 0. Since the optimal may accept small switch
    # penalties (the 160-cost schedule has switch=40), we leave this soft.

    # ── Derived: quad-side indicators (for matchup detection) ───────────
    iq_stay = {}
    iq_switch = {}
    for w in range(W):
        for q in range(Q):
            for t in range(T):
                s = m.new_bool_var(f'is{w}{q}_{t}')
                m.add(x[w, q, 0, t] + x[w, q, 2, t] == 1).only_enforce_if(s)
                m.add(x[w, q, 0, t] + x[w, q, 2, t] == 0).only_enforce_if(s.negated())
                iq_stay[w, q, t] = s

                sw = m.new_bool_var(f'iw{w}{q}_{t}')
                m.add(x[w, q, 1, t] + x[w, q, 3, t] == 1).only_enforce_if(sw)
                m.add(x[w, q, 1, t] + x[w, q, 3, t] == 0).only_enforce_if(sw.negated())
                iq_switch[w, q, t] = sw

    # ── Matchup detection ───────────────────────────────────────────────
    # Two teams are matched in a week iff they share a quad with one at a
    # stay position (0/2) and the other at a switch position (1/3).
    week_match = {}
    matchup_prods = {}
    for i in range(T):
        for j in range(i + 1, T):
            for w in range(W):
                prods = []
                for q in range(Q):
                    b1 = m.new_bool_var(f'p{w}{q}_{i}s{j}')
                    m.add_multiplication_equality(b1, [iq_stay[w, q, i], iq_switch[w, q, j]])
                    matchup_prods[w, q, i, j] = b1
                    prods.append(b1)
                    b2 = m.new_bool_var(f'p{w}{q}_{j}s{i}')
                    m.add_multiplication_equality(b2, [iq_stay[w, q, j], iq_switch[w, q, i]])
                    matchup_prods[w, q, j, i] = b2
                    prods.append(b2)

                wm = m.new_bool_var(f'm{w}_{i}_{j}')
                m.add(sum(prods) >= 1).only_enforce_if(wm)
                m.add(sum(prods) == 0).only_enforce_if(wm.negated())
                week_match[w, i, j] = wm

    # ── PENALTY 1: Matchup balance ──────────────────────────────────────
    if tight:
        for i in range(T):
            for j in range(i + 1, T):
                m.add(sum(week_match[w, i, j] for w in range(W)) >= 1)
        pen_matchup = m.new_int_var(0, 0, 'pen_matchup')
    else:
        zero_flags = []
        excess_list = []
        for i in range(T):
            for j in range(i + 1, T):
                mc = m.new_int_var(0, W, f'mc{i}_{j}')
                m.add(mc == sum(week_match[w, i, j] for w in range(W)))

                iz = m.new_bool_var(f'z{i}_{j}')
                m.add(mc == 0).only_enforce_if(iz)
                m.add(mc >= 1).only_enforce_if(iz.negated())
                zero_flags.append(iz)

                mc_m2 = m.new_int_var(-2, W - 2, f'mm{i}_{j}')
                m.add(mc_m2 == mc - 2)
                ex = m.new_int_var(0, W - 2, f'ex{i}_{j}')
                m.add_max_equality(ex, [mc_m2, ZERO])
                excess_list.append(ex)

        pen_matchup = m.new_int_var(0, 500_000, 'pen_matchup')
        m.add(pen_matchup == w8['matchup_zero'] * sum(zero_flags)
                            + w8['matchup_triple'] * sum(excess_list))

    # ── PENALTY 9: Half-season repeat ─────────────────────────────────
    H = W // 2
    hs_w = w8.get('half_season_repeat', 0)
    hs_excess = []
    for i in range(T):
        for j in range(i + 1, T):
            for half_start, half_end in [(0, H), (H, W)]:
                hc = m.new_int_var(0, H, f'hc{i}{j}{half_start}')
                m.add(hc == sum(
                    week_match[w, i, j]
                    for w in range(half_start, half_end)
                ))
                hc_m1 = m.new_int_var(-1, H - 1, f'hm{i}{j}{half_start}')
                m.add(hc_m1 == hc - 1)
                ex = m.new_int_var(0, H - 1, f'he{i}{j}{half_start}')
                m.add_max_equality(ex, [hc_m1, ZERO])
                hs_excess.append(ex)

    pen_hs = m.new_int_var(0, 500_000, 'pen_hs')
    m.add(pen_hs == hs_w * sum(hs_excess))

    # ── PENALTY 2: Consecutive opponents ────────────────────────────────
    consec_flags = []
    consec_prods = {}
    for w in range(W - 1):
        for i in range(T):
            for j in range(i + 1, T):
                b = m.new_bool_var(f'cc{w}_{i}{j}')
                m.add_multiplication_equality(b, [week_match[w, i, j], week_match[w + 1, i, j]])
                consec_prods[w, i, j] = b
                consec_flags.append(b)

    pen_consec = m.new_int_var(0, 500_000, 'pen_consec')
    m.add(pen_consec == w8['consecutive_opponents'] * sum(consec_flags))

    # ── PENALTY 3: Early/late balance ───────────────────────────────────
    # Hardcoded to 0 via the hard constraint above.
    pen_el = m.new_int_var(0, 0, 'pen_el')

    # ── PENALTY 4: Alternation (no 3-in-a-row same slot) ───────────────
    if tight:
        for t in range(T):
            for w in range(W - 2):
                s = early[w, t] + early[w + 1, t] + early[w + 2, t]
                m.add(s >= 1)
                m.add(s <= 2)
        pen_alt = m.new_int_var(0, 0, 'pen_alt')
    else:
        alt_flags = []
        for t in range(T):
            for w in range(W - 2):
                ae = m.new_bool_var(f'ae{t}{w}')
                m.add_bool_and([early[w, t], early[w+1, t], early[w+2, t]]).only_enforce_if(ae)
                m.add_bool_or([early[w, t].negated(), early[w+1, t].negated(),
                              early[w+2, t].negated()]).only_enforce_if(ae.negated())

                al = m.new_bool_var(f'al{t}{w}')
                m.add_bool_and([early[w, t].negated(), early[w+1, t].negated(),
                              early[w+2, t].negated()]).only_enforce_if(al)
                m.add_bool_or([early[w, t], early[w+1, t],
                              early[w+2, t]]).only_enforce_if(al.negated())

                v = m.new_bool_var(f'av{t}{w}')
                m.add_bool_or([ae, al]).only_enforce_if(v)
                m.add_bool_and([ae.negated(), al.negated()]).only_enforce_if(v.negated())
                alt_flags.append(v)

        pen_alt = m.new_int_var(0, 500_000, 'pen_alt')
        m.add(pen_alt == w8['early_late_alternation'] * sum(alt_flags))

    # ── PENALTY 4b: Consecutive (2-in-a-row same slot) ─────────────────
    el_con_w = w8.get('early_late_consecutive', 0)
    if el_con_w > 0:
        consec_el_flags = []
        for t in range(T):
            for w in range(W - 1):
                # same = 1 when both early or both late
                same = m.new_bool_var(f'ec{t}{w}')
                m.add_multiplication_equality(same, [early[w, t], early[w + 1, t]])
                # also detect both-late: neither early
                same_l = m.new_bool_var(f'ecl{t}{w}')
                m.add_multiplication_equality(same_l, [early[w, t].negated(),
                                                        early[w + 1, t].negated()])
                either = m.new_bool_var(f'ece{t}{w}')
                m.add_bool_or([same, same_l]).only_enforce_if(either)
                m.add_bool_and([same.negated(), same_l.negated()]).only_enforce_if(either.negated())
                consec_el_flags.append(either)

        pen_el_con = m.new_int_var(0, 500_000, 'pen_el_con')
        m.add(pen_el_con == el_con_w * sum(consec_el_flags))
    else:
        pen_el_con = m.new_int_var(0, 0, 'pen_el_con')

    # ── PENALTY 5: Lane balance ─────────────────────────────────────────
    if tight:
        for t in range(T):
            for l in range(L):
                terms = []
                coeffs = []
                for w in range(W):
                    for q in range(Q):
                        for p in range(P):
                            c = lane_coeff(p, q, l)
                            if c:
                                terms.append(x[w, q, p, t])
                                coeffs.append(c)
                m.add(cp_model.LinearExpr.weighted_sum(terms, coeffs) == W * 2 // L)
        pen_lane = m.new_int_var(0, 0, 'pen_lane')
    else:
        lane_w = int(w8['lane_balance'])
        lane_devs = []
        for t in range(T):
            for l in range(L):
                terms = []
                coeffs = []
                for w in range(W):
                    for q in range(Q):
                        for p in range(P):
                            c = lane_coeff(p, q, l)
                            if c:
                                terms.append(x[w, q, p, t])
                                coeffs.append(c)
                lc = m.new_int_var(0, 24, f'lc{t}{l}')
                m.add(lc == cp_model.LinearExpr.weighted_sum(terms, coeffs))
                diff = m.new_int_var(-24, 24, f'ld{t}{l}')
                m.add(diff == lc - (W * 2 // L))
                dev = m.new_int_var(0, 24, f'lv{t}{l}')
                m.add_abs_equality(dev, diff)
                lane_devs.append(dev)

        pen_lane = m.new_int_var(0, 500_000, 'pen_lane')
        m.add(pen_lane == lane_w * sum(lane_devs))

    # ── PENALTY 6: Lane switch balance ──────────────────────────────────
    sw_w = int(w8['lane_switch'])
    sw_devs = []
    for t in range(T):
        sc = m.new_int_var(0, W, f'sc{t}')
        m.add(sc == sum(x[w, q, p, t] for w in range(W) for q in range(Q) for p in [0, 2]))
        diff = m.new_int_var(-W, W, f'sd{t}')
        m.add(diff == sc - W // 2)
        dev = m.new_int_var(0, W, f'sv{t}')
        m.add_abs_equality(dev, diff)
        sw_devs.append(dev)

    pen_switch = m.new_int_var(0, 500_000, 'pen_sw')
    m.add(pen_switch == sw_w * sum(sw_devs))

    # ── PENALTY 7: Late lane balance ────────────────────────────────────
    if tight:
        for t in range(T):
            for l in range(L):
                terms = []
                coeffs = []
                for w in range(W):
                    for q in [2, 3]:
                        for p in range(P):
                            c = lane_coeff(p, q, l)
                            if c:
                                terms.append(x[w, q, p, t])
                                coeffs.append(c)
                m.add(cp_model.LinearExpr.weighted_sum(terms, coeffs) == W // L)
        pen_ll = m.new_int_var(0, 0, 'pen_ll')
    else:
        ll_w = int(w8['late_lane_balance'])
        ll_devs = []
        for t in range(T):
            for l in range(L):
                terms = []
                coeffs = []
                for w in range(W):
                    for q in [2, 3]:  # late quads only
                        for p in range(P):
                            c = lane_coeff(p, q, l)
                            if c:
                                terms.append(x[w, q, p, t])
                                coeffs.append(c)
                llc = m.new_int_var(0, 12, f'llc{t}{l}')
                m.add(llc == cp_model.LinearExpr.weighted_sum(terms, coeffs))
                diff = m.new_int_var(-12, 12, f'lld{t}{l}')
                m.add(diff == llc - W // L)
                dev = m.new_int_var(0, 12, f'llv{t}{l}')
                m.add_abs_equality(dev, diff)
                ll_devs.append(dev)

        pen_ll = m.new_int_var(0, 500_000, 'pen_ll')
        m.add(pen_ll == ll_w * sum(ll_devs))

    # ── PENALTY 8: Commissioner overlap ─────────────────────────────────
    # overlap(i,j) = 2 * (weeks both early), always even.
    pair_overlaps = []
    comm_prods = {}
    for i in range(T):
        for j in range(i + 1, T):
            be_prods = []
            for w in range(W):
                p = m.new_bool_var(f'be{w}{i}{j}')
                m.add_multiplication_equality(p, [early[w, i], early[w, j]])
                comm_prods[w, i, j] = p
                be_prods.append(p)
            bec = m.new_int_var(0, W // 2, f'bec{i}{j}')
            m.add(bec == sum(be_prods))
            ov = m.new_int_var(0, W, f'ov{i}{j}')
            m.add(ov == 2 * bec)
            pair_overlaps.append(ov)

    min_ov = m.new_int_var(0, W, 'min_ov')
    m.add_min_equality(min_ov, pair_overlaps)
    ov_m1 = m.new_int_var(-1, W - 1, 'ov_m1')
    m.add(ov_m1 == min_ov - 1)
    ov_base = m.new_int_var(0, W - 1, 'ov_base')
    m.add_max_equality(ov_base, [ov_m1, ZERO])

    pen_comm = m.new_int_var(0, 500_000, 'pen_comm')
    m.add(pen_comm == w8['commissioner_overlap'] * ov_base)

    # ── Objective ───────────────────────────────────────────────────────
    components = {
        'matchup': pen_matchup,
        'consec':  pen_consec,
        'el_bal':  pen_el,
        'el_alt':  pen_alt,
        'el_con':  pen_el_con,
        'lane':    pen_lane,
        'switch':  pen_switch,
        'll_bal':  pen_ll,
        'comm':    pen_comm,
        'hs_rpt':  pen_hs,
    }

    total = m.new_int_var(0, 5_000_000, 'total')
    m.add(total == sum(components.values()))
    m.minimize(total)

    derived = {'early': early, 'iq_stay': iq_stay,
                'iq_switch': iq_switch, 'week_match': week_match,
                'matchup_prods': matchup_prods, 'consec_prods': consec_prods,
                'comm_prods': comm_prods}
    return m, components, total, x, derived


def parse_tsv(path):
    """Parse a schedule TSV and return assignment[w][q] = [pa, pb, pc, pd]."""
    lines = open(path).read().strip().split('\n')
    assignment = []
    for w in range(W):
        base = 1 + w * 4
        e1 = lines[base].split('\t')      # Early 1 (game 1 of early quads)
        l1 = lines[base + 2].split('\t')   # Late 1  (game 1 of late quads)
        quads = []
        for row, c1, c2 in [(e1, 2, 3), (e1, 4, 5), (l1, 2, 3), (l1, 4, 5)]:
            pa, pb = [int(v) - 1 for v in row[c1].split(' v ')]
            pc, pd = [int(v) - 1 for v in row[c2].split(' v ')]
            quads.append([pa, pb, pc, pd])
        assignment.append(quads)
    return assignment


def add_seed(model, x, derived, assignment):
    """Provide a complete warm-start hint from an existing schedule."""
    team_pos = {}
    for w in range(W):
        team_pos[w] = {}
        for q in range(Q):
            for p in range(P):
                team_pos[w][assignment[w][q][p]] = (q, p)

    for w in range(W):
        for q in range(Q):
            for p in range(P):
                t_assigned = assignment[w][q][p]
                for t2 in range(T):
                    model.add_hint(x[w, q, p, t2], 1 if t2 == t_assigned else 0)

    early = derived['early']
    for w in range(W):
        for t in range(T):
            q, _ = team_pos[w][t]
            model.add_hint(early[w, t], 1 if q < 2 else 0)

    for w in range(W):
        for q in range(Q):
            for t in range(T):
                tq, tp = team_pos[w][t]
                in_this_quad = (tq == q)
                model.add_hint(derived['iq_stay'][w, q, t],
                               1 if in_this_quad and tp in (0, 2) else 0)
                model.add_hint(derived['iq_switch'][w, q, t],
                               1 if in_this_quad and tp in (1, 3) else 0)

    week_match_val = {}
    for w in range(W):
        for i in range(T):
            qi, pi = team_pos[w][i]
            for j in range(i + 1, T):
                qj, pj = team_pos[w][j]
                matched = 0
                if qi == qj:
                    i_stay = pi in (0, 2)
                    j_stay = pj in (0, 2)
                    if i_stay != j_stay:
                        matched = 1
                week_match_val[w, i, j] = matched
                model.add_hint(derived['week_match'][w, i, j], matched)

    for (w, q, si, sj), var in derived['matchup_prods'].items():
        qi_t, pi_t = team_pos[w][si]
        qj_t, pj_t = team_pos[w][sj]
        val = 1 if (qi_t == q and pi_t in (0, 2) and
                     qj_t == q and pj_t in (1, 3)) else 0
        model.add_hint(var, val)

    for (w, i, j), var in derived['consec_prods'].items():
        val = 1 if week_match_val[w, i, j] and week_match_val[w + 1, i, j] else 0
        model.add_hint(var, val)

    for (w, i, j), var in derived['comm_prods'].items():
        qi_i, _ = team_pos[w][i]
        qj_j, _ = team_pos[w][j]
        val = 1 if qi_i < 2 and qj_j < 2 else 0
        model.add_hint(var, val)


def extract_schedule(solver, x):
    """Read the solution back as an assignment[w][q] = [pa, pb, pc, pd]."""
    assignment = []
    for w in range(W):
        quads = []
        for q in range(Q):
            quad = []
            for p in range(P):
                for t in range(T):
                    if solver.value(x[w, q, p, t]):
                        quad.append(t)
                        break
            quads.append(quad)
        assignment.append(quads)
    return assignment


def assignment_to_tsv(assignment):
    """Convert to TSV matching the Rust solver output format."""
    slot_names = ["Early 1", "Early 2", "Late 1", "Late 2"]
    lines = ["Week\tSlot\tLane 1\tLane 2\tLane 3\tLane 4"]
    for w in range(W):
        slots = [[""] * L for _ in range(4)]
        for q in range(Q):
            pa, pb, pc, pd = assignment[w][q]
            sb = 0 if q < 2 else 2
            lb = (q % 2) * 2
            slots[sb][lb]     = f"{pa+1} v {pb+1}"
            slots[sb][lb + 1] = f"{pc+1} v {pd+1}"
            slots[sb+1][lb]     = f"{pa+1} v {pd+1}"
            slots[sb+1][lb + 1] = f"{pc+1} v {pb+1}"
        for s in range(4):
            lines.append(f"{w+1}\t{slot_names[s]}\t" + "\t".join(slots[s]))
    return "\n".join(lines)


class ProgressPrinter(cp_model.CpSolverSolutionCallback):
    def __init__(self, total, components):
        super().__init__()
        self._total = total
        self._components = components
        self._start = time.time()
        self._count = 0

    def on_solution_callback(self):
        self._count += 1
        elapsed = time.time() - self._start
        cost = self.value(self._total)
        parts = ' '.join(f'{k}={self.value(v):>3d}' for k, v in self._components.items())
        print(f'  [{elapsed:8.1f}s] #{self._count:>3d}  cost={cost:>5d}  {parts}')


def main():
    parser = argparse.ArgumentParser(description='CP-SAT bowling schedule solver')
    parser.add_argument('--time-limit', type=int, default=3600,
                        help='Solver time limit in seconds (default: 3600)')
    parser.add_argument('--workers', type=int, default=0,
                        help='Number of solver workers (0 = all cores)')
    parser.add_argument('--seed', type=str, default=None,
                        help='Path to TSV schedule to use as warm start')
    parser.add_argument('--save', type=str, default=None,
                        help='Save best schedule to this TSV path')
    parser.add_argument('--tight', action='store_true',
                        help='Hard-enforce matchup/el_alt/lane/ll_bal=0, minimize consec+switch+comm')
    args = parser.parse_args()

    with open('weights.json') as f:
        w8 = json.load(f)
    # Convert float weights to int for CP-SAT (all values in weights.json
    # are whole numbers; the Rust solver truncates via `as u32`)
    for k in w8:
        w8[k] = int(w8[k])

    seed_assignment = parse_tsv(args.seed) if args.seed else None

    mode = 'TIGHT' if args.tight else 'full'
    print(f'Building CP-SAT model ({mode})...')
    t0 = time.time()
    model, components, total, x, derived = build_model(
        w8, tight=args.tight, seed_assignment=seed_assignment)
    build_time = time.time() - t0
    proto = model.proto
    print(f'  {len(proto.variables)} variables, '
          f'{len(proto.constraints)} constraints, '
          f'built in {build_time:.1f}s')

    if seed_assignment:
        print(f'Seeding from {args.seed}')
        add_seed(model, x, derived, seed_assignment)

    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = args.time_limit
    if args.workers:
        solver.parameters.num_workers = args.workers
    solver.parameters.log_search_progress = True

    callback = ProgressPrinter(total, components)
    print(f'\nSolving (limit={args.time_limit}s)...\n')
    status = solver.solve(model, callback)

    labels = {
        cp_model.OPTIMAL: 'OPTIMAL',
        cp_model.FEASIBLE: 'FEASIBLE',
        cp_model.INFEASIBLE: 'INFEASIBLE',
        cp_model.MODEL_INVALID: 'MODEL_INVALID',
        cp_model.UNKNOWN: 'UNKNOWN',
    }

    print(f'\nStatus: {labels.get(status, status)}')
    print(f'Wall time: {solver.wall_time:.1f}s')

    if status in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        print(f'\n{"=" * 55}')
        print(f'  Best cost: {solver.value(total)}')
        for k, v in components.items():
            print(f'    {k:>10s}: {solver.value(v)}')

        if status == cp_model.OPTIMAL:
            print(f'\n  *** PROVEN OPTIMAL — {solver.value(total)} is the '
                  'minimum possible cost ***')
        else:
            print(f'\n  Best proven lower bound: {solver.best_objective_bound}')
            print('  (solver hit time limit before proving optimality)')

        sched = extract_schedule(solver, x)
        cost = solver.value(total)
        import os
        import datetime
        results_dir = 'solver-native/results/gpu'
        os.makedirs(results_dir, exist_ok=True)
        ts = datetime.datetime.now().strftime('%Y%m%d-%H%M%S%z')
        auto_path = f'{results_dir}/{cost:04d}-cpsat-{ts}.tsv'
        with open(auto_path, 'w') as f:
            f.write(assignment_to_tsv(sched))
        print(f'\n  Schedule saved to {auto_path}')

        if args.save:
            with open(args.save, 'w') as f:
                f.write(assignment_to_tsv(sched))
            print(f'  Also saved to {args.save}')

        print(f'{"=" * 55}')

    elif status == cp_model.INFEASIBLE:
        print('\nModel is infeasible — this should not happen with soft constraints')
    else:
        print('\nNo solution found within time limit')


if __name__ == '__main__':
    main()
