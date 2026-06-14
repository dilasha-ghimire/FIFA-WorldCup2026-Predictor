"""
simulate.py
────────────────────────────────────────────────────────────
Monte Carlo tournament simulator for WC 2026.

Instead of just predicting who wins each match, this simulates
the entire tournament 100,000 times. Each simulation rolls dice
according to the Elo match probabilities, advances 32 teams
through the knockout bracket, and crowns a winner.
After 100,000 runs, we count how often each team won — that
percentage is their probability of winning the World Cup.

Run:    python simulate.py
Output: predictions/wc2026_simulation.json
"""

import json
import random
from collections import defaultdict
from data import load_all
from train_model import build_h2h, predict_match

# ─────────────────────────────────────────────────────────────
# CONSTANTS
# ─────────────────────────────────────────────────────────────

# 100,000 is the standard for Monte Carlo simulations in research papers,
# giving probability estimates accurate to within +/- 0.1%.
N_SIMULATIONS = 100000


# ─────────────────────────────────────────────────────────────
# STEP 1 - Simulate a single match using probabilities as dice
# ─────────────────────────────────────────────────────────────

def simulate_match(team1, team2, rankings_2026, h2h, is_knockout=False):
    """
    Simulates one match by rolling a random number against
    the Elo win/draw/loss probabilities from predict_match().

    Group stage: draws are possible — returns team1, team2, or 'draw'.
    Knockout stage: no draws allowed — a draw triggers a 50/50
    penalty shootout coin flip between the two teams.
    """
    p1w, pdraw, p2w = predict_match(team1, team2, rankings_2026, h2h)

    roll = random.random()

    if roll < p1w:
        result = team1
    elif roll < p1w + pdraw:
        result = "draw"
    else:
        result = team2

    # In knockout rounds there must be a winner — simulate penalty shootout
    if is_knockout and result == "draw":
        result = random.choice([team1, team2])

    return result


# ─────────────────────────────────────────────────────────────
# STEP 2 - Simulate the full group stage
# ─────────────────────────────────────────────────────────────

def simulate_group_stage(predictions, rankings_2026, h2h):
    """
    Simulates all 72 group stage matches and returns the 32 teams
    that qualify for the Round of 32.

    WC 2026 format:
      - 12 groups of 4 teams, each team plays 3 matches
      - Top 2 from each group qualify (24 teams)
      - Best 8 third-place finishers also qualify (8 teams)
      - Total: 32 teams advance

    Points: Win = 3, Draw = 1, Loss = 0.
    Tiebreaker: goal difference, simulated with random scorelines.
    """
    groups = defaultdict(list)
    for p in predictions:
        groups[p["group"]].append(p)

    standings = defaultdict(lambda: defaultdict(lambda: {"pts": 0, "gd": 0}))

    for group, fixtures in groups.items():
        for fixture in fixtures:
            t1, t2 = fixture["team1"], fixture["team2"]
            result = simulate_match(t1, t2, rankings_2026, h2h)

            if result == t1:
                gf, ga = random.randint(1, 3), random.randint(0, 1)
                standings[group][t1]["pts"] += 3
                standings[group][t1]["gd"]  += (gf - ga)
                standings[group][t2]["gd"]  -= (gf - ga)
            elif result == t2:
                gf, ga = random.randint(1, 3), random.randint(0, 1)
                standings[group][t2]["pts"] += 3
                standings[group][t2]["gd"]  += (gf - ga)
                standings[group][t1]["gd"]  -= (gf - ga)
            else:
                # Draw: both teams get 1 point, goal difference unchanged
                standings[group][t1]["pts"] += 1
                standings[group][t2]["pts"] += 1

    qualifiers  = []
    third_place = []

    for group in sorted(standings.keys()):
        table = sorted(
            standings[group].items(),
            key=lambda x: (x[1]["pts"], x[1]["gd"]),
            reverse=True
        )
        qualifiers.append(table[0][0])  # 1st place
        qualifiers.append(table[1][0])  # 2nd place
        third_place.append((table[2][0], table[2][1]["pts"], table[2][1]["gd"]))

    # Best 8 third-place finishers also advance to the Round of 32
    third_place_sorted = sorted(third_place, key=lambda x: (x[1], x[2]), reverse=True)
    qualifiers += [t[0] for t in third_place_sorted[:8]]

    return qualifiers  # 32 teams


# ─────────────────────────────────────────────────────────────
# STEP 3 - Simulate the knockout rounds
# ─────────────────────────────────────────────────────────────

def simulate_knockout(teams, rankings_2026, h2h):
    """
    Simulates the full knockout bracket from Round of 32 to the Final.

    WC 2026 knockout path:
      Round of 32 (32 teams) -> Round of 16 -> Quarter-finals
      -> Semi-finals -> Final

    Bracket pairings are randomised since the actual R32 draw depends
    on group stage finishing positions which vary each simulation.
    Returns the tournament winner.
    """
    remaining = list(teams)
    random.shuffle(remaining)

    while len(remaining) > 1:
        next_round = []
        for i in range(0, len(remaining), 2):
            if i + 1 < len(remaining):
                # Knockout match — no draws allowed
                winner = simulate_match(
                    remaining[i], remaining[i + 1],
                    rankings_2026, h2h,
                    is_knockout=True
                )
                next_round.append(winner)
            else:
                next_round.append(remaining[i])
        remaining = next_round

    return remaining[0]


# ─────────────────────────────────────────────────────────────
# STEP 4 - Run N_SIMULATIONS full tournaments
# ─────────────────────────────────────────────────────────────

def run_simulation(predictions, rankings_2026, h2h, n=N_SIMULATIONS):
    """
    Runs the full tournament simulation n times and returns
    each team's win count and probability, sorted by probability.
    """
    win_counts = defaultdict(int)

    for i in range(n):
        if (i + 1) % 10000 == 0:
            print(f"  Simulating... {i + 1:,}/{n:,}")

        qualifiers = simulate_group_stage(predictions, rankings_2026, h2h)
        winner = simulate_knockout(qualifiers, rankings_2026, h2h)
        win_counts[winner] += 1

    return sorted(
        [
            {
                "team":        team,
                "wins":        count,
                "probability": round(count / n * 100, 2)
            }
            for team, count in win_counts.items()
        ],
        key=lambda x: x["probability"],
        reverse=True
    )


# ─────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("\nWC 2026 Monte Carlo Simulator\n")

    matches, rankings, rankings_2026, schedule = load_all()
    h2h = build_h2h(matches)

    with open("predictions/wc2026_predictions.json") as f:
        predictions = json.load(f)

    print(f"Running {N_SIMULATIONS:,} simulations...\n")
    random.seed(42)  # seed for reproducibility
    results = run_simulation(predictions, rankings_2026, h2h, N_SIMULATIONS)

    with open("predictions/wc2026_simulation.json", "w") as f:
        json.dump({"n_simulations": N_SIMULATIONS, "results": results}, f, indent=2)
    print(f"\nSaved -> predictions/wc2026_simulation.json\n")

    print(f"WC 2026 Win Probabilities ({N_SIMULATIONS:,} simulations):\n")
    print(f"  {'Rank':<6} {'Team':<25} {'Probability'}")
    print("  " + "-" * 42)
    for i, r in enumerate(results[:16], 1):
        print(f"  {i:<6} {r['team']:<25} {r['probability']:>6.2f}%")
    print(f"\n  ... and {len(results) - 16} more teams\n")
