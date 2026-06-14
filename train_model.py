"""
train_model.py
────────────────────────────────────────────────────────────
Generates match predictions for all 72 WC 2026 group stage matches.

Method: Elo-based win probability — the same formula used by FIFA
and major bookmakers — calibrated with:
  - June 2026 FIFA ranking points
  - Historical WC head-to-head records (1930-2022)
  - Host nation advantage (+80 ranking points)
  - Draw rate calibrated by ranking gap

Run:    python train_model.py
Output: outputs/wc2026_predictions.json
"""

import json
from collections import defaultdict
from data import load_all, HOST_NATIONS

# ─────────────────────────────────────────────────────────────
# TUNING CONSTANTS
# ─────────────────────────────────────────────────────────────

# Elo scale factor — FIFA uses 600, we use 400 (standard chess Elo)
# Lower value = ranking gap matters more
ELO_SCALE      = 400

# Base draw probability in WC group stage (historically ~24%)
DRAW_BASE      = 0.24

# How much the ranking gap reduces draw probability.
# A 200pt gap reduces draw chance by ~200/8000 = 2.5%
DRAW_GAP_SCALE = 8000

# Minimum draw probability even in very mismatched games
DRAW_MIN       = 0.10

# Extra ranking points given to host nations (home crowd advantage)
HOST_BOOST     = 80

# How much head-to-head history shifts the base probabilities.
# 0.08 = 8% weight on h2h, 92% on Elo — only kicks in with 2+ past meetings
H2H_WEIGHT     = 0.08


# ─────────────────────────────────────────────────────────────
# STEP 1 - Build head-to-head records from historical matches
# ─────────────────────────────────────────────────────────────

def build_h2h(matches):
    """
    Reads every historical WC group stage match and builds a
    nested dict of head-to-head records between each pair of teams.

    Returns: h2h[team1][team2] = {"wins": N, "draws": N, "losses": N}
    """
    h2h = defaultdict(lambda: defaultdict(lambda: {"wins": 0, "draws": 0, "losses": 0}))

    for _, row in matches.iterrows():
        home  = row["home_team"]
        away  = row["away_team"]
        hs    = row["home_score"]
        as_   = row["away_score"]

        if hs > as_:
            h2h[home][away]["wins"]   += 1
            h2h[away][home]["losses"] += 1
        elif hs == as_:
            h2h[home][away]["draws"]  += 1
            h2h[away][home]["draws"]  += 1
        else:
            h2h[home][away]["losses"] += 1
            h2h[away][home]["wins"]   += 1

    return h2h


# ─────────────────────────────────────────────────────────────
# STEP 2 - Core Elo prediction formula
# ─────────────────────────────────────────────────────────────

def elo_win_prob(pts1, pts2):
    """
    Standard Elo formula: returns probability that team1 beats team2.
    P = 1 / (1 + 10^(-(pts1 - pts2) / scale))
    """
    delta = (pts1 - pts2) / ELO_SCALE
    return 1.0 / (1.0 + 10.0 ** (-delta))


def predict_match(team1, team2, rankings_2026, h2h):
    """
    Predicts the outcome of a single match.
    Returns (p_team1_win, p_draw, p_team2_win) as floats that sum to 1.

    Steps:
      1. Get FIFA ranking points for each team
      2. Apply host nation boost if applicable
      3. Compute base Elo win probability
      4. Calibrate draw rate based on how mismatched the teams are
      5. Adjust slightly using head-to-head WC history
    """
    # Step 1: get ranking points (fallback to 1300 for unranked teams)
    r1 = rankings_2026.get(team1, 1300)
    r2 = rankings_2026.get(team2, 1300)

    # Step 2: host nation gets a small boost — playing at home matters
    if team1 in HOST_NATIONS:
        r1 += HOST_BOOST
    if team2 in HOST_NATIONS:
        r2 += HOST_BOOST

    # Step 3: base Elo probability — who is the stronger team?
    raw_p1 = elo_win_prob(r1, r2)
    raw_p2 = 1.0 - raw_p1

    # Step 4: calibrate draw rate.
    # Evenly matched teams draw more often; mismatches rarely end in draws.
    rank_gap  = abs(r1 - r2)
    draw_prob = max(DRAW_MIN, DRAW_BASE - (rank_gap / DRAW_GAP_SCALE))

    # Scale win probs so everything sums to 1
    win_scale = 1.0 - draw_prob
    p1_win    = raw_p1 * win_scale
    p2_win    = raw_p2 * win_scale

    # Step 5: blend in head-to-head history if enough meetings exist
    h = h2h[team1][team2]
    h_total = h["wins"] + h["draws"] + h["losses"]
    if h_total >= 2:
        h2h_win_rate  = h["wins"]  / h_total
        h2h_draw_rate = h["draws"] / h_total
        p1_win    = (1 - H2H_WEIGHT) * p1_win  + H2H_WEIGHT * (h2h_win_rate * win_scale)
        draw_prob = (1 - H2H_WEIGHT) * draw_prob + H2H_WEIGHT * h2h_draw_rate
        p2_win    = 1.0 - p1_win - draw_prob

    return p1_win, draw_prob, p2_win


# ─────────────────────────────────────────────────────────────
# STEP 3 - Generate predictions for all 72 fixtures
# ─────────────────────────────────────────────────────────────

def generate_predictions(schedule, rankings_2026, h2h):
    """
    Loops through every WC 2026 group stage fixture, runs predict_match,
    and returns a list of prediction dicts ready to be saved as JSON.

    Each dict contains:
      - match info (id, group, teams, date)
      - model prediction (winner, outcome, probabilities)
      - placeholder fields for actual results (filled later by fetch_results.py)
    """
    predictions = []

    for _, fixture in schedule.iterrows():
        t1 = fixture["team1"]
        t2 = fixture["team2"]

        p1w, pdraw, p2w = predict_match(t1, t2, rankings_2026, h2h)

        # Determine predicted outcome based on highest probability
        if p1w >= p2w and p1w >= pdraw:
            predicted_winner  = t1
            predicted_outcome = "team1_win"
        elif p2w >= p1w and p2w >= pdraw:
            predicted_winner  = t2
            predicted_outcome = "team2_win"
        else:
            predicted_winner  = "Draw"
            predicted_outcome = "draw"

        predictions.append({
            # Match info
            "match_id":           fixture["match_id"],
            "group":              fixture["group"],
            "date":               fixture["date"],
            "team1":              t1,
            "team2":              t2,
            "team1_fifa_pts":     round(rankings_2026.get(t1, 1300), 1),
            "team2_fifa_pts":     round(rankings_2026.get(t2, 1300), 1),
            # Model prediction
            "predicted_winner":   predicted_winner,
            "predicted_outcome":  predicted_outcome,
            "prob_team1_win":     round(p1w   * 100, 1),
            "prob_draw":          round(pdraw  * 100, 1),
            "prob_team2_win":     round(p2w   * 100, 1),
            # Actual result fields — filled by fetch_results.py on Day 2
            "actual_score_team1": None,
            "actual_score_team2": None,
            "actual_outcome":     None,
            "correct":            None,
            "status":             "upcoming",
        })

    return predictions


# ─────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("\nWC 2026 Match Predictor - Generating Predictions\n")

    # Load all datasets
    matches, rankings, rankings_2026, schedule = load_all()

    # Build head-to-head records from 688 historical matches
    print("Building head-to-head records...")
    h2h = build_h2h(matches)
    print(f"  Done - {len(matches)} matches processed\n")

    # Generate predictions for all 72 fixtures
    print("Generating predictions...")
    predictions = generate_predictions(schedule, rankings_2026, h2h)

    # Summary
    from collections import Counter
    outcomes = Counter(p["predicted_outcome"] for p in predictions)
    print(f"  {outcomes['team1_win']} team1 wins | {outcomes['draw']} draws | {outcomes['team2_win']} team2 wins\n")

    # Save to outputs/
    import os
    os.makedirs("predictions", exist_ok=True)
    output_path = "predictions/wc2026_predictions.json"
    with open(output_path, "w") as f:
        json.dump(predictions, f, indent=2)
    print(f"Saved {len(predictions)} predictions -> {output_path}\n")

    # Print all predictions grouped by group
    print("All predictions:")
    print(f"  {'ID':<5} {'Teams':<38} {'Prediction':<22} {'W%  / D%  / L%'}")
    print("  " + "-" * 78)
    current_group = None
    for p in predictions:
        if p["group"] != current_group:
            current_group = p["group"]
            print(f"\n  -- Group {current_group} --")
        teams = f"{p['team1']} vs {p['team2']}"
        pred  = p["predicted_winner"]
        probs = f"{p['prob_team1_win']}% / {p['prob_draw']}% / {p['prob_team2_win']}%"
        print(f"  {p['match_id']:<5} {teams:<38} {pred:<22} {probs}")
