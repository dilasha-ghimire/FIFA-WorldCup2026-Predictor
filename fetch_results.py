"""
fetch_results.py
────────────────────────────────────────────────────────────
Fetches live WC 2026 match results from the openfootball API
and compares them against our pre-generated predictions.

Data source: github.com/openfootball/worldcup.json
Updated:     Once daily by the maintainer (within 24hrs of match end)
Auth:        None required

Run:    python fetch_results.py
Output: predictions/wc2026_predictions.json (updated in place)
        outputs/accuracy_report.txt
"""

import json
import requests
from datetime import datetime, timezone

# ─────────────────────────────────────────────────────────────
# CONSTANTS
# ─────────────────────────────────────────────────────────────

# Raw GitHub URL — always returns the latest version of the file
API_URL = "https://raw.githubusercontent.com/openfootball/worldcup.json/master/2026/worldcup.json"

PREDICTIONS_PATH = "predictions/wc2026_predictions.json"
OUTPUT_PATH      = "outputs/accuracy_report.txt"

# The openfootball API uses slightly different team names to ours.
# This map converts API names to our standardized names.
API_NAME_MAP = {
    "USA":                    "United States",
    "Bosnia & Herzegovina":   "Bosnia and Herzegovina",
    "DR Congo":               "Congo DR",
    "Curaçao":                "Curacao",
}

def normalize_api_name(name):
    return API_NAME_MAP.get(name, name)


# ─────────────────────────────────────────────────────────────
# STEP 1 - Fetch live results from the API
# ─────────────────────────────────────────────────────────────

def fetch_results():
    """
    Fetches the worldcup.json file from GitHub.
    Returns a dict mapping (team1, team2) -> result dict for
    every completed group stage match.

    A match is considered completed if it has a "score" field
    with a "ft" (full time) key containing [home_score, away_score].
    """
    print("Fetching results from openfootball API...")

    try:
        response = requests.get(API_URL, timeout=10)
        response.raise_for_status()
        data = response.json()
    except requests.exceptions.RequestException as e:
        print(f"  ERROR: Could not fetch results — {e}")
        return {}

    # Filter to group stage matches only that have a final score
    results = {}
    completed = 0
    upcoming  = 0

    for match in data.get("matches", []):
        # Skip knockout stage matches
        if "Group" not in match.get("group", ""):
            continue

        t1 = normalize_api_name(match.get("team1", ""))
        t2 = normalize_api_name(match.get("team2", ""))

        score = match.get("score", {})
        ft    = score.get("ft")

        if ft and len(ft) == 2:
            # Match is completed — store the result
            home_score = ft[0]
            away_score = ft[1]

            if home_score > away_score:
                outcome = "team1_win"
            elif home_score == away_score:
                outcome = "draw"
            else:
                outcome = "team2_win"

            results[(t1, t2)] = {
                "home_score": home_score,
                "away_score": away_score,
                "outcome":    outcome,
            }
            completed += 1
        else:
            upcoming += 1

    print(f"  Completed matches found: {completed}")
    print(f"  Upcoming matches:        {upcoming}")
    return results


# ─────────────────────────────────────────────────────────────
# STEP 2 - Compare results to predictions
# ─────────────────────────────────────────────────────────────

def update_predictions(predictions, results):
    """
    Loops through every prediction and checks if a real result
    exists for that match. If it does:
      - Updates actual_score_team1, actual_score_team2
      - Sets actual_outcome
      - Sets correct = True/False
      - Sets status = "completed"

    Matches with no result yet stay as status = "upcoming".
    Returns the updated predictions list.
    """
    for pred in predictions:
        t1 = pred["team1"]
        t2 = pred["team2"]

        result = results.get((t1, t2))

        if result:
            pred["actual_score_team1"] = result["home_score"]
            pred["actual_score_team2"] = result["away_score"]
            pred["actual_outcome"]     = result["outcome"]
            pred["correct"]            = (pred["predicted_outcome"] == result["outcome"])
            pred["status"]             = "completed"
        else:
            # Keep existing values — don't overwrite if already set
            if pred["status"] != "completed":
                pred["status"] = "upcoming"

    return predictions


# ─────────────────────────────────────────────────────────────
# STEP 3 - Calculate accuracy
# ─────────────────────────────────────────────────────────────

def calculate_accuracy(predictions):
    """
    Calculates three accuracy metrics across all completed matches:

    1. Exact accuracy — predicted outcome matches actual outcome exactly
       (win/draw/loss all three must match)

    2. No upset accuracy — did the predicted stronger team avoid losing?
       A draw when we predicted a win still counts as "not wrong" here,
       since the stronger team wasn't beaten. Only counts as wrong when
       the team we predicted to win actually lost.

    3. Genuinely wrong — the wrong team won entirely (not just a draw)
    """
    completed = [p for p in predictions if p["status"] == "completed"]
    upcoming  = [p for p in predictions if p["status"] == "upcoming"]

    exact_correct   = []
    no_upset        = []
    genuinely_wrong = []

    for p in completed:
        predicted = p["predicted_outcome"]
        actual    = p["actual_outcome"]

        # Metric 1: exact match
        if predicted == actual:
            exact_correct.append(p)

        # Metric 2 & 3: was the predicted winner beaten?
        # A draw when we predicted team1_win or team2_win = not an upset
        # Only genuinely wrong if the OTHER team won
        if predicted == "team1_win":
            if actual == "team2_win":
                genuinely_wrong.append(p)
            else:
                no_upset.append(p)  # win or draw — team1 didn't lose
        elif predicted == "team2_win":
            if actual == "team1_win":
                genuinely_wrong.append(p)
            else:
                no_upset.append(p)  # win or draw — team2 didn't lose
        elif predicted == "draw":
            if actual == "draw":
                no_upset.append(p)  # predicted draw, got draw
            else:
                genuinely_wrong.append(p)  # predicted draw, one team won

    n = len(completed)
    exact_pct    = round(len(exact_correct)   / n * 100, 1) if n else 0.0
    no_upset_pct = round(len(no_upset)        / n * 100, 1) if n else 0.0
    wrong_pct    = round(len(genuinely_wrong) / n * 100, 1) if n else 0.0

    return {
        "total_matches":       len(predictions),
        "completed":           n,
        "upcoming":            len(upcoming),
        "exact_correct":       len(exact_correct),
        "exact_accuracy":      exact_pct,
        "no_upset":            len(no_upset),
        "no_upset_accuracy":   no_upset_pct,
        "genuinely_wrong":     len(genuinely_wrong),
        "genuinely_wrong_pct": wrong_pct,
        "last_updated":        datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"),
    }



# ─────────────────────────────────────────────────────────────
# STEP 4 - Save and report
# ─────────────────────────────────────────────────────────────

def save_predictions(predictions, path=PREDICTIONS_PATH):
    with open(path, "w") as f:
        json.dump(predictions, f, indent=2)
    print(f"  Saved updated predictions -> {path}")


def print_report(predictions, accuracy):
    """
    Prints a clean accuracy report and saves it to outputs/accuracy_report.txt.
    """
    lines = []
    lines.append("WC 2026 Prediction Accuracy Report")
    lines.append(f"Generated: {accuracy['last_updated']}")
    lines.append("-" * 55)
    lines.append(f"Total matches:         {accuracy['total_matches']}")
    lines.append(f"Completed:             {accuracy['completed']}")
    lines.append(f"Upcoming:              {accuracy['upcoming']}")
    lines.append("-" * 55)
    lines.append(f"Exact accuracy:        {accuracy['exact_correct']}/{accuracy['completed']} = {accuracy['exact_accuracy']}%")
    lines.append(f"  (predicted outcome matched exactly)")
    lines.append(f"Predicted team held:   {accuracy['no_upset']}/{accuracy['completed']} = {accuracy['no_upset_accuracy']}%")
    lines.append(f"  (our team won or drew — was not beaten)")
    lines.append(f"Genuinely wrong:       {accuracy['genuinely_wrong']}/{accuracy['completed']} = {accuracy['genuinely_wrong_pct']}%")
    lines.append(f"  (the other team won entirely)")
    lines.append("-" * 55)

    # Completed matches breakdown
    completed = [p for p in predictions if p["status"] == "completed"]
    if completed:
        lines.append("")
        lines.append("Completed matches:")
        lines.append(f"  {'Match':<6} {'Teams':<35} {'Predicted':<15} {'Actual':<10} {'Result'}")
        lines.append("  " + "-" * 75)
        for p in completed:
            teams     = f"{p['team1']} vs {p['team2']}"
            predicted = p["predicted_winner"]
            actual    = f"{p['actual_score_team1']}-{p['actual_score_team2']}"
            result    = "CORRECT" if p["correct"] else "WRONG"
            lines.append(f"  {p['match_id']:<6} {teams:<35} {predicted:<15} {actual:<10} {result}")

    # Upcoming matches
    upcoming = [p for p in predictions if p["status"] == "upcoming"]
    if upcoming:
        lines.append("")
        lines.append(f"Upcoming matches ({len(upcoming)} remaining):")
        for p in upcoming[:5]:
            lines.append(f"  {p['match_id']:<6} {p['team1']} vs {p['team2']} ({p['date']})")
        if len(upcoming) > 5:
            lines.append(f"  ... and {len(upcoming) - 5} more")

    report = "\n".join(lines)
    print("\n" + report)

    # Save to file
    import os
    os.makedirs("outputs", exist_ok=True)
    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        f.write(report)
    print(f"\n  Saved report -> {OUTPUT_PATH}")


# ─────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("\nWC 2026 Result Fetcher\n")

    # Load predictions
    with open(PREDICTIONS_PATH) as f:
        predictions = json.load(f)
    print(f"Loaded {len(predictions)} predictions\n")

    # Fetch live results
    results = fetch_results()

    # Update predictions with real results
    print("\nComparing results to predictions...")
    predictions = update_predictions(predictions, results)

    # Calculate accuracy
    accuracy = calculate_accuracy(predictions)

    # Save updated predictions
    print("\nSaving...")
    save_predictions(predictions)

    # Print and save report
    print_report(predictions, accuracy)

    print("\nDone. Run again after each matchday to update results.\n")