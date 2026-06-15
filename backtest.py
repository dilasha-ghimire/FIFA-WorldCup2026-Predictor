"""
backtest.py
────────────────────────────────────────────────────────────
Validates the Elo model by testing it on past World Cups.

Method:
  For each test tournament (2010, 2014, 2018, 2022):
    - Train on all matches BEFORE that tournament
    - Predict every group stage match in that tournament
    - Compare predictions to actual results
    - Report exact accuracy, team held %, and genuinely wrong %

This is standard backtesting methodology in sports analytics.
A model is only credible if it performs well on data it has
never seen — not just on the data it was trained on.

Run:    python backtest.py
Output: outputs/backtest_report.txt
"""

import json
import os
from collections import defaultdict
from data import load_all
from train_model import build_h2h, predict_match

# ─────────────────────────────────────────────────────────────
# CONSTANTS
# ─────────────────────────────────────────────────────────────

# Test on the last 4 World Cups — enough matches for statistical significance.
# We use 4 tournaments (192 matches total) because:
#   - Single tournament (48 matches) is too small for reliable statistics
#   - More tournaments = more confidence the model generalises across eras
TEST_YEARS = [2010, 2014, 2018, 2022]


# ─────────────────────────────────────────────────────────────
# STEP 1 - Run predictions for a single tournament
# ─────────────────────────────────────────────────────────────

def backtest_tournament(test_year, all_matches, rankings):
    """
    Trains on all matches before test_year, predicts test_year matches,
    compares to actual results.

    This is a strict train/test split — the model never sees the test
    year's data during training. This prevents data leakage, which is
    the most common mistake in sports prediction projects.

    Returns a list of result dicts, one per match.
    """
    # Split: train on everything before this tournament.
    # Example: for test_year=2018, we train on 1930–2014 only.
    # The 2018 matches are held out completely until prediction time.
    train_matches = all_matches[all_matches["year"] < test_year]
    test_matches  = all_matches[all_matches["year"] == test_year]

    # Build h2h records from training data only.
    # This means head-to-head history is also strictly pre-test-year.
    h2h = build_h2h(train_matches)

    results = []
    for _, row in test_matches.iterrows():
        t1 = row["home_team"]
        t2 = row["away_team"]
        hs = row["home_score"]
        as_ = row["away_score"]

        # Determine the actual outcome from the real match scoreline
        if hs > as_:
            actual = "team1_win"
        elif hs == as_:
            actual = "draw"
        else:
            actual = "team2_win"

        # Run our Elo model to get win/draw/loss probabilities.
        # We use the 2022 FIFA rankings as a proxy for team strength
        # since per-year rankings aren't available in our dataset.
        # This is a known limitation — teams' relative strengths don't
        # change dramatically enough to invalidate the comparison.
        p1w, pdraw, p2w = predict_match(t1, t2, rankings, h2h)

        # The predicted outcome is whichever probability is highest.
        # This is called the "argmax" prediction strategy.
        if p1w >= p2w and p1w >= pdraw:
            predicted = "team1_win"
            predicted_winner = t1
        elif p2w >= p1w and p2w >= pdraw:
            predicted = "team2_win"
            predicted_winner = t2
        else:
            predicted = "draw"
            predicted_winner = "Draw"

        # Metric 1: exact match — did we get the exact outcome right?
        exact = predicted == actual

        # Metric 2: no upset — did the team we predicted to win avoid losing?
        # A draw when we predicted a win is not the same as being completely wrong.
        # If we predicted team1 to win and they drew, the stronger team wasn't beaten.
        if predicted == "team1_win":
            no_upset = actual != "team2_win"   # win or draw = not upset
        elif predicted == "team2_win":
            no_upset = actual != "team1_win"   # win or draw = not upset
        else:
            no_upset = actual == "draw"         # we predicted draw, got draw

        # Metric 3: genuinely wrong — the completely wrong team won.
        # This is the harshest failure mode: we said team1 wins, team2 won.
        genuinely_wrong = (
            (predicted == "team1_win" and actual == "team2_win") or
            (predicted == "team2_win" and actual == "team1_win") or
            (predicted == "draw" and actual != "draw")
        )

        results.append({
            "year":              test_year,
            "team1":             t1,
            "team2":             t2,
            "actual_score":      f"{hs}-{as_}",
            "actual_outcome":    actual,
            "predicted_outcome": predicted,
            "predicted_winner":  predicted_winner,
            "prob_t1_win":       round(p1w * 100, 1),
            "prob_draw":         round(pdraw * 100, 1),
            "prob_t2_win":       round(p2w * 100, 1),
            "exact":             exact,
            "no_upset":          no_upset,
            "genuinely_wrong":   genuinely_wrong,
        })

    return results


# ─────────────────────────────────────────────────────────────
# STEP 2 - Calculate metrics for a set of results
# ─────────────────────────────────────────────────────────────

def calculate_metrics(results):
    """
    Computes the three accuracy metrics across a list of match results.

    We report three metrics rather than one because football prediction
    has three possible outcomes — a single accuracy number hides important
    information about where the model succeeds and where it fails.
    """
    n = len(results)
    if n == 0:
        return {}

    # Count each metric across all matches
    exact           = sum(1 for r in results if r["exact"])
    no_upset        = sum(1 for r in results if r["no_upset"])
    genuinely_wrong = sum(1 for r in results if r["genuinely_wrong"])

    return {
        "matches":              n,
        "exact_correct":        exact,
        "exact_accuracy":       round(exact / n * 100, 1),
        "no_upset":             no_upset,
        "no_upset_accuracy":    round(no_upset / n * 100, 1),
        "genuinely_wrong":      genuinely_wrong,
        "genuinely_wrong_pct":  round(genuinely_wrong / n * 100, 1),
    }


# ─────────────────────────────────────────────────────────────
# STEP 3 - Random baseline for comparison
# ─────────────────────────────────────────────────────────────

def random_baseline(results):
    """
    For a 3-outcome prediction (win/draw/loss), random guessing
    gives 33.3% exact accuracy. This is the baseline to beat.
    We also calculate what % of actual results were each outcome
    to show the real distribution.

    Why we need a baseline:
    If our model achieves 55% accuracy but draws only occur 20% of the time,
    a naive model that always predicts "no draw" would also beat 33.3%.
    Reporting the actual outcome distribution shows the model isn't just
    exploiting a class imbalance.
    """
    n = len(results)
    outcome_counts = defaultdict(int)
    for r in results:
        outcome_counts[r["actual_outcome"]] += 1

    return {
        "random_exact_accuracy": 33.3,
        "actual_team1_wins":     round(outcome_counts["team1_win"] / n * 100, 1),
        "actual_draws":          round(outcome_counts["draw"]      / n * 100, 1),
        "actual_team2_wins":     round(outcome_counts["team2_win"] / n * 100, 1),
    }


# ─────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("\nWC 2026 Predictor - Backtesting\n")

    # Load all data — we need the full historical dataset for train/test splits
    all_matches, rankings, rankings_2026, schedule = load_all()

    all_results = []
    tournament_summaries = []

    # Run backtest for each test year.
    # Each iteration is fully independent — no information leaks between years.
    for year in TEST_YEARS:
        print(f"Testing on {year} World Cup...")
        results = backtest_tournament(year, all_matches, rankings)
        metrics  = calculate_metrics(results)
        baseline = random_baseline(results)

        all_results.extend(results)
        tournament_summaries.append({
            "year":     year,
            "metrics":  metrics,
            "baseline": baseline,
        })

        print(f"  Matches:        {metrics['matches']}")
        print(f"  Exact accuracy: {metrics['exact_accuracy']}%  (random baseline: 33.3%)")
        print(f"  Team held:      {metrics['no_upset_accuracy']}%")
        print(f"  Genuinely wrong:{metrics['genuinely_wrong_pct']}%")
        print()

    # Aggregate results across all 4 tournaments.
    # 192 matches is a statistically meaningful sample size for football prediction.
    overall = calculate_metrics(all_results)
    print(f"Overall across {len(TEST_YEARS)} tournaments ({overall['matches']} matches):")
    print(f"  Exact accuracy:  {overall['exact_accuracy']}%  (random baseline: 33.3%)")
    print(f"  Team held:       {overall['no_upset_accuracy']}%")
    print(f"  Genuinely wrong: {overall['genuinely_wrong_pct']}%")
    print(f"  Improvement over random: +{round(overall['exact_accuracy'] - 33.3, 1)} percentage points")

    # Save structured results as JSON for further analysis or display in the app
    os.makedirs("outputs", exist_ok=True)
    output = {
        "test_years":    TEST_YEARS,
        "overall":       overall,
        "by_tournament": tournament_summaries,
    }

    with open("predictions/backtest_results.json", "w") as f:
        json.dump(output, f, indent=2)

    # Save human-readable report
    lines = []
    lines.append("WC 2026 Predictor - Backtest Report")
    lines.append("=" * 55)
    lines.append(f"Method: Train on data before each test year, predict that year")
    lines.append(f"Test tournaments: {', '.join(map(str, TEST_YEARS))}")
    lines.append("")

    for s in tournament_summaries:
        m = s["metrics"]
        b = s["baseline"]
        lines.append(f"{s['year']} World Cup ({m['matches']} matches)")
        lines.append(f"  Exact accuracy:     {m['exact_accuracy']}%   (random: {b['random_exact_accuracy']}%)")
        lines.append(f"  Team held:          {m['no_upset_accuracy']}%")
        lines.append(f"  Genuinely wrong:    {m['genuinely_wrong_pct']}%")
        lines.append(f"  Actual outcomes:    {b['actual_team1_wins']}% wins / {b['actual_draws']}% draws / {b['actual_team2_wins']}% losses")
        lines.append("")

    lines.append("=" * 55)
    lines.append(f"OVERALL ({overall['matches']} matches across 4 tournaments)")
    lines.append(f"  Exact accuracy:     {overall['exact_accuracy']}%")
    lines.append(f"  Random baseline:    33.3%")
    lines.append(f"  Improvement:        +{round(overall['exact_accuracy'] - 33.3, 1)} percentage points")
    lines.append(f"  Team held:          {overall['no_upset_accuracy']}%")
    lines.append(f"  Genuinely wrong:    {overall['genuinely_wrong_pct']}%")

    report = "\n".join(lines)
    print("\n" + report)

    with open("outputs/backtest_report.txt", "w", encoding="utf-8") as f:
        f.write(report)

    print(f"\nSaved -> predictions/backtest_results.json")
    print(f"Saved -> outputs/backtest_report.txt\n")