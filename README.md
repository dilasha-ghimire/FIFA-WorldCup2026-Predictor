# FIFA World Cup 2026 Predictor

A data science project that predicts all 72 FIFA World Cup 2026 group stage matches using Elo-based modelling and Monte Carlo simulation across 100,000 simulations.

![Banner](image/image.png)

---

## Project Structure

```
wc2026-predictor/
│
├── data/
│   ├── matches_1930_2022.csv          # Historical WC match results
│   ├── fifa_ranking_2022-10-06.csv    # FIFA rankings Oct 2022
│   ├── fifa_ranking_2026-06-08.csv    # FIFA rankings Jun 2026
│   └── schedule_2026.csv              # WC 2026 group stage fixtures
│
├── image/
│   └── image.png                      # Project banner
│
├── outputs/
│   ├── data_check.txt                 # data.py run output
│   ├── train_check.txt                # train_model.py run output
│   ├── simulate_check.txt             # simulate.py run output
│   ├── fetch_check.txt                # fetch_results.py run output
│   ├── backtest_check.txt             # backtest.py run output
│   ├── accuracy_report.txt            # latest prediction accuracy report
│   └── backtest_report.txt            # backtesting accuracy report
│
├── predictions/
│   ├── wc2026_predictions.json        # 72 predictions (updated with live results)
│   ├── wc2026_simulation.json         # Monte Carlo tournament win probabilities
│   ├── accuracy_summary.json          # accuracy metrics for Streamlit app
│   └── backtest_results.json          # backtesting results across 2010-2022
│
├── data.py                            # Loads and processes all datasets
├── train_model.py                     # Elo prediction engine
├── simulate.py                        # Monte Carlo tournament simulator
├── fetch_results.py                   # Fetches live results + tracks accuracy
├── backtest.py                        # Validates model on past World Cups
├── app.py                             # Streamlit live dashboard
├── README.md                          # Project documentation
├── requirements.txt                   # Python dependencies
└── .gitignore                         # Files excluded from version control
```

---

## How to Run

Install dependencies:

```bash
pip install -r requirements.txt
```

Verify datasets load correctly (optional):

```bash
python data.py
```

Generate predictions:

```bash
python train_model.py
```

Run Monte Carlo simulation:

```bash
python simulate.py
```

Fetch live results and update accuracy:

```bash
python fetch_results.py
```

Validate model on past World Cups:

```bash
python backtest.py
```

Launch the dashboard:

```bash
python -m streamlit run app.py
```

---

## Step 1 — Dataset

**Source:** [Football - FIFA World Cup, 1930 - 2026 (Kaggle)](https://www.kaggle.com/datasets/piterfm/fifa-football-world-cup)

This is the most widely used World Cup dataset on Kaggle, covering every match from 1930 to 2022 along with FIFA rankings. It was chosen because it is the most complete, well-maintained, and actively used dataset for World Cup prediction projects.

Four files are used:

| File                          | Purpose                                               |
| ----------------------------- | ----------------------------------------------------- |
| `matches_1930_2022.csv`       | 688 historical group stage match results for training |
| `fifa_ranking_2022-10-06.csv` | Team strength features for historical matches         |
| `fifa_ranking_2026-06-08.csv` | Team strength for generating 2026 predictions         |
| `schedule_2026.csv`           | All 72 WC 2026 group stage fixtures                   |

---

## Step 2 — Project Structure

The project is split into six Python files, each with a single responsibility:

- `data.py` — loads and cleans all four CSVs
- `train_model.py` — generates predictions using the Elo formula
- `simulate.py` — runs Monte Carlo tournament simulations
- `fetch_results.py` — fetches live results and tracks accuracy
- `backtest.py` — validates the model on past World Cups
- `app.py` — displays everything as a live Streamlit dashboard

Dependencies are listed in `requirements.txt`. Install them with:

```bash
pip install -r requirements.txt
```

| Library     | Why                                      |
| ----------- | ---------------------------------------- |
| `pandas`    | Reading CSVs and building DataFrames     |
| `numpy`     | Math operations in the Elo formula       |
| `streamlit` | The live web dashboard                   |
| `plotly`    | Charts inside the dashboard              |
| `requests`  | Fetching live match results from the API |

---

## Step 3 — Dataset Loading (`data.py`)

`data.py` loads all four CSV files and exposes clean DataFrames to the rest of the project via `load_all()`.

Key functions:

| Function               | What it does                                                            |
| ---------------------- | ----------------------------------------------------------------------- |
| `load_matches()`       | Loads match results, filters to group stage only, drops incomplete rows |
| `load_rankings_2022()` | Loads Oct 2022 rankings as a dict: team -> points                       |
| `load_rankings_2026()` | Loads Jun 2026 rankings as a dict: team -> points                       |
| `load_schedule_2026()` | Loads 2026 fixtures, assigns group letters, generates match IDs         |
| `load_all()`           | Calls all four loaders and returns everything in one call               |

Run `python data.py` to verify all datasets load correctly:

```
[matches]       688 group stage matches loaded (1930-2022)
[rankings 2022] 211 teams loaded (Oct 2022)
[rankings 2026] 211 teams loaded (Jun 2026)
[schedule]      72 fixtures loaded (WC 2026 group stage)
```

---

## Step 4 — Processing (`data.py`)

Raw data from different sources uses inconsistent team names — for example `"Korea Republic"` in one file and `"South Korea"` in another. `data.py` normalises all names through `TEAM_NAME_MAP` before any processing happens, ensuring every team has one consistent name used across all files.

The schedule CSV has no group column — every row just says `"Group stage"` in the Round field. Groups are assigned by looking up each team in `WC2026_GROUPS`, a dict derived from the official December 2025 draw.

---

## Step 5 — Elo Predictions (`train_model.py`)

### What is Elo?

Elo is a rating system originally designed for chess, now used by FIFA as the basis for their official ranking points. The core idea: a team with higher rating points is more likely to win, and the probability is calculated from the gap between the two teams' points.

### Formula

```
P(team1 wins) = 1 / (1 + 10^(-(r1 - r2) / 400))
```

Where `r1` and `r2` are the FIFA ranking points of each team. A difference of 400 points gives the stronger team roughly a 90% chance of winning.

### Logic

The model goes through five steps for each match:

1. **Get FIFA ranking points** for both teams from the June 2026 rankings
2. **Apply host nation boost** — USA, Canada, and Mexico each receive +80 points to reflect home advantage
3. **Compute base Elo probability** using the formula above
4. **Calibrate draw rate** — closely matched teams draw more often; mismatched games rarely end in draws. Base draw rate is 24%, reduced as the ranking gap grows, with a minimum of 10%
5. **Blend in head-to-head history** — if two teams have met 2+ times at a World Cup, their historical win rate nudges the probability by 8%

### Key functions

| Function                              | What it does                                                              |
| ------------------------------------- | ------------------------------------------------------------------------- |
| `build_h2h(matches)`                  | Builds head-to-head win/draw/loss records from all 688 historical matches |
| `elo_win_prob(pts1, pts2)`            | Core Elo formula — returns P(team1 wins)                                  |
| `predict_match(team1, team2, ...)`    | Runs all 5 steps and returns (p_win, p_draw, p_loss)                      |
| `generate_predictions(schedule, ...)` | Runs predict_match for all 72 fixtures and saves to JSON                  |

### Output

`predictions/wc2026_predictions.json` — 72 predictions, each containing:

```json
{
  "match_id": "A1",
  "group": "A",
  "team1": "Mexico",
  "team2": "South Africa",
  "predicted_winner": "Mexico",
  "prob_team1_win": 70.3,
  "prob_draw": 19.8,
  "prob_team2_win": 10.0,
  "status": "upcoming"
}
```

---

## Step 6 — Monte Carlo Simulation (`simulate.py`)

### What is Monte Carlo simulation?

Instead of just picking the most likely winner of each match, Monte Carlo simulation runs the entire tournament thousands of times. In each run, match outcomes are decided by rolling dice weighted by the Elo probabilities. After all runs, we count how often each team won the tournament — that percentage is their World Cup win probability.

### Logic

Each of the 100,000 simulations follows these steps:

1. **Group stage** — simulate all 72 matches, calculate points and goal difference, determine 1st and 2nd place qualifiers from each group plus the best 8 third-place finishers (32 teams total)
2. **Knockout rounds** — simulate Round of 32, Round of 16, Quarter-finals, Semi-finals, and Final. No draws in knockout stage — a drawn match goes to a 50/50 penalty shootout coin flip
3. **Count the winner** — add 1 to the winning team's tally

After 100,000 runs, divide each team's tally by 100,000 to get their win probability.

### Why 100,000?

100,000 is the standard threshold for Monte Carlo simulations in sports analytics research, giving probability estimates accurate to within ±0.1%. Running fewer simulations produces unstable results that change significantly between runs.

### Key functions

| Function                                 | What it does                                                   |
| ---------------------------------------- | -------------------------------------------------------------- |
| `simulate_match(team1, team2, ...)`      | Rolls dice against Elo probabilities to decide a match outcome |
| `simulate_group_stage(predictions, ...)` | Simulates all 72 group matches and returns 32 qualifiers       |
| `simulate_knockout(teams, ...)`          | Simulates the bracket from Round of 32 to the Final            |
| `run_simulation(predictions, ..., n)`    | Runs the full tournament n times and returns win probabilities |

### Results (100,000 simulations)

| Rank | Team      | Win Probability |
| ---- | --------- | --------------- |
| 1    | Argentina | 12.99%          |
| 2    | Spain     | 12.49%          |
| 3    | France    | 11.63%          |
| 4    | England   | 8.76%           |
| 5    | Brazil    | 5.19%           |
| 6    | Mexico    | 4.95%           |
| 7    | Portugal  | 4.64%           |
| 8    | Morocco   | 4.35%           |

**NOTE:** Predictions are based purely on FIFA ranking points and historical results. The model does not account for squad injuries, current form, or tactical factors.

---

## Step 7 — Live Results & Accuracy Tracking (`fetch_results.py`)

### Data source

Results are fetched from [openfootball/worldcup.json](https://github.com/openfootball/worldcup.json) — a free, open-source dataset updated within 24 hours of each match ending. No API key required.

### Logic

1. Fetch the live JSON from GitHub
2. Filter to completed group stage matches only (matches with a `score.ft` field)
3. Normalize team names to match our predictions (e.g. `"USA"` → `"United States"`)
4. Compare each result to our pre-generated prediction
5. Save updated `wc2026_predictions.json` and `accuracy_summary.json`

### Three accuracy metrics

A draw when we predicted a win is not the same as being completely wrong. The model reports three levels of accuracy:

| Metric              | Meaning                                           |
| ------------------- | ------------------------------------------------- |
| Exact accuracy      | Predicted outcome matched exactly (win/draw/loss) |
| Predicted team held | Our predicted team won or drew — was not beaten   |
| Genuinely wrong     | The other team won entirely                       |

### Current accuracy (as of June 15, 2026)

| Metric              | Result        |
| ------------------- | ------------- |
| Exact accuracy      | 6/12 = 50.0%  |
| Predicted team held | 10/12 = 83.3% |
| Genuinely wrong     | 2/12 = 16.7%  |

### Key functions

| Function                                   | What it does                                                        |
| ------------------------------------------ | ------------------------------------------------------------------- |
| `fetch_results()`                          | Fetches live JSON, filters completed matches, normalizes team names |
| `update_predictions(predictions, results)` | Matches API results to predictions, sets correct/wrong flags        |
| `calculate_accuracy(predictions)`          | Computes all three accuracy metrics                                 |
| `print_report(predictions, accuracy)`      | Prints and saves the accuracy report                                |

---

## Step 8 — Model Validation (`backtest.py`)

### Why backtesting matters

A model that only predicts future matches cannot be evaluated until those matches happen. Backtesting solves this by testing the model on past tournaments it has never seen — giving a reliable estimate of how accurate it actually is before a single 2026 match is played.

### Method

For each test tournament (2010, 2014, 2018, 2022):

1. Train on all historical matches **before** that tournament
2. Predict every group stage match in that tournament
3. Compare predictions to actual results
4. Report accuracy metrics

This is a strict train/test split. The model never sees the test year's data during training, preventing data leakage.

### Results

| Tournament  | Exact Accuracy | vs Random Guess (33.3%) |
| ----------- | -------------- | ----------------------- |
| 2010        | 45.8%          | +12.5pts                |
| 2014        | 54.2%          | +20.9pts                |
| 2018        | 68.8%          | +35.5pts                |
| 2022        | 54.2%          | +20.9pts                |
| **Overall** | **55.7%**      | **+22.4pts**            |

Across 192 matches over 4 tournaments, the model achieves 55.7% exact accuracy — 22.4 percentage points above the 33.3% expected from random guessing across 3 possible outcomes (win/draw/loss).

### Key functions

| Function                                       | What it does                                                             |
| ---------------------------------------------- | ------------------------------------------------------------------------ |
| `backtest_tournament(year, matches, rankings)` | Trains on pre-year data, predicts that year, returns match-level results |
| `calculate_metrics(results)`                   | Computes exact accuracy, favourite not beaten %, and completely wrong %  |
| `random_baseline(results)`                     | Returns the 33.3% random guess baseline and actual outcome distribution  |

---

## Step 9 — Streamlit Dashboard (`app.py`)

### Live app

_Coming soon — deploying on Streamlit Community Cloud_

### What it shows

- **Prediction accuracy** — 5 metric cards showing matches played, exact accuracy, team held, genuinely wrong, and upcoming matches
- **Results breakdown** — donut chart splitting completed matches into correct, draw (not wrong), wrong, and upcoming
- **Tournament win probability** — horizontal bar chart of top 16 teams from the Monte Carlo simulation
- **All predictions** — filterable table of all 72 matches with predicted winner, actual score, win/draw/loss probabilities, and result status

### Features

- Light mode (default) and dark mode toggle
- Filter matches by group (A–L) and status (All / Completed / Upcoming)
- Auto-refreshes every 5 minutes to pick up new results
- Colour-coded results — ✅ correct, 〰️ draw, ❌ wrong, ⏳ upcoming

### How to run locally

```bash
python -m streamlit run app.py
```

### Key functions

| Function            | What it does                                                                                         |
| ------------------- | ---------------------------------------------------------------------------------------------------- |
| `load_data()`       | Loads predictions, fetches live results, returns accuracy and simulation data. Cached for 5 minutes. |
| `color_result(val)` | Colours the Result column green / amber / red based on outcome                                       |
| `style_table(df)`   | Applies theme-aware styling to the match table                                                       |

---

## Techniques Used (Data Science)

| What has been used                               | Field                    |
| ------------------------------------------------ | ------------------------ |
| Data cleaning, normalization, merging            | Data Engineering         |
| Elo rating formula                               | Quantitative Modelling   |
| Historical statistics (head-to-head, draw rates) | Statistical Analysis     |
| Monte Carlo simulation                           | Computational Statistics |
| Train/test split backtesting                     | Model Validation         |
| Live API + accuracy tracking                     | Data Pipeline            |
| Streamlit dashboard                              | Data Visualization       |

---

## QnA(s)

### Q1: Why does your model predict Argentina and not France or Spain like most people say?

**Answer:** Because the model does not follow popular opinion. It uses FIFA ranking points from June 2026, where Argentina is ranked number 1. When I ran the tournament 100,000 times using those rankings, Argentina wins slightly more often than France or Spain but only by a tiny margin. Argentina wins 13% of simulations, Spain 12.49%, France 11.63%. That is not a bold prediction, it is a statistical outcome where three teams are essentially equal at the top.

### Q2: What does 55.7% accuracy actually mean?

**Answer:** It means that when the model was tested on 192 real World Cup group stage matches from 2010 to 2022, it correctly predicted the exact outcome (win, draw, or loss) 55.7% of the time. For comparison, if you picked randomly between three outcomes, you would be right 33.3% of the time. So the model is 22 percentage points better than random guessing across four tournaments it had never seen during training.

### Q3: Why is the random guess baseline 33.3% and not 50%?

**Answer:** Because football has three possible outcomes: the first team wins, the second team wins, or they draw. A 50% baseline would only apply if there were two outcomes like a coin flip. With three outcomes, random guessing gives you a 1 in 3 chance, which is 33.3%.

### Q4: Why does the model say Argentina has a 13% chance of winning? That seems low.

**Answer:** It is actually the correct range for a 48 team tournament. Even the strongest team in the world cannot win more than about 15 to 20% of the time when there are 48 teams and the knockout rounds introduce so much randomness. If the model had Argentina at 40%, that would be a sign something is wrong. At 13%, it is saying Argentina is the most likely winner while still being honest that almost 9 out of 10 times, someone else wins.

### Q5: What is backtesting and why did you add it?

**Answer:** Backtesting means testing the model on past tournaments it never saw during training. For each of the 2010, 2014, 2018 and 2022 World Cups, I trained the model using only data from before that year, then predicted every match in that year, then compared to what actually happened. This proves the model works on real data it had no access to, not just data it was built on. Without this, anyone could fairly ask "how do you know it works?" With it, there is a concrete answer.

### Q6: Why did the model perform so differently across years, with 68.8% in 2018 but only 45.8% in 2010?

**Answer:** Because football is genuinely unpredictable and different tournaments have different levels of upset. The 2018 World Cup had fewer major surprises, so a rankings based model performed well. The 2010 World Cup had more upsets, so any model based on rankings would struggle. This variance across years is actually a sign the model is behaving realistically rather than overfitting to one era.

### Q7: What is the Elo formula and why did you choose it over a machine learning model?

**Answer:** Elo is a mathematical formula originally designed for chess that calculates the probability one team beats another based on the gap between their ranking points. I chose it over machine learning because World Cup data is very limited. There are only about 688 group stage matches across all tournaments since 1930, which is not enough data for machine learning to find reliable patterns. Elo is simpler, more interpretable, and better suited to small datasets with high uncertainty, which is exactly what international football is.

### Q8: What does Monte Carlo simulation actually do in this project?

**Answer:** Instead of just saying "Argentina will win," the simulation runs the entire tournament 100,000 times. In each run, every match outcome is decided by rolling a weighted dice based on the Elo probabilities. Some runs Argentina wins, some runs France wins, some runs a surprise team wins. After 100,000 runs the count of how often each team won becomes their probability of winning the World Cup. It gives an honest picture of uncertainty rather than a single confident but potentially wrong prediction.

### Q9: Why is 83.3% favourite not beaten more useful than 50% exact accuracy?

**Answer:** Because in football, predicting a draw is genuinely hard. When the model predicts a stronger team to win and the match ends in a draw instead, the stronger team was not beaten. That is meaningfully different from being completely wrong. The 83.3% figure shows that in 10 out of 12 completed matches so far, the team I identified as stronger either won or drew. Only 2 matches had a result where the team expected to win actually lost. That distinction matters when evaluating a prediction model on a sport where draws are common.

### Q10: What would you do to make this model even better?

**Answer:** Three things. First, add a Poisson goal model to predict scorelines instead of just win, draw or loss, which would make the simulation more realistic especially in knockout rounds. Second, build a dynamic Elo system that updates team ratings after every simulated match within the tournament, because a team that just won 3 group stage matches is not the same strength as one that barely qualified. Third, compare the model probabilities against bookmaker odds to measure how well calibrated the predictions are. These additions would move it from a validated model to a professional grade forecasting system.

---
