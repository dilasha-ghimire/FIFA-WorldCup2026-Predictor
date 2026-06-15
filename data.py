"""
data.py
────────────────────────────────────────────────────────────
Loads and processes all raw datasets from the data/ folder.

Exposes via load_all():
    matches       — historical WC group stage matches (1930–2022)
    rankings      — FIFA rankings October 2022 (for training features)
    rankings_2026 — FIFA rankings June 2026 (for predictions)
    schedule      — WC 2026 group stage fixtures with group labels
"""

import pandas as pd
from pathlib import Path

DATA_DIR = Path(__file__).parent / "data"

# ─────────────────────────────────────────────────────────────
# CONSTANTS
# ─────────────────────────────────────────────────────────────

HOST_NATIONS = {"United States", "Canada", "Mexico"}

# The WC 2026 draw produced these 12 groups — derived from schedule_2026.csv.
# Used to assign a group letter to each fixture since the CSV has no group column.
WC2026_GROUPS = {
    "A": ["Mexico", "South Korea", "Czech Republic", "South Africa"],
    "B": ["Canada", "Bosnia and Herzegovina", "Qatar", "Switzerland"],
    "C": ["United States", "Paraguay", "Australia", "Turkey"],
    "D": ["Brazil", "Haiti", "Morocco", "Scotland"],
    "E": ["Germany", "Curacao", "Ivory Coast", "Ecuador"],
    "F": ["Netherlands", "Japan", "Sweden", "Tunisia"],
    "G": ["Belgium", "Egypt", "Iran", "New Zealand"],
    "H": ["Spain", "Cape Verde", "Saudi Arabia", "Uruguay"],
    "I": ["France", "Iraq", "Norway", "Senegal"],
    "J": ["Argentina", "Algeria", "Austria", "Jordan"],
    "K": ["Colombia", "Congo DR", "Portugal", "Uzbekistan"],
    "L": ["Croatia", "England", "Ghana", "Panama"],
}

# Build a reverse lookup: team name -> group letter
TEAM_TO_GROUP = {}
for group_letter, teams in WC2026_GROUPS.items():
    for team in teams:
        TEAM_TO_GROUP[team] = group_letter

# Name mismatches between datasets — normalize everything to one standard name.
# Different CSVs use different conventions (e.g. "Korea Republic" vs "South Korea"),
# so we map all variants to a single consistent name used throughout the project.
TEAM_NAME_MAP = {
    "USA":                      "United States",
    "US":                       "United States",
    "IR Iran":                  "Iran",
    "Korea Republic":           "South Korea",
    "Korea DPR":                "North Korea",
    "Türkiye":                  "Turkey",
    "Côte d'Ivoire":            "Ivory Coast",
    "Cote d'Ivoire":            "Ivory Coast",
    "Bosnia-Herzegovina":       "Bosnia and Herzegovina",
    "Bosnia & Herzegovina":     "Bosnia and Herzegovina",
    "West Germany":             "Germany",
    "Serbia and Montenegro":    "Serbia",
    "DR Congo":                 "Congo DR",
    "Czechia":                  "Czech Republic",
    "Curaçao":                  "Curacao",
    "Cabo Verde":               "Cape Verde",
}


# Look up the team name in TEAM_NAME_MAP and return the standardized version.
# If the name isn't in the map, just strip whitespace and return it as-is.
# This ensures team names are consistent across all 4 CSV files —
# e.g. "Korea Republic" in one file and "South Korea" in another both become "South Korea".
def normalize_name(name):
    if not isinstance(name, str):
        return name
    return TEAM_NAME_MAP.get(name.strip(), name.strip())


# ─────────────────────────────────────────────────────────────
# LOAD: Historical match results (1930–2022)
# ─────────────────────────────────────────────────────────────

def load_matches():
    """
    Loads matches_1930_2022.csv.
    Filters to group stage only, returns cleaned DataFrame with columns:
        year, home_team, away_team, home_score, away_score
    """
    df = pd.read_csv(DATA_DIR / "matches_1930_2022.csv")

    # Normalize team names to match rankings and schedule datasets
    df["home_team"] = df["home_team"].apply(normalize_name)
    df["away_team"] = df["away_team"].apply(normalize_name)

    # Keep only group stage matches.
    # "First round" is how older tournaments (1930–1950) labelled the group stage,
    # so we include both naming conventions to avoid losing early tournament data.
    group_keywords = ["Group stage", "First round", "First group stage",
                      "Second group stage", "Group stage play-off"]
    mask = df["Round"].apply(lambda r: any(kw in str(r) for kw in group_keywords))
    df = df[mask].copy()

    # Keep only the columns we need, drop any rows with missing scores
    df = df[["Year", "home_team", "away_team", "home_score", "away_score"]].copy()
    df = df.dropna(subset=["home_score", "away_score"])
    df["home_score"] = df["home_score"].astype(int)
    df["away_score"] = df["away_score"].astype(int)
    df = df.rename(columns={"Year": "year"})
    df = df.reset_index(drop=True)

    print(f"  [matches]       {len(df)} group stage matches loaded (1930–2022)")
    return df


# ─────────────────────────────────────────────────────────────
# LOAD: FIFA Rankings October 2022 (for training features)
# ─────────────────────────────────────────────────────────────

def load_rankings_2022():
    """
    Loads fifa_ranking_2022-10-06.csv.
    Returns dict: team_name -> ranking_points
    Used as the team strength feature when training on historical matches.
    """
    df = pd.read_csv(DATA_DIR / "fifa_ranking_2022-10-06.csv")
    df["team"] = df["team"].apply(normalize_name)
    rankings = dict(zip(df["team"], df["points"]))
    print(f"  [rankings 2022] {len(rankings)} teams loaded (Oct 2022)")
    return rankings


# ─────────────────────────────────────────────────────────────
# LOAD: FIFA Rankings June 2026 (for generating predictions)
# ─────────────────────────────────────────────────────────────

def load_rankings_2026():
    """
    Loads fifa_ranking_2026-06-08.csv.
    Returns dict: team_name -> ranking_points
    Used as team strength when predicting 2026 match outcomes.
    """
    df = pd.read_csv(DATA_DIR / "fifa_ranking_2026-06-08.csv")
    df["team"] = df["team"].apply(normalize_name)
    rankings = dict(zip(df["team"], df["points"]))
    print(f"  [rankings 2026] {len(rankings)} teams loaded (Jun 2026)")
    return rankings


# ─────────────────────────────────────────────────────────────
# LOAD: WC 2026 Group Stage Schedule
# ─────────────────────────────────────────────────────────────

def load_schedule_2026():
    """
    Loads schedule_2026.csv.
    Returns cleaned DataFrame with columns:
        match_id, group, team1, team2, date

    The CSV has no group column — every row just says "Group stage" in Round.
    So we assign group letters using WC2026_GROUPS, which maps each team to
    their group from the official December 2025 draw.
    """
    df = pd.read_csv(DATA_DIR / "schedule_2026.csv")

    # Normalize team names before doing any group lookups
    df["home_team"] = df["home_team"].apply(normalize_name)
    df["away_team"] = df["away_team"].apply(normalize_name)

    # Filter to group stage rows only
    df = df[df["Round"] == "Group stage"].copy()
    df = df.reset_index(drop=True)

    # Assign group letter by looking up team1 in TEAM_TO_GROUP
    df["group"] = df["home_team"].map(TEAM_TO_GROUP)

    # Generate clean match IDs: A1, A2 ... L6
    group_counters = {}
    match_ids = []
    for _, row in df.iterrows():
        g = row["group"]
        group_counters[g] = group_counters.get(g, 0) + 1
        match_ids.append(f"{g}{group_counters[g]}")
    df["match_id"] = match_ids

    df = df[["match_id", "group", "home_team", "away_team", "Date"]].copy()
    df = df.rename(columns={"home_team": "team1", "away_team": "team2", "Date": "date"})
    df = df.dropna(subset=["group"])
    df = df.reset_index(drop=True)

    print(f"  [schedule]      {len(df)} fixtures loaded (WC 2026 group stage)")
    return df


# ─────────────────────────────────────────────────────────────
# LOAD ALL — single entry point used by train_model.py
# ─────────────────────────────────────────────────────────────

def load_all():
    print("\nLoading datasets...")
    matches       = load_matches()
    rankings      = load_rankings_2022()
    rankings_2026 = load_rankings_2026()
    schedule      = load_schedule_2026()
    print("  Done.\n")
    return matches, rankings, rankings_2026, schedule


# ─────────────────────────────────────────────────────────────
# SANITY CHECK — run: python data.py
# ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    matches, rankings, rankings_2026, schedule = load_all()

    print("-- Sample matches (last 5) -----------------")
    print(matches.tail(5).to_string(index=False))

    print("\n-- Sample 2026 schedule (first 12) ---------")
    print(schedule.head(12).to_string(index=False))

    print("\n-- Top 10 FIFA rankings (Jun 2026) ---------")
    top10 = sorted(rankings_2026.items(), key=lambda x: x[1], reverse=True)[:10]
    for i, (team, pts) in enumerate(top10, 1):
        print(f"  {i:>2}. {team:<25} {pts:.1f}")

    print("\n-- 2026 schedule teams missing from rankings -")
    schedule_teams = set(schedule["team1"]) | set(schedule["team2"])
    missing = [t for t in sorted(schedule_teams) if t not in rankings_2026]
    print(f"  {missing if missing else 'None - all teams have rankings [OK]'}")

    print("\n-- Group assignments -------------------------")
    for g in sorted(schedule["group"].dropna().unique()):
        teams = schedule[schedule["group"] == g][["team1", "team2"]]
        group_teams = sorted(set(teams["team1"]) | set(teams["team2"]))
        print(f"  Group {g}: {', '.join(group_teams)}")
