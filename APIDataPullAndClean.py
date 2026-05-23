import os
import time
import cbbd
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime


#--------------------------------
# Config
#--------------------------------

SEASONS = list(range(2014,2025))

# Minimum games a team must have played before we include that row.
MIN_GAMES_PLAYED = 5

RAW_DIR       = Path("data/raw")
PROCESSED_DIR = Path("data/processed")


#--------------------------------
# Setup
#--------------------------------

def setup_directories():
    BASE_DIR  = Path(r"C:\Users\Drew\Documents\PersonalProjects\BasketballPrediction")
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    print("✓ Output directories ready")


def get_api_client():
    api_key = os.environ.get("CBBD_API_KEY")
    if not api_key:
        raise EnvironmentError(
            "Set CBBD_API_KEY environment variable before running.\n"
            "Example: export CBBD_API_KEY='your_key_here'"
        )
    configuration = cbbd.Configuration(access_token=api_key)
    client = cbbd.ApiClient(configuration)
    print("✓ API client authenticated")
    return client


#--------------------------------
# Pull Data
#--------------------------------

def pull_games(client, season):
    """
    Pull all completed games for a season.

    KEY FIELDS (confirmed from inspection):
      id                    → unique game ID
      start_date            → datetime with timezone
      home_team / away_team → team names
      home_points / away_points → final scores
      home_winner           → bool
      neutral_site          → bool
      season_type           → SeasonType.REGULAR or SeasonType.POSTSEASON
      tournament            → string (e.g. "NCAA") or None
      home_team_elo_start   → Elo BEFORE this game — safe predictive feature
      away_team_elo_start   → same for away team
      home_team_elo_end     → Elo AFTER — do NOT use as a feature (leakage)
      excitement            → game excitement index
      attendance            → crowd size
      home_seed / away_seed → seeding (tournament games only)
    """
    api = cbbd.GamesApi(client)
    print(f"  Pulling games for {season}...")

    regular    = api.get_games(season=season, season_type="regular")
    postseason = api.get_games(season=season, season_type="postseason")
    all_games  = regular + postseason

    rows = []
    for g in all_games:
        if str(g.status) != "GameStatus.FINAL":
            continue

        rows.append({
            "game_id":         g.id,
            "season":          season,
            "date":            g.start_date,
            "home_team":       g.home_team,
            "away_team":       g.away_team,
            "home_team_id":    g.home_team_id,
            "away_team_id":    g.away_team_id,
            "home_points":     g.home_points,
            "away_points":     g.away_points,
            "home_winner":     g.home_winner,
            "neutral_site":    g.neutral_site,
            "conference_game": g.conference_game,
            "season_type":     str(g.season_type),
            "tournament":      g.tournament,
            # Elo before the game — what we want for prediction
            "home_elo_pre":    g.home_team_elo_start,
            "away_elo_pre":    g.away_team_elo_start,
            # Context extras
            "excitement":      g.excitement,
            "attendance":      g.attendance,
            "venue":           g.venue,
            "city":            g.city,
            "state":           g.state,
            "home_seed":       g.home_seed,
            "away_seed":       g.away_seed,
        })

    df = pd.DataFrame(rows)
    print(f"    → {len(df)} completed games")
    return df


def pull_game_box_scores(client, season):
    """
    Pull per-game team box scores
    """
    api = cbbd.GamesApi(client)
    print(f"  Pulling box scores for {season}...")

    regular    = api.get_game_teams(season=season, season_type="regular")
    postseason = api.get_game_teams(season=season, season_type="postseason")
    results    = regular + postseason

    def extract(s, prefix):
        """Flatten a GameBoxScoreTeamStats object into prefixed dict keys."""
        ff = getattr(s, "four_factors",            None)
        fg = getattr(s, "field_goals",             None)
        tp = getattr(s, "three_point_field_goals", None)
        ft = getattr(s, "free_throws",             None)
        rb = getattr(s, "rebounds",                None)
        tv = getattr(s, "turnovers",               None)
        pt = getattr(s, "points",                  None)

        return {
            # Four factors — strongest basketball predictors
            f"{prefix}efg_pct":        getattr(ff, "effective_field_goal_pct", None),
            f"{prefix}to_ratio":       getattr(ff, "turnover_ratio",           None),
            f"{prefix}orb_pct":        getattr(ff, "offensive_rebound_pct",    None),
            f"{prefix}ft_rate":        getattr(ff, "free_throw_rate",          None),
            # Shooting
            f"{prefix}fg_pct":         getattr(fg, "pct",       None),
            f"{prefix}fg_made":        getattr(fg, "made",       None),
            f"{prefix}fg_att":         getattr(fg, "attempted",  None),
            f"{prefix}tp_pct":         getattr(tp, "pct",        None),
            f"{prefix}tp_made":        getattr(tp, "made",       None),
            f"{prefix}tp_att":         getattr(tp, "attempted",  None),
            f"{prefix}ft_pct":         getattr(ft, "pct",        None),
            f"{prefix}ft_made":        getattr(ft, "made",       None),
            f"{prefix}ft_att":         getattr(ft, "attempted",  None),
            # Rebounding
            f"{prefix}off_reb":        getattr(rb, "offensive",  None),
            f"{prefix}def_reb":        getattr(rb, "defensive",  None),
            f"{prefix}total_reb":      getattr(rb, "total",      None),
            # Turnovers
            f"{prefix}turnovers":      getattr(tv, "total",      None),
            # Scoring breakdown
            f"{prefix}points":         getattr(pt, "total",         None),
            f"{prefix}pts_paint":      getattr(pt, "in_paint",      None),
            f"{prefix}pts_fastbrk":    getattr(pt, "fast_break",    None),
            f"{prefix}pts_off_to":     getattr(pt, "off_turnovers", None),
            # Other
            f"{prefix}assists":        getattr(s,  "assists",        None),
            f"{prefix}blocks":         getattr(s,  "blocks",         None),
            f"{prefix}steals":         getattr(s,  "steals",         None),
            f"{prefix}possessions":    getattr(s,  "possessions",    None),
            f"{prefix}off_rating":     getattr(s,  "rating",         None),
            f"{prefix}true_shooting":  getattr(s,  "true_shooting",  None),
            f"{prefix}game_score":     getattr(s,  "game_score",     None),
        }

    seen_game_ids = set()
    rows = []

    for r in results:
        if r.game_id in seen_game_ids:
            continue
        # Non-neutral: only process the home team's record
        if not r.neutral_site and not r.is_home:
            continue

        seen_game_ids.add(r.game_id)

        row = {
            "game_id":   r.game_id,
            "season":    r.season,
            "date":      r.start_date,
            "pace":      r.pace,
            "home_team": r.team     if r.is_home else r.opponent,
            "away_team": r.opponent if r.is_home else r.team,
        }

        if r.is_home:
            row.update(extract(r.team_stats,     "home_"))
            row.update(extract(r.opponent_stats, "away_"))
        else:
            # Neutral site: r.team is arbitrarily the "home" side
            row.update(extract(r.team_stats,     "home_"))
            row.update(extract(r.opponent_stats, "away_"))

        rows.append(row)

    df = pd.DataFrame(rows)
    print(f"    → {len(df)} game box score rows")
    return df


def pull_adjusted_efficiency(client, season):
    """
    Season-level KenPom style efficiency ratings

    Note: lower defensive_rating = better defense (fewer points allowed)
    """
    api = cbbd.RatingsApi(client)
    print(f"  Pulling adjusted efficiency for {season}...")

    results = api.get_adjusted_efficiency(season=season)

    rows = []
    for r in results:
        rows.append({
            "team":         r.team,
            "season":       r.season,
            "adj_off":      r.offensive_rating,
            "adj_def":      r.defensive_rating,
            "adj_net":      r.net_rating,
            "adj_off_rank": getattr(r.rankings, "offense", None) if r.rankings else None,
            "adj_def_rank": getattr(r.rankings, "defense", None) if r.rankings else None,
            "adj_net_rank": getattr(r.rankings, "net",     None) if r.rankings else None,
        })

    df = pd.DataFrame(rows)
    print(f"    → {len(df)} efficiency ratings")
    return df


def pull_srs(client, season):
    """
    Simple Rating System — avg point margin adjusted for schedule strength
    """
    api = cbbd.RatingsApi(client)
    print(f"  Pulling SRS for {season}...")

    results = api.get_srs(season=season)
    rows = [{"team": r.team, "season": r.season, "srs": r.rating}
            for r in results]

    df = pd.DataFrame(rows)
    print(f"    → {len(df)} SRS ratings")
    return df


def pull_lines(client, season):
    """
    Betting lines — spread and over/under averaged across providers

    We average across all available providers for a consensus line
    """
    api = cbbd.LinesApi(client)
    print(f"  Pulling betting lines for {season}...")

    results = api.get_lines(season=season)

    rows = []
    for r in results:
        if not r.lines:
            continue

        spreads     = [l.spread     for l in r.lines if l.spread     is not None]
        over_unders = [l.over_under for l in r.lines if l.over_under is not None]

        rows.append({
            "game_id":       r.game_id,
            "spread":        sum(spreads)     / len(spreads)     if spreads     else None,
            "over_under":    sum(over_unders) / len(over_unders) if over_unders else None,
            "num_providers": len(r.lines),
        })

    df = pd.DataFrame(rows)
    print(f"    → {len(df)} games with lines")
    return df


def pull_rankings(client, season):
    """
    Weekly AP Top 25 poll rankings
    """
    api = cbbd.RankingsApi(client)
    print(f"  Pulling rankings for {season}...")

    results = api.get_rankings(season=season)

    rows = []
    for r in results:
        if r.poll_type != "AP Top 25":
            continue
        rows.append({
            "team":      r.team,
            "season":    r.season,
            "week":      r.week,
            "poll_date": r.poll_date,
            "ap_rank":   r.ranking,
        })

    df = pd.DataFrame(rows)
    print(f"    → {len(df)} AP Top 25 weekly entries")
    return df


def pull_all_seasons(client):
    """Pull all endpoints for all seasons and save raw CSVs"""
    all_games, all_box, all_eff, all_srs, all_lines, all_rank = [], [], [], [], [], []

    for season in SEASONS:
        print(f"\n── Season {season} ──────────────────────────────")
        all_games.append(pull_games(client, season));                   time.sleep(1)
        all_box.append(pull_game_box_scores(client, season));           time.sleep(1)
        all_eff.append(pull_adjusted_efficiency(client, season));       time.sleep(1)
        all_srs.append(pull_srs(client, season));                       time.sleep(1)
        all_lines.append(pull_lines(client, season));                   time.sleep(1)
        all_rank.append(pull_rankings(client, season));                 time.sleep(1)

    games_df      = pd.concat(all_games, ignore_index=True)
    boxscores_df  = pd.concat(all_box,   ignore_index=True)
    efficiency_df = pd.concat(all_eff,   ignore_index=True)
    srs_df        = pd.concat(all_srs,   ignore_index=True)
    lines_df      = pd.concat(all_lines, ignore_index=True)
    rankings_df   = pd.concat(all_rank,  ignore_index=True)

    games_df.to_csv(RAW_DIR      / "games.csv",      index=False)
    boxscores_df.to_csv(RAW_DIR  / "boxscores.csv",  index=False)
    efficiency_df.to_csv(RAW_DIR / "efficiency.csv", index=False)
    srs_df.to_csv(RAW_DIR        / "srs.csv",        index=False)
    lines_df.to_csv(RAW_DIR      / "lines.csv",      index=False)
    rankings_df.to_csv(RAW_DIR   / "rankings.csv",   index=False)

    print(f"\n✓ Raw data saved to {RAW_DIR}/")
    return games_df, boxscores_df, efficiency_df, srs_df, lines_df, rankings_df

#--------------------------------
# Build Per Game Features
#--------------------------------
def compute_per_game_features(boxscores_df):
    """
    Creating some features that aren't already included
    """
    df = boxscores_df.copy()

    # Home Team Adjusted Rebound Rates
    # Offensive
    df["home_adj_orb_rate"] = (
        df["home_off_reb"] /
        (df["home_off_reb"] + df["away_def_reb"])
    ).replace([np.inf, -np.inf], np.nan)

    # Defensive
    df["home_adj_drb_rate"] = (
        df["home_def_reb"] /
        (df["home_def_reb"] + df["away_off_reb"])
    ).replace([np.inf, -np.inf], np.nan)

    # Away Team Adjusted Rebound Rates
    # Offensive
    df["away_adj_orb_rate"] = (
        df["away_off_reb"] /
        (df["away_off_reb"] + df["home_def_reb"])
    ).replace([np.inf, -np.inf], np.nan)

    # Defensive
    df["away_adj_drb_rate"] = (
        df["away_def_reb"] /
        (df["away_def_reb"] + df["home_off_reb"])
    ).replace([np.inf, -np.inf], np.nan)

    # Pythagorean Expectation (based on the score, win expectation)
    K=13.91 # Morey value, another suggested value is 16

    df["home_pyth_exp"] = (
        df["home_points"]**K /
        (df["home_points"]**K + df["away_points"]**K)
    ).replace([np.inf, -np.inf], np.nan)
    
    df["away_pyth_exp"] = 1 - df["home_pyth_exp"]
 
    # Luck (difference betweten PythEx and Win/Loss)
    home_win = (df["home_points"] > df["away_points"]).astype(float)
    away_win = 1 - home_win

    df["home_pyth_luck"] = home_win - df["home_pyth_exp"]
    df["away_pyth_luck"] = away_win - df["away_pyth_exp"]
    
    # 3PT Shot Selection Rate
    # Style marker, will be used for disruption stats
    df["home_shot_sel_rate"] = (df["home_tp_att"] / df["home_fg_att"]).replace([np.inf, -np.inf], np.nan)
    df["away_shot_sel_rate"] = (df["away_tp_att"] / df["away_fg_att"]).replace([np.inf, -np.inf], np.nan)

    # Turnover rate per possession
    # Style marker, will be used for disruption stats
    df["home_to_rate_per_poss"] = (df["home_turnovers"] / df["home_possessions"]).replace([np.inf, -np.inf], np.nan)
    df["away_to_rate_per_poss"] = (df["away_turnovers"] / df["away_possessions"]).replace([np.inf, -np.inf], np.nan)

    # Forceed Turnover Rate
    # What percentage of turnovers were forced by the opponent's defense (susceptibility to defensive pressure)
    home_turnovers_safe = df["home_turnovers"].replace(0, np.nan)
    away_turnovers_safe = df["away_turnovers"].replace(0, np.nan)
    
    df["home_forced_to_rate"] = (df["away_steals"] / home_turnovers_safe).replace([np.inf, -np.inf], np.nan).clip(0,1)
    df["away_forced_to_rate"] = (df["home_steals"] / away_turnovers_safe).replace([np.inf, -np.inf], np.nan).clip(0,1)

    # Unforced Turnover Rate
    # What percentage of possessions result in unforced turnovers (offensive sloppiness)
    df["home_unforced_to_rate"] = ((df["home_turnovers"] - df["away_steals"]) / df["home_possessions"]).replace([np.inf, -np.inf], np.nan).clip(0,None)
    df["away_unforced_to_rate"] = ((df["away_turnovers"] - df["home_steals"]) / df["away_possessions"]).replace([np.inf, -np.inf], np.nan).clip(0,None)

    # Rebuilding efficiency metrics to avoid temporal leakage
    df["home_off_eff_game"] = (df["home_points"] / df["home_possessions"] * 100).replace([np.inf, -np.inf], np.nan)
 
    df["home_def_eff_game"] = (df["away_points"] / df["away_possessions"] * 100).replace([np.inf, -np.inf], np.nan)
 
    df["away_off_eff_game"] = (df["away_points"] / df["away_possessions"] * 100).replace([np.inf, -np.inf], np.nan)
 
    df["away_def_eff_game"] = (df["home_points"] / df["home_possessions"] * 100).replace([np.inf, -np.inf], np.nan)
    
    print(f"  Per-game features computed:")
    print(f"    adj rebound rates: {df['home_adj_orb_rate'].notna().sum():,} non-null")
    print(f"    pythagorean exp:   {df['home_pyth_exp'].notna().sum():,} non-null")
    print(f"    pythagorean luck:  {df['home_pyth_luck'].notna().sum():,} non-null")
 
    return df

#--------------------------------
# Weighting to be used for momentum stats
#--------------------------------

def ewm_shift(col, span=5):
    """
    Exponentially weighted mean with shift(1) for temporal integrity.
    Standard .ewm() doesn't support expanding shift directly, so we
    apply shift first then compute EWM on the shifted series.
    NOTE: this means game 1 uses no data, game 2 uses game 1 only.
    """
    return col.shift(1).ewm(span=span, adjust=True, min_periods=1).mean()

#--------------------------------
# Build Rolling Stats
#--------------------------------

def build_rolling_stats(boxscores_df):
    """
    Convert per-game box scores into rolling averages of prior games.

    We go from wide format (one row per game, home/away columns)
    to long format (two rows per game, one per team), compute the rolling
    average within each team-season group, then shift(1) so game N only
    uses stats from games 1 through N-1.
    """
    print("\nBuilding rolling stats...")

     # Compute per-game derived features
    df = compute_per_game_features(boxscores_df)
    df["date"] = pd.to_datetime(df["date"]).dt.tz_localize(None)
 
    # Go to long format 
    # Identify all home_ and away_ stat columns
    # (exclude team name columns — those are identifiers, not stats)
    home_stat_cols = [c for c in df.columns
                      if c.startswith("home_") and c != "home_team"]
    away_stat_cols = [c for c in df.columns
                      if c.startswith("away_") and c != "away_team"]
 
    # Home team view: rename home_* → * (strip prefix)
    home_view = df[
        ["game_id", "season", "date", "home_team", "pace"] + home_stat_cols
    ].copy().rename(columns={
        "home_team": "team",
        **{c: c.replace("home_", "") for c in home_stat_cols}
    })
 
    # Away team view: rename away_* → * (strip prefix)
    away_view = df[
        ["game_id", "season", "date", "away_team", "pace"] + away_stat_cols
    ].copy().rename(columns={
        "away_team": "team",
        **{c: c.replace("away_", "") for c in away_stat_cols}
    })
 
    # Stack into long format: two rows per game (one per team)
    long = pd.concat([home_view, away_view], ignore_index=True)
 
    # Sort chronologically 
    long = long.sort_values(["team", "season", "date"]).reset_index(drop=True)

    # Season league-average efficiencies
    # Needed for opponent adjustment efficiency calculations
    season_league_avg = (
        long.groupby("season")
        .agg(
            league_avg_off_eff=("off_eff_game", "mean"),
            league_avg_def_eff=("def_eff_game", "mean"),
        )
        .reset_index()
    )

    long = long.merge(
        season_league_avg,
        on="season",
        how="left"
    )
    # Identify stat cols to roll 
    id_cols = ["game_id", "season", "date", "team"]
 
    # Possessions handled separately for tempo control
    # so exclude it from the standard mean rolling
    # Also excluding additional helper columns
    non_roll = id_cols + [
    "possessions",
    "league_avg_off_eff",
    "league_avg_def_eff",
    "home_team",
    "away_team",
    "opponent",
]
    stat_cols = [c for c in long.columns if c not in non_roll]
 
    # Rolling mean for all stats
    #
    # groupby(["team", "season"]) ensures we only look within one team's
    # season — we don't want 2023 games influencing 2024 rolling stats.
    #
    # .transform() applies the function and returns a Series aligned to
    # the original DataFrame index, which is what we need for concat.
    #
    # lambda col: col.expanding().mean().shift(1)
    #   expanding() → grow window with each game
    #   mean()      → average of that growing window
    #   shift(1)    → slide down one: game N gets avg of games 1..N-1
 
    print("  Computing rolling means...")
    rolling_means = (
        long
        .groupby(["team", "season"])[stat_cols]
        .transform(lambda col: col.expanding().mean().shift(1))
    )
    rolling_means.columns = [f"roll_{c}" for c in stat_cols]
    
    # Attach rolling means back to long dataframe
    # Needed to build opponent-adjusted efficiencies at the GAME level
    long = pd.concat([long, rolling_means], axis=1)

    # Opponent Mapping
    game_pairs = df[["game_id", "home_team", "away_team"]].drop_duplicates()

    long = long.merge(
        game_pairs,
        on="game_id",
        how="left"
    )

    long["opponent"] = np.where(
        long["team"] == long["home_team"],
        long["away_team"],
        long["home_team"]
    )

    opponent_lookup = long[[
        "game_id",
        "team",
        "roll_off_eff_game",
        "roll_def_eff_game"
    ]].rename(columns={
        "team": "opponent",
        "roll_off_eff_game": "opp_roll_off_eff",
        "roll_def_eff_game": "opp_roll_def_eff",
    })

    long = long.merge(
        opponent_lookup,
        on=["game_id", "opponent"],
        how="left"
    )

    #Strength of Schedule efficiency adjustments
    long["adj_off_eff_game"] = (
        long["off_eff_game"] +
        (
            long["league_avg_def_eff"] -
            long["opp_roll_def_eff"]
        )
    )

    long["adj_def_eff_game"] = (
        long["def_eff_game"] -
        (
            long["league_avg_off_eff"] -
            long["opp_roll_off_eff"]
        )
    )

    long["adj_net_eff_game"] = (
        long["adj_off_eff_game"] -
        long["adj_def_eff_game"]
    )

    # Rolling opponent-adjusted efficiencies
    adj_eff_cols = [
        "adj_off_eff_game",
        "adj_def_eff_game",
        "adj_net_eff_game",
    ]

    rolling_adj_eff = (
        long
        .groupby(["team", "season"])[adj_eff_cols]
        .transform(lambda col: col.expanding().mean().shift(1))
    )

    rolling_adj_eff.columns = [f"roll_{c}" for c in adj_eff_cols]

    long = pd.concat([long, rolling_adj_eff], axis=1)

    # Momentum to include recent form
    rolling_momentum = (
    long
    .groupby(["team", "season"])[stat_cols]
    .transform(lambda col: ewm_shift(col, span=5))
)
    rolling_momentum.columns = [f"mom_{c}" for c in stat_cols]

    # Tempo control — rolling std dev of possessions 
    #
    # Standard deviation of possessions measures pace consistency.
    # A team with low std dev plays at the same tempo every game.
    # A team with high std dev gets pulled into different paces.
    #
    # Use ddof=1 (sample std dev, pandas default) since we're working
    # with a sample of games from the season.
    #
    # min_periods=2 because std dev requires at least 2 data points.
    # Games 1 and 2 will have NaN tempo_control which is fine —
    # MIN_GAMES_PLAYED filter will remove those rows anyway.
 
    print("  Computing tempo control (rolling possession std dev)...")
    rolling_tempo = (
        long
        .groupby(["team", "season"])["possessions"]
        .transform(lambda col: col.expanding(min_periods=2).std().shift(1))
    )
    rolling_tempo.name = "roll_tempo_control"
 
    # Rolling mean of possessions (avg preferred pace) 
    # Separate from tempo control — this tells us WHAT pace they prefer
    # while tempo control tells us HOW CONSISTENTLY they play that pace
    rolling_avg_poss = (
        long
        .groupby(["team", "season"])["possessions"]
        .transform(lambda col: col.expanding().mean().shift(1))
    )
    rolling_avg_poss.name = "roll_avg_poss"

    # Games played counter 
    # How many games has this team played BEFORE this game this season?
    # Used for the MIN_GAMES_PLAYED filter in build_training_rows.
    long["games_played"] = long.groupby(["team", "season"]).cumcount()
 
    # Assemble intermediate result with rolling means attached
    # Need this to compute disruption stats
    intermediate = pd.concat([
        long[id_cols + ["games_played", "possessions",
                        "shot_sel_rate", "to_rate_per_poss", "adj_orb_rate"]],
        rolling_means,
        rolling_adj_eff,
        rolling_momentum,
        rolling_tempo,
        rolling_avg_poss,
    ], axis=1)
 
    # Disruption susceptibility stats
    # How disruptable is this team ON AVERAGE across their season?
    # Low = resilient, plays their game regardless of opponent
    # High = susceptible, easily pushed off their game plan
 
    print("  Computing disruption susceptibility stats...")
 
    # Shot selection disruption
    # How far does this game's 3pt attempt rate deviate from their rolling norm?
    # High roll_shot_sel_disruption = easily pushed off preferred shot mix.
 
    intermediate["shot_sel_dev_game"] = abs(
        intermediate["shot_sel_rate"] - intermediate["roll_shot_sel_rate"]
    )
    intermediate["roll_shot_sel_disruption"] = (
        intermediate
        .groupby(["team", "season"])["shot_sel_dev_game"]
        .transform(lambda col: col.expanding().mean().shift(1))
    )
 
    # Turnover disruption
    # How far does this game's turnover rate deviate from their rolling norm?
    # Low roll_to_disruption = disciplined and consistent with the ball.
 
    intermediate["to_dev_game"] = abs(
        intermediate["to_rate_per_poss"] - intermediate["roll_to_rate_per_poss"]
    )
    intermediate["roll_to_disruption"] = (
        intermediate
        .groupby(["team", "season"])["to_dev_game"]
        .transform(lambda col: col.expanding().mean().shift(1))
    )
 
    # Offensive rebounding disruption
    # How far does this game's adjusted offensive rebound rate deviate from normal?
    # High roll_orb_disruption = rebounding effectiveness varies a lot = vulnerable.
 
    intermediate["orb_dev_game"] = abs(
        intermediate["adj_orb_rate"] - intermediate["roll_adj_orb_rate"]
    )
    intermediate["roll_orb_disruption"] = (
        intermediate
        .groupby(["team", "season"])["orb_dev_game"]
        .transform(lambda col: col.expanding().mean().shift(1))
    )

    # Step 8 — assemble final result
    # Drop the intermediate per-game deviation columns (not needed downstream)
    # Keep only the rolled disruption stats and everything from rolling_means
 
    disruption_cols = [
        "roll_shot_sel_disruption",
        "roll_to_disruption",
        "roll_orb_disruption",
    ]
    
    result = pd.concat([
        intermediate[id_cols + ["games_played"]],
        rolling_means,
        rolling_adj_eff,
        rolling_momentum,
        rolling_tempo,
        rolling_avg_poss,
        intermediate[disruption_cols],
    ], axis=1)
 
    # Drop first game of each team-season (NaN — no prior games to average)
    result = result.dropna(subset=["roll_points"])
 
    print(f"  → {len(result):,} team-game rolling stat rows")
    print(f"  → {len(result.columns)} columns per row")
    print(f"  → New disruption stats: {disruption_cols}")
 
    return result


#--------------------------------
# Join Rankings
#--------------------------------

def join_rankings(games_df, rankings_df):
    print("\nJoining rankings...")

    games = games_df.copy()
    if games.index.name == "game_id":
        games = games.reset_index()

    games["date"] = pd.to_datetime(games["date"]).dt.tz_localize(None)

    rankings = rankings_df.copy()
    rankings["poll_date"] = pd.to_datetime(rankings["poll_date"]).dt.tz_localize(None)

    # Drop any rows where poll_date or team is null — merge_asof requires
    # the join key to be non-null on both sides
    rankings = rankings.dropna(subset=["poll_date", "team", "season"])
    rankings = rankings.sort_values("poll_date")

    # Also drop any games with null dates (shouldn't exist, but defensive)
    games = games.dropna(subset=["date"])

    print(f"  Rankings rows after cleaning: {len(rankings)}")
    print(f"  Unique poll dates: {rankings['poll_date'].nunique()}")

    def get_rank_before_game(team_col, out_col):
        left = games[["game_id", "date", "season", team_col]].dropna(subset=["date", team_col])
        left = left.sort_values("date")

        right = (
            rankings[["team", "poll_date", "ap_rank", "season"]]
            .rename(columns={
                "team":      team_col,
                "poll_date": "date",
                "ap_rank":   out_col,
            })
            .dropna(subset=["date", team_col, out_col])  # belt and suspenders
            .sort_values("date")
        )

        merged = pd.merge_asof(
            left, right,
            on="date",
            by=[team_col, "season"],
            direction="backward",
        )
        return merged.set_index("game_id")[out_col]

    home_ranks = get_rank_before_game("home_team", "home_ap_rank")
    away_ranks = get_rank_before_game("away_team", "away_ap_rank")

    games = games.merge(home_ranks.reset_index(), on="game_id", how="left")
    games = games.merge(away_ranks.reset_index(), on="game_id", how="left")

    games["home_ap_rank"] = games["home_ap_rank"].fillna(26).astype(int)
    games["away_ap_rank"] = games["away_ap_rank"].fillna(26).astype(int)

    if games.index.name == "game_id":
        games = games.reset_index()

    print("  → Rankings joined")
    return games

#--------------------------------
# Add Engineered Features
#--------------------------------

def add_engineered_diff_features(games):
    """
    Add difference features for the newly engineered stats
    Called inside build_training_rows()
    """
 
    # Adjusted rebound rate advantage
    if "home_roll_adj_orb_rate" in games.columns:
        games["diff_adj_orb_rate"] = (
            games["home_roll_adj_orb_rate"] - games["away_roll_adj_orb_rate"]
        )
        games["diff_adj_drb_rate"] = (
            games["home_roll_adj_drb_rate"] - games["away_roll_adj_drb_rate"]
        )
 
    # Pythagorean expectation advantage
    # Higher pyth_exp = team has been "deserving" to win more
    if "home_roll_pyth_exp" in games.columns:
        games["diff_pyth_exp"] = (
            games["home_roll_pyth_exp"] - games["away_roll_pyth_exp"]
        )
 
    # Luck differential
    # Positive = home team has been luckier = due for regression
    # Negative = away team has been luckier = away team due for regression
    # NOTE: a NEGATIVE value here is actually good for the home team
    if "home_roll_pyth_luck" in games.columns:
        games["diff_pyth_luck"] = (
            games["home_roll_pyth_luck"] - games["away_roll_pyth_luck"]
        )
 
    # Tempo control advantage 
    # REVERSED: away minus home because lower std = more consistent = better
    # Positive = home team is more consistent (better tempo control)
    if "home_roll_tempo_control" in games.columns:
        games["diff_tempo_control"] = (
            games["away_roll_tempo_control"] - games["home_roll_tempo_control"]
        )
 
    # Pace mismatch — absolute difference in preferred pace
    # Captures style clash games
    if "home_roll_avg_poss" in games.columns:
        games["pace_mismatch"] = abs(
            games["away_roll_avg_poss"] - games["home_roll_avg_poss"]
        )
    
    # Shot selection disruption advantage
    # Positive = home team maintains their shot mix more consistently
    if "home_roll_shot_sel_disruption" in games.columns:
        games["diff_shot_sel_disruption"] = (
            games["away_roll_shot_sel_disruption"] - games["home_roll_shot_sel_disruption"]
        )

    # Turnover disruption advantage
    # Positive = home team has more consistent ball control under pressure
    if "home_roll_to_disruption" in games.columns:
        games["diff_turnover_disruption"] = (
            games["away_roll_to_disruption"] - games["home_roll_to_disruption"]
        )

    # Offensive Rebounding disruption advantage
    # Positive = home team maintains rebounding effectiveness under pressure
    if "home_roll_orb_disruption" in games.columns:
        games["diff_orb_disruption"] = (
            games["away_roll_orb_disruption"] - games["home_roll_orb_disruption"]
        )

    # Forced Turnover Rate differential
    # Positive = away team's turnovers are more often forced by opponent's defense (rather than self inflicted)
    if "home_roll_forced_to_rate" in games.columns:
        games["diff_forced_to_rate"] = (
            games["away_roll_forced_to_rate"] - games["home_roll_forced_to_rate"]
        )
    
    # Unforced Turnover Rate differential
    # Positive = home team has more possessions ending in unforced turnovers
    if "home_roll_unforced_to_rate" in games.columns:
        games["diff_unforced_to_rate"] = (
            games["home_roll_unforced_to_rate"] - games["away_roll_unforced_to_rate"]
        )

    # Rolling Net Efficency differentials
    if "home_roll_off_eff_game" in games.columns:
        games["home_roll_net_eff"] = (
            games["home_roll_off_eff_game"] - games["home_roll_def_eff_game"]
        )
        games["away_roll_net_eff"] = (
            games["away_roll_off_eff_game"] - games["away_roll_def_eff_game"]
        )
        games["diff_roll_net_eff"] = (
            games["home_roll_net_eff"] - games["away_roll_net_eff"]
        )
        games["diff_roll_off_eff"] = (
            games["home_roll_off_eff_game"] - games["away_roll_off_eff_game"]
        )
        games["diff_roll_def_eff"] = (
            # reversed — lower def_eff = better defense
            games["away_roll_def_eff_game"] - games["home_roll_def_eff_game"]
        )

    # Opponent-adjusted rolling efficiency
    if "home_roll_adj_net_eff_game" in games.columns:

        games["diff_adj_off_eff"] = (
            games["home_roll_adj_off_eff_game"] -
            games["away_roll_adj_off_eff_game"]
        )

        games["diff_adj_def_eff"] = (
            games["away_roll_adj_def_eff_game"] -
            games["home_roll_adj_def_eff_game"]
        )

        games["diff_adj_net_eff"] = (
            games["home_roll_adj_net_eff_game"] -
            games["away_roll_adj_net_eff_game"]
        )

    # Momentum Four Factors
    games["diff_mom_efg"] = (games["home_mom_efg_pct"] - games["away_mom_efg_pct"])

    games["diff_mom_to_ratio"] = (games["home_mom_to_ratio"] - games["away_mom_to_ratio"])

    games["diff_mom_orb_pct"] = (games["home_mom_orb_pct"] - games["away_mom_orb_pct"])

    games["diff_mom_ft_rate"] = (games["home_mom_ft_rate"] - games["away_mom_ft_rate"])
    return games
 

#--------------------------------
# Create Training Rows
#--------------------------------

def build_training_rows(games_df, rolling_df, efficiency_df, srs_df, lines_df):
    """
    Final join — one row per game
    """
    print("\nAssembling training rows...")

    games = games_df.copy()
    games["date"] = pd.to_datetime(games["date"]).dt.tz_localize(None)

    # Target variables
    games["point_diff"] = games["home_points"] - games["away_points"]
    games["home_won"]   = games["home_winner"].astype(int)

    feature_cols = [c for c in rolling_df.columns if c.startswith(("roll_", "mom_"))
]
    
    # Rolling stats — home team
    roll_home = rolling_df.rename(columns={
        "team": "home_team", "games_played": "home_games_played",
        **{c: f"home_{c}" for c in feature_cols}
    })
    games = games.merge(
        roll_home[["game_id", "home_team", "home_games_played"] +
                  [c for c in roll_home.columns if c.startswith(("home_roll_", "home_mom_"))]],
        on=["game_id", "home_team"], how="inner"
    )

    # Rolling stats — away team
    roll_away = rolling_df.rename(columns={
        "team": "away_team", "games_played": "away_games_played",
        **{c: f"away_{c}" for c in feature_cols}
    })
    games = games.merge(
        roll_away[["game_id", "away_team", "away_games_played"] +
                  [c for c in roll_away.columns if c.startswith(("away_roll_", "away_mom_"))]],
        on=["game_id", "away_team"], how="inner"
    )

    # Adjusted efficiency
    for side in ["home", "away"]:
        eff = efficiency_df.rename(columns={
            "team": f"{side}_team",
            "adj_off": f"{side}_adj_off", "adj_def": f"{side}_adj_def",
            "adj_net": f"{side}_adj_net", "adj_net_rank": f"{side}_adj_rank",
        })
        games = games.merge(
            eff[[f"{side}_team", "season",
                 f"{side}_adj_off", f"{side}_adj_def",
                 f"{side}_adj_net", f"{side}_adj_rank"]],
            on=[f"{side}_team", "season"], how="left"
        )

    # SRS
    for side in ["home", "away"]:
        srs = srs_df.rename(columns={"team": f"{side}_team", "srs": f"{side}_srs"})
        games = games.merge(
            srs[[f"{side}_team", "season", f"{side}_srs"]],
            on=[f"{side}_team", "season"], how="left"
        )

    # Betting lines
    games = games.merge(
        lines_df[["game_id", "spread", "over_under"]],
        on="game_id", how="left"
    )

    # Add engineered feature diffs
    games = add_engineered_diff_features(games)

    # Difference features — positive = home team advantage
    games["diff_adj_net"]    = games["home_adj_net"]         - games["away_adj_net"]
    games["diff_srs"]        = games["home_srs"]             - games["away_srs"]
    games["diff_elo"]        = games["home_elo_pre"]         - games["away_elo_pre"]
    games["diff_efg"]        = games["home_roll_efg_pct"]    - games["away_roll_efg_pct"]
    games["diff_to_ratio"]   = games["home_roll_to_ratio"]   - games["away_roll_to_ratio"]
    games["diff_orb_pct"]    = games["home_roll_orb_pct"]    - games["away_roll_orb_pct"]
    games["diff_ft_rate"]    = games["home_roll_ft_rate"]    - games["away_roll_ft_rate"]
    games["diff_off_rating"] = games["home_roll_off_rating"] - games["away_roll_off_rating"]
    # Rank: reversed so positive still = home advantage
    games["diff_ap_rank"]    = games["away_ap_rank"]         - games["home_ap_rank"]
    games["diff_seed"] = np.where(
        games["home_seed"].notna() & games["away_seed"].notna(),
        games["away_seed"] - games["home_seed"], 0    # neutral for non-tournament games, not imputed
    )
    games["has_seed"] = (games["home_seed"].notna()).astype(int)
    games["adj_net_available"] = (games["season_type"] == "SeasonType.POSTSEASON").astype(int)
    games["adj_net_if_available"] = games["diff_adj_net"] * games["adj_net_available"]
    games["diff_best_net_eff"] = np.where(
        games["season_type"] == "SeasonType.POSTSEASON",
        games["diff_adj_net"],
        games["diff_adj_net_eff"]
    )

    # Filter early-season noise
    before = len(games)
    games = games[
        (games["home_games_played"] >= MIN_GAMES_PLAYED) &
        (games["away_games_played"] >= MIN_GAMES_PLAYED)
    ]
    print(f"  → Dropped {before - len(games)} rows (< {MIN_GAMES_PLAYED} games played)")

    tournament_games = games[games["season_type"]=="SeasonType.POSTSEASON"].copy()
    print(f"  → {len(games):,} total training rows")
    print(f"  → {len(tournament_games):,} tournament rows")
    return games, tournament_games


#--------------------------------
# Save and Summarize
#--------------------------------

def save_and_summarise(training_df, tournament_df):
    training_path   = PROCESSED_DIR / "training_rows.csv"
    tournament_path = PROCESSED_DIR / "tournament_rows.csv"

    training_df.to_csv(training_path,   index=False)
    tournament_df.to_csv(tournament_path, index=False)

    print(f"\n✓ {training_path}")
    print(f"✓ {tournament_path}")

    print("\n── Dataset Summary ─────────────────────────────────────────")
    print(f"  Total rows:         {len(training_df):,}")
    print(f"  Tournament rows:    {len(tournament_df):,}")
    print(f"  Seasons:            {sorted(training_df['season'].unique())}")
    print(f"  Columns:            {len(training_df.columns)}")
    print(f"  Home win rate:      {training_df['home_won'].mean():.1%}")
    print(f"  Avg point diff:     {training_df['point_diff'].mean():+.1f}")
    print(f"  Games with lines:   {training_df['spread'].notna().sum():,}")

    print("\n  Sample rows:")
    cols = ["home_team", "away_team", "season", "diff_adj_net",
            "diff_elo", "spread", "point_diff", "home_won"]
    available = [c for c in cols if c in training_df.columns]
    print(training_df[available].head(5).to_string(index=False))


#--------------------------------
# Main
#--------------------------------

def main():
    print("=" * 60)
    print("  College Basketball Pipeline")
    print(f"  {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 60)

    setup_directories()
    client = get_api_client()

    print("\n[Step 1] Pulling raw data...")
    games_df, boxscores_df, efficiency_df, srs_df, lines_df, rankings_df = \
        pull_all_seasons(client)

    print("\n[Step 2] Building rolling stats...")
    rolling_df = build_rolling_stats(boxscores_df)

    print("\n[Step 3] Joining rankings...")
    games_df = join_rankings(games_df, rankings_df)

    print("\n[Step 4] Assembling training rows...")
    training_df, tournament_df = build_training_rows(
        games_df, rolling_df, efficiency_df, srs_df, lines_df
    )

    print("\n[Step 5] Saving...")
    save_and_summarise(training_df, tournament_df)

    print("\n✓ Pipeline complete!")
    print("Next: run EDA")



if __name__ == "__main__":
    main()
