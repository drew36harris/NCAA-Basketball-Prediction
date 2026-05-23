#--------------------------------
# Libraries
#--------------------------------

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
from pathlib import Path
from scipy import stats

#--------------------------------
# Directories
#--------------------------------

BaseDirectory = Path(r"C:\Users\Drew\Documents\PersonalProjects\BasketballPrediction")
TrainingCSV = BaseDirectory / "data" / "processed" / "training_rows.csv"
OutputDirectory = BaseDirectory / "eda_output"
OutputDirectory.mkdir(parents=True, exist_ok=True)

#--------------------------------
# MatPlotLib Style Guide
#--------------------------------

plt.style.use("seaborn-v0_8-whitegrid")
Colors = {
    "win": "#1D9E75", # Home Win = Teal
    "loss": "#D85A30",  # Home Loss = Orange
    "neutral": "#7F77DD",  # Neutral = Purple
    "highlight": "#EF9F27" #Callout = Amber
}

#--------------------------------
# Helper Functions
#--------------------------------

def save(fig, filename, notes=None):
    """Save figure and optionally append notes to the report"""
    path = OutputDirectory / filename
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {filename}")
    if notes:
        with open(OutputDirectory / "eda_report.txt", "a") as f:
            f.write(f"\n{'='*60}\n{filename}\n{'='*60}\n")
            f.write(notes + "\n")

def section(title):
    """Terminal Section Divider"""

    print(f"\n── {title} {'─' * (50 - len(title))}")

#--------------------------------
# Load Data
#--------------------------------

print("Loading training data...")
df = pd.read_csv(TrainingCSV)
df["date"] = pd.to_datetime(df["date"])

regseason = df[df["season_type"] == "SeasonType.REGULAR"].copy()
postseason = df[df["season_type"] == "SeasonType.POSTSEASON"].copy()

print(f"  Total rows:      {len(df):,}")
print(f"  Regular season:  {len(regseason):,}")
print(f"  Tournament:      {len(postseason):,}")
print(f"  Columns:         {len(df.columns)}")
print(f"  Seasons:         {sorted(df['season'].unique())}")

#--------------------------------
# Targets
#--------------------------------

section("Target Distibution")

# Win/Loss Balance
fig, axes = plt.subplots(1, 3, figsize=(15, 5))
fig.suptitle("Target variable distributions", fontsize=14, fontweight="bold")

win_counts = df["home_won"].value_counts()
axes[0].bar(["Home Win", "Home Loss"],
            [win_counts.get(1, 0), win_counts.get(0,0)],
            color=[Colors["win"], Colors["loss"]], edgecolor="white", linewidth=0.5)
axes[0].set_title("Win/Loss Balance")
axes[0].set_ylabel("Games")
for i, v, in enumerate([win_counts.get(1, 0), win_counts.get(0,0)]):
    axes[0].text(i, v + 20, f"{v:,}\n({v/len(df):.1%})", ha="center", fontsize=10)

# Point Differential Distribution
axes[1].hist(df["point_diff"], bins=40, color=Colors["neutral"],
             edgecolor="white", linewidth=0.3)
axes[1].axvline(0, color=Colors["loss"], linestyle="--", linewidth=1.5, label="Even game")
axes[1].axvline(df["point_diff"].mean(), color=Colors["highlight"],
                linestyle="--", linewidth=1.5, label=f"Mean: {df['point_diff'].mean():.1f}")
axes[1].set_title("Point differential (home - away)")
axes[1].set_xlabel("Points")
axes[1].set_ylabel("Games")
axes[1].legend()

 
# Point Differential by Game Type
axes[2].hist(regseason["point_diff"],    bins=35, alpha=0.6,
             color=Colors["win"],     label="Regular season", edgecolor="white")
axes[2].hist(postseason["point_diff"], bins=35, alpha=0.6,
             color=Colors["loss"],    label="Tournament",     edgecolor="white")
axes[2].axvline(0, color="black", linestyle="--", linewidth=1)
axes[2].set_title("Point diff: regular vs tournament")
axes[2].set_xlabel("Points")
axes[2].legend()
 
plt.tight_layout()
notes = (
    f"Home win rate: {df['home_won'].mean():.1%}\n"
    f"Mean point diff: {df['point_diff'].mean():+.2f}\n"
    f"Std point diff: {df['point_diff'].std():.2f}\n"
    f"Tournament home win rate: {postseason['home_won'].mean():.1%} "
    f"(neutral site games included)\n"
    f"Regular season home win rate: {regseason['home_won'].mean():.1%}"
)
save(fig, "target_distribution.png", notes)

#--------------------------------
# Season Coverage
#--------------------------------

section("Season Coverage")

season_stats = df.groupby("season").agg(
    games = ("game_id", "count"),
    home_wr = ("home_won", "mean"),
    avg_diff = ("point_diff", "mean"),
    tourn_games = ("tournament", "count")
).reset_index()

# Recalculating tournament count
tourn_by_season = postseason.groupby("season").size().reset_index(name="tourn_games")
season_stats = season_stats.drop(columns=["tourn_games"])
season_stats = season_stats.merge(tourn_by_season, on="season", how="left")
season_stats["tourn_games"] = season_stats["tourn_games"].fillna(0).astype(int)

fig, axes = plt.subplots(2, 2, figsize = (14,8))
fig.suptitle("Season Coverage", fontsize=14, fontweight="bold")

# Games per season
axes[0,0].bar(season_stats["season"], season_stats["games"], color=Colors["neutral"], edgecolor="white")
axes[0,0].set_title("Games per season")
axes[0,0].set_ylabel("Games")
axes[0,0].tick_params(axis="x", rotation=45)

# Tournament games per season
axes[0,1].bar(season_stats["season"], season_stats["tourn_games"], color=Colors["highlight"], edgecolor="white")
axes[0,1].set_title("Tournament games per season")
axes[0,1].set_ylabel("Games")
axes[0,1].tick_params(axis="x", rotation=45)

# Home WR per season
axes[1,0].plot(season_stats["season"], season_stats["home_wr"], marker="o", color=Colors["win"], linewidth=2)
axes[1,0].set_title("Home win rate per season")
axes[1,0].set_ylabel("Home win rate")
axes[1,0].set_ylim(0.45, 0.75)
axes[1,0].tick_params(axis="x", rotation=45)

# Avg point diff per season
axes[1,1].plot(season_stats["season"], season_stats["avg_diff"], marker="o", color=Colors["loss"], linewidth=2)
axes[1,1].set_title("Average point diff per season")
axes[1,1].set_ylabel("Points")
axes[1,1].tick_params(axis="x", rotation=45)

plt.tight_layout()
save(fig, "season_coverage.png", season_stats.to_string(index=False))

#--------------------------------
# Missing Values
#--------------------------------

section("Missing Values")

# Percent Missing by Column
missing = (df.isnull().sum()/len(df)*100).sort_values(ascending=False)
missing = missing[missing > 0]

fig, ax = plt.subplots(figsize = (10, max(4, len(missing)*.3)))
colors = [Colors["loss"] if v > 20 else Colors["highlight"] if v > 5 else Colors["neutral"] for v in missing.values]
ax.barh(missing.index, missing.values, color=colors, edgecolor="white")
ax.axvline(5, color=Colors["highlight"], linestyle="--", linewidth=1, label="5% - Check")
ax.axvline(20, color=Colors["loss"], linestyle="--", linewidth=1, label="20% - Warning")
ax.set_xlabel("Percent missing")
ax.set_title("Missing values by column")
ax.legend()
plt.tight_layout()

notes = "Columns with missing data:\n" + missing.to_string()
save(fig, "missing_values.png", notes)

missing_by_year = (
    df
    .groupby("season")
    .apply(lambda x: x.isnull().mean() * 100)
)

# optional: drop columns that are never missing
missing_by_year = missing_by_year.loc[:, missing_by_year.max() > 0]

fig, ax = plt.subplots(figsize=(12, 6))

im = ax.imshow(missing_by_year.values, aspect="auto", cmap="coolwarm")

ax.set_yticks(range(len(missing_by_year.index)))
ax.set_yticklabels(missing_by_year.index)

ax.set_xticks(range(len(missing_by_year.columns)))
ax.set_xticklabels(missing_by_year.columns, rotation=90)

ax.set_title("Missing Values by Season")
ax.set_xlabel("Features")
ax.set_ylabel("Season")

cbar = plt.colorbar(im, ax=ax)
cbar.set_label("% Missing")

plt.tight_layout()

notes_year = "Missing values by season (rows = seasons, columns = features)"

save(fig, "missing_values_by_year.png", notes_year)
plt.close(fig)

print(f" Columns with any missing: {len(missing)}")
print(f" Worst offenders:\n{missing.head(10).to_string()}")

#--------------------------------
# Correlation Check
#--------------------------------

section("Correlation Check")

# Excluding identifiers, metadata, target-adjacent, etc.
exclude = {
    "game_id", "season", "date", "season_type", "tournament",
    "home_team_id", "home_team_name", "away_team_id", 
    "away_team_name", "home_points", "away_points",
    "point_diff", "home_elo_post", "away_elo_post",
    "home_games_played", "away_games_played"
}

numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
feature_cols = [c for c in numeric_cols if c not in exclude]

# Correlation with point_diff
corr_diff = (
    df[feature_cols + ["point_diff"]]
    .corr()["point_diff"]
    .drop("point_diff")
    .sort_values(key=abs, ascending=False)
)

# Top 25 correlation
top25 = corr_diff.head(25)

fig, ax = plt.subplots(figsize=(10, 10))
colors = [Colors["win"] if v > 0 else Colors["loss"] for v in top25.values]
ax.barh(top25.index[::-1], top25.values[::-1], color=colors[::-1], edgecolor="white")
ax.axvline(0, color="black", linewidth = .8)
ax.axvline(.1, color="gray", linestyle="--", linewidth=0.8, label=".1 Threshold")
ax.axvline(-.1, color="gray", linestyle="--", linewidth=0.8)
ax.set_xlabel("Correlation with point_diff")
ax.set_title("Top 25 features correlated with point differential")
ax.legend()
plt.tight_layout()

notes = (
    "Top 15 most correlated:\n" + corr_diff.head(15).to_string() +
    "\n\nBottom 5 (weakest signal):\n" + corr_diff.tail(5).to_string()
)

save(fig, "correlation_check.png", notes)

print(f"Top 10 features correlated with point_diff:\n{corr_diff.head(10).to_string()}")

#--------------------------------
# Multicollinearity
#--------------------------------

section("Multicollinearity")

diff_cols = [c for c in df.columns if c.startswith("diff_")]
diff_cols += ["spread", "home_elo_pre", "away_elo_pre"]
diff_cols  = [c for c in diff_cols if c in df.columns]

corr_matrix = df[diff_cols].corr()

fig, ax = plt.subplots(figsize=(12, 10))
sns.heatmap(
    corr_matrix,
    annot=True, fmt=".2f",
    cmap="RdYlGn", center=0,
    vmin=-1, vmax=1,
    ax=ax,
    annot_kws={"size": 7}
)
ax.set_title("Feature intercorrelation\n"
             "(values near +-1 = redundant features)")
plt.tight_layout()
save(fig, "multicollinearity.png")

# Flag highly correlated pairs
print("\n  Highly correlated pairs (|r| > .7):")
found = False
for i in range(len(corr_matrix.columns)):
    for j in range(i + 1, len(corr_matrix.columns)):
        r = corr_matrix.iloc[i, j]
        if abs(r) > 0.7:
            print(f"    {corr_matrix.columns[i]} <-> "
                  f"{corr_matrix.columns[j]}: r={r:.2f}")
            found = True
if not found:
    print("None found above .7 threshold")

#--------------------------------
# Four Factors Sanity Check
#--------------------------------

# Just a quick check to make sure the four factors show a relationship with point differential.
# If they don't, it's a sign that something is really wrong with the data.

section("Four Factors Sanity Check")

four_factors = {
    "eFG diff": "diff_efg",
    "Turnover ratio diff": "diff_to_ratio",
    "Off reb diff": "diff_adj_orb_rate" if "diff_adj_orb_rate" in df.columns else "diff_orb_pct",
    "Free throw rate diff": "diff_ft_rate"
}

fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle("Four Factors vs Point Differential", fontsize=14, fontweight="bold")

for ax, (label, col) in zip(axes.flatten(), four_factors.items()):
    if col not in df.columns:
        ax.text(.5, .5, f"{col} not found",
                transform = ax.transAxes, ha="center")
        continue
    won = df[df["home_won"] == 1]
    lost = df[df["home_won"] == 0]

    ax.scatter(won[col], won["point_diff"], color=Colors["win"], alpha=.15, 
               s=8, label="Home Win")
    ax.scatter(lost[col], lost["point_diff"], color=Colors["loss"], alpha=.15,
               s=8, label="Home Loss")
    # Regession line
    clean = df[[col, "point_diff"]].dropna()
    if len(clean) > 10:
        m, b, r, p, _ = stats.linregress(clean[col], clean["point_diff"])
        x_range = np.linspace(clean[col].min(), clean[col].max(), 100)
        ax.plot(x_range, m * x_range + b, color="black", linewidth=1.5, label=f"r={r:.2f}")

    ax.axhline(0, color="gray", linestyle="--", linewidth=.8)
    ax.axvline(0, color="gray", linestyle="--", linewidth=.8)
    ax.set_xlabel(label)
    ax.set_ylabel("Point diff")
    ax.legend(fontsize=8, markerscale=3)

plt.tight_layout()
save(fig, "four_factors_check.png")

#--------------------------------
# Feature Selection Decisions
#--------------------------------

include_list = [
    # Four Factors
    #"diff_efg",
    "diff_to_ratio", 
    #"diff_orb_pct", 
    "diff_ft_rate",

    # Efficiencies
    "diff_best_net_eff",
    #"diff_roll_net_eff",
    "diff_adj_off_eff",
    "diff_adj_def_eff",

    # Ratings and Context
    "diff_elo", 
    #"diff_ap_rank", 
    "neutral_site",
    "diff_seed",
    "has_seed",

    # Regression Indicators
    "diff_pyth_exp",
    "diff_pyth_luck",

    # Game Control
    "diff_tempo_control",
    "diff_shot_sel_disruption",
    "diff_turnover_disruption",
    "diff_orb_disruption",
    "pace_mismatch",
    "diff_forced_to_rate",
    "diff_unforced_to_rate",

    # Momentum Four Factors
    "diff_mom_efg",
    #"diff_mom_to_ratio",
    "diff_mom_orb_pct",
    #"diff_mom_ft_rate",
]
model_features = df[include_list]

# Save for model
feature_list_path = OutputDirectory / "model_features.txt"
with open(feature_list_path, "w") as f:
    f.write("# Model features\n")
    f.write("\n# Keep:\n")
    for feat in model_features:
        f.write(f"{feat}\n")

print(f"\n  Saved to {feature_list_path}")
print("\n  EDA complete — ready for benchmarking")