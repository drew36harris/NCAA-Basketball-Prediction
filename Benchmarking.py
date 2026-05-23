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
OutputDirectory = BaseDirectory / "benchmark_outputs"
OutputDirectory.mkdir(parents=True, exist_ok=True)

# Clear the summary file so each run starts fresh
with open(OutputDirectory / "benchmark_summary.txt", "w") as f:
    f.write("Basketball Prediction — Benchmark Summary\n")
    f.write(f"Generated: {pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S")}\n")

#--------------------------------
# MatPlotLib Style Guide
#--------------------------------

plt.style.use("seaborn-v0_8-whitegrid")
Colors = {
    "win": "#1D9E75", # Home Win = Teal
    "loss": "#D85A30",  # Home Loss = Orange
    "neutral": "#7F77DD",  # Neutral = Purple
    "highlight": "#EF9F27", # Callout = Amber
    "naive": "#AAAAAA" # Naive = Gray
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
        with open(OutputDirectory / "benchmark_report.txt", "a") as f:
            f.write(f"\n{"="*60}\n{filename}\n{"="*60}\n")
            f.write(notes + "\n")

def section(title):
    """Terminal Section Divider"""
    print(f"\n── {title} {"─" * (50 - len(title))}")


def write_summary(text):
    """Append plain text to the summary report."""
    with open(OutputDirectory / "benchmark_summary.txt", "a") as f:
        f.write(text + "\n") 

#--------------------------------
# Classification Metrics
#--------------------------------

def win_accuracy(predictions, actuals):
    """
    Fraction of games where the predicted winner was correct
    Higher is better.
    """
    return (np.array(predictions) == np.array(actuals)).mean()

def log_loss(probs, actuals):
    """ 
    Measures confidence and accuracy - penalizes confident wrong predictions more than uncertain ones.
    Lower is better, 0 is perfect.
    """
    p = np.clip(np.array(probs, dtype=float), 1e-7, 1-1e-7) # Avoid log(0) issues
    y = np.array(actuals, dtype=int)
    return -np.mean(y * np.log(p) + (1 - y) * np.log(1 - p))

def brier_score(probs, actuals):
    """ 
    Mean squared error of predicted probabilities. Similar to log loss but less sensitive to extreme mistakes.
    Lower is better, 0 is perfect.
    """
    p = np.array(probs, dtype=float)
    y = np.array(actuals, dtype=int)
    return np.mean((p - y) ** 2)

#--------------------------------
# Regression Metrics
#--------------------------------

def rmse(predicted, actual):
    """
    Root Mean Squared Error - average prediction error in points
    Penalizes larger errors more than smaller ones
    "On average, predictions are off by X points" 
    """
    return np.sqrt(np.mean((np.array(predicted) - np.array(actual)) ** 2))

def mae(predicted, actual):
    """
    Mean Absolute Error - average absolute prediction error in points
    Treats all errors equally
    Lower is better.
    """
    return np.mean(np.abs(np.array(predicted) - np.array(actual)))

def r_squared(predicted, actual):
    """
    Coefficient of Determination - how much of the variance is explained by the predictions
    Higher is better, 1 is perfect, 0 means no better than predicting the mean.
    """
    actual    = np.array(actual, dtype=float)
    predicted = np.array(predicted, dtype=float)
    ss_res    = np.sum((actual - predicted) ** 2)
    ss_tot    = np.sum((actual - actual.mean()) ** 2)
    return 1 - (ss_res / ss_tot)

def point_diff_to_win_prob(point_diff_predictions, sigma=None, actual_diffs=None):
    """
    Convert predicted point differentials to win probabilities
    Bridges regression and classification outputs (assuming normal distribution)

    Sigma - standard deviation of error around predictions
    If not provided we estimate from residuals
    Larger = less confidence
    """

    predicted = np.array(point_diff_predictions, dtype=float)
 
    if sigma is None and actual_diffs is not None:
        # Estimate sigma from residuals if we have actual outcomes
        residuals = np.array(actual_diffs, dtype=float) - predicted
        sigma     = residuals.std()
    elif sigma is None:
        # Fallback: use empirical std of all point differentials
        sigma = 11.0   # typical college basketball game variability
 
    # stats.norm.cdf(x) = P(Z < x) for standard normal Z
    # We want P(actual > 0) = P(predicted + error > 0) = P(error > -predicted)
    # = 1 - P(error < -predicted) = 1 - CDF(-predicted/sigma) = CDF(predicted/sigma)
    return stats.norm.cdf(predicted / sigma)

#--------------------------------
# Load Data
#--------------------------------

section("Loading Data")

ValidationSeason = 2024
TestSeason = 2025

df = pd.read_csv(TrainingCSV)
df["date"] = pd.to_datetime(df["date"])
 
# Temporal splits
train_df = df[df["season"] <  ValidationSeason].copy()
val_df   = df[df["season"] == ValidationSeason].copy()
test_df  = df[df["season"] == TestSeason].copy()
 
# Game type splits
regseason  = df[df["season_type"] == "SeasonType.REGULAR"].copy()
postseason = df[df["season_type"] == "SeasonType.POSTSEASON"].copy()
ncaa_df    = df[df["tournament"] == "NCAA"].copy()
 
print(f"  Total rows:       {len(df):,}")
print(f"  Training:         {len(train_df):,}   (seasons < {ValidationSeason})")
print(f"  Validation:       {len(val_df):,}   (season {ValidationSeason})")
print(f"  Test:             {len(test_df):,}   (season {TestSeason})")
print(f"  Regular season:   {len(regseason):,}")
print(f"  Postseason:       {len(postseason):,}")
print(f"  NCAA tournament:  {len(ncaa_df):,}")
 
# Standard deviation of point differentials - Used as sigma for converting predicted margins to win probabilities
point_diff_std = train_df["point_diff"].std()
print(f"\n  Point diff std dev: {point_diff_std:.2f} points")

#--------------------------------
# Naive Baseline
#--------------------------------

# Always pick the home team

section("Naive Baseline: Always Pick Home Team")

# Classification
home_win_rate = train_df["home_won"].mean()
naive_pred_class = np.ones(len(df), dtype=int)
naive_prob = np.full(len(df), home_win_rate)

naive_acc = win_accuracy(naive_pred_class, df["home_won"])
naive_logloss = log_loss(naive_prob, df["home_won"])
naive_brier = brier_score(naive_prob, df["home_won"])

# Regression
mean_diff = train_df["point_diff"].mean()
naive_pred_reg = np.full(len(df), mean_diff)

naive_rmse = rmse(naive_pred_reg, df["point_diff"])
naive_mae = mae(naive_pred_reg, df["point_diff"])
naive_r2 = r_squared(naive_pred_reg, df["point_diff"])

naive_reg_prob = point_diff_to_win_prob(naive_pred_reg, sigma=point_diff_std)
naive_reg_logloss = log_loss(naive_reg_prob, df["home_won"])

# Summary
print(f"\nNaive Baseline (Always Pick Home Team):")
print(f"  Home win rate: {home_win_rate:.2%}")
print(f"  Classification - Accuracy: {naive_acc:.4f}, Log Loss: {naive_logloss:.4f}, Brier Score: {naive_brier:.4f}")
print(f"  Regression - RMSE: {naive_rmse:.2f} points, MAE: {naive_mae:.2f} points, R²: {naive_r2:.4f}, Log Loss (from reg): {naive_reg_logloss:.4f}")

#--------------------------------
# Elo Baseline (Classification)
#--------------------------------

# Pick whoever has the higher Elo rating pre-game

section("Elo Baseline: Pick Higher Elo Team (Classification)")

elo_df = df.dropna(subset=["home_elo_pre", "away_elo_pre"]).copy()
elo_df["elo_diff"] = elo_df["home_elo_pre"] - elo_df["away_elo_pre"]
# ELo win prob using logistic function with a scaling factor (commonly 400 in chess)
elo_df["elo_win_prob"] = 1 / (1 + 10 ** ((elo_df["away_elo_pre"] - elo_df["home_elo_pre"]) / 400))

# This is equivalent to picking the team with the higher Elo rating
elo_df["elo_pred"] = (elo_df["elo_win_prob"] >= .5).astype(int)

elo_acc = win_accuracy(elo_df["elo_pred"], elo_df["home_won"])
elo_logloss = log_loss(elo_df["elo_win_prob"], elo_df["home_won"])
elo_brier = brier_score(elo_df["elo_win_prob"], elo_df["home_won"])

# Accuracy across splits
elo_train = elo_df[elo_df["season"] < ValidationSeason]
elo_val   = elo_df[elo_df["season"] == ValidationSeason]
elo_test  = elo_df[elo_df["season"] == TestSeason]

# Summary
print(f"Elo accuracy (all): {elo_acc:.1%}")
print(f" Elo accuracy (train): {win_accuracy(elo_train["elo_pred"], elo_train["home_won"]):.1%}")
print(f" Elo accuracy (validation): {win_accuracy(elo_val["elo_pred"], elo_val["home_won"]):.1%}")
print(f" Elo accuracy (test): {win_accuracy(elo_test["elo_pred"], elo_test["home_won"]):.1%}")
print(f"Elo Log Loss: {elo_logloss:.4f}")
print(f"Elo Brier Score: {elo_brier:.4f}")
print(f" Games with Elo data: {len(elo_df):,}/{len(df):,}")

#-----------------------------------
# Elo Baseline (Regression)
#-----------------------------------

# Use Elo diff to predict point diff
# Simple linear regression

section("Elo Baseline: Predict Point Diff from Elo Difference (Regression)")

# Fit linear regression on training data to avoid leakage
elo_train_clean = elo_train.dropna(subset=["elo_diff", "point_diff"])

slope, intercept, r_val, p_val, std_err = stats.linregress(
    elo_train_clean["elo_diff"],
    elo_train_clean["point_diff"]
)

print(f"Fitted linear regression: point_diff = {intercept:.2f} + {slope:.4f} * elo_diff (R²={r_val**2:.4f})")
print(f" Slope interpretation: For every 100 point increase in Elo difference, we predict a {slope*100:.2f} point increase in margin")
print(f" Intercept interpretation: When Elo difference is 0, we predict a point differential of {intercept:.2f} points. This can be seen as a proxy for home court advantage in the absence of Elo difference.")

# Predict point differentials using the fitted regression
elo_df["elo_pred_diff"] = intercept + slope * elo_df["elo_diff"]

# Recreate splits so they include new columns
elo_train = elo_df[elo_df["season"] < ValidationSeason]
elo_val   = elo_df[elo_df["season"] == ValidationSeason]
elo_test  = elo_df[elo_df["season"] == TestSeason]

elo_rmse = rmse(elo_df["elo_pred_diff"], elo_df["point_diff"])
elo_mae = mae(elo_df["elo_pred_diff"], elo_df["point_diff"])
elo_r2 = r_squared(elo_df["elo_pred_diff"], elo_df["point_diff"])

# Convert to win probabilities
residuals = elo_train["point_diff"] - elo_train["elo_pred_diff"]
elo_sigma = residuals.std()

elo_df["elo_win_prob"] = point_diff_to_win_prob(elo_df["elo_pred_diff"], sigma=elo_sigma)
elo_df["elo_reg_pred"] = (elo_df["elo_win_prob"] >= .5).astype(int)

elo_reg_acc = win_accuracy(elo_df["elo_reg_pred"], elo_df["home_won"])
elo_reg_logloss = log_loss(elo_df["elo_win_prob"], elo_df["home_won"])
elo_reg_brier = brier_score(elo_df["elo_win_prob"], elo_df["home_won"])

print(f"Elo Regression - RMSE: {elo_rmse:.2f} points, MAE: {elo_mae:.2f} points, R²: {elo_r2:.4f}")
print(f"Accuracy: {elo_reg_acc:.1%}, Elo Regression - Log Loss: {elo_reg_logloss:.4f}, Brier Score: {elo_reg_brier:.4f}")

#---------------------------------
# Vegas Spread Baseline
#---------------------------------

# Using the Vegas spreead as a baseline
# NOT being used in the actual model, but serves as a strong benchmark

# Classification
# Negative spread = home team favored, Positive spread = away team favored
# Predict favored team, in case of 0 spread predict home team (tiebreaker)


# Regression
# Spread is already the predicted point margin, so we can directly compare to actual point differentials
# Tells us how accurate the Vegas market is

section("Vegas Spread Baseline")

spread_df = df.dropna(subset=["spread"]).copy()

# Classification
spread_df["spread_pred"] = (spread_df["spread"] < 0).astype(int)  # Home favored if spread < 0

spread_df["spread_pred_diff"] = -spread_df["spread"]  # Flip sign so positive means home favored

# Per-split accuracy
spread_train = spread_df[spread_df["season"] < ValidationSeason]
spread_val   = spread_df[spread_df["season"] == ValidationSeason]
spread_test  = spread_df[spread_df["season"] == TestSeason]

spread_residuals = spread_train["point_diff"] - spread_train["spread_pred_diff"]
spread_sigma = spread_residuals.std()

spread_df["spread_win_prob"] = point_diff_to_win_prob(spread_df["spread_pred_diff"], sigma=spread_sigma)

# Classification metrics
spread_acc = win_accuracy(spread_df["spread_pred"], spread_df["home_won"])
spread_logloss = log_loss(spread_df["spread_win_prob"], spread_df["home_won"])
spread_brier = brier_score(spread_df["spread_win_prob"], spread_df["home_won"])

# Regression metrics
spread_rmse = rmse(spread_df["spread_pred_diff"], spread_df["point_diff"])
spread_mae = mae(spread_df["spread_pred_diff"], spread_df["point_diff"])
spread_r2 = r_squared(spread_df["spread_pred_diff"], spread_df["point_diff"])

# Summary
print(f"Vegas Spread Baseline:")
print(f"  Classification - Accuracy: {spread_acc:.4f}, Log Loss: {spread_logloss:.4f}, Brier Score: {spread_brier:.4f}")
print(f"   Accuracy (train): {win_accuracy(spread_train["spread_pred"], spread_train["home_won"]):.1%}")
print(f"   Accuracy (validation): {win_accuracy(spread_val["spread_pred"], spread_val["home_won"]):.1%}")
print(f"   Accuracy (test): {win_accuracy(spread_test["spread_pred"], spread_test["home_won"]):.1%}")
print(f"  Regression - RMSE: {spread_rmse:.2f} points, MAE: {spread_mae:.2f} points, R²: {spread_r2:.4f}")
print(f"  Games with spread data: {len(spread_df):,}/{len(df):,}")

#--------------------------------
# Charts
#--------------------------------

section("Charts")

# Classification benchmark comparison

fig, axes = plt.subplots(1, 3, figsize=(15, 6))
fig.suptitle("Classification benchmarks - aim to beat", fontsize=14, fontweight="bold")
labels = ["Naive (Home Win)", "Elo (Pick Higher Elo)", "Vegas Spread"]
colors = [Colors["naive"], Colors["neutral"], Colors["highlight"]]

# Accuracy
accs = [naive_acc, elo_acc, spread_acc]
bars = axes[0].bar(labels, accs, color=colors, edgecolor="white")
axes[0].set_ylim(0.5, 1.0)
axes[0].set_title("Win/Loss Accuracy")
axes[0].set_ylabel("Accuracy")
axes[0].axhline(naive_acc, color=Colors["naive"], linestyle="--", label="Floor (Naive)",
                linewidth=1, alpha=.5)
for bar, val in zip(bars, accs):
    axes[0].text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.002, f"{val:.1%}", 
                 ha="center", fontsize=12, fontweight="bold")

# Log Loss
loglosses = [naive_logloss, elo_logloss, spread_logloss]
bars = axes[1].bar(labels, loglosses, color=colors, edgecolor="white")
axes[1].set_title("Log Loss (lower = better)")
axes[1].set_ylabel("Log Loss")
for bar, val in zip(bars, loglosses):
    axes[1].text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.002, f"{val:.4f}", 
                 ha="center", fontsize=12, fontweight="bold")
    
# Brier Score
briers = [naive_brier, elo_brier, spread_brier]
bars = axes[2].bar(labels, briers, color=colors, edgecolor="white")
axes[2].set_title("Brier Score (lower = better)")
axes[2].set_ylabel("Brier Score")
for bar, val in zip(bars, briers):
    axes[2].text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.002, f"{val:.4f}", 
                 ha="center", fontsize=12, fontweight="bold")
    
plt.tight_layout()
notes = (
    f"Naive: Acc = {naive_acc:.1%}, Log Loss = {naive_logloss:.4f}, Brier = {naive_brier:.4f}\n"
    f"Elo: Acc = {elo_acc:.1%}, Log Loss = {elo_logloss:.4f}, Brier = {elo_brier:.4f}\n"
    f"Vegas Spread: Acc = {spread_acc:.1%}, Log Loss = {spread_logloss:.4f}, Brier = {spread_brier:.4f}\n"
)

save(fig, "classification_benchmarks.png", notes)

# Regression benchmark comparison

fig, axes = plt.subplots(1, 3, figsize=(15, 6))
fig.suptitle("Regression benchmarks - Point diff prediction accuracy", fontsize=14, fontweight="bold")
reg_labels = ["Naive (Mean Point Diff)", "Elo (Linear Regression)", "Vegas Spread"]

# RMSE
rmses = [naive_rmse, elo_rmse, spread_rmse]
bars = axes[0].bar(reg_labels, rmses, color=colors, edgecolor="white")
axes[0].set_title("RMSE (lower = better)")
axes[0].set_ylabel("Points")
for bar, val in zip(bars, rmses):
    axes[0].text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.2, f"{val:.2f}", 
                 ha="center", fontsize=12, fontweight="bold")
    
# MAE
maes = [naive_mae, elo_mae, spread_mae]
bars = axes[1].bar(reg_labels, maes, color=colors, edgecolor="white")
axes[1].set_title("MAE (lower = better)")
axes[1].set_ylabel("Points")
for bar, val in zip(bars, maes):
    axes[1].text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.2, f"{val:.2f}", 
                 ha="center", fontsize=12, fontweight="bold")
    
# R2
r2s = [naive_r2, elo_r2, spread_r2]
bars = axes[2].bar(reg_labels, r2s, color=colors, edgecolor="white")
axes[2].set_title("R2 (higher = better)")
axes[2].set_ylabel("R2")
for bar, val in zip(bars, r2s):
    axes[2].text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.002, f"{val:.4f}", 
                 ha="center", fontsize=12, fontweight="bold")
    
plt.tight_layout()
notes = (
    f"Naive: RMSE = {naive_rmse:.2f}, MAE = {naive_mae:.2f}, R² = {naive_r2:.4f}\n"
    f"Elo Regression: RMSE = {elo_rmse:.2f}, MAE = {elo_mae:.2f}, R² = {elo_r2:.4f}\n"
    f"Vegas Spread: RMSE = {spread_rmse:.2f}, MAE = {spread_mae:.2f}, R² = {spread_r2:.4f}\n"
)
save(fig, "regression_benchmarks.png", notes)

# Accuracy by season

elo_df["elo_correct"] = (elo_df["elo_pred"] == elo_df["home_won"]).astype(int)
elo_by_season = (
    elo_df.groupby("season")["elo_correct"]
    .mean()
    .reset_index(name="elo_acc")
)
naive_by_season = (
    df.groupby("season")["home_won"]
    .mean()
    .reset_index(name="naive_acc")
)

spread_df["spread_correct"] = (spread_df["spread_pred"] == spread_df["home_won"]).astype(int)
spread_by_season = (
    spread_df.groupby("season")["spread_correct"]
    .mean()
    .reset_index(name="spread_acc")
)

season_acc = (
    naive_by_season[["season", "naive_acc"]]
    .merge(elo_by_season, on="season", how="left")
    .merge(spread_by_season, on="season", how="left")
)

fig, ax = plt.subplots(figsize=(12, 6))
ax.plot(season_acc["season"], season_acc["naive_acc"], label="Naive (Home Win)",
        color=Colors["naive"], marker="o", linewidth=2)
ax.plot(season_acc["season"], season_acc["elo_acc"], label="Elo (Pick Higher Elo)",
        color=Colors["neutral"], marker="o", linewidth=2)
ax.plot(season_acc["season"], season_acc["spread_acc"], label="Vegas Spread",
        color=Colors["highlight"], marker="o", linewidth=2)
ax.axvline(ValidationSeason, color=Colors["win"], linestyle="--", label=F"Validation ({ValidationSeason})", linewidth=1)
ax.axvline(TestSeason, color=Colors["loss"], linestyle="--", label=F"Test ({TestSeason})", linewidth=1)
ax.set_xlabel("Season")
ax.set_ylabel("Accuracy")
ax.set_title("Benchmark Accuracy by Season")
ax.legend()
ax.set_ylim(0.5, 1.0)
ax.tick_params(axis="x", rotation=45)
plt.tight_layout()
save(fig, "accuracy_by_season.png")

# Elo calibration

# Shows whether Elo win probabilities are trustworthy

elo_df["prob_bucket"] = pd.cut(
    elo_df["elo_win_prob"], 
    bins = [0, 0.35, 0.45, 0.5, 0.55, 0.65, 1],
    labels = ["<35%", "35-45%", "45-50%", "50-55%", "55-65%", ">65%"]
)

calibration = (
    elo_df.groupby("prob_bucket", observed=True)
    .agg(
        mean_pred = ("elo_win_prob", "mean"),
        actual_win_rate = ("home_won", "mean"),
        count = ("game_id", "count")
    )
    .reset_index()
    .dropna()
)

fig, axes = plt.subplots(1, 2, figsize=(15, 6))
fig.suptitle("Elo Probability Calibration", fontsize=14, fontweight="bold")
axes[0].scatter(
    calibration["mean_pred"], 
    calibration["actual_win_rate"], 
    s=calibration["count"]/3, 
    color = Colors["neutral"], alpha = .5, zorder = 5,
    edgecolor = "white"
)
axes[0].plot([0, 1], [0, 1], color="gray", linestyle="--", linewidth=1, label="Perfect Calibration")
axes[0].set_xlabel("Mean Predicted Win Probability")
axes[0].set_ylabel("Actual Win Rate")
axes[0].set_title("Elo Calibration Curve (points on diagonal are well calibrated)")
axes[0].legend()

axes[1].hist(elo_df["elo_win_prob"], bins=40, color=Colors["neutral"], edgecolor="white", linewidth=.3)
axes[1].axvline(.5, color=Colors["loss"], linestyle="--", label="Coin Flip Line", linewidth=1)
axes[1].set_xlabel("Elo Win Probability")
axes[1].set_ylabel("Games")
axes[1].set_title("Distribution of Elo Win Probabilities")
axes[1].legend()

plt.tight_layout()
save(fig, "elo_calibration.png")

# Tournament analysis

tourn_elo = elo_df[elo_df["season_type"] == "SeasonType.POSTSEASON"].copy()
ncaa_elo = elo_df[elo_df["tournament"] == "NCAA"].copy()

tourn_elo_acc = win_accuracy(tourn_elo["elo_pred"], tourn_elo["home_won"]) if len(tourn_elo) > 0 else 0
ncaa_elo_acc = win_accuracy(ncaa_elo["elo_pred"], ncaa_elo["home_won"])  if len(ncaa_elo) > 0 else 0

fig, axes = plt.subplots(1, 2, figsize=(15, 6))
fig.suptitle("Tourney Specific Benchmark Performance", fontsize=14, fontweight="bold")

# Accuracy by game type
categories = ["All", "Postseason", "NCAA"]
elo_accs = [elo_acc, tourn_elo_acc, ncaa_elo_acc]
naive_accs = [
    df["home_won"].mean(),
    postseason["home_won"].mean(),
    ncaa_df["home_won"].mean() if len(ncaa_df) > 0 else 0
]
x = np.arange(len(categories))
width = 0.35
b1 = axes[0].bar(x - width/2, naive_accs, width, label="Naive (Home Win)", 
                 color=Colors["naive"], edgecolor="white")
b2 = axes[0].bar(x + width/2, elo_accs, width, label="Elo (Pick Higher Elo)", 
                 color=Colors["neutral"], edgecolor="white")
axes[0].set_xticks(x)
axes[0].set_xticklabels(categories)
axes[0].set_ylabel("Accuracy")
axes[0].set_title("Accuracy by Game Type")
axes[0].legend()
axes[0].axhline(.5, color="gray", linestyle="--", label="Coin Flip Line", linewidth=1)
for bar in list(b1) + list(b2):
    axes[0].text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.002, f"{bar.get_height():.1%}", 
                 ha="center", fontsize=12, fontweight="bold")
    
# Elo diff distribution: Regular v Postseason
regseason["elo_diff"] = regseason["home_elo_pre"] - regseason["away_elo_pre"]
postseason["elo_diff"] = postseason["home_elo_pre"] - postseason["away_elo_pre"]

axes[1].hist(regseason["elo_diff"].dropna(), bins=35, alpha=0.5, label="Regular Season", 
             color=Colors["win"], edgecolor="white", density=True)
axes[1].hist(postseason["elo_diff"].dropna(), bins=35, alpha=0.5, label="Postseason", 
             color=Colors["loss"], edgecolor="white", density=True)
axes[1].axvline(0, color="black", linestyle="--", linewidth=1)
axes[1].set_xlabel("Elo Difference (Home - Away)")
axes[1].set_title("Distribution of Elo Differences: Regular vs Postseason")
axes[1].legend()

plt.tight_layout()
notes = (
    f"Elo Accuracy - All: {elo_acc:.1%}, Postseason: {tourn_elo_acc:.1%}, NCAA: {ncaa_elo_acc:.1%}\n"
)
save(fig, "tournament_benchmarks.png", notes)

#--------------------------------
# Final Summary
#--------------------------------

section("Final Summary")
summary = f"""
-----------------------------------------
Benchmaark Summary - Performance Ladders
-----------------------------------------

Classifcation (win/loss):
- Naive (Home Win): Accuracy = {naive_acc:.1%}, Log Loss = {naive_logloss:.4f}, Brier Score = {naive_brier:.4f}
- Elo (Pick Higher Elo): Accuracy = {elo_acc:.1%}, Log Loss = {elo_logloss:.4f}, Brier Score = {elo_brier:.4f}
- Vegas Spread: Accuracy = {spread_acc:.1%}, Log Loss = {spread_logloss:.4f}, Brier Score = {spread_brier:.4f}

Regression (point diff):
- Naive (Mean Point Diff): RMSE = {naive_rmse:.2f} points, MAE = {naive_mae:.2f} points, R² = {naive_r2:.4f}
- Elo (Linear Regression): RMSE = {elo_rmse:.2f} points, MAE = {elo_mae:.2f} points, R² = {elo_r2:.4f}
- Vegas Spread: RMSE = {spread_rmse:.2f} points, MAE = {spread_mae:.2f} points, R² = {spread_r2:.4f}

Elo Accuracy by Split:
- Training: {win_accuracy(elo_train["elo_pred"], elo_train["home_won"]):.1%}
- Validation: {win_accuracy(elo_val["elo_pred"], elo_val["home_won"]):.1%}
- Test: {win_accuracy(elo_test["elo_pred"], elo_test["home_won"]):.1%}

Tournament Specific:
- Postseason: {tourn_elo_acc:.1%}
- NCAA: {ncaa_elo_acc:.1%}

-----------------------------------------
What to Beat:

- Classifcation accuracy > {elo_acc:.1%} (Beat Elo, model is learning signal beyond just team strength)
- Classification accuracy > {spread_acc:.1%} (Beat Vegas Spread, model is strong enough to compete with market predictions)

- RMSE < {elo_rmse:.2f} points (Beat Elo regression, model is better at predicting point differentials than a simple linear regression on Elo)
- RMSE < {spread_rmse:.2f} points (Beat Vegas Spread, model is better at predicting point differentials than the Vegas market)

- R2 > {elo_r2:.4f} (Model explains more variance in point differentials than Elo regression)
- Log loss < {elo_logloss:.4f} (Model's probabilistic predictions are more accurate than those derived from Elo regression)

Tournament target:

- NCAA accuracy > {ncaa_elo_acc:.1%} (Model performs well in the most important games of the season)
"""

print(summary)
write_summary(summary)

print("\nBenchmarking complete. Time for the model training!")