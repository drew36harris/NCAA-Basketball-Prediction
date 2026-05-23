"""
Trains multiple regression models to predict point differentials in basketball games,
before converting those predictions to win probabilities.

Models:
Linear Regression - interpretable baseline
Ridge Regression - L2 Regularized
Lasso Regression - L1 Regularizeed and feature selection
Random Forest - Non-linear ensemble method robust to noise
XGBoost - gradient boosting

Model Evaluation:
Regression - RMSE, MAE, R2
Classification - Accuracy, Log Loss, Brier Score

Win Proability Conversion:
Predicted differential -> Win prob via normal CDF
    sigma = residual std from training set predictions
    As sigma increases, confidence decreases
    Same approach as baseline script

Temporal Split:
Train <2024
Validate = 2024
Test = 2025

Hyperparameter Tuning:
Random Forest and XGBoost use Optuna with TimeSeriesSplit CV
Linear models use built in cross validation (RidgeCV, LassoCV)
Early stopping on XGBoost prevents overfitting
"""

import json
import joblib
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from scipy import stats
from datetime import datetime
 
from sklearn.linear_model import LinearRegression, RidgeCV, LassoCV, ElasticNetCV
from sklearn.linear_model import RidgeCV as MetaLearner
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import TimeSeriesSplit, cross_val_score
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
 
import xgboost as xgb
import optuna
optuna.logging.set_verbosity(optuna.logging.WARNING)
 
warnings.filterwarnings("ignore")

#--------------------------------
# Directories and Config
#--------------------------------

BaseDirectory = Path(r"C:\Users\Drew\Documents\PersonalProjects\BasketballPrediction")
TrainingCSV = BaseDirectory / "data" / "processed" / "training_rows.csv"
FeaturesTxt = BaseDirectory / "eda_output" / "model_features.txt"
OutputDirectory = BaseDirectory / "model_output"
OutputDirectory.mkdir(parents=True, exist_ok=True)

ValidationSeason = 2024
TestSeason = 2025

# Optuna Tuning Trials
# More trials = better performance but longer runtime
n_optuna_trials = 150

# TimeSeriesSplit folds for cross validation during tuning
n_splits = 5

# Benchmark targets (from benchmarking script)
benchmark_targets = {
    "naive_rmse": 14.39,
    "elo_rmse": 12.34, 
    "spread_rmse": 11.30,
    "naive_acc": 0.661,
    "elo_acc": 0.673,
    "spread_acc": 0.744, 
    "elo_r2": 0.2639,
    "spread_r2": 0.3975,
    "ncaa_elo_acc": 0.672
}

plt.style.use("seaborn-v0_8-whitegrid")
Colors = {
    "win": "#1D9E75", # Home Win = Teal
    "loss": "#D85A30",  # Home Loss = Orange
    "neutral": "#7F77DD",  # Neutral = Purple
    "highlight": "#EF9F27", # Callout = Amber
    "naive": "#AAAAAA" # Naive = Gray
}

#--------------------------------
# Helpers
#--------------------------------

def save_fig(fig, filename, notes=None):
    """Save figure and optionally append notes to the report"""
    path = OutputDirectory / filename
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {filename}")
    if notes:
        with open(OutputDirectory / "model_report.txt", "a") as f:
            f.write(f"\n{"="*60}\n{filename}\n{"="*60}\n")
            f.write(notes + "\n")

def section(title):
    """Terminal Section Divider"""
    print(f"\n── {title} {"─" * (50 - len(title))}")


def write_summary(text):
    """Append plain text to the summary report."""
    with open(OutputDirectory / "model_summary.txt", "a") as f:
        f.write(text + "\n")

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

def point_diff_to_win_prob_dynamic(point_diff_predictions, sigmas):
    """
    Convert predicted point spreads into win probabilities
    using game-specific uncertainty estimates.
    """

    preds  = np.array(point_diff_predictions, dtype=float)
    sigmas = np.array(sigmas, dtype=float)

    return stats.norm.cdf(preds / sigmas)

def evaluate_model(name, y_pred, y_true, y_true_class, sigma=None):
    """
    Compute all metrics for a model's predictions.
    Returns a dict with regression and classification metrics.
    """
    # Compute sigma from residuals if not provided
    if sigma is None:
        sigma = (y_true - y_pred).std()
 
    # Regression metrics
    rmse_val = np.sqrt(mean_squared_error(y_true, y_pred))
    mae_val  = mean_absolute_error(y_true, y_pred)
    r2_val   = r2_score(y_true, y_pred)
 
    # Convert to win probability
    win_probs = point_diff_to_win_prob(y_pred, sigma)
    win_preds = (win_probs > 0.5).astype(int)
 
    # Classification metrics
    acc_val    = (win_preds == y_true_class).mean()
    ll_val     = log_loss(win_probs, y_true_class)
    brier_val  = brier_score(win_probs, y_true_class)
 
    return {
        "name":        name,
        "rmse":        rmse_val,
        "mae":         mae_val,
        "r2":          r2_val,
        "accuracy":    acc_val,
        "log_loss":    ll_val,
        "brier":       brier_val,
        "sigma":       sigma,
        "win_probs":   win_probs,
        "win_preds":   win_preds,
        "predictions": y_pred,
    }
 
 
def beats_benchmark(metric_name, value, higher_is_better=True):
    """Return a string indicator of whether this beats the Elo benchmark."""
    benchmarks = {
        "rmse":     (benchmark_targets["elo_rmse"],  False),
        "r2":       (benchmark_targets["elo_r2"],    True),
        "accuracy": (benchmark_targets["elo_acc"],   True),
        "log_loss": (benchmark_targets["elo_acc"],   False),
    }
    if metric_name not in benchmarks:
        return ""
    target, higher = benchmarks[metric_name]
    if higher:
        return "Beats Elo" if value > target else "Below Elo"
    else:
        return "Beats Elo" if value < target else "Below Elo"
    
def make_linear_pipeline(model):
    """
    Standard pipeline for linear models:
    - median imputation
    - feature scaling
    - model fitting

    Scaling is critical for Ridge/Lasso/ElasticNet because
    regularization penalizes coefficient magnitude.
    """

    return Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("model", model)
    ])
    
#--------------------------------
# Load Data
#--------------------------------

section("Loading Data")

# Clear summary file
with open(OutputDirectory / "model_summary.txt", "w") as f:
    f.write(f"Basketball Model Training Summary\n")
    f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
 
df = pd.read_csv(TrainingCSV)
df["date"] = pd.to_datetime(df["date"])
 
# Load feature list from EDA output
# Falls back to auto-detection if file not found
if FeaturesTxt.exists():
    with open(FeaturesTxt) as f:
        lines = f.readlines()
    features = [
        l.strip() for l in lines
        if l.strip() and not l.startswith("#") and l.strip() in df.columns
    ]
    duplicate_features = [f for f in features if features.count(f) > 1]
    duplicate_features = sorted(set(duplicate_features), key=features.index)
    if duplicate_features:
        print(f"  WARNING: duplicate feature names found in {FeaturesTxt.name}: {duplicate_features}")
    features = list(dict.fromkeys(features))
    print(f"  Loaded {len(features)} features from {FeaturesTxt.name}")
else:
    # Auto-detect: use all diff_ features plus key raw features
    # This is the fallback if EDA hasn't been run yet
    print("  model_features.txt not found — auto-detecting features")
    exclude = {
        "game_id", "season", "date", "home_team", "away_team",
        "home_team_id", "away_team_id", "season_type", "tournament",
        "venue", "city", "state", "home_points", "away_points",
        "point_diff", "home_won", "home_winner",
        "home_elo_post", "away_elo_post",
        "home_games_played", "away_games_played",
        "spread", "over_under", "num_providers",
        "diff_srs", "diff_orb_pct",   # dropped in EDA
        "conference_game", "excitement", "attendance",
    }
    features = [
        c for c in df.select_dtypes(include=[np.number]).columns
        if c not in exclude
    ]
    features = list(dict.fromkeys(features))
    print(f"  Auto-detected {len(features)} features")
 
# Targets
RegTarget   = "point_diff"    # regression target
ClassTarget = "home_won"      # classification target (derived)
 
# Temporal splits
train_df = df[df["season"] <  ValidationSeason].copy()
val_df   = df[df["season"] == ValidationSeason].copy()
test_df  = df[df["season"] == TestSeason].copy()
 
# Tournament subset for tournament-specific evaluation
ncaa_val  = val_df[val_df["tournament"] == "NCAA"].copy()
ncaa_test = test_df[test_df["tournament"] == "NCAA"].copy() if len(test_df) > 0 else pd.DataFrame()
 
print(f"  Total rows:      {len(df):,}")
print(f"  Train:           {len(train_df):,}  (seasons < {ValidationSeason})")
print(f"  Validation:      {len(val_df):,}  (season {ValidationSeason})")
print(f"  Test:            {len(test_df):,}  (season {TestSeason})")
print(f"  NCAA val games:  {len(ncaa_val):,}")
print(f"  Features:        {len(features)}")

#--------------------------------
# Prepare Feature Matrices
#--------------------------------

section("Preparing Feature Matrices")
 
def prepare_X_y(data, features, target):
    """
    Extract feature matrix X and target vector y from a DataFrame.
    Only returns rows where all values are present.
    """
    available = [f for f in features if f in data.columns]
    subset    = data[available + [target]].dropna(subset=[target])
    X = subset[available]
    y = subset[target]
    return X, y, subset.index
 
 
x_train, y_train, train_idx = prepare_X_y(train_df, features, RegTarget)
x_val, y_val, val_idx   = prepare_X_y(val_df, features, RegTarget)
x_test, y_test, test_idx  = prepare_X_y(test_df, features, RegTarget) if len(test_df) > 0 else (pd.DataFrame(), pd.Series(), [])
 
# Classification targets aligned to same indices
y_train_class = train_df.loc[train_idx, ClassTarget]
y_val_class   = val_df.loc[val_idx,     ClassTarget]
y_test_class  = test_df.loc[test_idx,   ClassTarget] if len(test_df) > 0 else pd.Series()
 
# Sigma from training residuals — used for all win probability conversions
# Computed here from a simple mean prediction to get scale, updated per model
train_sigma = y_train.std()   # rough initial estimate
 
print(f"  Train X shape:   {x_train.shape}")
print(f"  Val X shape:     {x_val.shape}")
print(f"  Features used:   {x_train.shape[1]}")
print(f"  Missing values in train: {x_train.isnull().sum().sum():,}")
print(f"  Missing values in val:   {x_val.isnull().sum().sum():,}")

#--------------------------------
# Imputation
#--------------------------------

# Strategy: median imputation — robust to outliers, doesn't create crazy values far from the distribution center.
# We fit the imputer on training data only — no leakage.
 
section("Imputation")
 
imputer = SimpleImputer(strategy="median")
imputer.fit(x_train)
 
x_train_imp = pd.DataFrame(
    imputer.transform(x_train),
    columns=x_train.columns,
    index=x_train.index
)
x_val_imp = pd.DataFrame(
    imputer.transform(x_val),
    columns=x_val.columns,
    index=x_val.index
)
if len(x_test) > 0:
    X_test_imp = pd.DataFrame(
        imputer.transform(x_test),
        columns=x_test.columns,
        index=x_test.index
    )
 
print(f"  Imputer fitted on training data")
print(f"  Strategy: median (robust to outliers)")
print(f"  Missing after imputation: {x_train_imp.isnull().sum().sum()}")

#--------------------------------
# Model 1 - Linear Regression
#--------------------------------

# Simplest model, coefficients say how much each feature is worth
section("Linear Model")

lr_model = make_linear_pipeline(LinearRegression())
lr_model.fit(x_train, y_train)

lr_pred_train = lr_model.predict(x_train)
lr_pred_val = lr_model.predict(x_val)

# Compute sigma
lr_sigma = (y_train - lr_pred_train).std()

# Evaluate
lr_train_metrics = evaluate_model(
    "Linear Regression (train)", lr_pred_train, y_train, y_train_class, lr_sigma
)
lr_val_metrics = evaluate_model(
    "Linear Regression (val)", lr_pred_val, y_val, y_val_class, lr_sigma
)
 
print(f"    Train   RMSE={lr_train_metrics['rmse']:.2f}  "
      f"R2={lr_train_metrics['r2']:.4f}  "
      f"Acc={lr_train_metrics['accuracy']:.1%}")
print(f"    Val     RMSE={lr_val_metrics['rmse']:.2f}  "
      f"R2={lr_val_metrics['r2']:.4f}  "
      f"Acc={lr_val_metrics['accuracy']:.1%}  "
      f"{beats_benchmark('rmse', lr_val_metrics['rmse'])}")
 
# Top coefficients — what is the model learning?
lr_coefs = pd.Series(lr_model.named_steps["model"].coef_, index=x_train.columns)
lr_coefs = lr_coefs.sort_values(key=abs, ascending=False)
print(f"\n  Top 10 coefficients (points per unit):")
print(lr_coefs.head(10).to_string())

#--------------------------------
# Model 2 - Ridge Regression
#--------------------------------

# Like near regression but penalizes large coefficients
# Helps with correlated features
# RidgeCV automatically selects alpha

section("Ridge Regression")

alphas = [0.001, 0.01, 0.1, 1.0, 10.0, 100.0, 1000.0]
 
ridge_model = make_linear_pipeline(RidgeCV(alphas=alphas, cv=5))
ridge_model.fit(x_train, y_train)
 
ridge_pred_train = ridge_model.predict(x_train)
ridge_pred_val   = ridge_model.predict(x_val)
ridge_sigma      = (y_train - ridge_pred_train).std()
 
ridge_train_metrics = evaluate_model(
    "Ridge (train)", ridge_pred_train, y_train, y_train_class, ridge_sigma
)
ridge_val_metrics = evaluate_model(
    "Ridge (val)", ridge_pred_val, y_val, y_val_class, ridge_sigma
)
 
print(f"  Best alpha: {ridge_model.named_steps["model"].alpha_:.4f}")
print(f"  Train  RMSE={ridge_train_metrics['rmse']:.2f}  "
      f"R2={ridge_train_metrics['r2']:.4f}  "
      f"Acc={ridge_train_metrics['accuracy']:.1%}")
print(f"  Val    RMSE={ridge_val_metrics['rmse']:.2f}  "
      f"R2={ridge_val_metrics['r2']:.4f}  "
      f"Acc={ridge_val_metrics['accuracy']:.1%}  "
      f"{beats_benchmark('rmse', ridge_val_metrics['rmse'])}")


#--------------------------------
# Model 3 - Lasso Regression
#--------------------------------

# Similar to ridge where it penalizes large coefficients
# Performs feature selection

#section("Lasso Regression")

#lasso_model = make_linear_pipeline(LassoCV(cv=5, max_iter=10000, random_state=42))
#lasso_model.fit(x_train, y_train)
 
#lasso_pred_train = lasso_model.predict(x_train)
#lasso_pred_val   = lasso_model.predict(x_val)
#lasso_sigma      = (y_train - lasso_pred_train).std()
 
#lasso_train_metrics = evaluate_model(
#    "Lasso (train)", lasso_pred_train, y_train, y_train_class, lasso_sigma
#)
#lasso_val_metrics = evaluate_model(
#    "Lasso (val)", lasso_pred_val, y_val, y_val_class, lasso_sigma
#)
 
# Which features did Lasso zero out
#lasso_coefs    = pd.Series(lasso_model.named_steps["model"].coef_, index=x_train.columns)
#zeroed_features = lasso_coefs[lasso_coefs == 0].index.tolist()
#kept_features   = lasso_coefs[lasso_coefs != 0].index.tolist()
 
#print(f"  Best alpha: {lasso_model.named_steps["model"].alpha_:.6f}")
#print(f"  Features kept:   {len(kept_features)}")
#print(f"  Features zeroed: {len(zeroed_features)}")
#if zeroed_features:
#    print(f"  Zeroed out: {zeroed_features[:10]}")
#print(f"  Train  RMSE={lasso_train_metrics['rmse']:.2f}  "
#      f"R²={lasso_train_metrics['r2']:.4f}  "
#      f"Acc={lasso_train_metrics['accuracy']:.1%}")
#print(f"  Val    RMSE={lasso_val_metrics['rmse']:.2f}  "
#      f"R²={lasso_val_metrics['r2']:.4f}  "
#      f"Acc={lasso_val_metrics['accuracy']:.1%}  "
#      f"{beats_benchmark('rmse', lasso_val_metrics['rmse'])}")

#--------------------------------
# Model 4 - Elastic Net
#--------------------------------

# Combines Lasso and Ridge
# Best of both worlds

section("Elastic Net")

elastic_model = make_linear_pipeline(ElasticNetCV(
    l1_ratio=[0.1, 0.3, 0.5, 0.7, 0.9, 0.95, 1.0],  # search ratio of L1 vs L2
    alphas=[0.001, 0.01, 0.1, 1.0, 10.0],
    cv=5,
    max_iter=10000,
    random_state=42
))
elastic_model.fit(x_train, y_train)
 
elastic_pred_train = elastic_model.predict(x_train)
elastic_pred_val   = elastic_model.predict(x_val)
elastic_sigma      = (y_train - elastic_pred_train).std()
 
elastic_train_metrics = evaluate_model(
    "Elastic Net (train)", elastic_pred_train, y_train, y_train_class, elastic_sigma
)
elastic_val_metrics = evaluate_model(
    "Elastic Net (val)", elastic_pred_val, y_val, y_val_class, elastic_sigma
)
 
elastic_coefs   = pd.Series(elastic_model.named_steps["model"].coef_, index=x_train.columns)
elastic_zeroed  = elastic_coefs[elastic_coefs == 0].index.tolist()
elastic_kept    = elastic_coefs[elastic_coefs != 0].index.tolist()
 
print(f"  Best l1_ratio: {elastic_model.named_steps["model"].l1_ratio_:.3f}")
print(f"  Best alpha:    {elastic_model.named_steps["model"].alpha_:.6f}")
print(f"  Features kept:   {len(elastic_kept)}")
print(f"  Features zeroed: {len(elastic_zeroed)}")
print(f"  Train  RMSE={elastic_train_metrics['rmse']:.2f}  "
      f"R2={elastic_train_metrics['r2']:.4f}  "
      f"Acc={elastic_train_metrics['accuracy']:.1%}" )
print(f"  Val    RMSE={elastic_val_metrics['rmse']:.2f}  "
      f"R2={elastic_val_metrics['r2']:.4f}  "
      f"Acc={elastic_val_metrics['accuracy']:.1%}  "
      f"{beats_benchmark('rmse', elastic_val_metrics['rmse'])}")

#--------------------------------
# Model 5 - Random Forest
#--------------------------------

# A forest of decision trees, each trained on random subset of data and features
# Predictions averaged
# Captures non-linear relationships, handles feature interactions, robust to outliers, built in feature importance
# Using Optuna to tune the key hyperparameters, to respect temporal integrity in CV

section("Random Forest")

print("  Tuning with Optuna (TimeSeriesSplit CV)...")
 
def rf_objective(trial):
    """
    Objective function for Optuna hyperparameter search.
    Returns negative RMSE (Optuna minimizes, lower RMSE = better).
    """
    params = {
        "n_estimators":      trial.suggest_int("n_estimators", 100, 500),
        "max_depth":         trial.suggest_int("max_depth", 3, 12),
        "min_samples_split": trial.suggest_int("min_samples_split", 2, 20),
        "min_samples_leaf":  trial.suggest_int("min_samples_leaf", 1, 10),
        "max_features":      trial.suggest_float("max_features", 0.3, 1.0),
        "random_state":      42,
        "n_jobs":            -1,
    }
 
    # TimeSeriesSplit — respects temporal ordering
    # Each fold trains on earlier data and validates on later data
    tscv   = TimeSeriesSplit(n_splits=n_splits)
    scores = []
 
    for train_idx, val_idx in tscv.split(x_train_imp):
        x_fold_train = x_train_imp.iloc[train_idx]
        x_fold_val   = x_train_imp.iloc[val_idx]
        y_fold_train = y_train.iloc[train_idx]
        y_fold_val   = y_train.iloc[val_idx]
 
        model = RandomForestRegressor(**params)
        model.fit(x_fold_train, y_fold_train)
        pred  = model.predict(x_fold_val)
        rmse  = np.sqrt(mean_squared_error(y_fold_val, pred))
        scores.append(rmse)
 
    return np.mean(scores)
 
rf_study = optuna.create_study(direction="minimize")
rf_study.optimize(rf_objective, n_trials=n_optuna_trials, show_progress_bar=False)
 
# Train final model with best parameters on full training set
rf_best_params = rf_study.best_params
rf_best_params["random_state"] = 42
rf_best_params["n_jobs"]       = -1
 
rf_model = RandomForestRegressor(**rf_best_params)
rf_model.fit(x_train_imp, y_train)
 
rf_pred_train = rf_model.predict(x_train_imp)
rf_pred_val   = rf_model.predict(x_val_imp)
rf_sigma      = (y_train - rf_pred_train).std()
 
rf_train_metrics = evaluate_model(
    "Random Forest (train)", rf_pred_train, y_train, y_train_class, rf_sigma
)
rf_val_metrics = evaluate_model(
    "Random Forest (val)", rf_pred_val, y_val, y_val_class, rf_sigma
)
 
print(f"  Best params: {rf_best_params}")
print(f"  Best CV RMSE: {rf_study.best_value:.2f}")
print(f"  Train  RMSE={rf_train_metrics['rmse']:.2f}  "
      f"R²={rf_train_metrics['r2']:.4f}  "
      f"Acc={rf_train_metrics['accuracy']:.1%}")
print(f"  Val    RMSE={rf_val_metrics['rmse']:.2f}  "
      f"R²={rf_val_metrics['r2']:.4f}  "
      f"Acc={rf_val_metrics['accuracy']:.1%}  "
      f"{beats_benchmark('rmse', rf_val_metrics['rmse'])}")

#--------------------------------
# Model 6 - XGBoost
#--------------------------------

# Gradient boosting - builds trees sequentially, each correcting the previous one's problems
# Natively handles missing values, built in regularization, early stopping prevents overfitting automatically
# Tune with Optuna

section("XGBoost")

print("  Tuning with Optuna + early stopping (TimeSeriesSplit CV)...")
 
def xgb_objective(trial):
    """
    XGBoost objective for Optuna.
 
    PARAMETERS BEING TUNED:
        learning_rate    -> how much each tree contributes
                           lower = more conservative, needs more trees
        max_depth        -> how deep each tree grows
                           deeper = more complex interactions, more overfit risk
        subsample        -> fraction of rows sampled per tree
                           <1.0 adds randomness -> helps generalization
        colsample_bytree -> fraction of features per tree
                           <1.0 adds randomness -> helps generalization
        min_child_weight -> min samples needed to split a node
                           higher = more conservative
        reg_alpha        -> L1 regularization (drives some weights to 0)
        reg_lambda       -> L2 regularization (shrinks weights toward 0)
 
    n_estimators is handled by early stopping — we set it high (1000)
    and let the validation performance tell us when to stop.
    """
    params = {
        "learning_rate":     trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
        "max_depth":         trial.suggest_int("max_depth", 3, 7),
        "subsample":         trial.suggest_float("subsample", 0.6, 1.0),
        "colsample_bytree":  trial.suggest_float("colsample_bytree", 0.6, 1.0),
        "min_child_weight":  trial.suggest_int("min_child_weight", 1, 10),
        "reg_alpha":         trial.suggest_float("reg_alpha", 1e-8, 1.0, log=True),
        "reg_lambda":        trial.suggest_float("reg_lambda", 1e-8, 1.0, log=True),
        "n_estimators":      1000,
        "early_stopping_rounds": 50,
        "eval_metric":       "rmse",
        "random_state":      42,
        "n_jobs":            -1,
    }
 
    tscv   = TimeSeriesSplit(n_splits=n_splits)
    scores = []
 
    for train_idx, val_idx in tscv.split(x_train_imp):
        x_fold_train = x_train_imp.iloc[train_idx]
        x_fold_val   = x_train_imp.iloc[val_idx]
        y_fold_train = y_train.iloc[train_idx]
        y_fold_val   = y_train.iloc[val_idx]
 
        model = xgb.XGBRegressor(**params, verbosity=0)
        model.fit(
            x_fold_train, y_fold_train,
            eval_set=[(x_fold_val, y_fold_val)],
            verbose=False
        )
        pred  = model.predict(x_fold_val)
        rmse  = np.sqrt(mean_squared_error(y_fold_val, pred))
        scores.append(rmse)
 
    return np.mean(scores)
 
xgb_study = optuna.create_study(direction="minimize")
xgb_study.optimize(xgb_objective, n_trials=n_optuna_trials, show_progress_bar=False)
 
# Train final XGBoost on full training set using best params
# Use validation set for early stopping on the final model
xgb_best_params = xgb_study.best_params
xgb_best_params.update({
    "n_estimators":          1000,
    "early_stopping_rounds": 50,
    "eval_metric":           "rmse",
    "random_state":          42,
    "n_jobs":                -1,
})
 
xgb_model = xgb.XGBRegressor(**xgb_best_params, verbosity=0)
xgb_model.fit(
    x_train_imp, y_train,
    eval_set=[(x_val_imp, y_val)],
    verbose=False
)
 
print(f"  Best iteration: {xgb_model.best_iteration}")
print(f"  Best CV RMSE:   {xgb_study.best_value:.2f}")
 
xgb_pred_train = xgb_model.predict(x_train_imp)
xgb_pred_val   = xgb_model.predict(x_val_imp)
xgb_sigma      = (y_train - xgb_pred_train).std()
 
xgb_train_metrics = evaluate_model(
    "XGBoost (train)", xgb_pred_train, y_train, y_train_class, xgb_sigma
)
xgb_val_metrics = evaluate_model(
    "XGBoost (val)", xgb_pred_val, y_val, y_val_class, xgb_sigma
)
 
print(f"  Best params: {xgb_best_params}")
print(f"  Train  RMSE={xgb_train_metrics['rmse']:.2f}  "
      f"R2={xgb_train_metrics['r2']:.4f}  "
      f"Acc={xgb_train_metrics['accuracy']:.1%}")
print(f"  Val    RMSE={xgb_val_metrics['rmse']:.2f}  "
      f"R2={xgb_val_metrics['r2']:.4f}  "
      f"Acc={xgb_val_metrics['accuracy']:.1%}  "
      f"{beats_benchmark('rmse', xgb_val_metrics['rmse'])}")

#--------------------------------
# Model 7 - Stacking Ensemble
#-------------------------------

# Base models to stack — use the already-fitted models
# For stacking we need out-of-fold predictions on training data
base_models = {
    "Linear":      lr_model,
    "Ridge":       ridge_model,
#    "Lasso":       lasso_model,
    "Elastic Net": elastic_model,
    "RF":          rf_model,
    "XGBoost":     xgb_model,
}
 
# Generate out-of-fold predictions on training set
# TimeSeriesSplit ensures temporal ordering is respected
print("  Generating out-of-fold predictions...")
tscv = TimeSeriesSplit(n_splits=n_splits)
 
oof_preds = np.full((len(x_train_imp), len(base_models)),np.nan)
 
for fold, (tr_idx, val_fold_idx) in enumerate(tscv.split(x_train_imp)):
    x_fold_tr  = x_train_imp.iloc[tr_idx]
    x_fold_val = x_train_imp.iloc[val_fold_idx]
    y_fold_tr  = y_train.iloc[tr_idx]
 
    for i, (name, _) in enumerate(base_models.items()):
        # Refit each model on this fold's training data
        if name == "XGBoost":
            fold_model = xgb.XGBRegressor(**xgb_best_params, verbosity=0)
            fold_model.fit(x_fold_tr, y_fold_tr,
                          eval_set=[(x_fold_val, y_train.iloc[val_fold_idx])],
                          verbose=False)
        elif name == "RF":
            fold_model = RandomForestRegressor(**rf_best_params)
            fold_model.fit(x_fold_tr, y_fold_tr)
        elif name == "Linear":
            fold_model = make_linear_pipeline(LinearRegression())
            fold_model.fit(x_fold_tr, y_fold_tr)
        elif name == "Ridge":
            fold_model = make_linear_pipeline(RidgeCV(alphas=alphas, cv=3))
            fold_model.fit(x_fold_tr, y_fold_tr)
        #elif name == "Lasso":
           # fold_model = LassoCV(cv=3, max_iter=10000, random_state=42)
            #fold_model.fit(x_fold_tr, y_fold_tr)
        elif name == "Elastic Net":
            fold_model = make_linear_pipeline(ElasticNetCV(
                l1_ratio=[0.1, 0.5, 0.9, 1.0], cv=3,
                max_iter=10000, random_state=42
            ))
            fold_model.fit(x_fold_tr, y_fold_tr)
 
        oof_preds[val_fold_idx, i] = fold_model.predict(x_fold_val)

# Keep only rows that actually received OOF predictions
valid_oof_mask = ~np.isnan(oof_preds).any(axis=1)

oof_preds_valid = oof_preds[valid_oof_mask]
y_train_valid = y_train.iloc[valid_oof_mask]
y_train_class_valid = y_train_class.iloc[valid_oof_mask]

print(f"  OOF predictions shape: {oof_preds.shape}")
 
# Train meta-learner on out-of-fold predictions
# Ridge is a good meta-learner: regularized, handles correlation between
# base model predictions (which will be high since they all predict the same target)
meta_learner = MetaLearner(alphas=[0.01, 0.1, 1.0, 10.0, 100.0], cv=5)
meta_learner.fit(oof_preds_valid, y_train_valid)
 
print(f"  Meta-learner alpha: {meta_learner.alpha_:.4f}")
print(f"  Meta-learner weights: {dict(zip(base_models.keys(), meta_learner.coef_.round(3)))}")
 
# Generate validation predictions from each base model
val_base_preds = np.column_stack([
    model.predict(x_val_imp) for model in base_models.values()
])
 
# Stack predictions through meta-learner
stack_pred_val   = meta_learner.predict(val_base_preds)
stack_sigma      = (y_train_valid - meta_learner.predict(oof_preds_valid)).std()
 
stack_val_metrics = evaluate_model(
    "Stacking (val)", stack_pred_val, y_val, y_val_class, stack_sigma
)
 
print(f"  Val    RMSE={stack_val_metrics['rmse']:.2f}  "
      f"R2={stack_val_metrics['r2']:.4f}  "
      f"Acc={stack_val_metrics['accuracy']:.1%}  "
      f"{beats_benchmark('rmse', stack_val_metrics['rmse'])}")
 
# Training set evaluation using OOF predictions (honest estimate)
stack_train_oof = evaluate_model(
    "Stacking OOF (train)", meta_learner.predict(oof_preds_valid),
    y_train_valid, y_train_class_valid, stack_sigma
)
print(f"  OOF    RMSE={stack_train_oof['rmse']:.2f}  "
      f"R2={stack_train_oof['r2']:.4f}  "
      f"Acc={stack_train_oof['accuracy']:.1%}")
 
# Save stacking components separately
joblib.dump({
    "base_models":   base_models,
    "meta_learner":  meta_learner,
    "imputer":       imputer,
    "features":      list(x_train_imp.columns),
    "sigma":         stack_sigma,
}, OutputDirectory / "stacking_ensemble.joblib")
print(f"  Saved stacking ensemble")

# Lightweight wrapper to make the stacking ensemble a single, picklable object
class StackingEnsemble:
    def __init__(self, base_models, meta_learner, imputer, features):
        self.base_models = base_models
        self.meta_learner = meta_learner
        self.imputer = imputer
        self.features = list(features)

    def predict(self, X):
        # Accept either DataFrame or array-like; return meta-learner predictions
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X, columns=self.features)
        X = X.copy()
        X = X[self.features]
        X_imp = pd.DataFrame(self.imputer.transform(X), columns=X.columns, index=X.index)
        base_preds = np.column_stack([m.predict(X_imp) for m in self.base_models.values()])
        return self.meta_learner.predict(base_preds)

# Create an instance we can save/serialize as the "stacking" model
stacking_ensemble_obj = StackingEnsemble(base_models, meta_learner, imputer, list(x_train_imp.columns))

#--------------------------------
# Elo Residual Model
#--------------------------------

section("Elo Residual Model")

# Idea:
#   Standard approach:  predict point_diff directly
#   Residual approach:  Elo predicts first, ML corrects what Elo got wrong
#
#   point_diff = elo_prediction + elo_error
#   ML target  = elo_error  ("when is Elo wrong, and by how much?")

from scipy.stats import linregress

# Fit Elo linear baseline on training data only
# This replicates what the baseline script computed.

print("  Fitting Elo linear baseline on training data...")

train_elo_diff = (
    train_df["home_elo_pre"] - train_df["away_elo_pre"]
).dropna()
train_elo_outcomes = train_df.loc[train_elo_diff.index, "point_diff"]

elo_slope, elo_intercept, elo_r, _, _ = linregress(
    train_elo_diff, train_elo_outcomes
)

print(f"  Elo linear fit (training data only):")
print(f"    slope:     {elo_slope:.5f} pts per Elo unit")
print(f"    intercept: {elo_intercept:+.3f} pts")
print(f"    R:         {elo_r:.4f}  R²: {elo_r**2:.4f}")
print(f"    (intercept ≈ home court advantage after controlling for Elo)")

# Compute Elo predictions and residuals for all splits
# Use training-fitted slope/intercept — no leakage

for name, dataset in [("train", train_df), ("val", val_df)]:
    elo_diff = dataset["home_elo_pre"] - dataset["away_elo_pre"]
    dataset["elo_pred_diff"] = elo_intercept + elo_slope * elo_diff
    dataset["elo_residual"]  = dataset["point_diff"] - dataset["elo_pred_diff"]

# Elo-only baseline for comparison
elo_only_rmse = np.sqrt(mean_squared_error(
    val_df["point_diff"].dropna(),
    val_df["elo_pred_diff"].dropna()
))
elo_only_acc = (
    (val_df["elo_pred_diff"] > 0) == val_df["home_won"]
).mean()

print(f"\n  Elo-only validation performance:")
print(f"    RMSE:     {elo_only_rmse:.2f} pts")
print(f"    Accuracy: {elo_only_acc:.1%}")
print(f"    Residual std: {val_df['elo_residual'].std():.2f} pts")
print(f"    (this is the ceiling — residual model must reduce this std)")

# Prepare residual model features
# Exclude anything that IS Elo or is derived from Elo
# Keep features that capture what Elo cannot see

residual_features = [
    f for f in features if f not in {
        "diff_elo",
        "diff_best_net_eff",   # highly correlated with Elo
        "diff_adj_off_eff",    # correlated with Elo
        "diff_adj_def_eff",    # correlated with Elo
    }
]
# Keep only what's available
residual_features = [f for f in residual_features if f in train_df.columns]

print(f"\n  Residual model features ({len(residual_features)}):")
print(f"    {residual_features}")

# Prepare feature matrices for residual model
x_res_train = train_df[residual_features].copy()
y_res_train = train_df["elo_residual"].copy()

x_res_val   = val_df[residual_features].copy()
y_res_val   = val_df["elo_residual"].copy()

# Align on non-null residuals (need Elo data for both teams)
res_train_mask = y_res_train.notna() & x_res_train.notna().all(axis=1)
res_val_mask   = y_res_val.notna()

x_res_train = x_res_train[res_train_mask]
y_res_train = y_res_train[res_train_mask]
x_res_val   = x_res_val[res_val_mask]
y_res_val   = y_res_val[res_val_mask]

# Impute residual features using same imputer fit on main features
# For features in both sets use existing imputer, otherwise median
res_imputer = SimpleImputer(strategy="median")
res_imputer.fit(x_res_train)
x_res_train_imp = pd.DataFrame(
    res_imputer.transform(x_res_train),
    columns=x_res_train.columns,
    index=x_res_train.index
)
x_res_val_imp = pd.DataFrame(
    res_imputer.transform(x_res_val),
    columns=x_res_val.columns,
    index=x_res_val.index
)

# Train residual model
# XGBoost with conservative settings — simpler model to avoid overfitting

print("\n  Training residual correction model...")

# Tune residual XGBoost with Optuna
def res_xgb_objective(trial):
    params = {
        "learning_rate":    trial.suggest_float("learning_rate", 0.01, 0.1, log=True),
        "max_depth":        trial.suggest_int("max_depth", 2, 5),
        "subsample":        trial.suggest_float("subsample", 0.6, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
        "min_child_weight": trial.suggest_int("min_child_weight", 5, 20),
        "reg_alpha":        trial.suggest_float("reg_alpha", 0.01, 1.0, log=True),
        "reg_lambda":       trial.suggest_float("reg_lambda", 0.01, 1.0, log=True),
        "n_estimators":     500,
        "early_stopping_rounds": 30,
        "eval_metric":      "rmse",
        "random_state":     42,
        "n_jobs":           -1,
    }
    tscv   = TimeSeriesSplit(n_splits=n_splits)
    scores = []
    for tr_idx, vl_idx in tscv.split(x_res_train_imp):
        m = xgb.XGBRegressor(**params, verbosity=0)
        m.fit(
            x_res_train_imp.iloc[tr_idx], y_res_train.iloc[tr_idx],
            eval_set=[(x_res_train_imp.iloc[vl_idx],
                       y_res_train.iloc[vl_idx])],
            verbose=False
        )
        scores.append(np.sqrt(mean_squared_error(
            y_res_train.iloc[vl_idx], m.predict(x_res_train_imp.iloc[vl_idx])
        )))
    return np.mean(scores)

res_study = optuna.create_study(direction="minimize")
res_study.optimize(res_xgb_objective, n_trials=50, show_progress_bar=False)

res_xgb_params = res_study.best_params
res_xgb_params.update({
    "n_estimators": 500,
    "early_stopping_rounds": 30,
    "eval_metric": "rmse",
    "random_state": 42,
    "n_jobs": -1,
})

residual_model = xgb.XGBRegressor(**res_xgb_params, verbosity=0)
residual_model.fit(
    x_res_train_imp, y_res_train,
    eval_set=[(x_res_val_imp, y_res_val)],
    verbose=False
)

# Evaluate residual model on its own target
res_pred_train = residual_model.predict(x_res_train_imp)
res_pred_val   = residual_model.predict(x_res_val_imp)

res_train_rmse = np.sqrt(mean_squared_error(y_res_train, res_pred_train))
res_val_rmse   = np.sqrt(mean_squared_error(y_res_val,   res_pred_val))
res_val_r2     = r2_score(y_res_val, res_pred_val)

print(f"  Residual model (predicting Elo's error):")
print(f"    Train RMSE: {res_train_rmse:.2f} pts")
print(f"    Val RMSE:   {res_val_rmse:.2f} pts")
print(f"    Val R²:     {res_val_r2:.4f}")
print(f"    Residual std (naive): {y_res_val.std():.2f} pts")
print(f"    (R² here = fraction of Elo's error explained by your features)")

# Combined prediction = Elo + residual correction
# This is the final prediction for each game
val_combined_mask   = val_df.index.isin(x_res_val_imp.index)
combined_pred_val   = (
    val_df.loc[x_res_val_imp.index, "elo_pred_diff"].values
    + res_pred_val
)
combined_actual     = val_df.loc[x_res_val_imp.index, "point_diff"].values
combined_actual_class = val_df.loc[x_res_val_imp.index, "home_won"].values

combined_rmse = np.sqrt(mean_squared_error(combined_actual, combined_pred_val))
combined_r2   = r2_score(combined_actual, combined_pred_val)
combined_acc  = ((combined_pred_val > 0) == combined_actual_class).mean()

# Compute sigma for win probability conversion
# Use training residuals from combined model
train_combined_pred = (
    train_df.loc[x_res_train_imp.index, "elo_pred_diff"].values
    + res_pred_train
)
combined_sigma = (
    train_df.loc[x_res_train_imp.index, "point_diff"].values
    - train_combined_pred
).std()

combined_win_probs = point_diff_to_win_prob(combined_pred_val, combined_sigma)

print(f"\n  Combined model (Elo + residual correction):")
print(f"    RMSE:     {combined_rmse:.2f} pts  "
      f"({'better' if combined_rmse < elo_only_rmse else 'worse'} than Elo-only {elo_only_rmse:.2f})")
print(f"    R²:       {combined_r2:.4f}")
print(f"    Accuracy: {combined_acc:.1%}  "
      f"({'better' if combined_acc > elo_only_acc else 'worse'} than Elo-only {elo_only_acc:.1%})")
print(f"    Log Loss: {log_loss(combined_win_probs, combined_actual_class):.4f}")

# Feature importance in residual model
# Which features catch Elo's errors most reliably
res_importance = pd.Series(
    residual_model.feature_importances_,
    index=x_res_train_imp.columns
).sort_values(ascending=False)

print(f"\n  Top features for correcting Elo's errors:")
print(res_importance.head(10).to_string())
print(f"\n  INTERPRETATION:")
print(f"  High importance = this feature explains when/why Elo is wrong")
print(f"  If diff_pyth_luck is high: lucky teams are being overrated by Elo")
print(f"  If diff_tempo_control high: style mismatches affect Elo accuracy")
print(f"  If neutral_site high: Elo's home court adjustment is imperfect")

# Step 8: Add combined model to evaluation
elo_resid_metrics = {
    "name":        "Elo + Residual (val)",
    "rmse":        combined_rmse,
    "mae":         mean_absolute_error(combined_actual, combined_pred_val),
    "r2":          combined_r2,
    "accuracy":    combined_acc,
    "log_loss":    log_loss(combined_win_probs, combined_actual_class),
    "brier":       brier_score(combined_win_probs, combined_actual_class),
    "sigma":       combined_sigma,
    "win_probs":   combined_win_probs,
    "win_preds":   (combined_win_probs > 0.5).astype(int),
    "predictions": combined_pred_val,
}

print(f"\n  Summary:")
print(f"  {'Model':<25} {'RMSE':>8} {'R²':>8} {'Accuracy':>10}")
print(f"  {'─'*53}")
print(f"  {'Elo only':<25} {elo_only_rmse:>8.2f} {'N/A':>8} {elo_only_acc:>10.1%}")
print(f"  {'Elo + Residual':<25} {combined_rmse:>8.2f} {combined_r2:>8.4f} {combined_acc:>10.1%}")
print(f"  {'Direct XGBoost':<25} {xgb_val_metrics['rmse']:>8.2f} "
      f"{xgb_val_metrics['r2']:>8.4f} {xgb_val_metrics['accuracy']:>10.1%}")

# Save residual model components for prediction script
joblib.dump({
    "elo_slope":      elo_slope,
    "elo_intercept":  elo_intercept,
    "residual_model": residual_model,
    "res_imputer":    res_imputer,
    "res_features":   list(x_res_train_imp.columns),
    "combined_sigma": combined_sigma,
}, OutputDirectory / "elo_residual_model.joblib")
print(f"\n  Saved → elo_residual_model.joblib")

#--------------------------------
# Uncertainty Modeling
#--------------------------------

section("Uncertainty Modeling")

base_margin_model = xgb_model

train_margin_preds = xgb_pred_train
val_margin_preds   = xgb_pred_val

xgb_oof_preds = np.zeros(len(x_train_imp))
tscv_unc = TimeSeriesSplit(n_splits=n_splits)
for tr_idx, val_idx in tscv_unc.split(x_train_imp):
    fold_xgb = xgb.XGBRegressor(**xgb_best_params, verbosity=0)
    fold_xgb.fit(x_train_imp.iloc[tr_idx], y_train.iloc[tr_idx],
                 eval_set=[(x_train_imp.iloc[val_idx], y_train.iloc[val_idx])],
                 verbose=False)
    xgb_oof_preds[val_idx] = fold_xgb.predict(x_train_imp.iloc[val_idx])

oof_abs_residuals = np.abs(y_train - xgb_oof_preds)

uncertainty_model = RandomForestRegressor(
    n_estimators=200,
    max_depth=4,
    min_samples_leaf=20,
    random_state=42,
    n_jobs=-1
)

uncertainty_model.fit(
    x_train_imp,
    oof_abs_residuals
)

dynamic_sigma_train = uncertainty_model.predict(x_train_imp)
dynamic_sigma_val   = uncertainty_model.predict(x_val_imp)

dynamic_sigma_train = np.clip(
    dynamic_sigma_train,
    5,
    25
)

dynamic_sigma_val = np.clip(
    dynamic_sigma_val,
    5,
    25
)

dynamic_win_probs_val = point_diff_to_win_prob_dynamic(val_margin_preds, dynamic_sigma_val)

static_probs = point_diff_to_win_prob(
    val_margin_preds,
    xgb_sigma
)

dynamic_probs = point_diff_to_win_prob_dynamic(
    val_margin_preds,
    dynamic_sigma_val
)

print(f"Static Log Loss:  {log_loss(static_probs, y_val_class):.4f}")
print(f"Dynamic Log Loss: {log_loss(dynamic_probs, y_val_class):.4f}")

print(f"Static Brier:     {brier_score(static_probs, y_val_class):.4f}")
print(f"Dynamic Brier:    {brier_score(dynamic_probs, y_val_class):.4f}")

val_actual_abs_error = np.abs(y_val.values - val_margin_preds)
sigma_error_corr = np.corrcoef(dynamic_sigma_val, val_actual_abs_error)[0, 1]
print(f"\n  Sigma-error correlation: {sigma_error_corr:.3f}")
print(f"  OOF mean abs residual:   {oof_abs_residuals.mean():.2f} pts")
print(f"  In-sample mean abs res:  {np.abs(y_train - xgb_pred_train).mean():.2f} pts")
print(f"  Dynamic sigma range:     {dynamic_sigma_val.min():.2f} – {dynamic_sigma_val.max():.2f}")
print(f"  Dynamic sigma mean:      {dynamic_sigma_val.mean():.2f}")

#--------------------------------
# Model Selection
#--------------------------------

section("Model Selection")
 
all_val_metrics = [
    lr_val_metrics,
    ridge_val_metrics,
#    lasso_val_metrics,
    elastic_val_metrics,
    rf_val_metrics,
    xgb_val_metrics,
    stack_val_metrics,
    elo_resid_metrics,
]
 
# Select by validation RMSE — lower is better
best_metrics = min(all_val_metrics, key=lambda x: x["rmse"])
best_name    = best_metrics["name"].replace(" (val)", "")
 
print(f"\n  Model comparison (validation set):")
print(f"  {'Model':<25} {'RMSE':>8} {'R²':>8} {'Accuracy':>10} {'vs Elo':>12}")
print(f"  {'─'*65}")
for m in all_val_metrics:
    name = m["name"].replace(" (val)", "")
    flag = " <- BEST" if m == best_metrics else ""
    print(f"  {name:<25} {m['rmse']:>8.2f} {m['r2']:>8.4f} "
          f"{m['accuracy']:>10.1%} "
          f"{'✓' if m['rmse'] < benchmark_targets['elo_rmse'] else '✗':>12}{flag}")
 
print(f"\n  -> Best model: {best_name}")
print(f"  -> Val RMSE:   {best_metrics['rmse']:.2f} pts")
print(f"  -> Val Acc:    {best_metrics['accuracy']:.1%}")
print(f"  -> Elo RMSE:   {benchmark_targets['elo_rmse']:.2f} pts (benchmark)")
print(f"  -> Vegas RMSE: {benchmark_targets['spread_rmse']:.2f} pts (ceiling)")

#--------------------------------
# Tourney Evaluation
#--------------------------------

section("Tournament Evaluation")
 
# Map model names to their fitted model objects and sigma
model_registry = {
    "Linear Regression": (lr_model,    lr_sigma),
    "Ridge":             (ridge_model, ridge_sigma),
#    "Lasso":             (lasso_model, lasso_sigma),
    "Random Forest":     (rf_model,    rf_sigma),
    "XGBoost":           (xgb_model,   xgb_sigma),
    "Ensemble Stack":    (None, stack_sigma)
}
 
if len(ncaa_val) > 0:
    x_ncaa, y_ncaa, ncaa_idx = prepare_X_y(ncaa_val, features, RegTarget)
    x_ncaa_imp = pd.DataFrame(
        imputer.transform(x_ncaa),
        columns=x_ncaa.columns,
        index=x_ncaa.index
    )
    y_ncaa_class = ncaa_val.loc[ncaa_idx, ClassTarget]
 
    print(f"\n  NCAA tournament games (validation season): {len(x_ncaa)}")
    print(f"  {'Model':<25} {'RMSE':>8} {'Accuracy':>10} {'vs Elo NCAA':>12}")
    print(f"  {'─'*57}")
 
    for model_name, (model_obj, model_sigma) in model_registry.items():
    # Stacking requires two-step prediction through meta-learner
        if model_obj is None:
            ncaa_base_preds = np.column_stack([
                m.predict(x_ncaa_imp) for m in base_models.values()
            ])
            ncaa_pred = meta_learner.predict(ncaa_base_preds)
        else:
            ncaa_pred = model_obj.predict(x_ncaa_imp)
        
        ncaa_metrics = evaluate_model(
            model_name, ncaa_pred, y_ncaa, y_ncaa_class, model_sigma
        )
        flag = "✓" if ncaa_metrics["accuracy"] > benchmark_targets["ncaa_elo_acc"] else "✗"
        print(f"  {model_name:<25} {ncaa_metrics['rmse']:>8.2f} "
            f"{ncaa_metrics['accuracy']:>10.1%} {flag:>12}")
else:
    print("  No NCAA tournament games in validation season")

#--------------------------------
# Charts
#--------------------------------

section("Generating Charts")

# Model Comparison

model_names_short = ["Linear", 
                     "Ridge", 
                     #"Lasso",
                     "Elastic Net", 
                     "RF", 
                     "XGBoost", 
                     "Stacking",
                     "Elo_Resid"]
val_rmses = [m["rmse"]     for m in all_val_metrics]
val_r2s   = [m["r2"]       for m in all_val_metrics]
val_accs  = [m["accuracy"] for m in all_val_metrics]
 
fig, axes = plt.subplots(1, 3, figsize=(15, 5))
fig.suptitle("Model comparison — validation set performance",
             fontsize=14, fontweight="bold")
 
bar_colors = [Colors["highlight"] if m == best_metrics else Colors["neutral"]
              for m in all_val_metrics]
 
### RMSE
bars = axes[0].bar(model_names_short, val_rmses, color=bar_colors, edgecolor="white")
axes[0].axhline(benchmark_targets["elo_rmse"],   color=Colors["neutral"], linestyle="--",
                linewidth=1.5, label=f"Elo ({benchmark_targets['elo_rmse']:.2f})")
axes[0].axhline(benchmark_targets["spread_rmse"], color=Colors["highlight"], linestyle="--",
                linewidth=1.5, label=f"Vegas ({benchmark_targets['spread_rmse']:.2f})")
axes[0].set_title("RMSE (lower = better)")
axes[0].set_ylabel("Points")
axes[0].legend(fontsize=8)
for bar, val in zip(bars, val_rmses):
    axes[0].text(bar.get_x() + bar.get_width() / 2,
                 bar.get_height() + 0.05, f"{val:.2f}",
                 ha="center", fontsize=9)
 
### R2
bars = axes[1].bar(model_names_short, val_r2s, color=bar_colors, edgecolor="white")
axes[1].axhline(benchmark_targets["elo_r2"],    color=Colors["neutral"], linestyle="--",
                linewidth=1.5, label=f"Elo ({benchmark_targets['elo_r2']:.3f})")
axes[1].axhline(benchmark_targets["spread_r2"], color=Colors["highlight"], linestyle="--",
                linewidth=1.5, label=f"Vegas ({benchmark_targets['spread_r2']:.3f})")
axes[1].set_title("R2 (higher = better)")
axes[1].set_ylabel("R2")
axes[1].legend(fontsize=8)
for bar, val in zip(bars, val_r2s):
    axes[1].text(bar.get_x() + bar.get_width() / 2,
                 val + 0.005, f"{val:.3f}",
                 ha="center", fontsize=9)
 
### Accuracy
bars = axes[2].bar(model_names_short, val_accs, color=bar_colors, edgecolor="white")
axes[2].axhline(benchmark_targets["elo_acc"],   color=Colors["neutral"], linestyle="--",
                linewidth=1.5, label=f"Elo ({benchmark_targets['elo_acc']:.1%})")
axes[2].axhline(benchmark_targets["spread_acc"], color=Colors["highlight"], linestyle="--",
                linewidth=1.5, label=f"Vegas ({benchmark_targets['spread_acc']:.1%})")
axes[2].set_title("Win/loss accuracy (higher = better)")
axes[2].set_ylabel("Accuracy")
axes[2].set_ylim(0.55, 0.85)
axes[2].legend(fontsize=8)
for bar, val in zip(bars, val_accs):
    axes[2].text(bar.get_x() + bar.get_width() / 2,
                 val + 0.003, f"{val:.1%}",
                 ha="center", fontsize=9)
 
plt.tight_layout()
save_fig(fig, "model_comparison.png")

# XGBoost Feature Importance
fig, ax = plt.subplots(figsize=(10, 12))
 
xgb_importance = pd.Series(
    xgb_model.feature_importances_,
    index=x_train_imp.columns
).sort_values(ascending=False).head(30)
 
colors_fi = [Colors["win"] if i < 10 else Colors["neutral"]
             for i in range(len(xgb_importance))]
ax.barh(xgb_importance.index[::-1], xgb_importance.values[::-1],
        color=colors_fi[::-1], edgecolor="white")
ax.set_xlabel("Feature importance (gain)")
ax.set_title("XGBoost — top 30 feature importances\n"
             "(teal = top 10, shows what the model is actually using)")
plt.tight_layout()
save_fig(fig, "feature_importance.png",
         "Top 30 XGBoost features:\n" + xgb_importance.to_string())

# Predicted v. Actual Point Differential
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle("XGBoost — predicted vs actual point differential",
             fontsize=14, fontweight="bold")
 
## Validation set scatter
axes[0].scatter(xgb_pred_val, y_val, alpha=0.15, s=8,
                color=Colors["neutral"])
lim = max(abs(y_val.min()), abs(y_val.max()), abs(xgb_pred_val.min()),
          abs(xgb_pred_val.max()))
axes[0].plot([-lim, lim], [-lim, lim], color="gray",
             linestyle="--", linewidth=1, label="Perfect prediction")
axes[0].axhline(0, color="gray", linestyle=":", linewidth=0.5)
axes[0].axvline(0, color="gray", linestyle=":", linewidth=0.5)
axes[0].set_xlabel("Predicted point diff")
axes[0].set_ylabel("Actual point diff")
axes[0].set_title(f"Validation set\nRMSE={xgb_val_metrics['rmse']:.2f}  "
                  f"R²={xgb_val_metrics['r2']:.4f}")
axes[0].legend()
 
## Residual distribution
residuals = y_val - xgb_pred_val
axes[1].hist(residuals, bins=40, color=Colors["neutral"],
             edgecolor="white", linewidth=0.3)
axes[1].axvline(0, color=Colors["loss"], linestyle="--",
                linewidth=1.5, label="Zero error")
axes[1].axvline(residuals.mean(), color=Colors["highlight"],
                linestyle="--", linewidth=1.5,
                label=f"Mean: {residuals.mean():+.2f}")
axes[1].set_xlabel("Residual (actual - predicted)")
axes[1].set_ylabel("Games")
axes[1].set_title("Residual distribution\n"
                  "(symmetric around 0 = no systematic bias)")
axes[1].legend()
plt.tight_layout()
save_fig(fig, "predicted_vs_actual.png")

# Win Prob Calibration
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle("Win probability calibration — XGBoost",
             fontsize=14, fontweight="bold")
 
win_probs_val = dynamic_win_probs_val
val_df_eval   = val_df.loc[x_val_imp.index].copy()
val_df_eval["predicted_diff"]  = xgb_pred_val
val_df_eval["win_prob"]        = win_probs_val
val_df_eval["prob_bucket"]     = pd.cut(
    val_df_eval["win_prob"],
    bins=[0, 0.35, 0.45, 0.50, 0.55, 0.65, 1.0],
    labels=["<35%", "35-45%", "45-50%", "50-55%", "55-65%", ">65%"]
)
 
calibration = (
    val_df_eval
    .groupby("prob_bucket", observed=True)
    .agg(
        mean_predicted  = ("win_prob",  "mean"),
        actual_win_rate = (ClassTarget, "mean"),
        count           = ("game_id",   "count"),
    )
    .reset_index()
    .dropna()
)
 
axes[0].scatter(calibration["mean_predicted"],
                calibration["actual_win_rate"],
                s=calibration["count"] * 2,
                color=Colors["neutral"], alpha=0.8, zorder=5,
                label="Calibration\n(size = game count)")
axes[0].plot([0, 1], [0, 1], color="gray", linestyle="--",
             linewidth=1.5, label="Perfect calibration")
axes[0].set_xlabel("Predicted win probability")
axes[0].set_ylabel("Actual win rate")
axes[0].set_title("Calibration curve\nPoints on diagonal = well calibrated")
axes[0].set_xlim(0.2, 0.8)
axes[0].set_ylim(0.2, 0.8)
axes[0].legend()
 
axes[1].hist(win_probs_val, bins=40, color=Colors["neutral"],
             edgecolor="white", linewidth=0.3)
axes[1].axvline(0.5, color=Colors["loss"], linestyle="--",
                linewidth=1.5, label="50% (coin flip)")
axes[1].set_xlabel("Predicted win probability")
axes[1].set_ylabel("Games")
axes[1].set_title("Win probability distribution\n"
                  "(spread = model confidence)")
axes[1].legend()
plt.tight_layout()
save_fig(fig, "win_probability_calibration.png")

# Errors by Game Type
fig, axes = plt.subplots(1, 3, figsize=(15, 5))
fig.suptitle("XGBoost error analysis — where does the model struggle?",
             fontsize=14, fontweight="bold")
 
val_df_eval["abs_error"] = abs(y_val - xgb_pred_val)
val_df_eval["elo_diff_abs"] = abs(
    val_df_eval.get("diff_elo", 0)
)
 
# Error by game type
for game_type, subset in [
    ("Regular season", val_df_eval[val_df_eval["season_type"] == "SeasonType.REGULAR"]),
    ("Postseason",     val_df_eval[val_df_eval["season_type"] == "SeasonType.POSTSEASON"]),
]:
    if len(subset) > 0:
        axes[0].hist(subset["abs_error"], bins=30, alpha=0.6,
                     label=f"{game_type} (n={len(subset)})",
                     edgecolor="white")
axes[0].set_xlabel("Absolute error (points)")
axes[0].set_ylabel("Games")
axes[0].set_title("Error distribution by game type")
axes[0].legend()
 
## Error vs predicted margin — do we get close games right?
axes[1].scatter(abs(val_df_eval["predicted_diff"]),
                val_df_eval["abs_error"],
                alpha=0.15, s=8, color=Colors["neutral"])
axes[1].set_xlabel("Absolute predicted margin")
axes[1].set_ylabel("Absolute error")
axes[1].set_title("Error vs predicted margin\n"
                  "(close predicted games = harder to get right)")
 
## Win/loss accuracy by predicted confidence bucket
val_df_eval["confidence"] = abs(val_df_eval["win_prob"] - 0.5)
val_df_eval["conf_bucket"] = pd.cut(
    val_df_eval["confidence"],
    bins=[0, 0.05, 0.10, 0.15, 0.20, 0.50],
    labels=["<5%", "5-10%", "10-15%", "15-20%", ">20%"]
)
val_df_eval["correct"] = (
    val_df_eval["win_prob"] > 0.5
).astype(int) == val_df_eval[ClassTarget]
 
conf_acc = val_df_eval.groupby("conf_bucket", observed=True)["correct"].agg(
    ["mean", "count"]
).reset_index()
 
bars = axes[2].bar(conf_acc["conf_bucket"], conf_acc["mean"],
                   color=Colors["win"], edgecolor="white")
axes[2].axhline(0.5, color="gray", linestyle="--", linewidth=1)
axes[2].set_xlabel("Model confidence (distance from 50%)")
axes[2].set_ylabel("Accuracy")
axes[2].set_title("Accuracy by model confidence\n"
                  "(confident predictions should be more accurate)")
axes[2].set_ylim(0.4, 1.0)
for bar, (_, row) in zip(bars, conf_acc.iterrows()):
    axes[2].text(bar.get_x() + bar.get_width() / 2,
                 bar.get_height() + 0.01,
                 f"{bar.get_height():.0%}\n(n={int(row['count'])})",
                 ha="center", fontsize=8)
plt.tight_layout()
save_fig(fig, "error_analysis.png")

# Dynamic sigma diagnostics
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

fig.suptitle("Dynamic uncertainty modeling diagnostics",fontsize=14,fontweight="bold")
actual_abs_error = np.abs(
    y_val - val_margin_preds
)

axes[0].scatter(
    dynamic_sigma_val,
    actual_abs_error,
    alpha=0.15,
    s=8,
    color=Colors["neutral"]
)

axes[0].set_xlabel("Predicted sigma")
axes[0].set_ylabel("Actual absolute error")
axes[0].set_title("Predicted uncertainty vs actual error")

axes[1].hist(
    dynamic_sigma_val,
    bins=40,
    color=Colors["neutral"],
    edgecolor="white",
    linewidth=0.3
)

axes[1].set_xlabel("Predicted sigma")
axes[1].set_ylabel("Games")
axes[1].set_title("Distribution of predicted uncertainty")

plt.tight_layout()

save_fig(fig,"dynamic_uncertainty.png")

#--------------------------------
# Save Best Model
#--------------------------------

section("Saving Best Model")

# Identify best model object
model_obj_map = {
    "Linear Regression": lr_model,
    "Ridge":             ridge_model,
#    "Lasso":             lasso_model,
    "Elastic Net":       elastic_model,
    "Random Forest":     rf_model,
    "XGBoost":           xgb_model,
    "Stacking":          stacking_ensemble_obj,
}
best_model_obj = model_obj_map[best_name]
best_sigma     = best_metrics["sigma"]
 
# Save model
model_path = OutputDirectory / "best_model.joblib"
joblib.dump({
    "model":    best_model_obj,
    "uncertainty_model": uncertainty_model,
    "imputer":  imputer,
    "name":     best_name,
    "features": list(x_train_imp.columns),
    "sigma":    best_sigma,
}, model_path)
print(f"  Saved model → {model_path}")
 
# Save metadata for prediction script
metadata = {
    "model_name":         best_name,
    "sigma":              best_sigma,
    "features":           list(x_train_imp.columns),
    "validation_season":  ValidationSeason,
    "test_season":        TestSeason,
    "val_rmse":           best_metrics["rmse"],
    "val_r2":             best_metrics["r2"],
    "val_accuracy":       best_metrics["accuracy"],
    "benchmarks":         benchmark_targets,
    "trained_at":         datetime.now().isoformat(),
}
meta_path = OutputDirectory / "model_metadata.json"
with open(meta_path, "w") as f:
    json.dump(metadata, f, indent=2)
print(f"  Saved metadata → {meta_path}")

#--------------------------------
# Summary
#--------------------------------

summary = f"""
Model Training Summary

Best Model: {best_name}

Validation Performance:
  RMSE:      {best_metrics['rmse']:.2f} pts
  R²:        {best_metrics['r2']:.4f}
  Accuracy:  {best_metrics['accuracy']:.1%}
  Log Loss:  {best_metrics['log_loss']:.4f}
  Brier:     {best_metrics['brier']:.4f}
  Sigma:     {best_sigma:.2f} pts (used for win probability)

Benchmark Comparison:
              RMSE      R²        Accuracy
  Naive:      {benchmark_targets['naive_rmse']:.2f}      {benchmark_targets['elo_r2']:.4f}    {benchmark_targets['naive_acc']:.1%}
  Elo:        {benchmark_targets['elo_rmse']:.2f}      {benchmark_targets['elo_r2']:.4f}    {benchmark_targets['elo_acc']:.1%}
  Vegas:      {benchmark_targets['spread_rmse']:.2f}      {benchmark_targets['spread_r2']:.4f}    {benchmark_targets['spread_acc']:.1%}
  Best model: {best_metrics['rmse']:.2f}      {best_metrics['r2']:.4f}    {best_metrics['accuracy']:.1%}
 
BEATS ELO:     {'YES' if best_metrics['rmse'] < benchmark_targets['elo_rmse'] else 'NO'}
BEATS VEGAS:   {'YES' if best_metrics['rmse'] < benchmark_targets['spread_rmse'] else 'NO'}
 
Elastic Net FEATURE SELECTION:
  Features kept:   {len(elastic_kept)}
  Features zeroed: {len(elastic_zeroed)}
  (zeroed features add no value beyond other features)
 
XGBOOST TOP FEATURES:
{xgb_importance.head(10).to_string()}
"""

print(summary)
write_summary(summary)
print(f"\nAll outputs saved to {OutputDirectory}")

df[df["home_seed"].notna() & 
             (df["season_type"]=="SeasonType.REGULAR")]