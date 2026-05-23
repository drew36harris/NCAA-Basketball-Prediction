# NCAA-Basketball-Prediction
Machine learning system for predicting NCAA basketball point differentials and win probabilities using Elo ratings, efficiency metrics, rolling team statistics, and engineered stylistic matchup features.

# College Basketball Game Prediction Model

A machine learning system that predicts NCAA basketball game outcomes 
and win probabilities using rolling performance statistics, efficiency 
ratings, and game control metrics.

## Project Overview

Predicts point differential for college basketball games using:
- Pre-game team quality metrics (Elo ratings, adjusted efficiency)
- Rolling season statistics (four factors, momentum-weighted)
- Novel game control features (tempo disruption, turnover decomposition)
- Pythagorean expectation as a luck/regression signal

## Results

| Model          | Val RMSE | Val Accuracy | vs Elo Benchmark |
|----------------|----------|--------------|------------------|
| Linear         | 12.46    | 69.5%        | Above Validation |
| Ridge          | 12.46    | 69.2%        | Above Validation |
| Elastic Net    | 12.50    | 69.7%        | Above Validation |
| Random Forest  | 12.69    | 69.7%        | Above Validation |
| XGBoost        | 12.58    | 69.9%        | Above Validation |
| Stacking       | 12.50    | 69.9%        | Above Validation |
| Elo + Residual | 12.61    | 68.0%        | Above Validation |


**Historical Benchmarks:** Elo = 12.34 RMSE / 67.4% accuracy, Vegas = 11.30 RMSE / 74.4% accuracy

**Validation Season Benchmark:** Elo = 62.9% accuracy

## Architecture
APIDataPullAndClean.py - data ingestion, feature engineering, rolling stats
EDA.py - exploratory data analysis and feature selection
Benchmarking.py - baseline performance targets
ModelTraining.py - model training, evaluation, and selection

## Key Features

**Rolling Statistics**— all features use expanding().mean().shift(1) to 
ensure no future data leaks into any training row.

**Disruption Metrics** — novel features measuring how consistently teams 
maintain their style (tempo, shot selection, rebounding) across games.

**Elo Residual Model** — two-stage approach where Elo predicts the baseline 
margin and ML features correct systematic Elo errors.

**Dynamic Uncertainty** — game-specific sigma estimates for win probability 
conversion using OOF-trained Random Forest.

## Setup

```bash
pip install -r requirements.txt
export CBBD_API_KEY='your_key_here'
python pipeline/data_pipeline.py
python analysis/eda.py
python analysis/benchmarks.py
python modeling/model_training.py
```

## Data Source

[CollegeBasketballData.com API](https://api.collegebasketballdata.com) — requires free API key.
