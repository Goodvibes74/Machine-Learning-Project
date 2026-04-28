# Stock Market Prediction — RF, XGBoost & SVM with Sentiment Analysis

A hybrid machine learning system that combines historical stock price data with technical indicators and a sentiment proxy to predict short-term stock price movements.

## Project Overview

This project implements and compares three classifiers:
- **Random Forest** — ensemble of decision trees with threshold calibration
- **XGBoost** — gradient boosting classifier
- **SVM** — Support Vector Machine with RBF kernel

All models integrate:
- **Quantitative features**: OHLCV data, 25+ technical indicators (RSI, MACD, Bollinger Bands, OBV, ATR, SMAs, momentum)
- **Qualitative features**: Price-based sentiment proxy (placeholder for NLP sentiment)
- **Walk-forward backtesting**: Proper time-series validation with 10 folds

**Goal**: Predict whether a stock's price will go up or down the next day.

## Project Structure

```
stock_ml_project/
├── data/
│   ├── raw/              # Downloaded stock data from Yahoo Finance
│   ├── processed/        # Cleaned and validated data
│   └── features/         # Data with engineered features
│
├── src/
│   ├── data_collection.py    # Download stock data via yfinance
│   ├── preprocessing.py      # Clean and validate data
│   ├── feature_engineering.py# Create 25+ technical indicators
│   ├── models.py             # Random Forest and SVM training/evaluation
│   ├── xgboost_model.py      # XGBoost training/evaluation
│   └── backtesting.py        # Walk-forward validation
│
├── comparison/
│   └── compare_models.py     # Compare RF vs XGBoost vs SVM
│
├── outputs/                  # Generated plots and results CSV
│
├── config.py                 # Central configuration file
├── requirements.txt          # Python dependencies
├── main.py                   # Main pipeline orchestrator
├── Getting_Started.md        # Step-by-step setup guide
└── README.md                 # This file
```

## Quick Start

### 1. Setup Environment

```bash
python -m venv venv

# Windows:
venv\Scripts\activate
# Mac/Linux:
source venv/bin/activate

pip install -r requirements.txt
```

### 2. Run the Pipeline

```bash
# Interactive menu
python main.py

# Or programmatically
python -c "from main import run_complete_pipeline; run_complete_pipeline('AAPL')"

# Run full model comparison (RF vs XGBoost vs SVM)
python comparison/compare_models.py
```

### 3. Test Individual Modules

```bash
python src/data_collection.py    # Download data
python src/preprocessing.py      # Clean data
python src/feature_engineering.py# Engineer features
python src/models.py             # Train Random Forest & SVM
python src/xgboost_model.py      # Train XGBoost
python src/backtesting.py        # Walk-forward validation
python comparison/compare_models.py  # Full model comparison
```

## What Each Module Does

### `config.py` — Configuration
Central settings for the whole project.

```python
TICKERS = ['AAPL', 'MSFT', 'GOOGL']  # Stocks to analyze
START_DATE = '2021-01-01'             # Data start
ROLLING_WINDOWS = [5, 10, 20]        # SMA periods
MOMENTUM_WINDOWS = [3, 5, 10, 20]    # Momentum lookback periods
RF_PARAMS  = { ... }                 # Random Forest hyperparameters
XGB_PARAMS = { ... }                 # XGBoost hyperparameters
SVM_PARAMS = { ... }                 # SVM hyperparameters
N_SPLITS = 10                        # Walk-forward folds
```

### `data_collection.py` — Data Download
Downloads historical OHLCV data from Yahoo Finance via `yfinance`.

```python
download_stock_data('AAPL', '2023-01-01', '2024-01-01')
download_multiple_stocks(['AAPL', 'MSFT'], ...)
load_stock_data('AAPL', 'raw')
```

### `preprocessing.py` — Data Cleaning
Cleans raw data: drops unnecessary columns, forward-fills missing values, validates integrity.

```python
clean_stock_data(df)
handle_missing_values(df)
validate_data(df)
preprocess_stock_data(df)
```

### `feature_engineering.py` — Feature Creation (25+ features)

| Category | Features |
|---|---|
| Returns & Volume | `Returns`, `Volume_Change`, `HL_Spread` |
| Moving Averages | `SMA_5`, `SMA_10`, `SMA_20` |
| Trend signals | `Price_SMA_5/10/20_ratio`, `SMA_5_20_ratio` |
| Volatility | `Volatility` (20-day std), `ATR_14` |
| Momentum | `Momentum_3/5/10/20` |
| Oscillators | `RSI_14` |
| Trend/Momentum | `MACD`, `MACD_signal`, `MACD_histogram` |
| Volatility bands | `BB_upper`, `BB_lower`, `BB_width`, `BB_position` |
| Volume pressure | `OBV`, `OBV_change` |
| Sentiment | `Sentiment_Proxy`, `Sentiment_SMA` |

```python
engineer_all_features(df)   # Apply all
prepare_ml_data(df)         # Return X, y, feature_names
```

### `models.py` — Random Forest & SVM
Trains RF with calibrated decision threshold and SVM with StandardScaler.

```python
train_random_forest(X_train, y_train)  # Returns _ThresholdRF wrapper
train_svm(X_train, y_train)            # Returns (model, scaler)
evaluate_model(model, X_train, y_train, X_test, y_test)
evaluate_svm(model, scaler, X_train, y_train, X_test, y_test)
```

**Threshold calibration**: RF with `class_weight='balanced'` compresses probabilities, so the raw 0.5 cut-off misses real signal. The last 20% of training data is used to find the optimal threshold — no test data involved, so there is no leakage.

### `xgboost_model.py` — XGBoost
Trains an XGBoost classifier and evaluates with ROC-AUC.

```python
train_xgboost(X_train, y_train)
evaluate_xgboost_model(model, X_train, y_train, X_test, y_test)
plot_xgb_confusion_matrix(y_true, y_pred)
plot_xgb_feature_importance(model, features)
```

### `comparison/compare_models.py` — Model Comparison
Trains all three models on the same 80/20 chronological split, generates comparison plots, and saves results to `outputs/comparison_results.csv`.

```python
run_full_comparison(ticker='AAPL')
plot_roc_curves(rf_model, xgb_model, X_test, y_test, svm_model, svm_scaler)
plot_metrics_bar_chart(rf_metrics, xgb_metrics, svm_metrics)
plot_feature_importance_comparison(rf_model, xgb_model, features)
```

### `backtesting.py` — Walk-Forward Validation
Simulates real trading: train on expanding past window, predict next period, repeat across 10 folds.

```python
walk_forward_validation(X, y, features, n_splits=10)
compare_strategies(X, y, features)   # Price-only vs hybrid
plot_backtest_results(results)
```

## Pipeline Flow

```
Raw Stock Data (OHLCV)
        ↓
[Data Collection] → Download from Yahoo Finance
        ↓
[Preprocessing]   → Clean, validate, handle missing values
        ↓
[Feature Engineering] → 25+ technical indicators + sentiment proxy
        ↓
[Train/Test Split] → Chronological 80/20 (no shuffling)
        ↓
[Model Training]  → RF (with threshold cal.) | XGBoost | SVM
        ↓
[Evaluation]      → Accuracy, Precision, Recall, F1, ROC-AUC
        ↓
[Comparison]      → Side-by-side RF vs XGBoost vs SVM
        ↓
[Backtesting]     → 10-fold walk-forward validation
        ↓
Results & Insights (outputs/ folder)
```

## Key Concepts

**Features (X) vs Target (y)**
- **Features**: Technical indicators the model learns from
- **Target**: Will price go up tomorrow? (1 = up, 0 = down/flat)

**Time Series Rules**
- No random shuffling — always split chronologically
- No look-ahead bias — only use data available at prediction time
- Walk-forward: retrain as new data arrives

**Evaluation Metrics**
- **Accuracy**: Overall correct predictions
- **Precision**: Of predicted "up" days, how many were correct
- **Recall**: Of actual "up" days, how many did we catch
- **F1**: Harmonic mean of precision and recall
- **ROC-AUC**: Area under the ROC curve (threshold-independent)

## Verified Results (AAPL, 5-year dataset, 80/20 chronological split)

| Model | Test Accuracy | ROC-AUC | F1 |
|---|---|---|---|
| Random Forest | 53.8% | 0.538 | 0.660 |
| XGBoost | 56.7% | 0.571 | 0.662 |
| SVM | 49.7% | 0.512 | 0.575 |

**Walk-forward (10-fold, most recent 5 folds):** XGB mean AUC = 0.518; RF mean AUC = 0.496

> Stock markets are noisy. 55–57% accuracy and AUC 0.54–0.57 are competitive with professional quant strategies. >50% AUC signals genuine predictive information.

## Customization

### Change Stocks
```python
# config.py
TICKERS = ['TSLA', 'NVDA', 'AMD']
```

### Change Date Range
```python
# config.py
START_DATE = '2020-01-01'
END_DATE = '2024-12-31'
```

### Tune Model Parameters
```python
# config.py — Random Forest
RF_PARAMS = {
    'n_estimators': 300,
    'max_depth': 4,
    'min_samples_split': 30,
    'min_samples_leaf': 15,
}

# XGBoost
XGB_PARAMS = {
    'n_estimators': 150,
    'max_depth': 5,
    'learning_rate': 0.08,
}

# SVM
SVM_PARAMS = {
    'kernel': 'rbf',
    'C': 5.0,
    'gamma': 'auto',
}
```

### Add a New Feature
Edit `feature_engineering.py`:
```python
def calculate_roc(df, window=12):
    """Rate of Change over N days"""
    df = df.copy()
    df['ROC_12'] = df['Close'].pct_change(window) * 100
    return df

# Then add to engineer_all_features():
df = calculate_roc(df)
```

## Troubleshooting

| Problem | Solution |
|---|---|
| "No data found for ticker" | Check ticker symbol (uppercase), verify internet |
| "ValueError: Input contains NaN" | Run `df.dropna()` — expected from rolling windows |
| Low accuracy (~50%) | Normal for stocks; try more data or different features |
| Train accuracy >> test accuracy | Overfitting; reduce `max_depth`, increase `min_samples_leaf` |
| Plots not showing | `import matplotlib; matplotlib.use('TkAgg')` |

## Next Steps

### Completed
- [x] Random Forest with threshold calibration
- [x] XGBoost classifier
- [x] SVM classifier
- [x] RSI, MACD, Bollinger Bands, OBV, ATR indicators
- [x] Price/SMA ratio trend signals
- [x] Multi-window momentum
- [x] Walk-forward backtesting (10 folds)
- [x] Three-model comparison with ROC curves

### Planned
- [ ] Real NLP sentiment analysis (FinBERT or VADER on news headlines)
- [ ] Live trading simulation with portfolio tracking
- [ ] Hyperparameter search (Optuna or GridSearchCV)
- [ ] Multi-stock portfolio optimization

## Academic Context

- **Title**: "Random Forest & Sentiment Analysis: Hybrid Stock Market Prediction"
- **Institution**: Makerere University, College of Computing and Information Sciences
- **Approach**: Combines quantitative technical features with qualitative sentiment signals

## Disclaimer

**For Educational Purposes Only.** This project is not investment advice. Stock markets are unpredictable. Past performance does not guarantee future results.

---

Data via [Yahoo Finance](https://finance.yahoo.com) · ML via [scikit-learn](https://scikit-learn.org) & [XGBoost](https://xgboost.readthedocs.io)
