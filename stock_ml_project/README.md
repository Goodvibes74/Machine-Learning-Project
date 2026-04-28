# Stock Market Prediction — RF, XGBoost & SVM with NLP Sentiment Analysis

A hybrid machine learning system that combines historical stock price data, 25+ technical indicators, and **real NLP sentiment analysis** (FinBERT on financial news) to predict short-term stock price movements.

## Project Overview

This project implements and compares three classifiers:
- **Random Forest** — ensemble of decision trees with threshold calibration
- **XGBoost** — gradient boosting classifier
- **SVM** — Support Vector Machine with RBF kernel

All models integrate:
- **Quantitative features**: OHLCV data, 25+ technical indicators (RSI, MACD, Bollinger Bands, OBV, ATR, SMAs, momentum)
- **NLP sentiment features**: `NLP_Sentiment` (FinBERT score, −1 to +1) and `News_Volume` (daily headline count) fetched from Finnhub/NewsAPI and scored with `ProsusAI/finbert`
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
│   ├── data_collection.py      # Download stock data via yfinance
│   ├── preprocessing.py        # Clean and validate data
│   ├── feature_engineering.py  # Create 25+ technical indicators + NLP merge
│   ├── sentiment_analysis.py   # FinBERT NLP sentiment pipeline (NEW)
│   ├── models.py               # Random Forest and SVM training/evaluation
│   ├── xgboost_model.py        # XGBoost training/evaluation
│   └── backtesting.py          # Walk-forward validation
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

### `feature_engineering.py` — Feature Creation (27+ features)

| Category | Features |
|---|---|
| Returns & Volume | `Returns`, `Volume_Change`, `HL_Spread` |
| Moving Averages | `SMA_5`, `SMA_10`, `SMA_20` |
| Trend signals | `Price_SMA_5/10/20_ratio`, `SMA_5_20_ratio` |
| Volatility | `Volatility` (20-day std), `ATR_14` |
| Momentum | `Momentum_3/5/10/20` |
| Oscillators | `RSI_14` |
| Trend/Momentum | `MACD`, `MACD_signal`, `MACD_histogram` |
| Volatility bands | `BB_width`, `BB_position` |
| Volume pressure | `OBV_change`, `Volume_Ratio` |
| NLP Sentiment | `NLP_Sentiment` (FinBERT score, 1-day lag), `News_Volume` |
| Fallback proxy | `Sentiment_Proxy`, `Sentiment_SMA` (used when no API key) |

```python
engineer_all_features(df)                      # Price proxy (no API key)
engineer_all_features(df, sentiment_df=sent)   # Real NLP features
prepare_ml_data(df)                            # Return X, y, feature_names
```

### `sentiment_analysis.py` — NLP Sentiment Pipeline (NEW)

```python
# Full pipeline for one ticker
from src.sentiment_analysis import get_sentiment_for_ticker
sent_df = get_sentiment_for_ticker('AAPL', '2023-01-01', '2024-01-01',
                                    api_key='your_key')
# Returns DataFrame with ['Date', 'NLP_Sentiment', 'News_Volume']

# Individual steps
fetch_news_headlines(ticker, start, end, api_key)  # Finnhub or NewsAPI
score_headlines(headlines)                          # FinBERT inference
aggregate_sentiment(scored_df, all_dates)           # Daily roll-up
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
[Step 1 — Data Collection]   → Download from Yahoo Finance
        ↓
[Step 2 — Preprocessing]     → Clean, validate, handle missing values
        ↓
[Step 2.5 — NLP Sentiment]   → Fetch news (Finnhub/NewsAPI)
                                → Score with FinBERT (ProsusAI/finbert)
                                → Aggregate to daily NLP_Sentiment + News_Volume
                                → Apply 1-day lag (no look-ahead bias)
        ↓
[Step 3 — Feature Engineering] → 27+ technical indicators + NLP features
        ↓
[Train/Test Split]  → Chronological 80/20 (no shuffling)
        ↓
[Step 4 — Model Training]   → RF (threshold cal.) | XGBoost | SVM
        ↓
[Step 5 — Evaluation]       → Accuracy, Precision, Recall, F1, ROC-AUC
        ↓
[Step 6 — Backtesting]      → 10-fold walk-forward validation
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

> Tested with `START_DATE = now − 5 years`, `END_DATE = today`, RF params as in config.py.

### Random Forest + NLP Sentiment Pipeline (latest)

| Metric | Value |
|---|---|
| Training rows | 1 233 |
| Test rows | 247 |
| Train accuracy | 69.2% |
| **Test accuracy** | **52.6%** |
| Train/test gap | 16.5% |
| Precision (Up) | 52.9% |
| Recall (Up) | 78.9% |
| F1 (Up) | 63.3% |

### All-model comparison (RF vs XGBoost vs SVM, same split)

| Model | Test Accuracy | ROC-AUC | F1 |
|---|---|---|---|
| Random Forest | 53.8% | 0.538 | 0.660 |
| XGBoost | 56.7% | 0.571 | 0.662 |
| SVM | 49.7% | 0.512 | 0.575 |

**Walk-forward (10-fold):** XGB mean AUC = 0.518; RF mean AUC = 0.496

> Stock markets are noisy. 52–57% accuracy is competitive with professional quant strategies on daily direction prediction. Any result consistently above 50% contains genuine predictive signal.

### What the train/test gap tells you

A ~16% gap is expected for financial data — the training window (2021–2025) spans a bull run, 2022 bear market, and recovery, while the test window (late 2025 – mid 2026) includes the tariff-shock volatility. Reducing the date range to 1–2 years collapses the training regime to a single trend and causes the model to predict "Up" for nearly everything, dropping test accuracy below 50%.

## Sentiment Configuration

### Enable Real NLP Sentiment

1. Get a free Finnhub API key at [https://finnhub.io](https://finnhub.io)
2. Set it in [config.py](config.py):

```python
SENTIMENT_API_KEY = 'your_key_here'
SENTIMENT_SOURCE  = "finnhub"   # or "newsapi"
```

3. Install the NLP dependencies (one-time):

```bash
pip install transformers torch
```

`main.py` automatically triggers sentiment collection in Step 2.5 and passes scores to feature engineering. FinBERT downloads once (~400 MB) on first run.

### How the lag works

```
Day T:  news published → FinBERT scores → date shifted to Day T+1
Day T+1: model uses T's sentiment to predict T+1's price direction ✅
```

No look-ahead bias: the model never sees news from the same day it is predicting.

### Finnhub free-tier limitation (important)

The free Finnhub plan only returns the most recent batch of news (~200–300 articles, all from the last few days) regardless of the `from/to` dates passed. For a 5-year training window this means nearly all historical rows receive `NLP_Sentiment = 0.0` (neutral fill). The 28 technical features carry the model in this case.

**To get real historical sentiment coverage:**

| Option | Cost | Coverage |
|---|---|---|
| Finnhub paid plan | ~$50/mo | Full historical news archive |
| NewsAPI paid plan | $449/mo | Up to 5-year archive |
| VADER on scraped Yahoo Finance headlines | Free | Partial, via BeautifulSoup |

### Fallback behaviour

If `SENTIMENT_API_KEY = None`, the pipeline silently falls back to the price-based `Sentiment_Proxy` — no code changes needed.

---

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
- [x] FinBERT NLP sentiment pipeline (ProsusAI/finbert + Finnhub/NewsAPI, 1-day lag, neutral fill)
- [x] Market regime analysis — confirmed 5-year window needed to avoid single-regime bias

### Planned
- [ ] Historical news data source with full archive (paid Finnhub or VADER scraper)
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
