# Getting Started Guide

## Step-by-Step Instructions for Running Your Project

### Prerequisites
- [ ] Python 3.8+ installed
- [ ] Internet connection (to download stock data)
- [ ] Text editor or IDE (VS Code, PyCharm, etc.)
- [ ] Terminal / Command Prompt access

---

## Part 1: Initial Setup (One-Time)

### Step 1: Navigate to Project Directory

```bash
cd "path/to/stock_ml_project"
```

Your project folder should look like this:

```
stock_ml_project/
├── src/
│   ├── data_collection.py
│   ├── preprocessing.py
│   ├── feature_engineering.py
│   ├── models.py
│   ├── xgboost_model.py
│   └── backtesting.py
├── comparison/
│   └── compare_models.py
├── data/
│   ├── raw/
│   ├── processed/
│   └── features/
├── outputs/
├── config.py
├── main.py
├── requirements.txt
├── Getting_Started.md
└── README.md
```

### Step 2: Create Virtual Environment

```bash
python -m venv venv

# Windows:
venv\Scripts\activate

# Mac/Linux:
source venv/bin/activate

# You should see (venv) at the start of the command line
```

### Step 3: Install Packages

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

This installs: `pandas`, `numpy`, `scikit-learn`, `xgboost`, `yfinance`, `matplotlib`, `seaborn`, `tqdm`, `transformers`, `torch`, and more.

> **Note on `transformers` / `torch`**: These are large packages (~1–2 GB total including the FinBERT model weights downloaded on first use). If disk space is limited and you don't need live NLP sentiment, you can skip them — the pipeline falls back to the price-based proxy automatically.

### Step 4: Verify Installation

Create `test_setup.py` (already in the project root):

```python
import pandas as pd
import numpy as np
import yfinance as yf
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
import matplotlib.pyplot as plt

print("✅ All packages imported successfully!")

data = yf.download("AAPL", period="5d", progress=False)
print(f"✅ Downloaded {len(data)} days of AAPL data")
print("\n🎉 Setup complete!")
```

```bash
python test_setup.py
```

---

## Part 2: Running the Project

### Option A: Interactive Menu (Recommended)

```bash
python main.py
```

Menu options:
```
1. Run pipeline for a single stock
2. Run pipeline for all configured stocks
3. Download data for all stocks (preparation)
4. Exit
```

**First time? Choose option 1, then enter `AAPL`.**

What happens:
1. Downloads AAPL data (3 years from Yahoo Finance)
2. Cleans and validates the data
3. Engineers 25+ technical indicators
4. Trains a Random Forest model (with threshold calibration)
5. Evaluates accuracy, precision, recall, F1
6. Runs 10-fold walk-forward backtesting

**Expected runtime**: 3–5 minutes for one stock (first run ~15 min — FinBERT model downloads ~400 MB).

### Option B: Full Model Comparison (RF vs XGBoost vs SVM)

```bash
python comparison/compare_models.py
```

This trains all three models on the same data, generates ROC curve and metrics comparison plots, and saves results to `outputs/comparison_results.csv`.

### Option C: Test Individual Modules

```bash
python src/data_collection.py      # Download & load data
python src/preprocessing.py        # Clean data
python src/feature_engineering.py  # Engineer features
python src/sentiment_analysis.py   # Test NLP sentiment (NEW)
python src/models.py               # Train Random Forest + SVM
python src/xgboost_model.py        # Train XGBoost
python src/backtesting.py          # Walk-forward validation
```

Each module has a built-in `run_example()` or `__main__` test block.

### Option D: Python Script

```python
from main import run_complete_pipeline

results = run_complete_pipeline('AAPL')

if results:
    accuracy = results['model_results']['metrics']['test_accuracy']
    print(f"Test Accuracy: {accuracy:.2%}")
```

---

## Part 3: Understanding the Output

### Sample Output

```
STEP 1: DATA COLLECTION
✅ Downloaded 1254 days of data for AAPL

STEP 2.5: NLP SENTIMENT COLLECTION
  [sentiment] Collecting sentiment for AAPL…
  [sentiment] 247 headlines fetched for AAPL.
  [sentiment] Loading FinBERT model (first run downloads ~400 MB)…
  [sentiment] FinBERT ready.
Sentiment data shape: (1826, 3)

STEP 3: FEATURE ENGINEERING
✅ Added 'RSI_14' feature
✅ Added 'MACD', 'MACD_signal', 'MACD_histogram' features (% of price)
✅ Added 'BB_width', 'BB_position' features
✅ Merged NLP sentiment (1-day lag applied; 1252 days filled neutral)
Feature engineering complete! Total columns: 39

STEP 4: MODEL TRAINING
✅ Model training complete!

TEST SET PERFORMANCE:
  Accuracy:  0.5263
  Precision: 0.5288
  Recall:    0.7891
  F1 Score:  0.6332
```

### Metrics Explained
| Metric | Value | Meaning |
|---|---|---|
| Accuracy 52.6% | ✅ | Correct direction predictions on unseen data — above 50% random baseline |
| Precision 52.9% | ✅ | When we predict "up", we're right 53% of the time |
| Recall 78.9% | ✅ | We catch 79% of actual "up" days |
| F1 63.3% | ✅ | Balanced precision/recall score |
| Train/test gap 16.5% | ⚠️ | Expected for financial data spanning multiple market regimes |

### Why the sentiment shows "X days filled neutral"

The Finnhub **free tier** only returns recent news (last few days), not the full 5-year archive. Rows without matched news receive `NLP_Sentiment = 0.0` and `News_Volume = 0`. Random Forest ignores constant-value features automatically, so the 28 technical indicators carry the model. Upgrading to a paid Finnhub plan unlocks full historical coverage.

### Generated Files

```
data/raw/AAPL_raw.csv          ← Raw downloaded data
data/processed/AAPL_processed.csv  ← After cleaning
data/features/AAPL_features.csv    ← After feature engineering

outputs/
├── roc_comparison.png              ← ROC curves (all models)
├── metrics_comparison.png          ← Bar chart comparison
├── feature_importance_comparison.png
└── comparison_results.csv         ← Metrics table
```

---

## Part 3.5: Setting Up NLP Sentiment (Optional but Recommended)

### Step 1: Get a Finnhub API Key

Sign up for free at [https://finnhub.io](https://finnhub.io).

> **Free-tier limitation**: The free plan only returns the most recent ~250 news articles (all from the last few days), regardless of the date range you request. For a 5-year training window, most rows will receive `NLP_Sentiment = 0.0`. This is handled gracefully by the pipeline — the 28 technical features still drive the model. A paid Finnhub plan is required for full historical coverage.

### Step 2: Set the Key in config.py

```python
# config.py
SENTIMENT_API_KEY = 'your_finnhub_key_here'
SENTIMENT_SOURCE  = "finnhub"
```

### Step 3: Test the Sentiment Module

```bash
python src/sentiment_analysis.py
```

Expected output:
```
[sentiment] Collecting sentiment for AAPL…
[sentiment] 87 headlines fetched for AAPL.
[sentiment] Loading FinBERT model (first run downloads ~400 MB)…
[sentiment] FinBERT ready.
Sentiment for AAPL — last 5 days:
         Date  NLP_Sentiment  News_Volume
...
Mean NLP_Sentiment : 0.142
Days with news     : 63 / 92
```

After this, `python main.py` will automatically run Step 2.5 (sentiment collection) before feature engineering.

---

## Part 4: Customization

### Change Which Stocks to Analyze

```python
# config.py
TICKERS = ['NVDA', 'AMD', 'INTC']
```

### Change Date Range

```python
# config.py
START_DATE = '2022-01-01'
```

### Tune Model Parameters

```python
# config.py — Random Forest (more conservative → less overfitting)
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
    'C': 5.0,
    'kernel': 'rbf',
    'gamma': 'auto',
}
```

---

## Part 5: Troubleshooting

### "pip: command not found"
```bash
python -m pip install -r requirements.txt
```

### "python: command not found"
```bash
python3 main.py
```

### PowerShell won't activate venv
```powershell
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
venv\Scripts\Activate.ps1
```

### "No data found for ticker"
- Use uppercase ticker symbols: `AAPL` not `aapl`
- Check internet connection
- Try a well-known ticker first

### "ValueError: Input contains NaN"
Expected from rolling windows — `prepare_ml_data()` handles this automatically via `dropna()`.

### Low accuracy (~50%)
Normal for financial data. Try a longer date range or more features.

### Plots not showing
```python
import matplotlib
matplotlib.use('TkAgg')  # or 'Qt5Agg'
import matplotlib.pyplot as plt
```

### FinBERT / transformers errors

```
ImportError: No module named 'transformers'
```
```bash
pip install transformers torch
```

```
OSError: Can't load tokenizer for 'ProsusAI/finbert'
```
Check your internet connection — the model is downloaded from Hugging Face on first run.

### Sentiment returns all zeros
- Verify `SENTIMENT_API_KEY` is set in `config.py` (not `None`)
- Check your Finnhub key is valid: `curl "https://finnhub.io/api/v1/company-news?symbol=AAPL&from=2024-01-01&to=2024-01-07&token=YOUR_KEY"`

### Out of memory
```python
# config.py — use less data
START_DATE = '2023-01-01'
TICKERS = ['AAPL']  # start with one stock
```

---

## Part 6: Daily Workflow

```bash
# Every session:
venv\Scripts\activate        # Windows
# source venv/bin/activate  # Mac/Linux

python main.py               # Run the pipeline

deactivate                   # When done
```

---

## Part 7: Learning Path

### Week 1: Get it Running
- [ ] Complete setup and verify installation
- [ ] Run pipeline for AAPL (`python main.py` → option 1)
- [ ] Read `config.py` and understand the settings
- [ ] Open the generated CSV files in Excel

### Week 2: Explore Models
- [ ] Run `python comparison/compare_models.py`
- [ ] Compare RF vs XGBoost vs SVM performance
- [ ] Look at feature importance plots
- [ ] Try different tickers (MSFT, TSLA, NVDA)

### Week 3: Customize
- [ ] Add a new technical indicator in `feature_engineering.py`
- [ ] Tune hyperparameters in `config.py`
- [ ] Extend `MOMENTUM_WINDOWS` or `ROLLING_WINDOWS`
- [ ] Experiment with `N_SPLITS` for backtesting

### Week 4: Go Deeper
- [ ] Read about each indicator (RSI, MACD, Bollinger Bands)
- [ ] Understand walk-forward vs random CV
- [ ] Learn why `class_weight='balanced'` needs threshold calibration
- [ ] Explore overfitting signals (train vs test accuracy gap)

---

## Quick Reference

```bash
# Setup (one time)
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
python test_setup.py

# Daily use
venv\Scripts\activate
python main.py                        # Full pipeline (menu)
python comparison/compare_models.py   # Compare all three models
deactivate

# Individual modules
python src/data_collection.py
python src/feature_engineering.py
python src/models.py
python src/xgboost_model.py
python src/backtesting.py

# Useful checks
pip list                   # See installed packages
pip install --upgrade pandas  # Update a package
```

---

## Common Questions

**Q: How long does it take to run?**
A: First run ~15 minutes for one stock (FinBERT downloads ~400 MB). Subsequent runs 3–5 minutes. Five stocks ~20–30 minutes.

**Q: What's good accuracy?**
A: 52–56% is realistic for daily direction prediction. Anything consistently above 50% contains genuine signal. >58% is exceptional.

**Q: Why does the test accuracy drop below 50% when I use 1–2 years of data?**
A: A short window often captures a single market regime (e.g., 2024–2025 bull run). The model learns "mostly predict Up" and fails when the test period has a correction. Use the 5-year window (`5 * 365` in config.py) to train across bear, recovery, and bull regimes.

**Q: Why does the pipeline show "1252 days filled neutral" for sentiment?**
A: Finnhub's free tier only returns the most recent ~250 news articles. Rows without matched news are filled with `NLP_Sentiment = 0.0`. Random Forest ignores these constant-value rows automatically. A paid Finnhub plan provides full historical coverage.

**Q: Why does Random Forest use a calibrated threshold?**
A: The threshold calibration finds the optimal probability cut-off on the last 20% of training data — no test data is touched. This is especially important when class imbalance shifts the raw 0.5 cut-off.

**Q: Can I use this for real trading?**
A: No. This is for learning only. Real trading needs risk management, transaction costs, slippage, and much more extensive testing.

**Q: Why no `shuffle=True` in the train/test split?**
A: Stock prices are time-ordered. Shuffling would let the model "see the future" during training, inflating accuracy (data leakage).

---

**You're ready to start! Take it step by step, read the code comments, and experiment freely.**
