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

This installs: `pandas`, `numpy`, `scikit-learn`, `xgboost`, `yfinance`, `matplotlib`, `seaborn`, `tqdm`, and more.

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

**Expected runtime**: 3–5 minutes for one stock.

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
python src/models.py               # Train Random Forest + SVM
python src/xgboost_model.py        # Train XGBoost
python src/backtesting.py          # Walk-forward validation
```

Each module has a built-in `run_example()` function.

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
✅ Downloaded 756 days of data for AAPL

STEP 3: FEATURE ENGINEERING
✅ Added 'RSI_14' feature
✅ Added 'MACD', 'MACD_signal', 'MACD_histogram' features
✅ Added Bollinger Bands features
Feature engineering complete! Total columns: 33

STEP 4: MODEL TRAINING
✅ Calibrated threshold: 0.43 (val acc=0.5821)
✅ Model training complete!

TEST SET PERFORMANCE:
  Accuracy:  0.6021
  Precision: 0.6134
  Recall:    0.6812
  F1 Score:  0.6456
```

### Metrics Explained
| Metric | Meaning |
|---|---|
| Accuracy 60% | Correct predictions 60% of the time |
| Precision 61% | When we predict "up", we're right 61% of the time |
| Recall 68% | We catch 68% of actual "up" days |
| >50% accuracy | Theoretically profitable — this is the target |

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
A: 3–5 minutes for one stock; 15–25 minutes for five stocks.

**Q: What's good accuracy?**
A: 55–65% is realistic. >60% is strong for daily stock direction prediction.

**Q: Why does Random Forest use a calibrated threshold?**
A: `class_weight='balanced'` compresses predicted probabilities, making the raw 0.5 cut-off too conservative. The last 20% of training data is used to find a better threshold — no test data is touched.

**Q: Can I use this for real trading?**
A: No. This is for learning only. Real trading needs risk management, transaction costs, slippage, and much more extensive testing.

**Q: Why no `shuffle=True` in the train/test split?**
A: Stock prices are time-ordered. Shuffling would let the model "see the future" during training, inflating accuracy (data leakage).

---

**You're ready to start! Take it step by step, read the code comments, and experiment freely.**
