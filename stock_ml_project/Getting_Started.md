# 🚀 Getting Started Guide

## Step-by-Step Instructions for Running Your Project

### Prerequisites Checklist
- [ ] Python 3.8+ installed
- [ ] Internet connection (to download stock data)
- [ ] Text editor or IDE (VS Code, PyCharm, or even Notepad)
- [ ] Terminal/Command Prompt access

---

## 📦 Part 1: Initial Setup (One-Time)

### Step 1: Navigate to Project Directory
```bash
# Open terminal/command prompt
# Navigate to where you want to create the project
cd Desktop
# Or wherever you want the project folder

# Create and enter the project directory
mkdir stock-prediction
cd stock-prediction
```

### Step 2: Copy All Project Files
Copy all the files from the guides into your `stock-prediction` folder:
- `requirements.txt`
- `config.py`
- `main.py`
- `README.md`
- All files in `src/` folder

Your folder should look like this:
```
stock-prediction/
├── src/
│   ├── data_collection.py
│   ├── preprocessing.py
│   ├── feature_engineering.py
│   ├── models.py
│   └── backtesting.py
├── config.py
├── main.py
├── requirements.txt
└── README.md
```

### Step 3: Create Virtual Environment
```bash
# Create virtual environment named 'venv'
python -m venv venv

# Activate it (choose based on your OS):

# Windows:
venv\Scripts\activate

# Mac/Linux:
source venv/bin/activate

# You should see (venv) at start of command line
```

### Step 4: Install Packages
```bash
# First, upgrade pip
python -m pip install --upgrade pip

# Install setuptools (if you had the error before)
pip install setuptools wheel

# Install all project dependencies
pip install -r requirements.txt

# This will take a few minutes - be patient!
```

### Step 5: Verify Installation
Create a file called `test_install.py`:

```python
# test_install.py
import pandas as pd
import numpy as np
import yfinance as yf
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt

print("✅ All packages imported successfully!")

# Test downloading data
print("\nTesting data download...")
data = yf.download("AAPL", period="5d", progress=False)
print(f"✅ Downloaded {len(data)} days of AAPL data")

print("\n🎉 Setup complete! You're ready to start!")
```

Run it:
```bash
python test_install.py
```

If you see "✅ Setup complete!", you're good to go!

---

## 🎮 Part 2: Running the Project

### Option A: Interactive Mode (Recommended for Beginners)

```bash
python main.py
```

You'll see a menu:
```
What would you like to do?
1. Run pipeline for a single stock
2. Run pipeline for all configured stocks
3. Download data for all stocks (preparation)
4. Exit

Enter your choice (1-4):
```

**First time? Choose option 1:**
- Enter `1`
- When asked for ticker, type: `AAPL`
- Press Enter and watch it run!

**What will happen**:
1. Downloads AAPL data from 2021-2024
2. Cleans the data
3. Creates features
4. Trains model
5. Shows accuracy
6. Generates plots (close them to continue)

**Expected runtime**: 2-3 minutes

### Option B: Run Specific Steps Only

#### Just Download Data
```bash
python src/data_collection.py
```

#### Just Test Feature Engineering
```bash
python src/feature_engineering.py
```

#### Just Train a Model
```bash
python src/models.py
```

Each module has examples built in!

### Option C: Python Script

Create `my_analysis.py`:
```python
from main import run_complete_pipeline

# Run pipeline for Apple stock
results = run_complete_pipeline('AAPL')

# Check accuracy
if results:
    accuracy = results['model_results']['metrics']['test_accuracy']
    print(f"\nFinal Accuracy: {accuracy:.2%}")
```

Run it:
```bash
python my_analysis.py
```

---

## 📊 Part 3: Understanding the Output

### What You'll See

1. **Data Collection Output**:
```
Downloading AAPL data from 2021-01-01 to 2024-01-01...
✅ Downloaded 756 days of data for AAPL
✅ Saved 756 rows to data/raw/AAPL_raw.csv
```

2. **Feature Engineering Output**:
```
✅ Added 'Returns' feature
✅ Added 'SMA_5' feature
✅ Added 'SMA_10' feature
...
Feature engineering complete!
Total columns: 18
```

3. **Model Training Output**:
```
Training model...
✅ Model training complete!

TEST SET PERFORMANCE:
  Accuracy:  0.6622
  Precision: 0.6891
  Recall:    0.7234
  F1 Score:  0.7058
```

4. **What the Metrics Mean**:
- **Accuracy 66%**: Correct predictions 66% of the time
- **Precision 69%**: When we predict "up", we're right 69% of the time
- **Recall 72%**: We catch 72% of actual "up" days
- **This is GOOD for stock prediction!** (>50% is profitable)

### Generated Files

After running, you'll have:
```
data/
├── raw/
│   └── AAPL_raw.csv          # Downloaded data
├── processed/
│   └── AAPL_processed.csv    # Cleaned data
└── features/
    └── AAPL_features.csv     # Data with all features
```

You can open these CSV files in Excel to see the data!

---

## 🔧 Part 4: Customization

### Change Which Stock to Analyze

Edit `config.py`, find this line:
```python
TICKERS = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA']
```

Change to your preferred stocks:
```python
TICKERS = ['NVDA', 'AMD', 'INTC']  # Semiconductor stocks
```

### Change Date Range

Edit `config.py`:
```python
START_DATE = '2022-01-01'  # Start from 2022
END_DATE = '2024-12-31'    # Up to end of 2024
```

### Change Model Parameters

Edit `config.py`, find `RF_PARAMS`:
```python
RF_PARAMS = {
    'n_estimators': 200,    # Try more trees (slower but potentially better)
    'max_depth': 15,        # Allow deeper trees
    'min_samples_split': 10 # More conservative
}
```

**After any config change**:
```bash
# Re-run the pipeline
python main.py
```

---

## 🐛 Part 5: Troubleshooting

### Problem: "pip: command not found"
**Solution**: Use `python -m pip` instead:
```bash
python -m pip install -r requirements.txt
```

### Problem: "python: command not found"
**Solution**: Try `python3` instead:
```bash
python3 -m venv venv
python3 main.py
```

### Problem: Virtual environment won't activate
**Windows PowerShell**:
```powershell
# Enable script execution first
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
# Then activate
venv\Scripts\Activate.ps1
```

### Problem: "No data found for ticker"
**Possible causes**:
1. Typo in ticker symbol (use uppercase: AAPL not aapl)
2. No internet connection
3. Yahoo Finance is down (try again later)

**Solution**: Start with well-known tickers:
```python
# These always work:
'AAPL'  # Apple
'MSFT'  # Microsoft
'GOOGL' # Google
```

### Problem: Plots not showing
**Solution**: 
```python
# Add this at top of any file using plots
import matplotlib
matplotlib.use('TkAgg')  # Or 'Qt5Agg'
import matplotlib.pyplot as plt
```

### Problem: Out of memory
**Solution**: Use less data:
```python
# In config.py
START_DATE = '2023-01-01'  # Just 1 year instead of 3
```

---

## 📝 Part 6: Daily Workflow

### First Time Setup (Already done above)
1. Install Python
2. Create virtual environment
3. Install packages
4. Verify installation

### Every Time You Work on Project
```bash
# 1. Navigate to project folder
cd path/to/stock-prediction

# 2. Activate virtual environment
# Windows:
venv\Scripts\activate
# Mac/Linux:
source venv/bin/activate

# 3. Run your code
python main.py

# 4. When done, deactivate
deactivate
```

### Typical Development Session
```bash
# Activate environment
source venv/bin/activate  # or venv\Scripts\activate on Windows

# Make changes to code in your editor
# ... edit config.py, modify features, etc ...

# Test your changes
python main.py

# Look at results
# Open generated CSV files in Excel
# Review plots

# Deactivate when done
deactivate
```

---

## 🎯 Part 7: Learning Path

### Week 1: Get it Running
- [ ] Complete setup
- [ ] Run pipeline for AAPL
- [ ] Understand what each file does
- [ ] Read all the comments in the code

### Week 2: Experiment
- [ ] Try different stocks
- [ ] Change date ranges
- [ ] Modify model parameters
- [ ] Compare results

### Week 3: Customize
- [ ] Add a new feature
- [ ] Try different window sizes
- [ ] Experiment with hyperparameters
- [ ] Document what works better

### Week 4: Understand Deeply
- [ ] Read about Random Forests
- [ ] Understand each evaluation metric
- [ ] Learn about overfitting
- [ ] Explore backtesting results

---

## 💡 Quick Commands Reference

```bash
# Setup (one time)
python -m venv venv
venv\Scripts\activate  # Windows
pip install -r requirements.txt

# Daily use
venv\Scripts\activate  # Start
python main.py         # Run
deactivate            # Stop

# Testing individual parts
python src/data_collection.py
python src/models.py
python src/backtesting.py

# Check what's installed
pip list

# Update a package
pip install --upgrade pandas
```

---

## 📞 Getting Help

### If Something Doesn't Work

1. **Read the error message carefully**
   - It usually tells you what's wrong
   - Google the exact error message

2. **Check the code comments**
   - Every function has documentation
   - Examples show how to use it

3. **Try the examples**
   - Each module has a `run_example()` function
   - These are guaranteed to work

4. **Start simple**
   - Use one stock, short date range
   - Get that working first
   - Then expand

### Common Questions

**Q: How long should it take to run?**
A: 2-5 minutes for one stock, 10-20 for five stocks

**Q: What's good accuracy?**
A: 60-70% is excellent for stock prediction

**Q: Can I use this for real trading?**
A: NO! This is for learning only

**Q: Why do I need virtual environment?**
A: Keeps project dependencies isolated from other Python projects

---

## ✅ Checklist for Success

Before you start:
- [ ] Python installed and working
- [ ] Virtual environment activated
- [ ] All packages installed
- [ ] Test script runs successfully

When running pipeline:
- [ ] Internet connection active
- [ ] Sufficient disk space (~100MB)
- [ ] Config.py settings reviewed
- [ ] Terminal window large enough to see output

After completion:
- [ ] Check data/ folder for CSV files
- [ ] Review accuracy metrics
- [ ] Examine feature importance plot
- [ ] Check confusion matrix

---

**You're now ready to start your ML journey! 🚀**

**Remember**: Every expert was once a beginner. Take it step by step, read the code, and don't hesitate to experiment!

Good luck! 🍀