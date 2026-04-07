# Stock Market ML Project - Setup Guide

## Project Structure
```
stock-prediction/
├── data/
│   ├── raw/              # Raw downloaded data
│   ├── processed/        # Cleaned data
│   └── features/         # Engineered features
├── notebooks/
│   ├── 01_data_exploration.ipynb
│   ├── 02_feature_engineering.ipynb
│   ├── 03_model_training.ipynb
│   └── 04_backtesting.ipynb
├── src/
│   ├── data_collection.py
│   ├── preprocessing.py
│   ├── feature_engineering.py
│   ├── models.py
│   └── backtesting.py
├── requirements.txt
├── config.py
└── main.py
```

## Environment Setup

### Step 1: Create Virtual Environment
```bash
# Create project directory
mkdir stock-prediction
cd stock-prediction

# Create virtual environment
python -m venv venv

# Activate it
# On Windows:
venv\Scripts\activate
# On Mac/Linux:
source venv/bin/activate
```

### Step 2: Install Dependencies
Create `requirements.txt` with these packages:

```
# Data handling
pandas==2.1.0
numpy==1.25.0

# Data collection
yfinance==0.2.28
requests==2.31.0

# Machine Learning
scikit-learn==1.3.0

# Visualization
matplotlib==3.7.2
seaborn==0.12.2

# Sentiment Analysis (for later)
vaderSentiment==3.3.2
transformers==4.30.0
torch==2.0.1

# Jupyter for notebooks
jupyter==1.0.0
ipython==8.14.0

# Utilities
python-dotenv==1.0.0
tqdm==4.65.0
```

Install with:
```bash
pip install -r requirements.txt
```

### Step 3: Verify Installation
```python
# test_setup.py
import pandas as pd
import numpy as np
import yfinance as yf
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt

print("✅ All packages imported successfully!")

# Test data download
test_stock = yf.download("AAPL", start="2024-01-01", end="2024-01-10")
print(f"✅ Downloaded {len(test_stock)} days of AAPL data")
```

## Your First Task (Day 1)

### Task 1.1: Setup the structure
Create all the folders and files shown above

### Task 1.2: Basic Data Exploration
Create `notebooks/01_data_exploration.ipynb` and:
1. Download 1 year of data for AAPL, MSFT, GOOGL
2. Print the first 10 rows
3. Check for missing values
4. Create a simple line plot of closing prices

**Starter snippet:**
```python
import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt

# Download data
ticker = "AAPL"
data = yf.download(ticker, start="2023-01-01", end="2024-01-01")

# YOUR CODE HERE:
# 1. Print first 10 rows using .head(10)
# 2. Check missing values using .isnull().sum()
# 3. Plot closing price using data['Close'].plot()
```

### Task 1.3: Understanding the Data
Answer these questions in your notebook (use code):
- What are the columns in the data?
- What's the date range?
- What's the average daily price change?
- On what date was the highest closing price?

**Learning Checkpoint:** You should understand what OHLCV data means by end of Day 1
