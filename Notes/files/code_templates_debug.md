# 🛠️ Code Templates & Debugging Guide

## Essential Code Snippets (Copy-Paste Starters)

### 1. Data Loading Template
```python
import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

def load_stock_data(ticker, start_date, end_date):
    """
    Load stock data with error handling
    """
    try:
        data = yf.download(ticker, start=start_date, end=end_date, progress=False)
        
        if data.empty:
            raise ValueError(f"No data found for {ticker}")
        
        # Reset index to make Date a column
        data.reset_index(inplace=True)
        
        print(f"✅ Loaded {len(data)} days of data for {ticker}")
        print(f"Date range: {data['Date'].min()} to {data['Date'].max()}")
        
        return data
    
    except Exception as e:
        print(f"❌ Error loading data: {e}")
        return None

# Usage
data = load_stock_data("AAPL", "2023-01-01", "2024-01-01")
```

### 2. Feature Engineering Template
```python
def add_technical_indicators(df):
    """
    Add technical indicators to dataframe
    """
    df = df.copy()
    
    # 1. Returns
    df['Returns'] = df['Close'].pct_change()
    
    # 2. Rolling averages
    for window in [5, 10, 20]:
        df[f'SMA_{window}'] = df['Close'].rolling(window=window).mean()
    
    # 3. Volatility (rolling standard deviation)
    df['Volatility'] = df['Returns'].rolling(window=20).std()
    
    # 4. Price momentum
    df['Momentum'] = df['Close'] - df['Close'].shift(5)
    
    # 5. Volume change
    df['Volume_Change'] = df['Volume'].pct_change()
    
    # 6. High-Low spread
    df['HL_Spread'] = (df['High'] - df['Low']) / df['Close']
    
    return df

def create_target_variable(df, horizon=1):
    """
    Create binary target: 1 if price goes up tomorrow, 0 otherwise
    
    horizon: how many days ahead to predict
    """
    df = df.copy()
    
    # Get future price
    df['Future_Close'] = df['Close'].shift(-horizon)
    
    # Create binary target
    df['Target'] = (df['Future_Close'] > df['Close']).astype(int)
    
    # Remove last row (no future data)
    df = df[:-horizon]
    
    return df
```

### 3. Data Splitting (Critical for Time Series!)
```python
def prepare_train_test_data(df, test_size=0.2, feature_cols=None):
    """
    Split data chronologically and prepare features/target
    
    IMPORTANT: No random shuffling for time series!
    """
    # Drop NaN rows (from rolling windows and target creation)
    df = df.dropna()
    
    # If no features specified, use all except target and date
    if feature_cols is None:
        exclude_cols = ['Date', 'Target', 'Future_Close']
        feature_cols = [col for col in df.columns if col not in exclude_cols]
    
    # Chronological split
    split_idx = int(len(df) * (1 - test_size))
    
    train = df.iloc[:split_idx]
    test = df.iloc[split_idx:]
    
    # Separate features and target
    X_train = train[feature_cols]
    y_train = train['Target']
    X_test = test[feature_cols]
    y_test = test['Target']
    
    print(f"Training samples: {len(X_train)}")
    print(f"Testing samples: {len(X_test)}")
    print(f"Features: {len(feature_cols)}")
    print(f"\nFeature list: {feature_cols}")
    
    return X_train, X_test, y_train, y_test, feature_cols
```

### 4. Model Training Template
```python
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

def train_and_evaluate_model(X_train, y_train, X_test, y_test, feature_names):
    """
    Train Random Forest and show comprehensive evaluation
    """
    # Initialize model
    model = RandomForestClassifier(
        n_estimators=100,
        max_depth=10,
        min_samples_split=5,
        random_state=42,
        n_jobs=-1
    )
    
    # Train
    print("Training model...")
    model.fit(X_train, y_train)
    
    # Predict
    train_pred = model.predict(X_train)
    test_pred = model.predict(X_test)
    
    # Evaluate
    print("\n" + "="*50)
    print("TRAINING SET PERFORMANCE")
    print("="*50)
    print(f"Accuracy: {accuracy_score(y_train, train_pred):.4f}")
    
    print("\n" + "="*50)
    print("TEST SET PERFORMANCE")
    print("="*50)
    print(f"Accuracy: {accuracy_score(y_test, test_pred):.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, test_pred))
    
    # Confusion Matrix
    cm = confusion_matrix(y_test, test_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title('Confusion Matrix')
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.show()
    
    # Feature Importance
    plot_feature_importance(model, feature_names)
    
    return model

def plot_feature_importance(model, feature_names, top_n=10):
    """
    Plot top N important features
    """
    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1][:top_n]
    
    plt.figure(figsize=(10, 6))
    plt.title(f'Top {top_n} Feature Importances')
    plt.bar(range(top_n), importances[indices])
    plt.xticks(range(top_n), [feature_names[i] for i in indices], rotation=45, ha='right')
    plt.tight_layout()
    plt.show()
    
    # Print numerical values
    print("\nTop Feature Importances:")
    for idx in indices:
        print(f"{feature_names[idx]}: {importances[idx]:.4f}")
```

### 5. Sentiment Analysis Template
```python
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

def analyze_text_sentiment(text):
    """
    Analyze sentiment of text using VADER
    
    Returns:
        compound score: -1 (most negative) to 1 (most positive)
    """
    analyzer = SentimentIntensityAnalyzer()
    scores = analyzer.polarity_scores(text)
    
    return {
        'compound': scores['compound'],  # Overall score
        'positive': scores['pos'],
        'negative': scores['neg'],
        'neutral': scores['neu']
    }

def create_price_based_sentiment(df, window=5):
    """
    Create sentiment proxy from price momentum
    (Use this while learning, before implementing real NLP)
    """
    df = df.copy()
    
    # Calculate momentum
    df['Price_Momentum'] = df['Close'].pct_change(window)
    
    # Normalize to -1 to 1 range (simple approach)
    df['Sentiment_Proxy'] = df['Price_Momentum'].clip(-0.1, 0.1) / 0.1
    
    # Add smoothed sentiment
    df['Sentiment_SMA'] = df['Sentiment_Proxy'].rolling(window=3).mean()
    
    return df
```

### 6. Backtesting Template
```python
from sklearn.model_selection import TimeSeriesSplit

def walk_forward_backtest(df, feature_cols, n_splits=5):
    """
    Perform walk-forward validation
    This simulates real trading where you:
    1. Train on past data
    2. Predict next period
    3. Retrain with new data
    4. Repeat
    """
    tscv = TimeSeriesSplit(n_splits=n_splits)
    
    # Prepare data
    df_clean = df.dropna()
    X = df_clean[feature_cols]
    y = df_clean['Target']
    
    results = []
    all_predictions = []
    
    for fold, (train_idx, test_idx) in enumerate(tscv.split(X)):
        print(f"\nFold {fold + 1}/{n_splits}")
        print(f"Train size: {len(train_idx)}, Test size: {len(test_idx)}")
        
        # Split data
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
        
        # Train model for this fold
        model = RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            random_state=42,
            n_jobs=-1
        )
        model.fit(X_train, y_train)
        
        # Predict
        predictions = model.predict(X_test)
        accuracy = accuracy_score(y_test, predictions)
        
        print(f"Accuracy: {accuracy:.4f}")
        
        results.append({
            'fold': fold + 1,
            'train_size': len(train_idx),
            'test_size': len(test_idx),
            'accuracy': accuracy
        })
        
        all_predictions.extend(predictions)
    
    # Summary
    results_df = pd.DataFrame(results)
    print("\n" + "="*50)
    print("BACKTEST SUMMARY")
    print("="*50)
    print(results_df)
    print(f"\nMean Accuracy: {results_df['accuracy'].mean():.4f}")
    print(f"Std Accuracy: {results_df['accuracy'].std():.4f}")
    
    return results_df
```

---

## 🐛 Common Errors & Solutions

### Error 1: "ValueError: Input contains NaN"
**Cause:** Rolling windows create NaN values at the start of your data

**Solution:**
```python
# After feature engineering, always:
df = df.dropna()

# Or be more specific:
df = df.dropna(subset=['SMA_20', 'Volatility'])  # Drop if these are NaN
```

### Error 2: "Look-ahead bias detected"
**Cause:** Using future data to predict the past

**Solution:**
```python
# WRONG:
df['Target'] = (df['Close'].shift(1) > df['Close']).astype(int)  # Uses past to predict past

# RIGHT:
df['Target'] = (df['Close'].shift(-1) > df['Close']).astype(int)  # Predicts future
```

### Error 3: "Different lengths for X and y"
**Cause:** Dropping NaN inconsistently

**Solution:**
```python
# Do this ONCE at the end of feature engineering
df_clean = df.dropna()
X = df_clean[feature_cols]
y = df_clean['Target']
```

### Error 4: Low accuracy (~50%)
**Possible causes:**
1. Model is basically guessing (random)
2. Features are not predictive
3. Target is too noisy

**Debug steps:**
```python
# Check target distribution
print(y_train.value_counts())  # Should be roughly balanced

# Check for data leakage
print(X_train.columns)  # Should NOT include 'Target', 'Future_Close'

# Try simpler model first
from sklearn.tree import DecisionTreeClassifier
simple_model = DecisionTreeClassifier(max_depth=3)
# If this also gets ~50%, problem is with features/data
```

### Error 5: "Model performs better on training than test"
**Cause:** Overfitting

**Solutions:**
```python
# Reduce model complexity
model = RandomForestClassifier(
    n_estimators=50,      # Fewer trees
    max_depth=5,          # Shallower trees
    min_samples_split=10  # Need more samples to split
)

# Or add regularization via min_samples_leaf
model = RandomForestClassifier(
    min_samples_leaf=5    # Each leaf needs 5+ samples
)
```

---

## 📊 Quick Reference: Key Pandas Operations

```python
# Data inspection
df.head()              # First 5 rows
df.info()              # Column types and non-null counts
df.describe()          # Statistical summary
df.isnull().sum()      # Count missing values

# Feature engineering
df['new_col'] = df['old_col'].pct_change()     # Percentage change
df['rolling'] = df['col'].rolling(5).mean()    # 5-period average
df['future'] = df['col'].shift(-1)             # Next value
df['past'] = df['col'].shift(1)                # Previous value

# Data cleaning
df.dropna()                                    # Remove NaN rows
df.fillna(0)                                   # Fill NaN with 0
df.drop(columns=['col1', 'col2'])             # Remove columns

# Filtering
df[df['Close'] > 100]                         # Rows where Close > 100
df.loc[df['Date'] > '2023-01-01']            # Rows after date

# Sorting
df.sort_values('Date')                        # Sort by date
df.sort_values('Close', ascending=False)      # Sort descending
```

---

## 🎯 Debugging Checklist Before Asking for Help

When stuck, go through this checklist:

- [ ] Did I print the shape of my data? `print(df.shape)`
- [ ] Did I check for NaN values? `print(df.isnull().sum())`
- [ ] Did I verify my train/test split is chronological?
- [ ] Did I check target distribution? `print(y.value_counts())`
- [ ] Did I verify no future data leaks into features?
- [ ] Did I check feature correlations? `df.corr()`
- [ ] Did I try a simpler model first?
- [ ] Did I read the error message carefully?
- [ ] Did I Google the exact error message?

---

## 💡 Pro Tips

1. **Start Simple:** Build the simplest version first, then add complexity
2. **Print Everything:** When debugging, print shapes, types, and sample values
3. **Comment Your Code:** Future you will thank present you
4. **Save Checkpoints:** Save data after each major step
```python
df.to_csv('data/checkpoint_features.csv', index=False)
```

5. **Use Version Control:** Even basic git commit messages help track progress
6. **Experiment in Notebooks:** Test ideas in Jupyter before adding to .py files
7. **Document Findings:** Keep a learning journal of what works and what doesn't

---

## 🎓 When You Get Stuck

1. **Read the error message** (seriously, read it twice)
2. **Print intermediate results** to see where it breaks
3. **Simplify the problem** - test with smaller data first
4. **Google the exact error** with "pandas" or "sklearn" keywords
5. **Check Stack Overflow** - someone has had your problem
6. **Review the learning resources** - maybe you missed a concept
7. **Take a break** - sometimes stepping away helps!

Remember: Every error is a learning opportunity. Don't get frustrated - even experienced developers debug constantly!

Good luck! 🚀
