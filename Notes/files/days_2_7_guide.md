# Days 2-7: Detailed Learning Roadmap

---

## 📊 **Day 2: Feature Engineering Fundamentals**

### Morning Learning (2-3 hours)
**Core Concept: Features are the variables/inputs your model uses to make predictions**

**Watch:**
1. "Feature Engineering for Machine Learning" - Krish Naik (YouTube, 30 mins)
2. "Time Series Feature Engineering" - ritvikmath (YouTube, 15 mins)

**Read:**
- [Feature Engineering Tutorial](https://www.kaggle.com/learn/feature-engineering) - Kaggle Course
- Your proposal Section 7.4 - understand what rolling averages mean

**Key Questions to Answer:**
- What's a rolling average and why use it?
- What's the difference between technical indicators and raw price data?
- How do you avoid "look-ahead bias"? (Critical for stock prediction!)

### Afternoon Coding (3-4 hours)

**Your Task:** Create `src/feature_engineering.py`

**Features to implement:**
1. **Price-based features:**
   - Daily returns: (Close_today - Close_yesterday) / Close_yesterday
   - 5-day, 10-day, 20-day rolling averages
   - Price volatility (rolling standard deviation)
   - Volume changes

2. **Target variable:**
   - Binary classification: 1 if tomorrow's price > today's price, else 0

**Starter code snippet:**
```python
import pandas as pd
import numpy as np

def calculate_returns(df):
    """Calculate daily percentage returns"""
    # YOUR CODE: Use .pct_change()
    pass

def add_rolling_features(df, windows=[5, 10, 20]):
    """Add rolling average features"""
    df = df.copy()
    for window in windows:
        # YOUR CODE: Use .rolling(window).mean()
        # Example: df[f'SMA_{window}'] = ...
        pass
    return df

def create_target(df):
    """Create binary target: 1 if next day price goes up"""
    # YOUR CODE: Use .shift(-1) to get tomorrow's price
    # Compare with today's price
    pass

# Test it!
if __name__ == "__main__":
    import yfinance as yf
    data = yf.download("AAPL", start="2023-01-01", end="2024-01-01")
    # Apply your functions and print results
```

**Challenge Task:** 
- Calculate the Relative Strength Index (RSI) - look up the formula
- Google "RSI indicator formula" and implement it

**End of Day Checkpoint:**
- You should have a DataFrame with at least 8 new features
- Understand what "shifting" data means in time series
- Know why we can't use tomorrow's data to predict tomorrow (!)

---

## 🤖 **Day 3: Machine Learning Basics - Random Forest**

### Morning Learning (2-3 hours)

**Watch (MUST WATCH):**
1. **StatQuest Random Forests** - Josh Starmer (20 mins, watch BOTH parts)
   - Part 1: Building intuition
   - Part 2: Missing data and clustering
2. "Decision Trees" - StatQuest (10 mins)
3. "Training and Testing Data" - StatQuest (10 mins)

**Read:**
- [Scikit-learn Random Forest Guide](https://scikit-learn.org/stable/modules/ensemble.html#forest)
- [Random Forest Explained](https://www.analyticsvidhya.com/blog/2021/06/understanding-random-forest/)

**Key Concepts to Master:**
- What's the difference between a decision tree and random forest?
- What is "bagging" (bootstrap aggregating)?
- Why does random forest prevent overfitting better than a single tree?
- What are hyperparameters? (n_estimators, max_depth, min_samples_split)

### Afternoon Coding (3-4 hours)

**Your Task:** Create `src/models.py` and `notebooks/03_model_training.ipynb`

**Step-by-step guide:**

#### Step 1: Train/Test Split (Time Series Way)
```python
from sklearn.model_selection import TimeSeriesSplit

# IMPORTANT: For time series, don't use random split!
# Use chronological split - train on past, test on future

def split_data(df, test_size=0.2):
    """
    Split time series data chronologically
    """
    split_idx = int(len(df) * (1 - test_size))
    
    train = df.iloc[:split_idx]
    test = df.iloc[split_idx:]
    
    return train, test
```

#### Step 2: Prepare Features and Target
```python
def prepare_ml_data(df, feature_cols, target_col='Target'):
    """
    Separate features (X) and target (y)
    Remove rows with NaN (from rolling windows)
    """
    # YOUR CODE:
    # 1. Drop NaN rows
    # 2. Separate X (features) and y (target)
    # 3. Return X, y
    pass
```

#### Step 3: Build Your First Model
```python
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

def train_baseline_model(X_train, y_train, X_test, y_test):
    """
    Train a simple Random Forest
    """
    # Start simple!
    model = RandomForestClassifier(
        n_estimators=100,      # Number of trees
        max_depth=10,          # How deep each tree can go
        random_state=42,       # For reproducibility
        n_jobs=-1              # Use all CPU cores
    )
    
    # YOUR CODE:
    # 1. Fit the model: model.fit(X_train, y_train)
    # 2. Make predictions: predictions = model.predict(X_test)
    # 3. Calculate accuracy: accuracy_score(y_test, predictions)
    # 4. Print classification_report
    
    return model

# In your notebook, run this and interpret the results!
```

**Critical Learning Exercise:**
After training your first model, answer:
1. What's the accuracy? Is 60% good or bad for stock prediction?
2. Look at the confusion matrix - what does it tell you?
3. What happens if you change n_estimators from 100 to 10? To 500?
4. What about max_depth from 10 to 3? To None?

**End of Day Checkpoint:**
- Trained your first ML model
- Understand train/test split for time series
- Know what accuracy, precision, recall mean
- Started thinking about evaluation metrics

---

## 📈 **Day 4: Model Evaluation & Hyperparameter Tuning**

### Morning Learning (2 hours)

**Watch:**
1. "Cross Validation" - StatQuest (6 mins)
2. "Confusion Matrix" - StatQuest (8 mins)  
3. "ROC and AUC" - StatQuest (16 mins)

**Read:**
- [Model Evaluation Metrics](https://scikit-learn.org/stable/modules/model_evaluation.html)
- Section 7.6 of your proposal - understand backtesting

**Key Questions:**
- Why is accuracy alone not enough?
- What's precision vs recall? Which matters more for stocks?
- What is overfitting and how do you detect it?

### Afternoon Coding (4 hours)

**Task 1: Comprehensive Evaluation**
```python
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

def evaluate_model(model, X_test, y_test):
    """
    Comprehensive model evaluation
    """
    predictions = model.predict(X_test)
    
    # 1. Classification Report
    print(classification_report(y_test, predictions))
    
    # 2. Confusion Matrix
    cm = confusion_matrix(y_test, predictions)
    
    # YOUR CODE: Plot the confusion matrix using seaborn heatmap
    # plt.figure(figsize=(8,6))
    # sns.heatmap(cm, annot=True, fmt='d')
    
    # 3. Feature Importance
    # YOUR CODE: Get feature_importances_ from the model
    # Create a bar plot showing top 10 features
    
    return predictions
```

**Task 2: Hyperparameter Tuning**

First, understand what each parameter does:
```python
# Create this as a learning exercise:
def experiment_with_parameters():
    """
    Test different parameter combinations
    """
    param_grid = {
        'n_estimators': [50, 100, 200],
        'max_depth': [5, 10, 15, None],
        'min_samples_split': [2, 5, 10]
    }
    
    # YOUR TASK:
    # 1. Loop through different combinations
    # 2. Train a model with each
    # 3. Record the accuracy
    # 4. Find the best combination
    
    # This teaches you what GridSearchCV does automatically!
```

Then use scikit-learn's GridSearchCV:
```python
from sklearn.model_selection import GridSearchCV

# YOUR CODE: Implement grid search
# Remember: Use TimeSeriesSplit for cv parameter!
```

**End of Day Checkpoint:**
- Know how to read a confusion matrix
- Understand feature importance
- Can tune hyperparameters systematically
- Understand the bias-variance tradeoff

---

## 🗣️ **Day 5: Sentiment Analysis Introduction**

### Morning Learning (3 hours)

**Watch:**
1. "Natural Language Processing" - Sentdex (YouTube series, first 3 videos)
2. "Sentiment Analysis Explained" - Code Emporium (15 mins)

**Read:**
- [VADER Sentiment Analysis](https://github.com/cjhutto/vaderSentiment)
- [Intro to NLP](https://www.analyticsvidhya.com/blog/2021/06/nlp-sentiment-analysis/)
- Your proposal Section 7.3

**Key Concepts:**
- What is sentiment analysis?
- What's the difference between rule-based (VADER) and ML-based (BERT)?
- How do you score "neutral" vs "positive" vs "negative"?

### Afternoon Coding (3 hours)

**Task: Implement Basic Sentiment Analysis**

```python
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import pandas as pd

def analyze_sentiment(text):
    """
    Analyze sentiment of a single text
    Returns compound score (-1 to 1)
    """
    analyzer = SentimentIntensityAnalyzer()
    scores = analyzer.polarity_scores(text)
    return scores['compound']  # -1 (negative) to 1 (positive)

# Test with stock news headlines
test_headlines = [
    "Apple stock soars to record high on strong earnings",
    "Tech stocks plunge amid recession fears",
    "Microsoft announces new AI partnership",
    "Market closes flat after mixed economic data"
]

# YOUR CODE: 
# 1. Analyze each headline
# 2. Print the sentiment scores
# 3. Categorize as positive (>0.05), negative (<-0.05), or neutral
```

**Task 2: Create Sentiment Proxy (Simplified Approach)**

Since getting real-time news is complex, create a price-based sentiment proxy:
```python
def create_sentiment_proxy(df):
    """
    Create a simple sentiment indicator from price action
    This mimics market sentiment based on momentum
    """
    # Idea: Strong upward momentum = positive sentiment
    df = df.copy()
    
    # YOUR CODE:
    # 1. Calculate 5-day return
    # 2. Normalize to -1 to 1 range
    # 3. This becomes your "sentiment" feature
    
    return df
```

**Challenge (Optional):** 
- Sign up for NewsAPI (free tier)
- Download headlines for a stock ticker
- Apply VADER sentiment analysis
- Aggregate daily sentiment scores

**End of Day Checkpoint:**
- Understand basics of NLP and sentiment analysis
- Can use VADER for text classification
- Have a sentiment feature in your dataset

---

## 🔄 **Day 6: Backtesting & Walk-Forward Validation**

### Morning Learning (2 hours)

**Watch:**
1. "Walk Forward Analysis" - Trading view tutorials
2. "Backtesting Trading Strategies" - QuantInsti (YouTube)

**Read:**
- [Time Series Cross-Validation](https://scikit-learn.org/stable/modules/cross_validation.html#time-series-split)
- Your proposal Section 7.6

**Critical Concept: Look-Ahead Bias**
- You can ONLY use information available at prediction time
- Features must be calculated using past data only
- This is why we use .shift() carefully!

### Afternoon Coding (4 hours)

**Task: Implement Walk-Forward Backtesting**

```python
from sklearn.model_selection import TimeSeriesSplit
import numpy as np

def backtest_model(df, feature_cols, n_splits=5):
    """
    Perform walk-forward backtesting
    
    Example with 5 splits:
    Split 1: Train [0:100]    Test [100:120]
    Split 2: Train [0:120]    Test [120:140]
    Split 3: Train [0:140]    Test [140:160]
    ...
    """
    tscv = TimeSeriesSplit(n_splits=n_splits)
    
    results = []
    
    for fold, (train_idx, test_idx) in enumerate(tscv.split(df)):
        print(f"\n{'='*50}")
        print(f"Fold {fold + 1}/{n_splits}")
        print(f"Train: {len(train_idx)} samples")
        print(f"Test: {len(test_idx)} samples")
        
        # YOUR CODE:
        # 1. Split data using train_idx and test_idx
        # 2. Prepare features for this fold
        # 3. Train model
        # 4. Evaluate on test set
        # 5. Store results
        
        results.append({
            'fold': fold,
            'accuracy': accuracy,
            'train_size': len(train_idx),
            'test_size': len(test_idx)
        })
    
    return pd.DataFrame(results)

# Analyze results
results_df = backtest_model(your_data, your_features)
print(results_df)
print(f"\nMean Accuracy: {results_df['accuracy'].mean():.3f}")
print(f"Std Accuracy: {results_df['accuracy'].std():.3f}")
```

**Task 2: Compare Price-Only vs Hybrid Model**

```python
def compare_models(df):
    """
    Compare model with only price features vs 
    model with price + sentiment features
    """
    # Model 1: Price features only
    price_features = ['Open', 'High', 'Low', 'Volume', 
                     'SMA_5', 'SMA_10', 'Returns']
    
    # Model 2: Price + Sentiment
    hybrid_features = price_features + ['Sentiment_Proxy', 'Sentiment_SMA']
    
    # YOUR CODE:
    # 1. Backtest both models
    # 2. Compare accuracies
    # 3. Which performs better?
    
    return results_comparison
```

**End of Day Checkpoint:**
- Understand walk-forward validation
- Can properly backtest a time series model
- Compared baseline vs hybrid approach
- Understood the importance of avoiding look-ahead bias

---

## 🎯 **Day 7: Integration, Documentation & Final Analysis**

### Morning (3 hours): Code Integration

**Task 1: Create End-to-End Pipeline**

```python
# main.py

class StockPredictionPipeline:
    def __init__(self, ticker, start_date, end_date):
        self.ticker = ticker
        self.start_date = start_date
        self.end_date = end_date
        
    def run_complete_pipeline(self):
        """
        Complete pipeline from data to prediction
        """
        # YOUR CODE: Tie everything together
        # 1. Download data
        # 2. Engineer features
        # 3. Create target
        # 4. Split train/test
        # 5. Train model
        # 6. Evaluate
        # 7. Backtest
        # 8. Generate report
        
        pass

# Usage
if __name__ == "__main__":
    pipeline = StockPredictionPipeline(
        ticker="AAPL",
        start_date="2022-01-01",
        end_date="2024-01-01"
    )
    
    results = pipeline.run_complete_pipeline()
```

### Afternoon (3-4 hours): Analysis & Documentation

**Task 2: Create Final Report Notebook**

Create `notebooks/final_report.ipynb` with:

1. **Introduction**
   - What problem are you solving?
   - What's your approach?

2. **Data Analysis**
   - Show exploratory plots
   - Feature distributions
   - Correlation matrices

3. **Model Performance**
   - Training/validation curves
   - Confusion matrices
   - Feature importance plots
   - Backtest results over time

4. **Key Findings**
   - What worked well?
   - What didn't work?
   - How does sentiment help (or not)?

5. **Limitations & Future Work**
   - What would you improve?
   - What did you learn?

**Task 3: Create README.md**

Document:
- How to set up the project
- How to run the code
- What each file does
- Your results summary

---

## 🎓 **Bonus Learning Resources**

### Books (Optional, for deeper understanding):
1. "Hands-On Machine Learning" - Aurélien Géron (Chapters 1-7)
2. "Python for Data Analysis" - Wes McKinney (great for pandas)

### YouTube Channels to Follow:
- StatQuest (best for ML fundamentals)
- Sentdex (Python + ML projects)
- Krish Naik (end-to-end ML projects)
- ritvikmath (statistics + ML)

### Practice Platforms:
- Kaggle Learn (free courses)
- Kaggle Competitions (practice on real datasets)
- Google Colab (free GPU for experimentation)

### Important Concepts to Master:
1. ✅ Train/Test split for time series
2. ✅ Feature engineering for financial data
3. ✅ Random Forest mechanics
4. ✅ Evaluation metrics (accuracy, precision, recall, F1)
5. ✅ Cross-validation
6. ✅ Overfitting vs underfitting
7. ✅ Look-ahead bias
8. ✅ Backtesting methodology

---

## ⚠️ Common Pitfalls to Avoid

1. **Look-Ahead Bias**: Never use future data to predict the past
2. **Random Train/Test Split**: Always split chronologically for time series
3. **Ignoring Missing Values**: NaN from rolling windows must be handled
4. **Overfitting**: Don't tune on test set; use cross-validation
5. **Unrealistic Expectations**: 60-70% accuracy is actually good for stocks!
6. **Not Understanding Metrics**: Know what your numbers mean

---

## 📊 Expected Learning Outcomes

By end of week, you should be able to:
- ✅ Collect and preprocess financial data
- ✅ Engineer meaningful features from time series
- ✅ Train and evaluate a Random Forest classifier
- ✅ Implement proper backtesting for time series
- ✅ Understand basic sentiment analysis
- ✅ Interpret model results and metrics
- ✅ Build an end-to-end ML pipeline
- ✅ Explain your model to others

Good luck! Remember: The goal is LEARNING, not perfect predictions. Ask questions, experiment, and document what you learn!
