# Stock Market Prediction with Random Forest & Sentiment Analysis

A hybrid machine learning system that combines historical stock price data with sentiment analysis to predict short-term stock price movements.

## 📋 Project Overview

This project implements a Random Forest classifier that integrates:
- **Quantitative features**: Historical OHLCV data, technical indicators
- **Qualitative features**: Market sentiment (price-based proxy)
- **Walk-forward backtesting**: Proper time-series validation

**Goal**: Predict whether a stock's price will go up or down the next day.

## 🗂️ Project Structure

```
stock-prediction/
├── data/
│   ├── raw/              # Downloaded stock data from Yahoo Finance
│   ├── processed/        # Cleaned and validated data
│   └── features/         # Data with engineered features
│
├── src/
│   ├── data_collection.py    # Download stock data
│   ├── preprocessing.py       # Clean and validate data
│   ├── feature_engineering.py # Create predictive features
│   ├── models.py             # Train and evaluate Random Forest models
│   ├── xgboost_model.py      # Train and evaluate XGBoost models
│   └── backtesting.py        # Walk-forward validation
│
├── comparison/
│   └── compare_models.py     # Compare RF vs XGBoost performance
│
├── outputs/                   # Generated plots and results
│
├── notebooks/            # Jupyter notebooks for exploration (to be created)
│   ├── 01_data_exploration.ipynb
│   ├── 02_feature_engineering.ipynb
│   ├── 03_model_training.ipynb
│   └── 04_backtesting.ipynb
│
├── config.py            # Central configuration file
├── requirements.txt     # Python dependencies
├── main.py             # Main pipeline orchestrator
└── README.md           # This file
```

## 🚀 Quick Start

### 1. Setup Environment

```bash
# Create virtual environment
python -m venv venv

# Activate it
# Windows:
venv\Scripts\activate
# Mac/Linux:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Run the Pipeline

```bash
# Run interactive menu
python main.py

# Or run programmatically
python -c "from main import run_complete_pipeline; run_complete_pipeline('AAPL')"
```

### 3. Test Individual Modules

Each module can be tested independently:

```bash
# Test data collection
python src/data_collection.py

# Test preprocessing
python src/preprocessing.py

# Test feature engineering
python src/feature_engineering.py

# Test Random Forest model training
python src/models.py

# Test XGBoost model training
python src/xgboost_model.py

# Compare RF vs XGBoost models
python comparison/compare_models.py

# Test backtesting
python src/backtesting.py
```

## 📊 What Each Module Does

### 1. `config.py` - Configuration
- **Purpose**: Central place for all settings
- **Contains**: 
  - Stock tickers to analyze
  - Date ranges
  - Feature engineering parameters
  - Model hyperparameters
  - File paths
  
**Key settings you can modify**:
```python
TICKERS = ['AAPL', 'MSFT', 'GOOGL']  # Stocks to analyze
START_DATE = '2021-01-01'             # Data start
ROLLING_WINDOWS = [5, 10, 20]         # Moving average periods
RF_PARAMS = {'n_estimators': 100}     # Random Forest settings
XGB_PARAMS = {'n_estimators': 100, 'max_depth': 6, 'learning_rate': 0.1}  # XGBoost settings
```

### 2. `data_collection.py` - Data Download
- **Purpose**: Download historical stock data from Yahoo Finance
- **Data source**: yfinance library (free Yahoo Finance API)
- **Output**: CSV files in `data/raw/`

**Main functions**:
```python
download_stock_data('AAPL', '2023-01-01', '2024-01-01')
download_multiple_stocks(['AAPL', 'MSFT'], ...)
load_stock_data('AAPL', 'raw')
```

### 3. `preprocessing.py` - Data Cleaning
- **Purpose**: Clean and validate raw data
- **Steps**:
  - Remove unnecessary columns
  - Handle missing values (forward fill)
  - Validate data quality
  - Sort by date
- **Output**: CSV files in `data/processed/`

**Main functions**:
```python
clean_stock_data(df)
handle_missing_values(df)
validate_data(df)
preprocess_stock_data(df)  # Complete pipeline
```

### 4. `feature_engineering.py` - Create Features
- **Purpose**: Transform raw prices into predictive features
- **Features created**:
  - **Returns**: Daily percentage change
  - **SMA**: Simple moving averages (5, 10, 20 days)
  - **Volatility**: Rolling standard deviation
  - **Momentum**: Price change over N days
  - **Volume Change**: Trading volume percentage change
  - **HL Spread**: High-Low range normalized
  - **Sentiment Proxy**: Price-based sentiment indicator
  - **Target**: Binary (1=up, 0=down)

**Main functions**:
```python
calculate_returns(df)
add_rolling_averages(df, windows=[5, 10, 20])
calculate_volatility(df)
create_target_variable(df, horizon=1)
engineer_all_features(df)  # Apply all
prepare_ml_data(df)        # Prepare X, y for ML
```

### 5. `models.py` - Machine Learning (Random Forest)
- **Purpose**: Train and evaluate Random Forest models
- **Model**: scikit-learn RandomForestClassifier
- **Evaluation**: Accuracy, precision, recall, F1-score
- **Visualization**: Confusion matrix, feature importance

**Main functions**:
```python
train_random_forest(X_train, y_train)
evaluate_model(model, X_train, y_train, X_test, y_test)
plot_confusion_matrix(y_true, y_pred)
plot_feature_importance(model, features)
compare_baseline(X_train, y_train, X_test, y_test)
```

### 6. `xgboost_model.py` - XGBoost Models
- **Purpose**: Train and evaluate XGBoost models
- **Model**: XGBoost XGBClassifier
- **Evaluation**: Accuracy, precision, recall, F1-score, ROC-AUC
- **Visualization**: Confusion matrix, feature importance

**Main functions**:
```python
train_xgboost(X_train, y_train)
evaluate_xgboost_model(model, X_train, y_train, X_test, y_test)
plot_xgb_confusion_matrix(y_true, y_pred)
plot_xgb_feature_importance(model, features)
```

### 7. `comparison/compare_models.py` - Model Comparison
- **Purpose**: Compare Random Forest vs XGBoost performance
- **Metrics**: Accuracy, precision, recall, F1, ROC-AUC, training time
- **Visualization**: ROC curves, metrics bar chart, feature importance comparison

**Main functions**:
```python
get_metrics(model, X_test, y_test, model_name)
build_comparison_table(rf_metrics, xgb_metrics)
plot_roc_curves(rf_model, xgb_model, X_test, y_test)
plot_metrics_bar_chart(rf_metrics, xgb_metrics)
plot_feature_importance_comparison(rf_model, xgb_model, features)
run_full_comparison(ticker='AAPL')
```

### 8. `backtesting.py` - Time Series Validation
- **Purpose**: Proper validation for time series models
- **Method**: Walk-forward validation
- **Why**: Simulates real trading (train on past, test on future)

**Main functions**:
```python
walk_forward_validation(X, y, features, n_splits=5)
compare_strategies(X, y, features)  # Compare price-only vs hybrid
plot_backtest_results(results)
```

## 🎯 Understanding the Pipeline

### Complete Flow

```
Raw Stock Data (OHLCV)
    ↓
[Data Collection] → Download from Yahoo Finance
    ↓
[Preprocessing] → Clean, validate, handle missing values
    ↓
[Feature Engineering] → Create technical indicators + sentiment
    ↓
[Data Preparation] → Split into features (X) and target (y)
    ↓
[Train/Test Split] → Chronological split (80% train, 20% test)
    ↓
[Model Training] → Random Forest Classifier
    ↓
[Evaluation] → Accuracy, confusion matrix, feature importance
    ↓
[Backtesting] → Walk-forward validation across time periods
    ↓
Results & Insights
```

### Key Concepts

**1. Features (X) vs Target (y)**
- **Features**: Input variables model learns from
  - Example: Closing price, moving averages, volume
- **Target**: What we're predicting
  - Example: Will price go up tomorrow? (1=yes, 0=no)

**2. Train/Test Split**
- **Training set**: Past data model learns from (e.g., 2021-2023)
- **Test set**: Future data we evaluate on (e.g., 2024)
- **Critical**: NEVER train on test data!

**3. Time Series Considerations**
- **No random shuffling**: Must split chronologically
- **No look-ahead bias**: Only use past data for features
- **Walk-forward**: Retrain as new data arrives

**4. Evaluation Metrics**
- **Accuracy**: Overall correct predictions
- **Precision**: Of predicted "up", how many were correct?
- **Recall**: Of actual "up" days, how many did we catch?
- **F1-Score**: Balance between precision and recall

## 📈 Expected Results

Based on literature and project proposal:

- **Target accuracy**: 60-70% (good for stock prediction!)
- **Baseline (price-only)**: ~55-60%
- **Hybrid (price + sentiment)**: ~65-70%
- **Important**: >50% is profitable (better than random)

**Reality check**: 
- Stock markets are noisy and unpredictable
- 60% accuracy is actually impressive
- This is what professional quant funds target

## 🛠️ Customization

### Change Stocks to Analyze

Edit `config.py`:
```python
TICKERS = ['TSLA', 'NVDA', 'AMD']  # Your stocks
```

### Change Date Range

Edit `config.py`:
```python
START_DATE = '2020-01-01'
END_DATE = '2024-12-31'
```

### Adjust Model Parameters

Edit `config.py` for Random Forest:
```python
RF_PARAMS = {
    'n_estimators': 200,    # More trees (slower but better)
    'max_depth': 15,        # Deeper trees (more complex)
    'min_samples_split': 10 # More conservative splitting
}
```

Or for XGBoost:
```python
XGB_PARAMS = {
    'n_estimators': 150,    # More boosting rounds
    'max_depth': 8,         # Deeper trees
    'learning_rate': 0.05,  # Lower learning rate for better generalization
    'random_state': 42
}
```

### Add More Features

Edit `feature_engineering.py`, add your function:
```python
def calculate_rsi(df, window=14):
    """Calculate Relative Strength Index"""
    # Your implementation
    pass

# Then add to engineer_all_features()
```

## 📚 Learning Resources

### Understanding the Code

1. **Start with**: `config.py` → `data_collection.py` → `main.py`
2. **Read in order**: Follow the pipeline flow
3. **Run examples**: Each module has `run_example()` function
4. **Experiment**: Change parameters and see what happens

### Machine Learning Concepts

- **Random Forest**: [StatQuest video](https://www.youtube.com/watch?v=J4Wdy0Wc_xQ)
- **Train/Test Split**: [StatQuest video](https://www.youtube.com/watch?v=fSytzGwwBVw)
- **Cross Validation**: [StatQuest video](https://www.youtube.com/watch?v=fSytzGwwBVw)

### Stock Market Concepts

- **Technical Indicators**: [Investopedia](https://www.investopedia.com/terms/t/technicalindicator.asp)
- **Moving Averages**: [Investopedia](https://www.investopedia.com/terms/m/movingaverage.asp)

## 🐛 Troubleshooting

### Common Issues

**"No data found for ticker"**
- Check ticker symbol is correct
- Check date range is valid
- Try with a well-known stock first (AAPL)

**"ValueError: Input contains NaN"**
- Run `df.dropna()` after feature engineering
- This is normal - rolling windows create NaN

**Low accuracy (~50%)**
- This might be normal for stocks!
- Try more features or different parameters
- Check if target distribution is balanced

**Model performs great on train, poor on test**
- This is overfitting
- Reduce `max_depth` in config
- Increase `min_samples_split`

## 📝 Next Steps

### For Learning
1. ✅ Understand each module's purpose
2. ✅ Run complete pipeline for one stock
3. ✅ Experiment with different parameters
4. ✅ Try adding new features
5. ✅ Compare different stocks

### For Improvement
1. ⬜ Implement real NLP sentiment analysis
2. ⬜ Add more technical indicators (RSI, MACD)
3. ⬜ Try other models (XGBoost, Neural Networks)
4. ⬜ Implement trading strategy backtesting
5. ⬜ Add risk management features

## 🎓 Academic Context

This project is based on the research proposal:
- **Title**: "Random Forest & Sentiment Analysis: Hybrid Stock Market Prediction"
- **Institution**: Makerere University, College of Computing and Information Sciences
- **Approach**: Combines quantitative and qualitative features
- **References**: Based on current ML and finance literature

## 📄 License & Disclaimer

**For Educational Purposes Only**

This project is for learning machine learning concepts. It is NOT:
- Investment advice
- A guaranteed profitable trading system
- Suitable for real trading without extensive testing

Stock markets are unpredictable. Past performance doesn't guarantee future results.

## 🙏 Acknowledgments

- **Data**: Yahoo Finance (via yfinance library)
- **ML Framework**: scikit-learn
- **Inspiration**: Academic research in ML for finance

---

**Happy Learning! 🚀📈**

If you have questions, review the code comments - every function is documented!