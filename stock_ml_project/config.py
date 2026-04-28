"""
Configuration file for Stock Prediction Project

This file contains all configuration parameters for the project.
Modify these settings to customize your analysis.
"""

import os
from datetime import datetime, timedelta

# PROJECT PATHS
# Base directory of the project
BASE_DIR = os.path.dirname(os.path.abspath(__file__)) # Directory where config.py is located

# Data directories
DATA_DIR = os.path.join(BASE_DIR, 'data') # Base data directory which is responsible for storing all data-related files (raw, processed, features)
RAW_DATA_DIR = os.path.join(DATA_DIR, 'raw') # Directory for raw data downloaded from Yahoo Finance
PROCESSED_DATA_DIR = os.path.join(DATA_DIR, 'processed') # Directory for processed data after cleaning and preprocessing
FEATURES_DATA_DIR = os.path.join(DATA_DIR, 'features') # Directory for feature data after feature engineering steps are applied

# Create directories if they don't exist
for directory in [RAW_DATA_DIR, PROCESSED_DATA_DIR, FEATURES_DATA_DIR]:
    os.makedirs(directory, exist_ok=True) # Ensure that the necessary directories exist, if not create them

# STOCK DATA SETTINGS
# List of stock tickers to analyze (S&P 500 stocks as per proposal)
# Start with a few for testing, expand later
TICKERS = [
    'AAPL',  # Apple
    'MSFT',  # Microsoft
    'GOOGL',  # Google
    'AMZN',  # Amazon
    'TSLA',  # Tesla
]

# Date range for historical data
# Default: Last 3 years of data
END_DATE = datetime.now().strftime('%Y-%m-%d') # Today's date
START_DATE = (datetime.now() - timedelta(days=3 * 365)).strftime('%Y-%m-%d') # 3 years ago

# FEATURE ENGINEERING SETTINGS
# Rolling window sizes for moving averages (in trading days)
ROLLING_WINDOWS = [5, 10, 20]  # Short, medium, long-term trends

# Volatility calculation window
VOLATILITY_WINDOW = 20  # 20-day volatility (approximately 1 month)

# Momentum calculation windows — multiple periods capture different regimes
MOMENTUM_WINDOWS = [3, 5, 10, 20]  # short, short-med, medium, long
MOMENTUM_WINDOW = 5  # kept for backward compat (sentiment proxy uses this)

# TARGET VARIABLE SETTINGS
# Prediction horizon (how many days ahead to predict) , prediction horizon is the number of days into the future that we want to predict whether the stock price will go up or down. For example, if PREDICTION_HORIZON is set to 1, we are trying to predict whether the stock price will be higher or lower tomorrow compared to today.
PREDICTION_HORIZON = 1  # Next day prediction (binary: up/down)

# Classification threshold
# If tomorrow's price is > today's price: class 1 (up)
# If tomorrow's price is <= today's price: class 0 (down)


# MODEL SETTINGS
# Train/Test split ratio
TEST_SIZE = 0.2  # 20% of data for testing

# Random Forest hyperparameters
RF_PARAMS = {
    'n_estimators': 300,       # More trees → lower variance
    'max_depth': 4,            # Shallow trees are key for noisy financial data (was 8)
    'min_samples_split': 30,   # Hard split threshold → forces generalization (was 10)
    'min_samples_leaf': 15,    # Larger leaves → smoother decision boundaries (was 5)
    'max_features': 'sqrt',    # Classic RF: sqrt(n_features) per split
    'class_weight': 'balanced',
    'random_state': 42,
    'n_jobs': -1,
}

# XGBoost hyperparameters - Optimized for 20 features
XGB_PARAMS = {
    'n_estimators': 150,  # Balanced
    'max_depth': 5,  # Moderate depth
    'learning_rate': 0.08,  # Moderate
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'scale_pos_weight': 1,  # Balanced classes
    'eval_metric': 'logloss',
    'random_state': 42
}

# SVM hyperparameters - Optimized for 20 features
SVM_PARAMS = {
    'kernel': 'rbf',
    'C': 5.0,  # Moderate regularization
    'gamma': 'auto',  # Different kernel sensitivity
    'class_weight': 'balanced',  # Handle class imbalance
    'probability': True,
    'random_state': 42
}

# BACKTESTING SETTINGS
# Number of splits for walk-forward validation
N_SPLITS = 10  # 10-fold time series cross-validation

# SENTIMENT ANALYSIS SETTINGS (Optional - for future use)
# News API settings (you'll need to sign up for API key)
NEWS_API_KEY = None  # Set this if you get a NewsAPI key

# Sentiment smoothing window
SENTIMENT_WINDOW = 1  # 3-day moving average of sentiment

# LOGGING SETTINGS
# Verbosity level , Verbosity is the level of detail in console output. Set to True for detailed logs, False for minimal output.
VERBOSE = True  # Set to False to reduce console output


# FEATURE COLUMNS
# Base price features from Yahoo Finance
BASE_FEATURES = ['Open', 'High', 'Low', 'Close', 'Volume']

# Features to create (will be added by feature engineering)
ENGINEERED_FEATURES = [
    'Returns',          # Daily pct return
    'Volatility',       # Rolling std of returns
    'Volume_Change',    # Pct change in volume
    'HL_Spread',        # (High-Low)/Close — intraday range
    # Technical indicators
    'RSI_14',
    'MACD',
    'MACD_signal',
    'MACD_histogram',   # MACD - signal: zero-cross is the key signal
    'BB_upper',
    'BB_lower',
    'BB_width',
    'BB_position',
    'OBV',
    'OBV_change',
    'ATR_14',
]

# Rolling average features
for window in ROLLING_WINDOWS:
    ENGINEERED_FEATURES.append(f'SMA_{window}')

# Multi-window normalized momentum (pct_change — scale-independent)
for window in MOMENTUM_WINDOWS:
    ENGINEERED_FEATURES.append(f'Momentum_{window}')

# Price/SMA ratios — how far price is from its trend
for window in ROLLING_WINDOWS:
    ENGINEERED_FEATURES.append(f'Price_SMA_{window}_ratio')
ENGINEERED_FEATURES.append('SMA_5_20_ratio')  # golden-cross signal

# Sentiment features (for future use)
SENTIMENT_FEATURES = [
    'Sentiment_Proxy',  # Price-based sentiment proxy
    'Sentiment_SMA',  # Smoothed sentiment
]

# All features that will be used for training
ALL_FEATURES = BASE_FEATURES + ENGINEERED_FEATURES + SENTIMENT_FEATURES



def get_data_path(ticker, data_type='raw'):
    """
    Get the file path for a specific ticker and data type

    Args:
        ticker (str): Stock ticker symbol
        data_type (str): 'raw', 'processed', or 'features'

    Returns:
        str: Full path to the data file
    """
    if data_type == 'raw':
        return os.path.join(RAW_DATA_DIR, f'{ticker}_raw.csv')
    elif data_type == 'processed':
        return os.path.join(PROCESSED_DATA_DIR, f'{ticker}_processed.csv')
    elif data_type == 'features':
        return os.path.join(FEATURES_DATA_DIR, f'{ticker}_features.csv')
    else:
        raise ValueError(f"Unknown data_type: {data_type}")


def print_config():
    """Print current configuration settings"""
    print("=" * 60)
    print("STOCK PREDICTION PROJECT CONFIGURATION")
    print("=" * 60)
    print(f"Tickers: {', '.join(TICKERS)}") # List of stock tickers being analyzed
    print(f"Date Range: {START_DATE} to {END_DATE}") # Date range for historical data
    print(f"Rolling Windows: {ROLLING_WINDOWS}") # Rolling window sizes for moving averages
    print(f"Volatility Window: {VOLATILITY_WINDOW} days") # Window size for calculating volatility
    print(f"Momentum Window: {MOMENTUM_WINDOW} days") # Window size for calculating momentum
    print(f"Prediction Horizon: {PREDICTION_HORIZON} day(s)") # How many days ahead to predict
    print(f"Test Size: {TEST_SIZE * 100}%") # Percentage of data used for testing
    print(f"Random Forest Estimators: {RF_PARAMS['n_estimators']}") # Number of trees in the Random Forest model
    print(f"Max Depth: {RF_PARAMS['max_depth']}") # Maximum depth of trees in the Random Forest model
    print(f"Min Samples Split: {RF_PARAMS['min_samples_split']}") # Minimum samples required to split a node in the Random Forest model
    print(f"Min Samples Leaf: {RF_PARAMS['min_samples_leaf']}") # Minimum samples required at a leaf node in the Random Forest model
    print(f"Backtesting Splits: {N_SPLITS}") # Number of splits for walk-forward validation in backtesting
    print(f"Verbose Logging: {VERBOSE}") # Whether verbose logging is enabled   
    print("=" * 60)


# Print configuration when module is imported (optional)
if __name__ == '__main__':
    print_config()