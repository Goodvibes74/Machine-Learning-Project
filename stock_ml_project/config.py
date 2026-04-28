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
# Default: Last 5 years of data — more data → better generalisation for financial ML
END_DATE = datetime.now().strftime('%Y-%m-%d')
START_DATE = (datetime.now() - timedelta(days=5 * 365)).strftime('%Y-%m-%d')

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
    'n_estimators': 300,       # More trees -> lower variance
    'max_depth': 4,            # Shallow trees for noisy financial data
    'min_samples_split': 30,   # Forces generalisation — prevents tiny splits
    'min_samples_leaf': 15,    # Large leaves -> smoother decision boundaries
    'max_features': 'sqrt',    # Classic RF: sqrt(n_features) per split
    # class_weight omitted: 53/47 imbalance is mild; balanced weighting
    # compresses probabilities and hurts threshold reliability.
    'random_state': 42,
    'n_jobs': 1,
}

# XGBoost hyperparameters
# Strong regularization is needed for financial data (~600 training samples,
# high noise).  Without it XGBoost trivially memorises the training set.
XGB_PARAMS = {
    'n_estimators': 200,
    'max_depth': 3,          # Shallow trees prevent memorisation
    'learning_rate': 0.05,   # Small steps → better generalisation
    'subsample': 0.7,
    'colsample_bytree': 0.7,
    'min_child_weight': 10,  # Require ≥10 samples per leaf
    'gamma': 0.2,            # Min gain to make a split
    'reg_alpha': 0.5,        # L1 regularisation
    'reg_lambda': 3.0,       # L2 regularisation (default=1)
    'scale_pos_weight': 1,
    'eval_metric': 'logloss',
    'random_state': 42
}

# SVM hyperparameters
# Low C keeps a wide margin (better generalisation on noisy data).
# gamma='scale' normalises by n_features * X.var() — more stable than 'auto'.
SVM_PARAMS = {
    'kernel': 'rbf',
    'C': 0.5,
    'gamma': 'scale',
    'class_weight': 'balanced',
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
    # Returns & basic price features
    'Returns',          # Daily pct return
    'Volatility',       # Rolling std of returns (20-day)
    'Volume_Change',    # Pct change in volume
    'HL_Spread',        # (High-Low)/Close — intraday range

    # Lagged return features (short-term momentum/mean-reversion signals)
    'Returns_lag1',     # Previous day's return
    'Returns_lag2',     # Two days ago return
    'Returns_lag3',     # Three days ago return

    # Volume features
    'Volume_Ratio',     # Current volume / 20-day avg volume

    # Calendar effect
    'DayOfWeek',        # 0=Monday … 4=Friday

    # Technical indicators — all normalised (% of price or unitless)
    'RSI_14',
    'MACD',             # (EMA12 - EMA26) / Close * 100
    'MACD_signal',      # 9-day EMA of MACD / Close * 100
    'MACD_histogram',   # MACD - signal (zero-cross = momentum flip)
    'BB_width',         # (upper - lower) / SMA  — normalised band width
    'BB_position',      # (Close - lower) / (upper - lower)  — 0..1
    'OBV_change',       # OBV pct change (raw cumulative OBV excluded)
    'ATR_14',           # Average True Range / Close * 100
]

# Rolling average features
for window in ROLLING_WINDOWS:
    ENGINEERED_FEATURES.append(f'SMA_{window}')

# Multi-window normalised momentum (pct_change — scale-independent)
for window in MOMENTUM_WINDOWS:
    ENGINEERED_FEATURES.append(f'Momentum_{window}')

# Price/SMA ratios — how far price is from its trend
for window in ROLLING_WINDOWS:
    ENGINEERED_FEATURES.append(f'Price_SMA_{window}_ratio')
ENGINEERED_FEATURES.append('SMA_5_20_ratio')  # golden-cross signal

# Sentiment features
SENTIMENT_FEATURES = [
    'Sentiment_Proxy',  # Price-based sentiment proxy
    'Sentiment_SMA',    # Smoothed sentiment
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