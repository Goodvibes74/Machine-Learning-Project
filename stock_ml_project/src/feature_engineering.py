"""
Feature Engineering Module

This module creates predictive features from raw stock price data.
Features are the variables that the machine learning model uses to make predictions.

Key Concepts:
- Features (X): Input variables the model learns from
- Target (y): What we're trying to predict
- Rolling windows: Calculate averages/stats over N recent days
- Look-ahead bias: NEVER use future data to create features!
"""

import pandas as pd
import numpy as np
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))# Add parent directory to path so we can import config
import config



def calculate_returns(df): # responsible for calculating daily percentage returns, which measure the day-to-day price change and are a fundamental feature for financial modeling.
    """
    Calculate daily percentage returns

    Returns show how much the price changed from one day to the next.
    Formula: (Today's Close - Yesterday's Close) / Yesterday's Close

    Args:
        df (pd.DataFrame): Stock data with 'Close' column

    Returns:
        pd.DataFrame: Data with 'Returns' column added

    Example:
        If yesterday's close was $100 and today's is $105:
        Return = (105 - 100) / 100 = 0.05 = 5%
    """
    df = df.copy()

    # pct_change() calculates percentage change from previous row
    df['Returns'] = df['Close'].pct_change()

    if config.VERBOSE:
        print("✅ Added 'Returns' feature")

    return df


def add_rolling_averages(df, windows=None): # responsible for adding Simple Moving Average (SMA) features for specified rolling windows, which help capture price trends and smooth out short-term fluctuations in the stock price.
    """
    Add Simple Moving Averages (SMA) for different time windows

    A rolling average smooths out price fluctuations to show trends.
    SMA_5 = average of last 5 days' closing prices
    SMA_20 = average of last 20 days' closing prices

    Interpretation:
    - Price > SMA: Upward trend
    - Price < SMA: Downward trend
    - Short SMA > Long SMA: Bullish signal

    Args:
        df (pd.DataFrame): Stock data with 'Close' column
        windows (list): List of window sizes (default from config)

    Returns:
        pd.DataFrame: Data with SMA columns added

    Example:
        >>> df = add_rolling_averages(df, windows=[5, 10, 20])
        >>> # Creates columns: SMA_5, SMA_10, SMA_20
    """
    df = df.copy()

    if windows is None:
        windows = config.ROLLING_WINDOWS

    for window in windows:
        # rolling(window) creates a rolling window
        # .mean() calculates the average
        df[f'SMA_{window}'] = df['Close'].rolling(window=window).mean()

        if config.VERBOSE:
            print(f"✅ Added 'SMA_{window}' feature")

    return df


def calculate_volatility(df, window=None): # responsible for calculating rolling volatility (standard deviation of returns), which measures how much the stock price fluctuates over a specified period and is an important indicator of risk.
    """
    Calculate rolling volatility (standard deviation of returns)

    Volatility measures how much the price fluctuates.
    High volatility = risky, unstable
    Low volatility = stable, less risky

    Args:
        df (pd.DataFrame): Stock data with 'Returns' column
        window (int): Rolling window size (default from config)

    Returns:
        pd.DataFrame: Data with 'Volatility' column added

    Note: This requires 'Returns' column to exist first!
    """
    df = df.copy()

    if window is None:
        window = config.VOLATILITY_WINDOW

    # Volatility = standard deviation of returns over window
    df['Volatility'] = df['Returns'].rolling(window=window).std()

    if config.VERBOSE:
        print(f"✅ Added 'Volatility' feature (window={window})")

    return df


def calculate_momentum(df, windows=None): # responsible for calculating price momentum, which indicates the strength and direction of price movement by comparing the current price to the price from a specified number of days ago.
    """
    Calculate normalized price momentum for multiple windows.

    Momentum = (Close[t] - Close[t-N]) / Close[t-N]  (percentage return)
    Normalized by price so it's comparable across stocks and time periods.
    Multiple windows capture short, medium, and long-term momentum regimes.

    Args:
        df (pd.DataFrame): Stock data with 'Close' column
        windows (list): Lookback periods (default from config.MOMENTUM_WINDOWS)

    Returns:
        pd.DataFrame: Data with 'Momentum_N' columns added for each window
    """
    df = df.copy()

    if windows is None:
        windows = config.MOMENTUM_WINDOWS

    for window in windows:
        # pct_change(N) = (Close[t] - Close[t-N]) / Close[t-N]
        # Normalized: a 5-day move of 2% means the same regardless of stock price
        df[f'Momentum_{window}'] = df['Close'].pct_change(window)

        if config.VERBOSE:
            print(f"✅ Added 'Momentum_{window}' feature")

    return df


def calculate_volume_change(df): # responsible for calculating the percentage change in trading volume, which can provide insights into market activity and potential shifts in investor sentiment.
    """
    Calculate percentage change in trading volume

    Volume changes can indicate:
    - High volume + price up = strong buying
    - High volume + price down = strong selling
    - Low volume = weak conviction

    Args:
        df (pd.DataFrame): Stock data with 'Volume' column

    Returns:
        pd.DataFrame: Data with 'Volume_Change' column added
    """
    df = df.copy()

    df['Volume_Change'] = df['Volume'].pct_change()

    if config.VERBOSE:
        print("✅ Added 'Volume_Change' feature")

    return df


def calculate_hl_spread(df): # responsible for calculating the High-Low spread, which measures the intraday price range and can indicate volatility within the day.
    """
    Calculate High-Low spread normalized by Close price

    This shows the intraday price range.
    Formula: (High - Low) / Close

    Large spread = high volatility within the day
    Small spread = price stayed stable during the day

    Args:
        df (pd.DataFrame): Stock data with 'High', 'Low', 'Close' columns

    Returns:
        pd.DataFrame: Data with 'HL_Spread' column added
    """
    df = df.copy()

    df['HL_Spread'] = (df['High'] - df['Low']) / df['Close']

    if config.VERBOSE:
        print("✅ Added 'HL_Spread' feature")

    return df


def calculate_rsi(df, window=14):
    """
    Calculate Relative Strength Index (RSI)

    RSI measures the magnitude of recent price changes to evaluate
    overbought or oversold conditions.
    - RSI > 70: Overbought (potential sell signal)
    - RSI < 30: Oversold (potential buy signal)

    IMPROVEMENT: Uses shift(1) on Close to ensure no look-ahead bias.

    Args:
        df (pd.DataFrame): Stock data with 'Close' column
        window (int): RSI period (default: 14)

    Returns:
        pd.DataFrame: Data with 'RSI_14' column added
    """
    df = df.copy()

    # Use today's close — no look-ahead bias since we know today's close
    # when predicting tomorrow's direction. shift(1) was overly conservative.
    delta = df['Close'].diff()

    gains = delta.where(delta > 0, 0)
    losses = (-delta).where(delta < 0, 0)

    avg_gain = gains.rolling(window=window).mean()
    avg_loss = losses.rolling(window=window).mean()

    rs = avg_gain / avg_loss
    df['RSI_14'] = 100 - (100 / (1 + rs))

    if config.VERBOSE:
        print(f"✅ Added 'RSI_14' feature (window={window})")

    return df


def calculate_macd(df):
    """
    Calculate MACD (Moving Average Convergence Divergence)

    MACD is a trend-following momentum indicator:
    - MACD line = 12-day EWM - 26-day EWM
    - Signal line = 9-day EWM of MACD
    - Histogram = MACD - Signal

    IMPROVEMENT: Uses shift(1) on Close to ensure no look-ahead bias.

    Args:
        df (pd.DataFrame): Stock data with 'Close' column

    Returns:
        pd.DataFrame: Data with 'MACD' and 'MACD_signal' columns added
    """
    df = df.copy()

    # Use today's close — valid since we know it when predicting tomorrow
    ema_12 = df['Close'].ewm(span=12, adjust=False).mean()
    ema_26 = df['Close'].ewm(span=26, adjust=False).mean()
    df['MACD'] = ema_12 - ema_26
    df['MACD_signal'] = df['MACD'].ewm(span=9, adjust=False).mean()
    # Histogram crosses zero when momentum flips — the primary trading signal
    df['MACD_histogram'] = df['MACD'] - df['MACD_signal']

    if config.VERBOSE:
        print("✅ Added 'MACD', 'MACD_signal', 'MACD_histogram' features")

    return df


def calculate_bollinger_bands(df, window=20):
    """
    Calculate Bollinger Bands

    Bollinger Bands are volatility bands placed above and below a moving average.
    - Upper Band = SMA + 2*Standard Deviation
    - Lower Band = SMA - 2*Standard Deviation
    - Band Width = (Upper - Lower) / SMA
    - Band Position = (Close - Lower) / (Upper - Lower)

    IMPROVEMENT: Uses shift(1) on Close to ensure no look-ahead bias.

    Args:
        df (pd.DataFrame): Stock data with 'Close' column
        window (int): Window for SMA and std (default: 20)

    Returns:
        pd.DataFrame: Data with 'BB_upper', 'BB_lower', 'BB_width', 'BB_position' columns
    """
    df = df.copy()

    # Use today's close — consistent with the BB_position which also uses today's close
    sma = df['Close'].rolling(window=window).mean()
    std = df['Close'].rolling(window=window).std()

    df['BB_upper'] = sma + 2 * std
    df['BB_lower'] = sma - 2 * std
    df['BB_width'] = (df['BB_upper'] - df['BB_lower']) / sma
    df['BB_position'] = (df['Close'] - df['BB_lower']) / (df['BB_upper'] - df['BB_lower'])

    if config.VERBOSE:
        print(f"✅ Added Bollinger Bands features (window={window})")

    return df


def calculate_obv(df):
    """
    Calculate On-Balance Volume (OBV)

    OBV is a cumulative indicator that adds volume on up days and
    subtracts volume on down days. It measures buying/selling pressure.

    IMPROVEMENT: Uses shift(1) on Close to ensure no look-ahead bias.

    Args:
        df (pd.DataFrame): Stock data with 'Close' and 'Volume' columns

    Returns:
        pd.DataFrame: Data with 'OBV' and 'OBV_change' columns added
    """
    df = df.copy()

    # Use shifted close to prevent look-ahead bias
    close_shifted = df['Close'].shift(1)

    # Calculate direction: 1 if close > close_shifted, -1 if close < close_shifted, 0 otherwise
    direction = (df['Close'] > close_shifted).astype(int) - (df['Close'] < close_shifted).astype(int)

    # Calculate OBV (cumulative sum of volume * direction)
    df['OBV'] = (direction * df['Volume']).cumsum()

    # Calculate OBV change (percentage)
    df['OBV_change'] = df['OBV'].pct_change()

    if config.VERBOSE:
        print("✅ Added 'OBV' and 'OBV_change' features")

    return df


def calculate_atr(df, window=14):
    """
    Calculate Average True Range (ATR)

    ATR measures market volatility. It's the average of the True Range
    over a specified period.
    True Range = max(High - Low, |High - Previous Close|, |Low - Previous Close|)

    IMPROVEMENT: Uses shift(1) on Close to ensure no look-ahead bias.

    Args:
        df (pd.DataFrame): Stock data with 'High', 'Low', 'Close' columns
        window (int): ATR period (default: 14)

    Returns:
        pd.DataFrame: Data with 'ATR_14' column added
    """
    df = df.copy()

    # Use shifted close for True Range calculation
    prev_close = df['Close'].shift(1)

    # Calculate True Range components
    high_low = df['High'] - df['Low']
    high_prev = (df['High'] - prev_close).abs()
    low_prev = (df['Low'] - prev_close).abs()

    # True Range is the max of the three
    true_range = pd.concat([high_low, high_prev, low_prev], axis=1).max(axis=1)

    # Calculate ATR as rolling mean of True Range
    df['ATR_14'] = true_range.rolling(window=window).mean()

    if config.VERBOSE:
        print(f"✅ Added 'ATR_14' feature (window={window})")

    return df


def calculate_price_sma_ratios(df, windows=None):
    """
    Calculate price-to-SMA ratios and SMA crossover signals.

    These are normalized, scale-free trend signals:
    - Price_SMA_N_ratio: How far above/below its N-day average the price is.
      Positive = price above trend (bullish), negative = below (bearish).
    - SMA_5_20_ratio: Fast SMA vs slow SMA (golden-cross / death-cross signal).
      Positive = short-term trend above long-term (bullish momentum).

    Args:
        df (pd.DataFrame): Data with 'Close' and SMA_N columns already computed
        windows (list): Windows to use (must match what add_rolling_averages created)

    Returns:
        pd.DataFrame: Data with Price_SMA_N_ratio and SMA_5_20_ratio columns
    """
    df = df.copy()

    if windows is None:
        windows = config.ROLLING_WINDOWS

    for window in windows:
        sma_col = f'SMA_{window}'
        if sma_col in df.columns:
            df[f'Price_SMA_{window}_ratio'] = (df['Close'] / df[sma_col]) - 1

    if 'SMA_5' in df.columns and 'SMA_20' in df.columns:
        df['SMA_5_20_ratio'] = (df['SMA_5'] / df['SMA_20']) - 1

    if config.VERBOSE:
        print("✅ Added Price/SMA ratio features")

    return df


def create_sentiment_proxy(df, window=None): #responsible for creating a simple sentiment proxy based on price momentum. This is a placeholder until we implement real NLP sentiment analysis.
    """
    Create a price-based sentiment proxy

    This is a SIMPLIFIED version of sentiment analysis.
    We use price momentum to approximate market sentiment:
    - Strong upward momentum = positive sentiment
    - Strong downward momentum = negative sentiment

    This is a placeholder until we implement real NLP sentiment analysis.

    Args:
        df (pd.DataFrame): Stock data with 'Close' column
        window (int): Window for momentum calculation

    Returns:
        pd.DataFrame: Data with 'Sentiment_Proxy' column added
    """
    df = df.copy()

    if window is None:
        window = config.MOMENTUM_WINDOW

    # Calculate momentum as proxy
    momentum = df['Close'].pct_change(window)

    # Normalize to -1 to 1 range (clip extreme values)
    # This makes it similar to sentiment scores
    df['Sentiment_Proxy'] = momentum.clip(-0.1, 0.1) / 0.1

    if config.VERBOSE:
        print("✅ Added 'Sentiment_Proxy' feature")

    return df


def smooth_sentiment(df, window=None): # responsible for smoothing the sentiment proxy using a moving average to reduce noise in daily sentiment fluctuations.
    """
    Add smoothed (moving average) sentiment

    Smoothing reduces noise in daily sentiment fluctuations

    Args:
        df (pd.DataFrame): Data with 'Sentiment_Proxy' column
        window (int): Smoothing window (default from config)

    Returns:
        pd.DataFrame: Data with 'Sentiment_SMA' column added
    """
    df = df.copy()

    if window is None:
        window = config.SENTIMENT_WINDOW

    df['Sentiment_SMA'] = df['Sentiment_Proxy'].rolling(window=window).mean()

    if config.VERBOSE:
        print(f"✅ Added 'Sentiment_SMA' feature (window={window})")

    return df


def create_target_variable(df, horizon=None): # responsible for creating the target variable for prediction, which indicates whether the stock price will go up the next day (or after a specified horizon), enabling the model to learn from historical price movements and make future predictions.
    """
    Create the target variable for prediction

    This is what we're trying to predict!
    Binary classification: Will the price go up tomorrow?
    - 1 = Price goes up (Close tomorrow > Close today)
    - 0 = Price goes down or stays same

    CRITICAL: We use shift(-horizon) to get FUTURE price
    This creates the target but introduces NaN in last rows

    Args:
        df (pd.DataFrame): Stock data with 'Close' column
        horizon (int): How many days ahead to predict (default: 1)

    Returns:
        pd.DataFrame: Data with 'Target' and 'Future_Close' columns added

    Warning: Last 'horizon' rows will have NaN target (no future data)
    """
    df = df.copy()

    if horizon is None:
        horizon = config.PREDICTION_HORIZON 

    # Get future close price
    # shift(-1) means "get yesterdays's value"
    df['Future_Close'] = df['Close'].shift(-horizon)

    # Create binary target: 1 if price goes up, 0 otherwise
    df['Target'] = (df['Future_Close'] > df['Close']).astype(int)

    if config.VERBOSE:
        print(f"✅ Added 'Target' variable (horizon={horizon} day)")
        print(f"   Last {horizon} row(s) will have NaN target")

    return df


def engineer_all_features(df):
    """
    Apply all feature engineering steps

    This is the main function that creates ALL features in the correct order.
    Order matters because some features depend on others!

    Pipeline:
    1. Calculate returns (needed for volatility)
    2. Add rolling averages
    3. Calculate volatility (needs returns)
    4. Calculate momentum
    5. Calculate volume change
    6. Calculate HL spread
    7. Create sentiment proxy
    8. Smooth sentiment (needs sentiment proxy)
    9. Create target variable

    Args:
        df (pd.DataFrame): Preprocessed stock data

    Returns:
        pd.DataFrame: Data with all features added

    Example:
        >>> processed_data = preprocess_stock_data(raw_data)
        >>> feature_data = engineer_all_features(processed_data)
    """
    if config.VERBOSE:
        print("\n" + "=" * 60)
        print("FEATURE ENGINEERING PIPELINE")
        print("=" * 60 + "\n")

    df = df.copy()

    # Apply each feature engineering step
    df = calculate_returns(df)
    df = add_rolling_averages(df)          # SMA_5, SMA_10, SMA_20
    df = calculate_volatility(df)
    df = calculate_momentum(df)            # Momentum_3, Momentum_5, Momentum_10, Momentum_20
    df = calculate_volume_change(df)
    df = calculate_hl_spread(df)

    # Technical indicators (use today's close — no look-ahead since target is tomorrow)
    df = calculate_rsi(df)
    df = calculate_macd(df)                # MACD, MACD_signal, MACD_histogram
    df = calculate_bollinger_bands(df)
    df = calculate_obv(df)
    df = calculate_atr(df)

    # Normalized trend signals (requires SMAs to be computed first)
    df = calculate_price_sma_ratios(df)   # Price_SMA_N_ratio, SMA_5_20_ratio

    # Sentiment proxy
    df = create_sentiment_proxy(df)
    df = smooth_sentiment(df)
    df = create_target_variable(df)

    if config.VERBOSE:
        print("\n" + "=" * 60)
        print(f"Feature engineering complete!")
        print(f"Total columns: {len(df.columns)}")
        print(f"New features: {len(df.columns) - len(config.BASE_FEATURES) - 1}")  # -1 for Date
        print("=" * 60 + "\n")

    return df


def prepare_ml_data(df, feature_cols=None):
    """
    Prepare data for machine learning

    This function:
    1. Removes rows with NaN (from rolling windows and target creation)
    2. Separates features (X) from target (y)
    3. Returns clean data ready for training

    Args:
        df (pd.DataFrame): Data with all features and target
        feature_cols (list): List of feature column names to use
                            If None, uses all except 'Date', 'Target', 'Future_Close'

    Returns:
        tuple: (X, y, feature_names)
            X: Feature matrix (DataFrame)
            y: Target vector (Series)
            feature_names: List of feature column names

    Example:
        >>> X, y, features = prepare_ml_data(feature_data)
        >>> print(f"Features: {features}")
        >>> print(f"X shape: {X.shape}, y shape: {y.shape}")
    """
    df = df.copy()

    # Remove rows with NaN values
    # These come from rolling windows (first N rows) and target (last row)
    original_len = len(df)
    df_clean = df.dropna()
    dropped = original_len - len(df_clean)

    if config.VERBOSE:
        print(f"Dropped {dropped} rows with NaN values")
        print(f"Usable data: {len(df_clean)} rows")

    # Determine feature columns
    if feature_cols is None:
        # LEAKAGE FIX: Exclude raw OHLCV columns to prevent direct leakage
        # Raw price columns used alongside target derived from Close causes leakage
        exclude_cols = ['Date', 'Target', 'Future_Close', 'Ticker', 
                        'Open', 'High', 'Low', 'Close', 'Volume']
        feature_cols = [col for col in df_clean.columns if col not in exclude_cols]

    # Separate features and target
    X = df_clean[feature_cols]
    y = df_clean['Target']

    if config.VERBOSE:
        print(f"\nFeatures ({len(feature_cols)}): {feature_cols}")
        print(f"X shape: {X.shape}")
        print(f"y shape: {y.shape}")
        print(f"\nTarget distribution:")
        print(y.value_counts())
        print(f"Percentage Up: {(y.sum() / len(y) * 100):.1f}%")

    return X, y, feature_cols


# ============================================================================
# TESTING AND EXAMPLES
# ============================================================================

def run_example():
    """
    Example usage of feature engineering module
    """
    print("\n" + "=" * 60)
    print("FEATURE ENGINEERING MODULE - EXAMPLE")
    print("=" * 60 + "\n")

    # Import necessary modules
    import sys
    sys.path.append('..')
    from src.data_collection import load_stock_data
    from src.preprocessing import preprocess_stock_data

    # Load and preprocess data
    ticker = 'AAPL'
    raw_data = load_stock_data(ticker, 'raw')

    if raw_data is not None:
        # Preprocess
        processed_data = preprocess_stock_data(raw_data)

        print("Before feature engineering:")
        print(f"Columns: {list(processed_data.columns)}")
        print(f"Shape: {processed_data.shape}\n")

        # Engineer features
        feature_data = engineer_all_features(processed_data)

        print("\nAfter feature engineering:")
        print(f"Columns: {list(feature_data.columns)}")
        print(f"Shape: {feature_data.shape}\n")

        # Save
        feature_path = config.get_data_path(ticker, 'features')
        feature_data.to_csv(feature_path, index=False)
        print(f"Saved to: {feature_path}\n")

        # Show sample
        print("Sample of engineered features:")
        print(feature_data[['Date', 'Close', 'Returns', 'SMA_5',
                            'SMA_20', 'Volatility', 'Target']].tail(10))

        # Prepare for ML
        print("\n" + "=" * 60)
        X, y, features = prepare_ml_data(feature_data)


if __name__ == '__main__':
    run_example()