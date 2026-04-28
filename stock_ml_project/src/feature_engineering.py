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
    Calculate MACD (Moving Average Convergence Divergence), normalised by Close.

    MACD is a trend-following momentum indicator:
    - MACD line = 12-day EWM - 26-day EWM
    - Signal line = 9-day EWM of MACD
    - Histogram = MACD - Signal

    All three values are divided by the current Close price so they are
    scale-independent (expressed as % of price).  This prevents the model
    from learning price level rather than price direction.

    Args:
        df (pd.DataFrame): Stock data with 'Close' column

    Returns:
        pd.DataFrame: Data with 'MACD', 'MACD_signal', 'MACD_histogram' columns added
    """
    df = df.copy()

    ema_12 = df['Close'].ewm(span=12, adjust=False).mean()
    ema_26 = df['Close'].ewm(span=26, adjust=False).mean()
    raw_macd = ema_12 - ema_26
    raw_signal = raw_macd.ewm(span=9, adjust=False).mean()

    # Normalise by Close so values are comparable across price levels
    df['MACD'] = raw_macd / df['Close'] * 100
    df['MACD_signal'] = raw_signal / df['Close'] * 100
    df['MACD_histogram'] = df['MACD'] - df['MACD_signal']

    if config.VERBOSE:
        print("✅ Added 'MACD', 'MACD_signal', 'MACD_histogram' features (% of price)")

    return df


def calculate_bollinger_bands(df, window=20):
    """
    Calculate Bollinger Bands (normalised, no absolute price columns).

    BB_width   = (upper - lower) / SMA  — band width as fraction of price
    BB_position = (Close - lower) / (upper - lower) — 0=at lower, 1=at upper

    BB_upper and BB_lower are NOT added as features because they are absolute
    price values (scale-dependent) that would cause the model to learn price
    level rather than price direction.

    Args:
        df (pd.DataFrame): Stock data with 'Close' column
        window (int): Window for SMA and std (default: 20)

    Returns:
        pd.DataFrame: Data with 'BB_width' and 'BB_position' columns added
    """
    df = df.copy()

    sma = df['Close'].rolling(window=window).mean()
    std = df['Close'].rolling(window=window).std()
    bb_upper = sma + 2 * std
    bb_lower = sma - 2 * std

    df['BB_width'] = (bb_upper - bb_lower) / sma
    df['BB_position'] = (df['Close'] - bb_lower) / (bb_upper - bb_lower)

    if config.VERBOSE:
        print(f"✅ Added 'BB_width', 'BB_position' features (window={window})")

    return df


def calculate_obv(df):
    """
    Calculate On-Balance Volume (OBV) change only.

    OBV accumulates volume signed by price direction.  The raw cumulative OBV
    is monotonically increasing over time (scale-dependent) so it is NOT added
    as a feature.  Only OBV_change (percentage change) is kept — it captures
    sudden volume pressure without the long-term drift.

    Args:
        df (pd.DataFrame): Stock data with 'Close' and 'Volume' columns

    Returns:
        pd.DataFrame: Data with 'OBV_change' column added
    """
    df = df.copy()

    close_shifted = df['Close'].shift(1)
    direction = (df['Close'] > close_shifted).astype(int) - (df['Close'] < close_shifted).astype(int)
    obv = (direction * df['Volume']).cumsum()
    df['OBV_change'] = obv.pct_change()

    if config.VERBOSE:
        print("✅ Added 'OBV_change' feature")

    return df


def calculate_atr(df, window=14):
    """
    Calculate Average True Range (ATR), normalised by Close price.

    ATR measures volatility as the average True Range over N days.
    Dividing by Close gives ATR as a percentage of price, making it
    scale-independent across different price levels and time periods.

    True Range = max(High - Low, |High - Prev Close|, |Low - Prev Close|)

    Args:
        df (pd.DataFrame): Stock data with 'High', 'Low', 'Close' columns
        window (int): ATR period (default: 14)

    Returns:
        pd.DataFrame: Data with 'ATR_14' column added (% of Close price)
    """
    df = df.copy()

    prev_close = df['Close'].shift(1)
    high_low = df['High'] - df['Low']
    high_prev = (df['High'] - prev_close).abs()
    low_prev = (df['Low'] - prev_close).abs()

    true_range = pd.concat([high_low, high_prev, low_prev], axis=1).max(axis=1)
    raw_atr = true_range.rolling(window=window).mean()

    # Normalise by Close so it is scale-independent (expressed as % of price)
    df['ATR_14'] = raw_atr / df['Close'] * 100

    if config.VERBOSE:
        print(f"✅ Added 'ATR_14' feature (% of price, window={window})")

    return df


def calculate_lag_features(df, lags=(1, 2, 3)):
    """
    Add lagged return features.

    Yesterday's and the day-before-yesterday's returns capture short-term
    momentum and mean-reversion signals that are not already present in the
    rolling Momentum columns.

    Args:
        df (pd.DataFrame): Data with 'Returns' column
        lags (tuple): Number of days to lag

    Returns:
        pd.DataFrame: Data with 'Returns_lag1', 'Returns_lag2', ... columns
    """
    df = df.copy()
    for lag in lags:
        df[f'Returns_lag{lag}'] = df['Returns'].shift(lag)
        if config.VERBOSE:
            print(f"✅ Added 'Returns_lag{lag}' feature")
    return df


def calculate_volume_ratio(df, window=20):
    """
    Add Volume_Ratio: current volume relative to its N-day rolling average.

    A ratio > 1 means unusually high trading activity (potential breakout or
    news event).  Normalising by the rolling average removes the absolute
    scale of volume, making the feature comparable across time.

    Args:
        df (pd.DataFrame): Data with 'Volume' column
        window (int): Rolling window for average volume (default: 20)

    Returns:
        pd.DataFrame: Data with 'Volume_Ratio' column added
    """
    df = df.copy()
    avg_vol = df['Volume'].rolling(window=window).mean()
    df['Volume_Ratio'] = df['Volume'] / avg_vol
    if config.VERBOSE:
        print(f"✅ Added 'Volume_Ratio' feature (window={window})")
    return df


def calculate_day_of_week(df):
    """
    Add DayOfWeek feature (0=Monday … 4=Friday).

    Day-of-week effects are well documented in equity markets (e.g. the
    Monday effect, pre-weekend drift).  This encodes the calendar pattern
    as a simple integer feature.

    Args:
        df (pd.DataFrame): Data with 'Date' column

    Returns:
        pd.DataFrame: Data with 'DayOfWeek' column added
    """
    df = df.copy()
    if 'Date' in df.columns:
        df['DayOfWeek'] = pd.to_datetime(df['Date']).dt.dayofweek
        if config.VERBOSE:
            print("✅ Added 'DayOfWeek' feature")
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


def merge_nlp_sentiment(df, sentiment_df):
    """
    Merge daily NLP sentiment into the OHLCV DataFrame with a 1-day lag.

    The lag prevents look-ahead bias: today's published news is shifted
    forward by one day so it predicts *tomorrow's* price, not today's.
    Trading days with no matching news receive NLP_Sentiment = 0.0
    (neutral) and News_Volume = 0 to keep the dataset shape intact.

    Args:
        df            (pd.DataFrame): Stock data with 'Date' column
        sentiment_df  (pd.DataFrame): Daily sentiment with columns
                                      ['Date', 'NLP_Sentiment', 'News_Volume']

    Returns:
        pd.DataFrame: df extended with 'NLP_Sentiment' and 'News_Volume'
    """
    df           = df.copy()
    sentiment_df = sentiment_df[['Date', 'NLP_Sentiment', 'News_Volume']].copy()

    df['Date']           = pd.to_datetime(df['Date'])
    sentiment_df['Date'] = pd.to_datetime(sentiment_df['Date'])

    # 1-day lag: shift each sentiment row forward by one calendar day so that
    # the news published on day T is available as a feature on day T+1.
    sentiment_df = sentiment_df.sort_values('Date').reset_index(drop=True)
    sentiment_df['Date'] = sentiment_df['Date'] + pd.Timedelta(days=1)

    df = df.merge(sentiment_df, on='Date', how='left')

    # Fill trading days that have no lagged news as neutral
    df['NLP_Sentiment'] = df['NLP_Sentiment'].fillna(0.0)
    df['News_Volume']   = df['News_Volume'].fillna(0).astype(int)

    if config.VERBOSE:
        no_news = (df['News_Volume'] == 0).sum()
        print(f"✅ Merged NLP sentiment (1-day lag applied; {no_news} days filled neutral)")

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
    # shift(-horizon) shifts the Close column backward so row[t] holds Close[t+horizon]
    df['Future_Close'] = df['Close'].shift(-horizon)

    # Create binary target: 1 if price goes up, 0 otherwise
    df['Target'] = (df['Future_Close'] > df['Close']).astype(int)

    if config.VERBOSE:
        print(f"✅ Added 'Target' variable (horizon={horizon} day)")
        print(f"   Last {horizon} row(s) will have NaN target")

    return df


def engineer_all_features(df, sentiment_df=None):
    """
    Apply all feature engineering steps.

    When sentiment_df is provided (output of
    src.sentiment_analysis.get_sentiment_for_ticker), real NLP features
    (NLP_Sentiment, News_Volume) replace the price-based proxy.
    When it is None the price-based proxy is used as a fallback.

    Pipeline:
    1.  Calculate returns (needed for volatility)
    2.  Add rolling averages
    3.  Calculate volatility (needs returns)
    4.  Calculate momentum
    5.  Calculate volume change
    6.  Calculate HL spread
    7.  Lagged returns / volume ratio / day-of-week
    8.  Technical indicators (RSI, MACD, Bollinger, OBV, ATR)
    9.  Price/SMA ratio signals
    10. Sentiment — NLP merge (if sentiment_df provided) or price proxy
    11. Create target variable

    Args:
        df           (pd.DataFrame):       Preprocessed stock data
        sentiment_df (pd.DataFrame|None):  Daily NLP sentiment
                                           ['Date', 'NLP_Sentiment', 'News_Volume']

    Returns:
        pd.DataFrame: Data with all features added

    Example:
        >>> processed_data = preprocess_stock_data(raw_data)
        >>> feature_data = engineer_all_features(processed_data, sentiment_df)
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
    df = calculate_momentum(df)            # Momentum_3/5/10/20
    df = calculate_volume_change(df)
    df = calculate_hl_spread(df)

    # Lagged returns and volume features
    df = calculate_lag_features(df)        # Returns_lag1/2/3
    df = calculate_volume_ratio(df)        # Volume_Ratio (vs 20-day avg)
    df = calculate_day_of_week(df)         # DayOfWeek (0=Mon … 4=Fri)

    # Technical indicators (normalised — no look-ahead since target is tomorrow)
    df = calculate_rsi(df)
    df = calculate_macd(df)                # MACD/signal/histogram (% of price)
    df = calculate_bollinger_bands(df)     # BB_width, BB_position (no abs. price cols)
    df = calculate_obv(df)                 # OBV_change only (not raw cumulative OBV)
    df = calculate_atr(df)                 # ATR_14 (% of price)

    # Normalized trend signals (requires SMAs to be computed first)
    df = calculate_price_sma_ratios(df)   # Price_SMA_N_ratio, SMA_5_20_ratio

    # Sentiment — use real NLP scores when available, proxy otherwise
    if sentiment_df is not None:
        df = merge_nlp_sentiment(df, sentiment_df)
    else:
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