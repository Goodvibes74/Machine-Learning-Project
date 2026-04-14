"""
Data Preprocessing Module

This module handles cleaning and preparing raw stock data for analysis.
It removes unnecessary columns, handles missing values, and ensures data quality.

Key Concepts:
- Data cleaning: Removing errors, handling missing values
- Data validation: Ensuring data is in the correct format
- Column selection: Keeping only relevant features
"""

import pandas as pd
import numpy as np
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))# Add parent directory to path so we can import config
import config


def clean_stock_data(df):
    """
    Clean raw stock data

    This function performs basic cleaning operations:
    1. Removes dividend and split columns (not needed for price prediction)
    2. Handles missing values
    3. Ensures proper data types
    4. Sorts by date

    Args:
        df (pd.DataFrame): Raw stock data

    Returns:
        pd.DataFrame: Cleaned stock data

    Example:
        >>> raw_data = load_stock_data('AAPL', 'raw')
        >>> clean_data = clean_stock_data(raw_data)
    """
    df = df.copy()  # Don't modify original dataframe because we might want to keep raw data for reference

    # Remove columns we don't need (dividends, stock splits)
    columns_to_drop = ['Dividends', 'Stock Splits']
    existing_drops = [col for col in columns_to_drop if col in df.columns]
    if existing_drops:
        df = df.drop(columns=existing_drops)
        if config.VERBOSE:
            print(f"Dropped columns: {existing_drops}")

    # Ensure Date column is datetime type
    if 'Date' in df.columns:
        df['Date'] = pd.to_datetime(df['Date'])

    # Sort by date (important for time series!)
    if 'Date' in df.columns:
        df = df.sort_values('Date').reset_index(drop=True)#reset index doesn't keep old index which might be out of order after sorting

    # Ensure price and volume columns are numeric
    numeric_cols = [col for col in ['Open', 'High', 'Low', 'Close', 'Volume'] if col in df.columns]
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors='coerce')

    # Check for and report missing values
    missing = df.isnull().sum() # .isnull().sum() counts missing values in each column
    if missing.sum() > 0:
        print(f"⚠️  Missing values found:")
        print(missing[missing > 0])

    return df


def handle_missing_values(df, method='forward_fill'):
    """
    Handle missing values in stock data

    For stock price data, we typically use forward fill (carry last known price)
    or backward fill. We DON'T use mean/median because prices have trends.

    Args:
        df (pd.DataFrame): Stock data with potential missing values
        method (str): Method to handle missing values
            - 'forward_fill': Use previous value (recommended for stock data)
            - 'backward_fill': Use next value
            - 'drop': Remove rows with missing values

    Returns:
        pd.DataFrame: Data with missing values handled

    Example:
        >>> df = handle_missing_values(df, method='forward_fill')
    """
    df = df.copy()

    if method == 'forward_fill':
        # Fill missing values with previous value
        df = df.ffill()
        if config.VERBOSE:
            print("Applied forward fill for missing values")

    elif method == 'backward_fill':
        # Fill missing values with next value
        df = df.bfill()
        if config.VERBOSE:
            print("Applied backward fill for missing values")

    elif method == 'drop':
        # Remove rows with any missing values
        original_len = len(df)
        df = df.dropna()# Drop any rows that have missing values in any column
        dropped = original_len - len(df) # Calculate how many rows were dropped
        if config.VERBOSE:
            print(f"Dropped {dropped} rows with missing values")
    else:
        raise ValueError(f"Unknown method: {method}")

    # If any missing values remain (e.g., first row after forward fill)
    # drop those rows
    if df.isnull().sum().sum() > 0:
        df = df.dropna()
        if config.VERBOSE:
            print("Dropped remaining rows with missing values")

    return df


def validate_data(df):
    """
    Validate that data is in correct format for further processing

    Checks:
    1. Required columns exist
    2. No missing values remain
    3. Prices are positive
    4. Volume is non-negative
    5. Dates are properly sorted

    Args:
        df (pd.DataFrame): Stock data to validate

    Returns:
        bool: True if validation passes, False otherwise

    Example:
        >>> if validate_data(df):
        >>>     print("Data is valid!")
    """
    is_valid = True # this flag will track if any validation checks fail. We start with True and set to False if any check fails

    # Check 1: Required columns
    required_cols = ['Date', 'Open', 'High', 'Low', 'Close', 'Volume']
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        print(f"❌ Missing required columns: {missing_cols}")
        is_valid = False

    # Check 2: No missing values
    if df.isnull().sum().sum() > 0:
        print(f"❌ Data still contains missing values")
        is_valid = False

    # Check 3: Prices are positive
    price_cols = ['Open', 'High', 'Low', 'Close']
    for col in price_cols:
        if col in df.columns and ((df[col]) <= 0).any():# Stock prices should never be zero or negative, if we find any it indicates a data error
            print(f"❌ {col} contains non-positive values") 
            is_valid = False

    # Check 4: Volume is non-negative
    if 'Volume' in df.columns and (df['Volume'] < 0).any():# Volume should never be negative, if we find any it indicates a data error.
        print(f"❌ Volume contains negative values")
        is_valid = False

    # Check 5: Dates are sorted
    if 'Date' in df.columns:
        if not df['Date'].is_monotonic_increasing:
            print(f"❌ Dates are not sorted in ascending order")
            is_valid = False

    # Check 6: High >= Low (sanity check)
    if 'High' in df.columns and 'Low' in df.columns:
        if (df['High'] < df['Low']).any():
            print(f"❌ Some High prices are less than Low prices")
            is_valid = False

    if is_valid and config.VERBOSE:
        print("✅ Data validation passed")

    return is_valid


def preprocess_stock_data(df, save_path=None):
    """
    Complete preprocessing pipeline

    This is the main function that runs all preprocessing steps:
    1. Clean the data
    2. Handle missing values
    3. Validate the result
    4. Optionally save to CSV

    Args:
        df (pd.DataFrame): Raw stock data
        save_path (str): Path to save processed data (optional)

    Returns:
        pd.DataFrame: Fully preprocessed data ready for feature engineering

    Example:
        >>> raw_data = load_stock_data('AAPL', 'raw')
        >>> processed = preprocess_stock_data(raw_data,
        >>>     save_path=config.get_data_path('AAPL', 'processed'))
    """
    if config.VERBOSE:
        print("\n" + "=" * 60)
        print("PREPROCESSING PIPELINE")
        print("=" * 60)

    # Step 1: Clean
    if config.VERBOSE:
        print("\nStep 1: Cleaning data...")
    df_clean = clean_stock_data(df)

    # Step 2: Handle missing values
    if config.VERBOSE:
        print("\nStep 2: Handling missing values...")
    df_processed = handle_missing_values(df_clean, method='forward_fill')

    # Step 3: Validate
    if config.VERBOSE:
        print("\nStep 3: Validating data...")
    is_valid = validate_data(df_processed)

    if not is_valid:
        print("⚠️  Warning: Data validation failed. Check the data carefully.")

    # Step 4: Save if path provided
    if save_path:
        df_processed.to_csv(save_path, index=False)
        if config.VERBOSE:
            print(f"\n✅ Saved processed data to {save_path}")

    if config.VERBOSE:
        print("\n" + "=" * 60)
        print(f"Preprocessing complete: {len(df_processed)} rows")
        print("=" * 60 + "\n")

    return df_processed


def get_data_summary(df):
    """
    Get a summary of the stock data

    Provides useful statistics about the data:
    - Date range
    - Number of trading days
    - Price ranges
    - Average volume

    Args:
        df (pd.DataFrame): Stock data

    Returns:
        dict: Summary statistics
    """
    summary = {
        'rows': len(df),
        'date_range': f"{df['Date'].min()} to {df['Date'].max()}",
        'trading_days': len(df),
        'avg_close': df['Close'].mean() if 'Close' in df.columns else None,
        'min_close': df['Close'].min() if 'Close' in df.columns else None,
        'max_close': df['Close'].max() if 'Close' in df.columns else None,
        'avg_volume': df['Volume'].mean() if 'Volume' in df.columns else None,
        'columns': list(df.columns)
    }

    return summary


# ============================================================================
# TESTING AND EXAMPLES
# ============================================================================

def run_example():
    """
    Example usage of preprocessing module
    """
    print("\n" + "=" * 60)
    print("PREPROCESSING MODULE - EXAMPLE")
    print("=" * 60 + "\n")

    # First, make sure we have some raw data
    import sys
    sys.path.append('..')
    from src.data_collection import download_stock_data, load_stock_data

    # Download if needed
    ticker = 'AAPL'
    raw_data = load_stock_data(ticker, 'raw')

    if raw_data is None:
        print("Downloading sample data first...")
        raw_data = download_stock_data(ticker, '2023-01-01', '2024-01-01', save=True)

    if raw_data is not None:
        print("Before preprocessing:")
        print(f"Shape: {raw_data.shape}")
        print(f"Columns: {list(raw_data.columns)}")

        # Preprocess
        processed_data = preprocess_stock_data(
            raw_data,
            save_path=config.get_data_path(ticker, 'processed')
        )

        print("\nAfter preprocessing:")
        print(f"Shape: {processed_data.shape}")
        print(f"Columns: {list(processed_data.columns)}")

        # Get summary
        print("\nData Summary:")
        summary = get_data_summary(processed_data)
        for key, value in summary.items():
            print(f"{key}: {value}")


if __name__ == '__main__':
    run_example()