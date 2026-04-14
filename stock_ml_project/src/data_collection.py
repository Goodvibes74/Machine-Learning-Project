# import yfinance as yf
# import pandas as pd
# import matplotlib.pyplot as plt
# import os
#
# # Download data
# ticker = "AAPL"
# data = yf.download(ticker, start="2023-01-01", end="2024-01-01")
#
# # YOUR CODE HERE:
# # 1. Print first 10 rows using .head(10)
# print (data)
# #Exporting Data to CSV
# destination_folder= r'C:\Users\user\Desktop\Machine Learning Project\stock_ml_project\data\raw'
# file_name = "APPL_data.csv"
# file_path = os.path.join(destination_folder, file_name)
# data.to_csv(file_path)
# print("Data exported")

"""
Data Collection Module

This module handles downloading stock market data from Yahoo Finance.
It provides functions to fetch historical OHLCV data and save it locally.

Key Concepts:
- OHLCV: Open, High, Low, Close, Volume - standard stock price data
- yfinance: Python library that downloads data from Yahoo Finance API
"""

import yfinance as yf
import pandas as pd
import os
from datetime import datetime
from tqdm import tqdm
print ("✅ Imported necessary libraries")
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))# Add parent directory to path so we can import config
import config
print ("✅ Imported config settings")


def download_stock_data(ticker, start_date, end_date, save=True):
    """
    Download historical stock data for a single ticker

    This function uses the yfinance library to download stock data from
    Yahoo Finance. The data includes daily OHLCV (Open, High, Low, Close, Volume).

    Args:
        ticker (str): Stock ticker symbol (e.g., 'AAPL', 'MSFT')
        start_date (str): Start date in 'YYYY-MM-DD' format
        end_date (str): End date in 'YYYY-MM-DD' format
        save (bool): Whether to save the data to CSV file

    Returns:
        pd.DataFrame: DataFrame with stock data, or None if download fails

    Example:
        >>> data = download_stock_data('AAPL', '2023-01-01', '2024-01-01')
        >>> print(data.head())
    """
    try:
        if config.VERBOSE:
            print(f"Downloading {ticker} data from {start_date} to {end_date}...")

        # Download data using yfinance
        # progress=False suppresses the download progress bar
        stock_data = yf.download(
            ticker,
            start=start_date,
            end=end_date,
            progress=False
        )

        # Check if data was successfully downloaded
        if stock_data.empty:
            print(f"❌ No data found for {ticker}")
            return None

        # Reset index to make 'Date' a column instead of index
        stock_data.reset_index(inplace=True)

        # Add ticker column for reference
        stock_data['Ticker'] = ticker # Add ticker symbol as a column for easier identification later

        # Save to CSV if requested
        if save:
            filepath = config.get_data_path(ticker, 'raw')
            stock_data.to_csv(filepath, index=False)
            if config.VERBOSE:
                print(f"✅ Saved {len(stock_data)} rows to {filepath}")

        if config.VERBOSE:
            print(f"✅ Downloaded {len(stock_data)} days of data for {ticker}")

        return stock_data

    except Exception as e:
        print(f"❌ Error downloading {ticker}: {e}")
        return None


def download_multiple_stocks(tickers, start_date, end_date, save=True):
    """
    Download data for multiple stock tickers

    This function loops through a list of tickers and downloads data for each.
    It includes error handling so that if one ticker fails, others still download.

    Args:
        tickers (list): List of ticker symbols
        start_date (str): Start date in 'YYYY-MM-DD' format
        end_date (str): End date in 'YYYY-MM-DD' format
        save (bool): Whether to save each stock's data to CSV

    Returns:
        dict: Dictionary mapping ticker -> DataFrame

    Example:
        >>> tickers = ['AAPL', 'MSFT', 'GOOGL']
        >>> data_dict = download_multiple_stocks(tickers, '2023-01-01', '2024-01-01')
        >>> print(data_dict['AAPL'].head())
    """
    stock_data_dict = {} # Initialize an empty dictionary to store data for each ticker

    # Use tqdm for a progress bar (makes it easier to track multiple downloads)
    for ticker in tqdm(tickers, desc="Downloading stocks"):
        data = download_stock_data(ticker, start_date, end_date, save=save)

        if data is not None:
            stock_data_dict[ticker] = data

    print(f"\n✅ Successfully downloaded {len(stock_data_dict)}/{len(tickers)} stocks")

    return stock_data_dict


def load_stock_data(ticker, data_type='raw'):
    """
    Load previously saved stock data from CSV

    This function loads data that was previously downloaded and saved.
    Useful to avoid re-downloading data every time you run your code.

    Args:
        ticker (str): Stock ticker symbol
        data_type (str): Type of data to load ('raw', 'processed', or 'features')

    Returns:
        pd.DataFrame: Loaded data, or None if file doesn't exist

    Example:
        >>> data = load_stock_data('AAPL', 'raw')
        >>> if data is not None:
        >>>     print(data.head())
    """
    filepath = config.get_data_path(ticker, data_type) # Get the correct file path based on ticker and data type

    if not os.path.exists(filepath):
        print(f"❌ File not found: {filepath}")
        print(f"   Run download_stock_data('{ticker}', ...) first")
        return None

    try:
        data = pd.read_csv(filepath)

        # Convert Date column to datetime if it exists
        if 'Date' in data.columns:
            data['Date'] = pd.to_datetime(data['Date'])

        if config.VERBOSE:
            print(f"✅ Loaded {len(data)} rows from {filepath}")

        return data

    except Exception as e:
        print(f"❌ Error loading {filepath}: {e}")
        return None


def get_data_info(ticker, data_type='raw'):
    """
    Get summary information about stored data

    This is a helper function to quickly check what data you have
    without loading the entire file.

    Args:
        ticker (str): Stock ticker symbol
        data_type (str): Type of data ('raw', 'processed', or 'features')

    Returns:
        dict: Information about the data (date range, row count, etc.)
    """
    data = load_stock_data(ticker, data_type)

    if data is None:
        return None

    info = {
        'ticker': ticker,
        'rows': len(data),
        'columns': list(data.columns),
        'date_range': f"{data['Date'].min()} to {data['Date'].max()}",
        'missing_values': data.isnull().sum().sum() # it counts total missing values across the entire DataFrame
    }

    return info


# ============================================================================
# TESTING AND EXAMPLES
# ============================================================================

def run_example():
    """
    Example usage of this module
    Run this to test if everything works correctly
    """
    print("\n" + "=" * 60)
    print("DATA COLLECTION MODULE - EXAMPLE")
    print("=" * 60 + "\n")

    # Example 1: Download single stock
    print("Example 1: Download single stock")
    print("-" * 40)
    data = download_stock_data('AAPL', '2023-01-01', '2024-01-01', save=True)
    if data is not None:
        print(f"Shape: {data.shape}")
        print(f"Columns: {list(data.columns)}\n")
        print("First 5 rows:")
        print(data.head())

    print("\n" + "-" * 60 + "\n")

    # Example 2: Download multiple stocks
    print("Example 2: Download multiple stocks")
    print("-" * 40)
    test_tickers = ['AAPL', 'MSFT']
    data_dict = download_multiple_stocks(
        test_tickers,
        '2023-01-01',
        '2024-01-01',
        save=True
    )

    print("\n" + "-" * 60 + "\n")

    # Example 3: Load saved data
    print("Example 3: Load saved data")
    print("-" * 40)
    loaded_data = load_stock_data('AAPL', 'raw')
    if loaded_data is not None:
        print(f"Loaded {len(loaded_data)} rows")

    # Example 4: Get data info
    print("\n" + "-" * 60 + "\n")
    print("Example 4: Get data info")
    print("-" * 40)
    info = get_data_info('AAPL', 'raw')
    if info:
        for key, value in info.items():
            print(f"{key}: {value}")


if __name__ == '__main__':
    # Run examples when this file is executed directly
    run_example()