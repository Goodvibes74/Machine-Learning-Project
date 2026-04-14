"""
Main Pipeline Script

This is the central script that orchestrates the entire stock prediction pipeline.
Run this file to execute the complete workflow from data collection to backtesting.

Usage:
    python main.py
"""

import sys
import os

# Add src directory to path so we can import modules
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

import config
from src.data_collection import (
    download_stock_data,
    download_multiple_stocks,
    load_stock_data
)
from src.preprocessing import preprocess_stock_data
from src.feature_engineering import engineer_all_features, prepare_ml_data
from src.models import train_and_evaluate_pipeline, compare_baseline
from src.backtesting import walk_forward_validation, compare_strategies


def run_complete_pipeline(ticker, start_date=None, end_date=None):
    """
    Run the complete ML pipeline for a single stock

    This function orchestrates all steps:
    1. Data Collection
    2. Preprocessing
    3. Feature Engineering
    4. Model Training
    5. Evaluation
    6. Backtesting

    Args:
        ticker (str): Stock ticker symbol (e.g., 'AAPL')
        start_date (str): Start date for data (default from config)
        end_date (str): End date for data (default from config)

    Returns:
        dict: Results from all pipeline steps
    """
    if start_date is None:
        start_date = config.START_DATE
    if end_date is None:
        end_date = config.END_DATE

    print("\n" + "=" * 80)
    print(" " * 20 + f"STOCK PREDICTION PIPELINE: {ticker}")
    print("=" * 80)
    print(f"Date Range: {start_date} to {end_date}")
    print("=" * 80 + "\n")

    # ========================================================================
    # STEP 1: DATA COLLECTION
    # ========================================================================
    print("\n" + "🔵 " * 30)
    print("STEP 1: DATA COLLECTION")
    print("🔵 " * 30)

    # Try to load existing data first
    raw_data = load_stock_data(ticker, 'raw')

    if raw_data is None:
        # Download if not found
        print(f"No existing data found. Downloading {ticker}...")
        raw_data = download_stock_data(ticker, start_date, end_date, save=True)

        if raw_data is None:
            print(f"❌ Failed to download data for {ticker}")
            return None
    else:
        print(f"✅ Loaded existing data for {ticker}")

    print(f"Raw data shape: {raw_data.shape}")

    # ========================================================================
    # STEP 2: DATA PREPROCESSING
    # ========================================================================
    print("\n" + "🟢 " * 30)
    print("STEP 2: DATA PREPROCESSING")
    print("🟢 " * 30)

    processed_data = preprocess_stock_data(
        raw_data,
        save_path=config.get_data_path(ticker, 'processed')
    )

    print(f"Processed data shape: {processed_data.shape}")

    # ========================================================================
    # STEP 3: FEATURE ENGINEERING
    # ========================================================================
    print("\n" + "🟡 " * 30)
    print("STEP 3: FEATURE ENGINEERING")
    print("🟡 " * 30)

    feature_data = engineer_all_features(processed_data)

    # Save feature data
    feature_path = config.get_data_path(ticker, 'features')
    feature_data.to_csv(feature_path, index=False)
    print(f"Saved feature data to: {feature_path}")

    # Prepare for ML
    X, y, feature_names = prepare_ml_data(feature_data)

    print(f"\nFinal dataset:")
    print(f"  Samples: {len(X)}")
    print(f"  Features: {len(feature_names)}")
    print(f"  Target distribution: {y.value_counts().to_dict()}")

    # ========================================================================
    # STEP 4: MODEL TRAINING & EVALUATION
    # ========================================================================
    print("\n" + "🔴 " * 30)
    print("STEP 4: MODEL TRAINING & EVALUATION")
    print("🔴 " * 30)

    # Run complete training pipeline
    model_results = train_and_evaluate_pipeline(X, y, feature_names)

    # ========================================================================
    # STEP 5: BASELINE COMPARISON
    # ========================================================================
    print("\n" + "🟣 " * 30)
    print("STEP 5: BASELINE COMPARISON")
    print("🟣 " * 30)

    comparison = compare_baseline(
        model_results['X_train'],
        model_results['y_train'],
        model_results['X_test'],
        model_results['y_test']
    )

    # ========================================================================
    # STEP 6: BACKTESTING
    # ========================================================================
    print("\n" + "🟠 " * 30)
    print("STEP 6: BACKTESTING (Walk-Forward Validation)")
    print("🟠 " * 30)

    backtest_results = walk_forward_validation(X, y, feature_names,
                                               n_splits=config.N_SPLITS)

    # ========================================================================
    # FINAL SUMMARY
    # ========================================================================
    print("\n" + "=" * 80)
    print(" " * 25 + "PIPELINE COMPLETE!")
    print("=" * 80)

    print(f"\n📊 FINAL RESULTS FOR {ticker}:")
    print("-" * 80)
    print(f"Test Set Accuracy:        {model_results['metrics']['test_accuracy']:.2%}")
    print(f"Backtesting Mean Accuracy: {backtest_results['accuracy'].mean():.2%}")
    print(f"Backtesting Std:          {backtest_results['accuracy'].std():.4f}")

    if comparison['improvement'] > 0:
        print(f"\nSentiment features improved accuracy by: {comparison['improvement']:.2%}")
    else:
        print(f"\nSentiment features did not improve accuracy")

    print("\n" + "=" * 80)

    # Return all results
    return {
        'ticker': ticker,
        'raw_data': raw_data,
        'processed_data': processed_data,
        'feature_data': feature_data,
        'X': X,
        'y': y,
        'feature_names': feature_names,
        'model_results': model_results,
        'comparison': comparison,
        'backtest_results': backtest_results
    }


def run_multiple_stocks(tickers=None):
    """
    Run pipeline for multiple stocks

    Args:
        tickers (list): List of ticker symbols (default from config)

    Returns:
        dict: Results for each ticker
    """
    if tickers is None:
        tickers = config.TICKERS

    print("\n" + "=" * 80)
    print(" " * 15 + f"RUNNING PIPELINE FOR {len(tickers)} STOCKS")
    print("=" * 80)
    print(f"Tickers: {', '.join(tickers)}")
    print("=" * 80 + "\n")

    results = {}

    for idx, ticker in enumerate(tickers, 1):
        print(f"\n{'=' * 80}")
        print(f" " * 20 + f"PROCESSING {ticker} ({idx}/{len(tickers)})")
        print(f"{'=' * 80}\n")

        try:
            ticker_results = run_complete_pipeline(ticker)
            if ticker_results is not None:
                results[ticker] = ticker_results
                print(f"\n✅ Successfully processed {ticker}")
        except Exception as e:
            print(f"\n❌ Error processing {ticker}: {e}")
            continue

    # Summary across all stocks
    print("\n" + "=" * 80)
    print(" " * 20 + "MULTI-STOCK SUMMARY")
    print("=" * 80)

    for ticker, result in results.items():
        test_acc = result['model_results']['metrics']['test_accuracy']
        backtest_acc = result['backtest_results']['accuracy'].mean()
        print(f"{ticker:6s} - Test: {test_acc:.2%}, Backtest: {backtest_acc:.2%}")

    return results


# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    """
    Main entry point

    This is what runs when you execute: python main.py
    """
    print("\n" + "🚀 " * 40)
    print(" " * 15 + "STOCK MARKET PREDICTION SYSTEM")
    print(" " * 10 + "Random Forest & Sentiment Analysis Hybrid Model")
    print("🚀 " * 40 + "\n")

    # Print configuration
    config.print_config()

    # Ask user what to do
    print("\nWhat would you like to do?")
    print("1. Run pipeline for a single stock")
    print("2. Run pipeline for all configured stocks")
    print("3. Download data for all stocks (preparation)")
    print("4. Exit")

    choice = input("\nEnter your choice (1-4): ").strip()

    if choice == '1':
        # Single stock
        ticker = input("Enter ticker symbol (e.g., AAPL): ").strip().upper()
        results = run_complete_pipeline(ticker)

    elif choice == '2':
        # Multiple stocks
        results = run_multiple_stocks()

    elif choice == '3':
        # Just download data
        print("\nDownloading data for all configured stocks...")
        data_dict = download_multiple_stocks(
            config.TICKERS,
            config.START_DATE,
            config.END_DATE,
            save=True
        )
        print(f"\n✅ Downloaded data for {len(data_dict)} stocks")

    elif choice == '4':
        print("\nGoodbye! 👋")
        return

    else:
        print("\n❌ Invalid choice. Please run again.")
        return

    print("\n" + "=" * 80)
    print(" " * 25 + "ALL DONE! 🎉")
    print("=" * 80 + "\n")


if __name__ == '__main__':
    # Run main when script is executed
    main()