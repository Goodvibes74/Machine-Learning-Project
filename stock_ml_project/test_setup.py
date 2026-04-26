# # test_setup.py
# import pandas as pd
# import numpy as np
# import yfinance as yf
# from sklearn.ensemble import RandomForestClassifier
# import matplotlib.pyplot as plt

# print("✅ All packages imported successfully!")

# # Test data download
# test_stock = yf.download("AAPL", start="2024-01-01", end="2024-01-10")
# print(f"✅ Downloaded {len(test_stock)} days of AAPL data")


import config
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

    print("\\n" + "=" * 80)
    print(" " * 20 + f"STOCK PREDICTION PIPELINE: {ticker}")
    print("=" * 80)
    print(f"Date Range: {start_date} to {end_date}")
    print("=" * 80 + "\\n")

    # ========================================================================
    # STEP 1: DATA COLLECTION
    # ========================================================================
    print("\\n" + "🔵 " * 30)
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
    print("\\n" + "🟢 " * 30)
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
    print("\\n" + "🟡 " * 30)
    print("STEP 3: FEATURE ENGINEERING")
    print("🟡 " * 30)

    feature_data = engineer_all_features(processed_data)

    # Save feature data
    feature_path = config.get_data_path(ticker, 'features')
    feature_data.to_csv(feature_path, index=False)
    print(f"Saved feature data to: {feature_path}")

    # Prepare for ML
    X, y, feature_names = prepare_ml_data(feature_data)

    print(f"\\nFinal dataset:")
    print(f"  Samples: {len(X)}")
    print(f"  Features: {len(feature_names)}")
    print(f"  Target distribution: {y.value_counts().to_dict()}")

    # ========================================================================
    # STEP 4: MODEL TRAINING & EVALUATION
    # ========================================================================
    print("\\n" + "🔴 " * 30)
    print("STEP 4: MODEL TRAINING & EVALUATION")
    print("🔴 " * 30)

    # Run complete training pipeline
    model_results = train_and_evaluate_pipeline(X, y, feature_names)

    # ========================================================================
    # STEP 5: BASELINE COMPARISON
    # ========================================================================
    print("\\n" + "🟣 " * 30)
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
    print("\\n" + "🟠 " * 30)
    print("STEP 6: BACKTESTING (Walk-Forward Validation)")
    print("🟠 " * 30)

    backtest_results = walk_forward_validation(X, y, feature_names,
                                               n_splits=config.N_SPLITS)

    # Plot backtesting results
    print("\nGenerating backtesting visualization...")
    plot_backtest_results(backtest_results)

    # ========================================================================
    # FINAL SUMMARY
    # ========================================================================
    print("\\n" + "=" * 80)
    print(" " * 25 + "PIPELINE COMPLETE!")
    print("=" * 80)

    print(f"\\n📊 FINAL RESULTS FOR {ticker}:")
    print("-" * 80)
    print(f"Test Set Accuracy:        {model_results['metrics']['test_accuracy']:.2%}")
    print(f"Backtesting Mean Accuracy: {backtest_results['accuracy'].mean():.2%}")
    print(f"Backtesting Std:          {backtest_results['accuracy'].std():.4f}")

    if comparison['improvement'] > 0:
        print(f"\\nSentiment features improved accuracy by: {comparison['improvement']:.2%}")
    else:
        print(f"\\nSentiment features did not improve accuracy")

    print("\\n" + "=" * 80)

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

print("✅ Function defined: run_complete_pipeline()")
