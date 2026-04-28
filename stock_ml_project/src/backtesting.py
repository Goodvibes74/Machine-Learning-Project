"""
Backtesting Module

This module implements walk-forward validation for time series models.
Backtesting simulates real trading by training on past data and testing on future.

Key Concepts:
- Walk-forward: Train on expanding window, test on next period
- Time Series CV: Like cross-validation but respects time order
- No look-ahead: Only use data available at prediction time
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import accuracy_score, classification_report
import matplotlib.pyplot as plt
import config
from src.models import train_random_forest


def walk_forward_validation(X, y, feature_names, n_splits=None):
    """
    Perform walk-forward validation

    This is the GOLD STANDARD for evaluating time series models.

    How it works:
    Split 1: Train [0:100]     Test [100:120]
    Split 2: Train [0:120]     Test [120:140]
    Split 3: Train [0:140]     Test [140:160]
    ...

    Each iteration:
    1. Train on all past data
    2. Predict next period
    3. Expand training window
    4. Repeat

    This simulates real trading where you retrain as new data arrives.

    Args:
        X (pd.DataFrame): Features
        y (pd.Series): Target
        feature_names (list): List of feature column names
        n_splits (int): Number of splits (default from config)

    Returns:
        pd.DataFrame: Results for each fold

    Example:
        >>> results = walk_forward_validation(X, y, features, n_splits=5)
        >>> print(f"Mean accuracy: {results['accuracy'].mean():.4f}")
    """
    if n_splits is None:
        n_splits = config.N_SPLITS

    print("\n" + "=" * 70)
    print(" " * 20 + "WALK-FORWARD VALIDATION")
    print("=" * 70)
    print(f"Number of splits: {n_splits}")
    print(f"Total samples: {len(X)}")

    # Initialize time series cross-validator
    tscv = TimeSeriesSplit(n_splits=n_splits)

    results = []
    all_predictions = []
    all_actuals = []

    # Iterate through splits
    for fold_idx, (train_idx, test_idx) in enumerate(tscv.split(X)):
        fold = fold_idx + 1

        print(f"\n{'=' * 70}")
        print(f"Fold {fold}/{n_splits}")
        print(f"{'=' * 70}")

        # Split data
        X_train = X.iloc[train_idx]
        X_test = X.iloc[test_idx]
        y_train = y.iloc[train_idx]
        y_test = y.iloc[test_idx]

        print(f"Training samples: {len(X_train)}")
        print(f"Testing samples: {len(X_test)}")

        # Train model with threshold calibration for this fold
        model = train_random_forest(X_train, y_train)

        # Predict
        predictions = model.predict(X_test)

        # Evaluate
        accuracy = accuracy_score(y_test, predictions)

        print(f"Accuracy: {accuracy:.4f}")

        # Store results
        results.append({
            'fold': fold,
            'train_size': len(X_train),
            'test_size': len(X_test),
            'train_start_idx': train_idx[0],
            'train_end_idx': train_idx[-1],
            'test_start_idx': test_idx[0],
            'test_end_idx': test_idx[-1],
            'accuracy': accuracy
        })

        all_predictions.extend(predictions)
        all_actuals.extend(y_test)

    # Convert to DataFrame
    results_df = pd.DataFrame(results)

    # Print summary
    print("\n" + "=" * 70)
    print(" " * 20 + "BACKTESTING SUMMARY")
    print("=" * 70)
    print(results_df[['fold', 'train_size', 'test_size', 'accuracy']])
    print(f"\nMean Accuracy: {results_df['accuracy'].mean():.4f}")
    print(f"Std Accuracy:  {results_df['accuracy'].std():.4f}")
    print(f"Min Accuracy:  {results_df['accuracy'].min():.4f}")
    print(f"Max Accuracy:  {results_df['accuracy'].max():.4f}")

    # Overall classification report
    print("\nOverall Classification Report:")
    print(classification_report(all_actuals, all_predictions,
                                target_names=['Down (0)', 'Up (1)']))

    return results_df


def plot_backtest_results(results_df):
    """
    Visualize backtesting results over time

    This plot shows:
    - How accuracy changes across different time periods
    - Whether model performance is stable or degrading

    Args:
        results_df (pd.DataFrame): Results from walk_forward_validation
    """
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))

    # Plot 1: Accuracy over folds
    ax1.plot(results_df['fold'], results_df['accuracy'],
             marker='o', linewidth=2, markersize=8)
    ax1.axhline(results_df['accuracy'].mean(),
                color='red', linestyle='--', label='Mean Accuracy')
    ax1.set_xlabel('Fold')
    ax1.set_ylabel('Accuracy')
    ax1.set_title('Model Accuracy Across Time Periods')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Plot 2: Train and test set sizes
    ax2.plot(results_df['fold'], results_df['train_size'],
             marker='s', label='Train Size', linewidth=2)
    ax2.plot(results_df['fold'], results_df['test_size'],
             marker='^', label='Test Size', linewidth=2)
    ax2.set_xlabel('Fold')
    ax2.set_ylabel('Number of Samples')
    ax2.set_title('Training and Test Set Sizes')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()


def compare_strategies(X, y, feature_names, n_splits=None):
    """
    Compare different modeling strategies using backtesting

    This compares:
    1. Baseline: Price features only
    2. Hybrid: Price + sentiment features

    Args:
        X (pd.DataFrame): All features
        y (pd.Series): Target
        feature_names (list): All feature names
        n_splits (int): Number of CV splits

    Returns:
        dict: Comparison results
    """
    print("\n" + "=" * 70)
    print(" " * 15 + "STRATEGY COMPARISON WITH BACKTESTING")
    print("=" * 70)

    # Define feature sets
    price_features = [f for f in feature_names if 'Sentiment' not in f]

    print(f"\nPrice-only features: {len(price_features)}")
    print(f"Hybrid features: {len(feature_names)}")

    # Backtest baseline (price-only)
    print("\n" + "=" * 70)
    print("1. BASELINE MODEL (Price Features Only)")
    print("=" * 70)
    baseline_results = walk_forward_validation(
        X[price_features], y, price_features, n_splits
    )

    # Backtest hybrid (price + sentiment)
    print("\n" + "=" * 70)
    print("2. HYBRID MODEL (Price + Sentiment)")
    print("=" * 70)
    hybrid_results = walk_forward_validation(
        X, y, feature_names, n_splits
    )

    # Compare
    print("\n" + "=" * 70)
    print(" " * 20 + "COMPARISON")
    print("=" * 70)

    baseline_mean = baseline_results['accuracy'].mean()
    hybrid_mean = hybrid_results['accuracy'].mean()
    improvement = hybrid_mean - baseline_mean

    print(f"\nBaseline mean accuracy: {baseline_mean:.4f}")
    print(f"Hybrid mean accuracy:   {hybrid_mean:.4f}")
    print(f"Improvement:            {improvement:+.4f} ({improvement * 100:+.2f}%)")

    if improvement > 0:
        print("\n✅ Sentiment features improve predictions!")
    else:
        print("\n⚠️  Sentiment features don't help (or hurt)")

    # Plot comparison
    fig, ax = plt.subplots(figsize=(12, 6))

    x = np.arange(len(baseline_results))
    width = 0.35

    ax.bar(x - width / 2, baseline_results['accuracy'], width,
           label='Baseline (Price Only)', alpha=0.8)
    ax.bar(x + width / 2, hybrid_results['accuracy'], width,
           label='Hybrid (Price + Sentiment)', alpha=0.8)

    ax.set_xlabel('Fold')
    ax.set_ylabel('Accuracy')
    ax.set_title('Baseline vs Hybrid Model Performance Across Folds')
    ax.set_xticks(x)
    ax.set_xticklabels(baseline_results['fold'])
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    plt.show()

    return {
        'baseline_results': baseline_results,
        'hybrid_results': hybrid_results,
        'baseline_mean': baseline_mean,
        'hybrid_mean': hybrid_mean,
        'improvement': improvement
    }


def analyze_prediction_distribution(y_true, y_pred):
    """
    Analyze the distribution of predictions

    Checks:
    - Are we predicting mostly up or down?
    - Are predictions balanced?
    - Are we too conservative or aggressive?

    Args:
        y_true: Actual values
        y_pred: Predicted values
    """
    print("\n" + "=" * 60)
    print("PREDICTION DISTRIBUTION ANALYSIS")
    print("=" * 60)

    # Actual distribution
    actual_up = sum(y_true)
    actual_down = len(y_true) - actual_up
    actual_up_pct = actual_up / len(y_true) * 100

    # Predicted distribution
    pred_up = sum(y_pred)
    pred_down = len(y_pred) - pred_up
    pred_up_pct = pred_up / len(y_pred) * 100

    print(f"\nActual Distribution:")
    print(f"  Up (1):   {actual_up} ({actual_up_pct:.1f}%)")
    print(f"  Down (0): {actual_down} ({100 - actual_up_pct:.1f}%)")

    print(f"\nPredicted Distribution:")
    print(f"  Up (1):   {pred_up} ({pred_up_pct:.1f}%)")
    print(f"  Down (0): {pred_down} ({100 - pred_up_pct:.1f}%)")

    # Check for bias
    bias = pred_up_pct - actual_up_pct
    print(f"\nBias: {bias:+.1f}%")

    if abs(bias) < 5:
        print("✅ Predictions are well-balanced")
    elif bias > 0:
        print("⚠️  Model is too optimistic (predicting up too often)")
    else:
        print("⚠️  Model is too pessimistic (predicting down too often)")


# ============================================================================
# TESTING AND EXAMPLES
# ============================================================================

def run_example():
    """
    Example usage of backtesting module
    """
    print("\n" + "=" * 70)
    print(" " * 20 + "BACKTESTING MODULE - EXAMPLE")
    print("=" * 70 + "\n")

    # Import necessary modules
    import sys
    sys.path.append('..')
    from src.data_collection import load_stock_data
    from src.preprocessing import preprocess_stock_data
    from src.feature_engineering import engineer_all_features, prepare_ml_data

    # Load and prepare data
    ticker = 'AAPL'
    raw_data = load_stock_data(ticker, 'raw')

    if raw_data is not None:
        # Prepare data
        processed = preprocess_stock_data(raw_data)
        featured = engineer_all_features(processed)
        X, y, features = prepare_ml_data(featured)

        # Run walk-forward validation
        results = walk_forward_validation(X, y, features, n_splits=5)

        # Visualize results
        plot_backtest_results(results)

        # Compare strategies
        comparison = compare_strategies(X, y, features, n_splits=5)

        print("\n📊 Backtesting Complete!")
        print(f"Average Performance: {results['accuracy'].mean():.2%}")


if __name__ == '__main__':
    run_example()