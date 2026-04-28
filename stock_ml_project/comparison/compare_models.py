"""
Model Comparison Module

This module compares Random Forest and XGBoost models for stock prediction.
It provides functions to evaluate, visualize, and compare both models.

Key Concepts:
- Model Comparison: Evaluating multiple models on the same data
- ROC Curve: Receiver Operating Characteristic curve
- Feature Importance: Which features matter most for each model
- Performance Metrics: Accuracy, Precision, Recall, F1, ROC-AUC
"""

import os
import sys
import time

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    roc_curve
)

# Import from existing modules
from src.models import train_random_forest, evaluate_model, train_svm, evaluate_svm
from src.xgboost_model import train_xgboost, evaluate_xgboost_model
import config


def get_metrics(model, X_test, y_test, model_name):
    """
    Get performance metrics for a trained model

    Args:
        model: Trained model
        X_test: Test features
        y_test: Test labels
        model_name: Name of the model for reporting

    Returns:
        dict: Dictionary containing model name and metrics
              {model, accuracy, precision, recall, f1, roc_auc, training_time}
    """
    # Make predictions
    y_pred = model.predict(X_test)

    # Get probability predictions for ROC-AUC
    try:
        y_proba = model.predict_proba(X_test)[:, 1]
        roc_auc = roc_auc_score(y_test, y_proba)
    except Exception:
        roc_auc = None

    # Calculate metrics
    metrics = {
        'model': model_name,
        'accuracy': accuracy_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred, zero_division=0),
        'recall': recall_score(y_test, y_pred, zero_division=0),
        'f1': f1_score(y_test, y_pred, zero_division=0),
        'roc_auc': roc_auc,
    }

    return metrics


def get_svm_metrics(model, scaler, X_test, y_test, model_name):
    """
    Get performance metrics for a trained SVM model.

    Respects the calibrated threshold stored on the model during training
    so predictions are consistent with the two-sided behaviour seen at
    training time (not all-one-class from the raw 0.5 cut-off).

    Args:
        model: Trained SVM model (with ._threshold attribute)
        scaler: Fitted StandardScaler
        X_test: Test features
        y_test: Test labels
        model_name: Name of the model for reporting

    Returns:
        dict: Dictionary containing model name and metrics
    """
    X_test_scaled = scaler.transform(X_test)

    # Use calibrated threshold if set during training
    thresh = getattr(model, '_threshold', 0.5)
    try:
        y_proba = model.predict_proba(X_test_scaled)[:, 1]
        y_pred = (y_proba >= thresh).astype(int)
        roc_auc = roc_auc_score(y_test, y_proba)
    except Exception:
        y_pred = model.predict(X_test_scaled)
        y_proba = None
        roc_auc = None

    metrics = {
        'model': model_name,
        'accuracy': accuracy_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred, zero_division=0),
        'recall': recall_score(y_test, y_pred, zero_division=0),
        'f1': f1_score(y_test, y_pred, zero_division=0),
        'roc_auc': roc_auc,
    }

    return metrics


def build_comparison_table(rf_metrics, xgb_metrics, svm_metrics=None):
    """
    Build a formatted comparison table for all models

    Args:
        rf_metrics: Metrics dict for Random Forest
        xgb_metrics: Metrics dict for XGBoost
        svm_metrics: Metrics dict for SVM (optional)

    Returns:
        pd.DataFrame: Formatted comparison table
    """
    # Create list of metrics
    metrics_list = [rf_metrics, xgb_metrics]
    if svm_metrics is not None:
        metrics_list.append(svm_metrics)

    # Create DataFrame
    df = pd.DataFrame(metrics_list)

    # Rename columns for display
    df = df.rename(columns={
        'model': 'Model',
        'accuracy': 'Accuracy',
        'precision': 'Precision',
        'recall': 'Recall',
        'f1': 'F1',
        'roc_auc': 'ROC-AUC',
        'training_time': 'Training Time(s)'
    })

    # Set Model as index
    df = df.set_index('Model')

    # Print table with highlighting
    print("\n" + "=" * 70)
    print(" " * 20 + "MODEL COMPARISON")
    print("=" * 70)

    # Print each row
    for idx, row in df.iterrows():
        print(f"\n{idx}:")
        print(f"  Accuracy:      {row['Accuracy']:.4f}")
        print(f"  Precision:     {row['Precision']:.4f}")
        print(f"  Recall:        {row['Recall']:.4f}")
        print(f"  F1 Score:      {row['F1']:.4f}")
        print(f"  ROC-AUC:       {row['ROC-AUC']:.4f}" if pd.notna(row['ROC-AUC']) else f"  ROC-AUC:       N/A")
        print(f"  Training Time: {row['Training Time(s)']:.4f}s")

    # Determine winners for each metric
    print("\n" + "-" * 70)
    print("WINNERS BY METRIC:")
    print("-" * 70)

    metric_cols = ['Accuracy', 'Precision', 'Recall', 'F1', 'ROC-AUC']
    for col in metric_cols:
        rf_val = rf_metrics.get(col.lower().replace('-', '_'), rf_metrics.get(col, 0))
        xgb_val = xgb_metrics.get(col.lower().replace('-', '_'), xgb_metrics.get(col, 0))
        
        # Get SVM value if available
        svm_val = None
        if svm_metrics is not None:
            svm_val = svm_metrics.get(col.lower().replace('-', '_'), svm_metrics.get(col, 0))

        # Find the best model for this metric
        values = {'Random Forest': rf_val, 'XGBoost': xgb_val}
        if svm_val is not None:
            values['SVM'] = svm_val
        
        best_model = max(values, key=lambda k: values[k] if values[k] is not None else 0)
        best_val = values[best_model]
        
        # Check for ties
        tied = [k for k, v in values.items() if v is not None and abs(v - best_val) < 0.0001]
        
        if len(tied) > 1:
            print(f"  {col}: {' & '.join(tied)} (tie)")
        else:
            print(f"  {col}: {best_model} [BEST]")

    return df


def plot_roc_curves(rf_model, xgb_model, X_test, y_test, svm_model=None, svm_scaler=None):
    """
    Plot ROC curves for all models on the same figure

    Args:
        rf_model: Trained Random Forest model
        xgb_model: Trained XGBoost model
        X_test: Test features
        y_test: Test labels
        svm_model: Trained SVM model (optional)
        svm_scaler: Fitted StandardScaler for SVM (optional)
    """
    plt.figure(figsize=(10, 8))

    # Get ROC curve for Random Forest
    rf_proba = rf_model.predict_proba(X_test)[:, 1]
    rf_fpr, rf_tpr, _ = roc_curve(y_test, rf_proba)
    rf_auc = roc_auc_score(y_test, rf_proba)

    # Get ROC curve for XGBoost
    xgb_proba = xgb_model.predict_proba(X_test)[:, 1]
    xgb_fpr, xgb_tpr, _ = roc_curve(y_test, xgb_proba)
    xgb_auc = roc_auc_score(y_test, xgb_proba)

    # Plot both curves
    plt.plot(rf_fpr, rf_tpr, color='blue', lw=2,
             label=f'Random Forest (AUC = {rf_auc:.4f})')
    plt.plot(xgb_fpr, xgb_tpr, color='orange', lw=2,
             label=f'XGBoost (AUC = {xgb_auc:.4f})')

    # Plot SVM if available
    if svm_model is not None and svm_scaler is not None:
        X_test_scaled = svm_scaler.transform(X_test)
        svm_proba = svm_model.predict_proba(X_test_scaled)[:, 1]
        svm_fpr, svm_tpr, _ = roc_curve(y_test, svm_proba)
        svm_auc = roc_auc_score(y_test, svm_proba)
        plt.plot(svm_fpr, svm_tpr, color='green', lw=2,
                 label=f'SVM (AUC = {svm_auc:.4f})')

    # Plot diagonal (random classifier)
    plt.plot([0, 1], [0, 1], color='gray', lw=1, linestyle='--',
             label='Random Classifier')

    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate', fontsize=12)
    plt.ylabel('True Positive Rate', fontsize=12)
    plt.title('ROC Curve Comparison', fontsize=14)
    plt.legend(loc='lower right', fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    # Save plot
    output_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'outputs')
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(os.path.join(output_dir, 'roc_comparison.png'), dpi=150)
    print(f"[OK] ROC curve saved to {output_dir}/roc_comparison.png")

    plt.show()


def plot_metrics_bar_chart(rf_metrics, xgb_metrics, svm_metrics=None):
    """
    Plot grouped bar chart comparing performance metrics

    Args:
        rf_metrics: Metrics dict for Random Forest
        xgb_metrics: Metrics dict for XGBoost
        svm_metrics: Metrics dict for SVM (optional)
    """
    # Define metrics to compare
    metrics = ['accuracy', 'precision', 'recall', 'f1', 'roc_auc']
    metric_labels = ['Accuracy', 'Precision', 'Recall', 'F1', 'ROC-AUC']

    # Extract values
    rf_values = [rf_metrics.get(m, 0) for m in metrics]
    xgb_values = [xgb_metrics.get(m, 0) for m in metrics]
    svm_values = [svm_metrics.get(m, 0) for m in metrics] if svm_metrics else None

    # Handle None values
    rf_values = [v if v is not None else 0 for v in rf_values]
    xgb_values = [v if v is not None else 0 for v in xgb_values]
    if svm_values:
        svm_values = [v if v is not None else 0 for v in svm_values]

    # Create figure
    if svm_values:
        plt.figure(figsize=(14, 6))
        x = np.arange(len(metrics))
        width = 0.25
        
        bars1 = plt.bar(x - width, rf_values, width, label='Random Forest',
                        color='steelblue', edgecolor='black')
        bars2 = plt.bar(x, xgb_values, width, label='XGBoost',
                        color='darkorange', edgecolor='black')
        bars3 = plt.bar(x + width, svm_values, width, label='SVM',
                        color='green', edgecolor='black')
    else:
        plt.figure(figsize=(12, 6))
        x = np.arange(len(metrics))
        width = 0.35

        bars1 = plt.bar(x - width/2, rf_values, width, label='Random Forest',
                        color='steelblue', edgecolor='black')
        bars2 = plt.bar(x + width/2, xgb_values, width, label='XGBoost',
                        color='darkorange', edgecolor='black')

    # Customize plot
    plt.xlabel('Metrics', fontsize=12)
    plt.ylabel('Score', fontsize=12)
    plt.title('Model Performance Comparison', fontsize=14)
    plt.xticks(x, metric_labels, fontsize=11)
    plt.ylim(0, 1.1)
    plt.legend(fontsize=11)
    plt.grid(axis='y', alpha=0.3)

    # Add value labels on bars
    for bar in bars1:
        height = bar.get_height()
        plt.annotate(f'{height:.3f}',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3),
                    textcoords="offset points",
                    ha='center', va='bottom', fontsize=9)

    for bar in bars2:
        height = bar.get_height()
        plt.annotate(f'{height:.3f}',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3),
                    textcoords="offset points",
                    ha='center', va='bottom', fontsize=9)
    
    if svm_values:
        for bar in bars3:
            height = bar.get_height()
            plt.annotate(f'{height:.3f}',
                        xy=(bar.get_x() + bar.get_width() / 2, height),
                        xytext=(0, 3),
                        textcoords="offset points",
                        ha='center', va='bottom', fontsize=9)

    plt.tight_layout()

    # Save plot
    output_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'outputs')
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(os.path.join(output_dir, 'metrics_comparison.png'), dpi=150)
    print(f"[OK] Metrics comparison saved to {output_dir}/metrics_comparison.png")

    plt.show()


def plot_feature_importance_comparison(rf_model, xgb_model, feature_names):
    """
    Plot side-by-side feature importance comparison

    Args:
        rf_model: Trained Random Forest model
        xgb_model: Trained XGBoost model
        feature_names: List of feature names
    """
    # Get feature importances
    rf_importances = rf_model.feature_importances_
    xgb_importances = xgb_model.feature_importances_

    # Create DataFrames
    rf_df = pd.DataFrame({
        'feature': feature_names,
        'importance': rf_importances
    }).sort_values('importance', ascending=False).head(10)

    xgb_df = pd.DataFrame({
        'feature': feature_names,
        'importance': xgb_importances
    }).sort_values('importance', ascending=False).head(10)

    # Create figure with two subplots
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Random Forest subplot
    axes[0].barh(range(len(rf_df)), rf_df['importance'], color='steelblue')
    axes[0].set_yticks(range(len(rf_df)))
    axes[0].set_yticklabels(rf_df['feature'])
    axes[0].set_xlabel('Importance')
    axes[0].set_title('Random Forest - Top 10 Features')
    axes[0].invert_yaxis()

    # XGBoost subplot
    axes[1].barh(range(len(xgb_df)), xgb_df['importance'], color='darkorange')
    axes[1].set_yticks(range(len(xgb_df)))
    axes[1].set_yticklabels(xgb_df['feature'])
    axes[1].set_xlabel('Importance')
    axes[1].set_title('XGBoost - Top 10 Features')
    axes[1].invert_yaxis()

    plt.suptitle('Feature Importance Comparison', fontsize=14, y=1.02)
    plt.tight_layout()

    # Save plot
    output_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'outputs')
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(os.path.join(output_dir, 'feature_importance_comparison.png'), dpi=150)
    print(f"[OK] Feature importance comparison saved to {output_dir}/feature_importance_comparison.png")

    plt.show()


def run_full_comparison(ticker='AAPL'):
    """
    Run complete comparison pipeline between Random Forest and XGBoost

    This function:
    1. Loads stock data
    2. Preprocesses and engineers features
    3. Prepares ML data (same split for both models)
    4. Trains both models with timing
    5. Generates all comparison plots
    6. Prints comparison table
    7. Saves results to CSV

    Args:
        ticker: Stock ticker symbol (default: 'AAPL')

    Returns:
        dict: Results containing both models and their metrics
    """
    print("\n" + "=" * 70)
    print(" " * 15 + "MODEL COMPARISON PIPELINE")
    print("=" * 70)
    print(f"\nStock: {ticker}")

    # Import data processing modules
    from src.data_collection import load_stock_data
    from src.preprocessing import preprocess_stock_data
    from src.feature_engineering import engineer_all_features, prepare_ml_data

    # Load data
    print("\n1. Loading stock data...")
    raw_data = load_stock_data(ticker, 'raw')

    # IMPROVEMENT: If no cached data, download with new date range
    if raw_data is None:
        print("No cached data found. Downloading with new date range...")
        from src.data_collection import download_stock_data
        raw_data = download_stock_data(ticker, config.START_DATE, config.END_DATE, save=True)
        
        if raw_data is None:
            print("❌ Failed to download data.")
            return None

    # Process data
    print("2. Preprocessing data...")
    processed = preprocess_stock_data(raw_data)

    print("3. Engineering features...")
    featured = engineer_all_features(processed)

    print("4. Preparing ML data (80/20 chronological split)...")
    X, y, features = prepare_ml_data(featured)

    if X is None or len(X) == 0:
        print("❌ Failed to prepare ML data.")
        return None

    print(f"\nData shape: X={X.shape}, y={y.shape}")
    print(f"Number of features: {len(features)}")

    # LEAKAGE FIX: Add chronological train/test split
    # Use 80% for training, 20% for testing (chronological, not random)
    split_idx = int(len(X) * 0.8)
    X_train = X.iloc[:split_idx]
    X_test = X.iloc[split_idx:]
    y_train = y.iloc[:split_idx]
    y_test = y.iloc[split_idx:]
    
    print(f"\nTrain set: {len(X_train)} samples (index 0 to {split_idx-1})")
    print(f"Test set: {len(X_test)} samples (index {split_idx} to {len(X)-1})")

    # Train Random Forest with timing
    print("\n" + "=" * 70)
    print("5. TRAINING RANDOM FOREST")
    print("=" * 70)
    start_time = time.time()
    rf_model = train_random_forest(X_train, y_train)
    rf_training_time = time.time() - start_time
    print(f"Training time: {rf_training_time:.4f}s")

    # Train XGBoost with timing
    print("\n" + "=" * 70)
    print("6. TRAINING XGBOOST")
    print("=" * 70)
    start_time = time.time()
    xgb_model = train_xgboost(X_train, y_train)
    xgb_training_time = time.time() - start_time
    print(f"Training time: {xgb_training_time:.4f}s")

    # Train SVM with timing
    print("\n" + "=" * 70)
    print("7. TRAINING SVM")
    print("=" * 70)
    start_time = time.time()
    svm_model, svm_scaler = train_svm(X_train, y_train)
    svm_training_time = time.time() - start_time
    print(f"Training time: {svm_training_time:.4f}s")

    # Get metrics for all models (use test set for evaluation)
    print("\n" + "=" * 70)
    print("8. EVALUATING MODELS")
    print("=" * 70)

    rf_metrics = get_metrics(rf_model, X_test, y_test, 'Random Forest')
    rf_metrics['training_time'] = rf_training_time

    xgb_metrics = get_metrics(xgb_model, X_test, y_test, 'XGBoost')
    xgb_metrics['training_time'] = xgb_training_time

    svm_metrics = get_svm_metrics(svm_model, svm_scaler, X_test, y_test, 'SVM')
    svm_metrics['training_time'] = svm_training_time

    # Build and display comparison table
    comparison_df = build_comparison_table(rf_metrics, xgb_metrics, svm_metrics)

    # Data summary — reuse `featured` already in memory (no reload needed)
    print("\n" + "=" * 70)
    print(" " * 20 + "DATA SUMMARY")
    print("=" * 70)

    print(f"\nNumber of features: {len(features)}")
    print(f"Feature names: {features}")

    featured_clean = featured.dropna()
    if 'Date' in featured_clean.columns:
        dates = featured_clean['Date'].reset_index(drop=True)
        train_dates = dates.iloc[:split_idx]
        test_dates  = dates.iloc[split_idx:]
        print(f"\nTraining data: {train_dates.min().strftime('%Y-%m-%d')} to {train_dates.max().strftime('%Y-%m-%d')}")
        print(f"Test data:     {test_dates.min().strftime('%Y-%m-%d')} to {test_dates.max().strftime('%Y-%m-%d')}")
    
    # Class balance in test set
    up_pct = (y_test.sum() / len(y_test) * 100)
    down_pct = 100 - up_pct
    print(f"\nClass balance in test set:")
    print(f"  Up days:   {up_pct:.1f}%")
    print(f"  Down days: {down_pct:.1f}%")

    # Generate plots
    print("\n" + "=" * 70)
    print("9. GENERATING VISUALIZATIONS")
    print("=" * 70)

    print("\nPlotting ROC curves...")
    plot_roc_curves(rf_model, xgb_model, X_test, y_test, svm_model, svm_scaler)

    print("\nPlotting metrics comparison...")
    plot_metrics_bar_chart(rf_metrics, xgb_metrics, svm_metrics)

    print("\nPlotting feature importance comparison...")
    plot_feature_importance_comparison(rf_model, xgb_model, features)

    # Save results to CSV
    output_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'outputs')
    os.makedirs(output_dir, exist_ok=True)

    # Create results DataFrame
    results_df = pd.DataFrame([rf_metrics, xgb_metrics, svm_metrics])
    results_df = results_df.rename(columns={
        'model': 'Model',
        'accuracy': 'Accuracy',
        'precision': 'Precision',
        'recall': 'Recall',
        'f1': 'F1',
        'roc_auc': 'ROC_AUC',
        'training_time': 'Training_Time'
    })

    # Save to CSV
    csv_path = os.path.join(output_dir, 'comparison_results.csv')
    results_df.to_csv(csv_path, index=False)
    print(f"\n[OK] Results saved to {csv_path}")

    print("\n" + "=" * 70)
    print(" " * 20 + "COMPARISON COMPLETE!")
    print("=" * 70)

    # Sanity check: very high AUC on held-out test data may indicate leakage
    rf_auc = rf_metrics.get('roc_auc', 0)
    xgb_auc = xgb_metrics.get('roc_auc', 0)
    max_auc = max(rf_auc if rf_auc is not None else 0, xgb_auc if xgb_auc is not None else 0)

    if max_auc > 0.85:
        print("\n⚠️  WARNING: ROC-AUC > 0.85 on test set — review feature_engineering.py for look-ahead bias")
        print(f"   Max ROC-AUC: {max_auc:.4f}")

    return {
        'rf_model': rf_model,
        'xgb_model': xgb_model,
        'svm_model': svm_model,
        'svm_scaler': svm_scaler,
        'rf_metrics': rf_metrics,
        'xgb_metrics': xgb_metrics,
        'svm_metrics': svm_metrics,
        'comparison_table': comparison_df
    }


if __name__ == '__main__':
    run_full_comparison('AAPL')