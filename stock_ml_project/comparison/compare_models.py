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
from src.models import train_random_forest, evaluate_model
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


def build_comparison_table(rf_metrics, xgb_metrics):
    """
    Build a formatted comparison table for both models

    Args:
        rf_metrics: Metrics dict for Random Forest
        xgb_metrics: Metrics dict for XGBoost

    Returns:
        pd.DataFrame: Formatted comparison table
    """
    # Create DataFrame
    df = pd.DataFrame([rf_metrics, xgb_metrics])

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

        if rf_val is not None and xgb_val is not None:
            if rf_val > xgb_val:
                print(f"  {col}: Random Forest ⬆️")
            elif xgb_val > rf_val:
                print(f"  {col}: XGBoost ⬆️")
            else:
                print(f"  {col}: Tie")
        else:
            print(f"  {col}: N/A")

    return df


def plot_roc_curves(rf_model, xgb_model, X_test, y_test):
    """
    Plot ROC curves for both models on the same figure

    Args:
        rf_model: Trained Random Forest model
        xgb_model: Trained XGBoost model
        X_test: Test features
        y_test: Test labels
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

    # Plot diagonal (random classifier)
    plt.plot([0, 1], [0, 1], color='gray', lw=1, linestyle='--',
             label='Random Classifier')

    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate', fontsize=12)
    plt.ylabel('True Positive Rate', fontsize=12)
    plt.title('ROC Curve Comparison: Random Forest vs XGBoost', fontsize=14)
    plt.legend(loc='lower right', fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    # Save plot
    output_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'outputs')
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(os.path.join(output_dir, 'roc_comparison.png'), dpi=150)
    print(f"✅ ROC curve saved to {output_dir}/roc_comparison.png")

    plt.show()


def plot_metrics_bar_chart(rf_metrics, xgb_metrics):
    """
    Plot grouped bar chart comparing performance metrics

    Args:
        rf_metrics: Metrics dict for Random Forest
        xgb_metrics: Metrics dict for XGBoost
    """
    # Define metrics to compare
    metrics = ['accuracy', 'precision', 'recall', 'f1', 'roc_auc']
    metric_labels = ['Accuracy', 'Precision', 'Recall', 'F1', 'ROC-AUC']

    # Extract values
    rf_values = [rf_metrics.get(m, 0) for m in metrics]
    xgb_values = [xgb_metrics.get(m, 0) for m in metrics]

    # Handle None values
    rf_values = [v if v is not None else 0 for v in rf_values]
    xgb_values = [v if v is not None else 0 for v in xgb_values]

    # Create figure
    plt.figure(figsize=(12, 6))

    # Set up bar positions
    x = np.arange(len(metrics))
    width = 0.35

    # Create bars
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

    plt.tight_layout()

    # Save plot
    output_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'outputs')
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(os.path.join(output_dir, 'metrics_comparison.png'), dpi=150)
    print(f"✅ Metrics comparison saved to {output_dir}/metrics_comparison.png")

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
    print(f"✅ Feature importance comparison saved to {output_dir}/feature_importance_comparison.png")

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

    if raw_data is None:
        print("❌ Failed to load data. Please run data collection first.")
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

    # Get metrics for both models (use test set for evaluation)
    print("\n" + "=" * 70)
    print("7. EVALUATING MODELS")
    print("=" * 70)

    rf_metrics = get_metrics(rf_model, X_test, y_test, 'Random Forest')
    rf_metrics['training_time'] = rf_training_time

    xgb_metrics = get_metrics(xgb_model, X_test, y_test, 'XGBoost')
    xgb_metrics['training_time'] = xgb_training_time

    # Build and display comparison table
    comparison_df = build_comparison_table(rf_metrics, xgb_metrics)

    # Generate plots
    print("\n" + "=" * 70)
    print("8. GENERATING VISUALIZATIONS")
    print("=" * 70)

    print("\nPlotting ROC curves...")
    plot_roc_curves(rf_model, xgb_model, X_test, y_test)

    print("\nPlotting metrics comparison...")
    plot_metrics_bar_chart(rf_metrics, xgb_metrics)

    print("\nPlotting feature importance comparison...")
    plot_feature_importance_comparison(rf_model, xgb_model, features)

    # Save results to CSV
    output_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'outputs')
    os.makedirs(output_dir, exist_ok=True)

    # Create results DataFrame
    results_df = pd.DataFrame([rf_metrics, xgb_metrics])
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
    print(f"\n✅ Results saved to {csv_path}")

    print("\n" + "=" * 70)
    print(" " * 20 + "COMPARISON COMPLETE!")
    print("=" * 70)

    # LEAKAGE FIX: Check for remaining leakage indicators
    rf_auc = rf_metrics.get('roc_auc', 0)
    xgb_auc = xgb_metrics.get('roc_auc', 0)
    max_auc = max(rf_auc if rf_auc is not None else 0, xgb_auc if xgb_auc is not None else 0)
    
    if max_auc > 0.85:
        print("\n⚠️  WARNING: Leakage may still exist — review feature_engineering.py manually")
        print(f"   Max ROC-AUC detected: {max_auc:.4f}")

    return {
        'rf_model': rf_model,
        'xgb_model': xgb_model,
        'rf_metrics': rf_metrics,
        'xgb_metrics': xgb_metrics,
        'comparison_table': comparison_df
    }


if __name__ == '__main__':
    run_full_comparison('AAPL')