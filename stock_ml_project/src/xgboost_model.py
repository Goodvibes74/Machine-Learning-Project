"""
XGBoost Model Module

This module contains functions for training and evaluating XGBoost models
for stock prediction. XGBoost is an alternative to Random Forest that often
provides better performance through gradient boosting.

Key Concepts:
- Gradient Boosting: Sequential ensemble method that corrects errors
- XGBClassifier: Extreme Gradient Boosting classifier
- Supervised Learning: Using labeled data to predict stock direction
"""

import os
import sys

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score
)
import config

# Try to import XGBoost, handle if not installed
try:
    from xgboost import XGBClassifier
except ImportError:
    raise ImportError(
        "XGBoost is not installed. Please install it with:\n"
        "pip install xgboost\n"
        "or: conda install xgboost"
    )


def train_xgboost(X_train, y_train, params=None):
    """
    Train an XGBoost classifier

    XGBoost works by:
    1. Building decision trees sequentially
    2. Each tree corrects errors from previous trees
    3. Final prediction is weighted sum of all trees

    Args:
        X_train (pd.DataFrame or np.array): Training features
        y_train (pd.Series or np.array): Training target
        params (dict): Model hyperparameters (default from config)

    Returns:
        XGBClassifier: Trained model

    Hyperparameters explained:
    - n_estimators: Number of boosting rounds (trees)
    - max_depth: Maximum depth of each tree
    - learning_rate: Step size shrinkage (lower = more trees needed but better generalization)
    - eval_metric: Evaluation metric for training
    - random_state: Seed for reproducibility
    """
    if params is None:
        params = config.XGB_PARAMS

    if config.VERBOSE:
        print("\n" + "=" * 60)
        print("TRAINING XGBOOST MODEL")
        print("=" * 60)
        print(f"Parameters: {params}")

    # Initialize the model
    model = XGBClassifier(**params)

    # Train the model
    if config.VERBOSE:
        print("\nTraining model...")

    model.fit(X_train, y_train)

    if config.VERBOSE:
        print("✅ Model training complete!")

    return model


def evaluate_xgboost_model(model, X_train, y_train, X_test, y_test):
    """
    Evaluate XGBoost model performance on both training and test sets

    We evaluate on BOTH sets to detect overfitting:
    - If train accuracy >> test accuracy: Model is overfitting
    - If both are similar: Model is generalizing well

    Args:
        model: Trained XGBoost model
        X_train: Training features
        y_train: Training target
        X_test: Test features
        y_test: Test target

    Returns:
        dict: Dictionary with performance metrics including:
              accuracy, precision, recall, f1, roc_auc
    """
    if config.VERBOSE:
        print("\n" + "=" * 60)
        print("XGBOOST MODEL EVALUATION")
        print("=" * 60)

    # Make predictions
    train_pred = model.predict(X_train)
    test_pred = model.predict(X_test)

    # Get probability predictions for ROC-AUC
    try:
        train_proba = model.predict_proba(X_train)[:, 1]
        test_proba = model.predict_proba(X_test)[:, 1]
        roc_auc_train = roc_auc_score(y_train, train_proba)
        roc_auc_test = roc_auc_score(y_test, test_proba)
    except Exception:
        # Handle case where probabilities cannot be computed
        roc_auc_train = None
        roc_auc_test = None

    # Calculate metrics
    results = {
        'accuracy': accuracy_score(y_test, test_pred),
        'precision': precision_score(y_test, test_pred, zero_division=0),
        'recall': recall_score(y_test, test_pred, zero_division=0),
        'f1': f1_score(y_test, test_pred, zero_division=0),
        'roc_auc': roc_auc_test,
        'train_accuracy': accuracy_score(y_train, train_pred),
        'train_precision': precision_score(y_train, train_pred, zero_division=0),
        'train_recall': recall_score(y_train, train_pred, zero_division=0),
        'train_f1': f1_score(y_train, train_pred, zero_division=0),
        'train_roc_auc': roc_auc_train,
    }

    # Print classification report
    print("\n" + "=" * 60)
    print("CLASSIFICATION REPORT (Test Set)")
    print("=" * 60)
    print(classification_report(y_test, test_pred, target_names=['Down (0)', 'Up (1)']))

    # Print results summary
    if config.VERBOSE:
        print("\nTRAINING SET PERFORMANCE:")
        print(f"  Accuracy:  {results['train_accuracy']:.4f}")
        print(f"  Precision: {results['train_precision']:.4f}")
        print(f"  Recall:    {results['train_recall']:.4f}")
        print(f"  F1 Score:  {results['train_f1']:.4f}")
        if results['train_roc_auc'] is not None:
            print(f"  ROC-AUC:   {results['train_roc_auc']:.4f}")

        print("\nTEST SET PERFORMANCE:")
        print(f"  Accuracy:  {results['accuracy']:.4f}")
        print(f"  Precision: {results['precision']:.4f}")
        print(f"  Recall:    {results['recall']:.4f}")
        print(f"  F1 Score:  {results['f1']:.4f}")
        if results['roc_auc'] is not None:
            print(f"  ROC-AUC:   {results['roc_auc']:.4f}")

        # Check for overfitting
        accuracy_diff = results['train_accuracy'] - results['accuracy']
        if accuracy_diff > 0.1:
            print(f"\n⚠️  Warning: Possible overfitting detected!")
            print(f"   Train accuracy is {accuracy_diff:.2%} higher than test")

    return results, test_pred


def plot_xgb_confusion_matrix(y_true, y_pred, title='XGBoost Confusion Matrix'):
    """
    Plot confusion matrix to visualize XGBoost prediction results

    Confusion Matrix shows:
    - True Negatives (TN): Correctly predicted down
    - False Positives (FP): Predicted up, actually down
    - False Negatives (FN): Predicted down, actually up
    - True Positives (TP): Correctly predicted up

    Args:
        y_true: Actual labels
        y_pred: Predicted labels
        title: Plot title (default: 'XGBoost Confusion Matrix')
    """
    cm = confusion_matrix(y_true, y_pred)

    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Down (0)', 'Up (1)'],
                yticklabels=['Down (0)', 'Up (1)'])
    plt.title(title)
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.tight_layout()

    # Save plot if outputs directory exists
    output_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'outputs')
    if os.path.exists(output_dir):
        plt.savefig(os.path.join(output_dir, 'xgboost_confusion_matrix.png'), dpi=150)
        print(f"✅ Confusion matrix saved to {output_dir}/xgboost_confusion_matrix.png")

    plt.show()

    # Print interpretation
    tn, fp, fn, tp = cm.ravel()
    print(f"\nConfusion Matrix Breakdown:")
    print(f"  True Negatives (predicted down, was down):  {tn}")
    print(f"  False Positives (predicted up, was down):   {fp}")
    print(f"  False Negatives (predicted down, was up):   {fn}")
    print(f"  True Positives (predicted up, was up):      {tp}")


def plot_xgb_feature_importance(model, feature_names, top_n=15):
    """
    Plot which features are most important for XGBoost predictions

    Feature importance shows which variables the model relies on most.
    Higher value = more important for making predictions

    Args:
        model: Trained XGBoost model
        feature_names: List of feature names
        top_n: Number of top features to display (default: 15)
    """
    # Get feature importances
    importances = model.feature_importances_

    # Create DataFrame for easy sorting
    feature_imp = pd.DataFrame({
        'feature': feature_names,
        'importance': importances
    }).sort_values('importance', ascending=False)

    # Plot top N features
    plt.figure(figsize=(10, 8))
    top_features = feature_imp.head(top_n)
    plt.barh(range(len(top_features)), top_features['importance'], color='steelblue')
    plt.yticks(range(len(top_features)), top_features['feature'])
    plt.xlabel('Importance')
    plt.title(f'XGBoost Top {top_n} Feature Importances')
    plt.gca().invert_yaxis()  # Highest at top
    plt.tight_layout()

    # Save plot if outputs directory exists
    output_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'outputs')
    if os.path.exists(output_dir):
        plt.savefig(os.path.join(output_dir, 'xgboost_feature_importance.png'), dpi=150)
        print(f"✅ Feature importance plot saved to {output_dir}/xgboost_feature_importance.png")

    plt.show()

    # Print numerical values
    if config.VERBOSE:
        print(f"\nTop {top_n} Features:")
        for idx, row in top_features.iterrows():
            print(f"  {row['feature']}: {row['importance']:.4f}")


def run_example():
    """
    Example usage of XGBoost model module

    This function:
    1. Loads AAPL stock data
    2. Processes and engineers features
    3. Prepares ML data (80/20 chronological split)
    4. Trains XGBoost model
    5. Evaluates and visualizes results
    6. Saves plots to outputs/ folder
    """
    print("\n" + "=" * 60)
    print("XGBOOST MODEL - EXAMPLE")
    print("=" * 60 + "\n")

    # Import necessary modules
    from src.data_collection import load_stock_data
    from src.preprocessing import preprocess_stock_data
    from src.feature_engineering import engineer_all_features, prepare_ml_data

    # Create outputs directory if it doesn't exist
    output_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'outputs')
    os.makedirs(output_dir, exist_ok=True)
    print(f"Outputs directory: {output_dir}")

    # Load data
    ticker = 'AAPL'
    print(f"\nLoading {ticker} stock data...")
    raw_data = load_stock_data(ticker, 'raw')

    if raw_data is None:
        print("❌ Failed to load data. Please run data collection first.")
        return

    # Prepare data through the pipeline
    print("Preprocessing data...")
    processed = preprocess_stock_data(raw_data)

    print("Engineering features...")
    featured = engineer_all_features(processed)

    print("Preparing ML data (80/20 chronological split)...")
    X, y, features = prepare_ml_data(featured)

    if X is None or len(X) == 0:
        print("❌ Failed to prepare ML data.")
        return

    print(f"\nData shape: X={X.shape}, y={y.shape}")
    print(f"Number of features: {len(features)}")

    # Train XGBoost model
    print("\n" + "=" * 60)
    print("TRAINING XGBOOST MODEL")
    print("=" * 60)
    model = train_xgboost(X, y)

    # Evaluate model
    print("\n" + "=" * 60)
    print("EVALUATING MODEL")
    print("=" * 60)
    results, y_pred = evaluate_xgboost_model(model, X, y, X, y)

    # Plot confusion matrix
    print("\nGenerating confusion matrix...")
    plot_xgb_confusion_matrix(y, y_pred, title='XGBoost Confusion Matrix (Full Data)')

    # Plot feature importance
    print("\nGenerating feature importance plot...")
    plot_xgb_feature_importance(model, features)

    # Print final summary
    print("\n" + "=" * 60)
    print("FINAL RESULTS SUMMARY")
    print("=" * 60)
    print(f"Accuracy:  {results['accuracy']:.4f}")
    print(f"Precision: {results['precision']:.4f}")
    print(f"Recall:    {results['recall']:.4f}")
    print(f"F1 Score:  {results['f1']:.4f}")
    if results['roc_auc'] is not None:
        print(f"ROC-AUC:   {results['roc_auc']:.4f}")

    print("\n✅ XGBoost model training and evaluation complete!")
    print(f"📁 Plots saved to: {output_dir}")

    return {
        'model': model,
        'results': results,
        'predictions': y_pred
    }


if __name__ == '__main__':
    run_example()