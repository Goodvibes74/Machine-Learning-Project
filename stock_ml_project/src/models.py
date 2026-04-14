"""
Machine Learning Models Module

This module contains functions for training and evaluating ML models.
We use Random Forest Classifier as specified in the project proposal.

Key Concepts:
- Supervised Learning: We have labeled data (features + target)
- Classification: Predicting categories (up/down), not continuous values
- Random Forest: Ensemble of decision trees that vote on the prediction
- Train/Test Split: Use old data to train, recent data to test
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    precision_score,
    recall_score,
    f1_score
)
import matplotlib.pyplot as plt
import seaborn as sns
import config


def split_train_test(X, y, test_size=None):
    """
    Split data into training and testing sets

    CRITICAL FOR TIME SERIES:
    We MUST split chronologically (not randomly)!
    - Train on past data
    - Test on future data
    This simulates real trading where you can only use past data.

    Args:
        X (pd.DataFrame): Features
        y (pd.Series): Target variable
        test_size (float): Fraction of data for testing (default from config)

    Returns:
        tuple: (X_train, X_test, y_train, y_test)

    Example:
        If you have 1000 days and test_size=0.2:
        - Train on first 800 days
        - Test on last 200 days
    """
    if test_size is None:
        test_size = config.TEST_SIZE

    # Calculate split index
    split_idx = int(len(X) * (1 - test_size))

    # Split chronologically (NOT random!)
    X_train = X.iloc[:split_idx]
    X_test = X.iloc[split_idx:]
    y_train = y.iloc[:split_idx]
    y_test = y.iloc[split_idx:]

    if config.VERBOSE:
        print(f"Train set: {len(X_train)} samples")
        print(f"Test set: {len(X_test)} samples")
        print(f"Train period: index {X_train.index[0]} to {X_train.index[-1]}")
        print(f"Test period: index {X_test.index[0]} to {X_test.index[-1]}")

    return X_train, X_test, y_train, y_test


def train_random_forest(X_train, y_train, params=None):
    """
    Train a Random Forest classifier

    Random Forest works by:
    1. Creating many decision trees
    2. Each tree is trained on a random subset of data
    3. Each tree votes on the prediction
    4. Final prediction is the majority vote

    Args:
        X_train (pd.DataFrame): Training features
        y_train (pd.Series): Training target
        params (dict): Model hyperparameters (default from config)

    Returns:
        RandomForestClassifier: Trained model

    Hyperparameters explained:
    - n_estimators: Number of trees (more = better but slower)
    - max_depth: How deep each tree grows (deeper = more complex)
    - min_samples_split: Minimum samples to split a node
    - random_state: Seed for reproducibility
    """
    if params is None:
        params = config.RF_PARAMS

    if config.VERBOSE:
        print("\n" + "=" * 60)
        print("TRAINING RANDOM FOREST MODEL")
        print("=" * 60)
        print(f"Parameters: {params}")

    # Initialize the model
    model = RandomForestClassifier(**params)

    # Train the model (this is where the learning happens!)
    if config.VERBOSE:
        print("\nTraining model...")

    model.fit(X_train, y_train)

    if config.VERBOSE:
        print("✅ Model training complete!")

    return model


def evaluate_model(model, X_train, y_train, X_test, y_test):
    """
    Evaluate model performance on both training and test sets

    We evaluate on BOTH sets to detect overfitting:
    - If train accuracy >> test accuracy: Model is overfitting
    - If both are similar: Model is generalizing well

    Args:
        model: Trained model
        X_train, y_train: Training data
        X_test, y_test: Test data

    Returns:
        dict: Dictionary with performance metrics
    """
    if config.VERBOSE:
        print("\n" + "=" * 60)
        print("MODEL EVALUATION")
        print("=" * 60)

    # Make predictions
    train_pred = model.predict(X_train)
    test_pred = model.predict(X_test)

    # Calculate metrics
    results = {
        'train_accuracy': accuracy_score(y_train, train_pred),
        'test_accuracy': accuracy_score(y_test, test_pred),
        'train_precision': precision_score(y_train, train_pred),
        'test_precision': precision_score(y_test, test_pred),
        'train_recall': recall_score(y_train, train_pred),
        'test_recall': recall_score(y_test, test_pred),
        'train_f1': f1_score(y_train, train_pred),
        'test_f1': f1_score(y_test, test_pred),
    }

    # Print results
    if config.VERBOSE:
        print("\nTRAINING SET PERFORMANCE:")
        print(f"  Accuracy:  {results['train_accuracy']:.4f}")
        print(f"  Precision: {results['train_precision']:.4f}")
        print(f"  Recall:    {results['train_recall']:.4f}")
        print(f"  F1 Score:  {results['train_f1']:.4f}")

        print("\nTEST SET PERFORMANCE:")
        print(f"  Accuracy:  {results['test_accuracy']:.4f}")
        print(f"  Precision: {results['test_precision']:.4f}")
        print(f"  Recall:    {results['test_recall']:.4f}")
        print(f"  F1 Score:  {results['test_f1']:.4f}")

        # Check for overfitting
        accuracy_diff = results['train_accuracy'] - results['test_accuracy']
        if accuracy_diff > 0.1:
            print(f"\n⚠️  Warning: Possible overfitting detected!")
            print(f"   Train accuracy is {accuracy_diff:.2%} higher than test")

        # Classification report
        print("\nDETAILED CLASSIFICATION REPORT (Test Set):")
        print(classification_report(y_test, test_pred,
                                    target_names=['Down (0)', 'Up (1)']))

    return results


def plot_confusion_matrix(y_true, y_pred, title="Confusion Matrix"):
    """
    Plot confusion matrix to visualize prediction results

    Confusion Matrix shows:
    - True Negatives (TN): Correctly predicted down
    - False Positives (FP): Predicted up, actually down
    - False Negatives (FN): Predicted down, actually up
    - True Positives (TP): Correctly predicted up

    Args:
        y_true: Actual labels
        y_pred: Predicted labels
        title: Plot title
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
    plt.show()

    # Print interpretation
    tn, fp, fn, tp = cm.ravel()
    print(f"\nConfusion Matrix Breakdown:")
    print(f"  True Negatives (predicted down, was down):  {tn}")
    print(f"  False Positives (predicted up, was down):   {fp}")
    print(f"  False Negatives (predicted down, was up):   {fn}")
    print(f"  True Positives (predicted up, was up):      {tp}")


def plot_feature_importance(model, feature_names, top_n=10):
    """
    Plot which features are most important for predictions

    Feature importance shows which variables the model relies on most.
    Higher value = more important for making predictions

    Args:
        model: Trained Random Forest model
        feature_names: List of feature names
        top_n: Number of top features to display
    """
    # Get feature importances
    importances = model.feature_importances_

    # Create DataFrame for easy sorting
    feature_imp = pd.DataFrame({
        'feature': feature_names,
        'importance': importances
    }).sort_values('importance', ascending=False)

    # Plot top N features
    plt.figure(figsize=(10, 6))
    top_features = feature_imp.head(top_n)
    plt.barh(range(len(top_features)), top_features['importance'])
    plt.yticks(range(len(top_features)), top_features['feature'])
    plt.xlabel('Importance')
    plt.title(f'Top {top_n} Feature Importances')
    plt.gca().invert_yaxis()  # Highest at top
    plt.tight_layout()
    plt.show()

    # Print numerical values
    if config.VERBOSE:
        print(f"\nTop {top_n} Features:")
        for idx, row in top_features.iterrows():
            print(f"  {row['feature']}: {row['importance']:.4f}")


def compare_baseline(X_train, y_train, X_test, y_test, price_features_only=True):
    """
    Compare hybrid model (price + sentiment) with price-only baseline

    This helps answer: Does adding sentiment features actually help?

    Args:
        X_train, y_train: Training data
        X_test, y_test: Test data
        price_features_only: If True, uses only price-based features for baseline

    Returns:
        dict: Comparison results
    """
    print("\n" + "=" * 60)
    print("BASELINE COMPARISON")
    print("=" * 60)

    # Define baseline features (only price-based, no sentiment)
    baseline_features = [col for col in X_train.columns
                         if 'Sentiment' not in col]

    # Train baseline model (price-only)
    print("\n1. Training BASELINE model (price features only)...")
    baseline_model = train_random_forest(
        X_train[baseline_features],
        y_train,
        params=config.RF_PARAMS
    )
    baseline_pred = baseline_model.predict(X_test[baseline_features])
    baseline_accuracy = accuracy_score(y_test, baseline_pred)

    # Train hybrid model (price + sentiment)
    print("\n2. Training HYBRID model (price + sentiment features)...")
    hybrid_model = train_random_forest(X_train, y_train, params=config.RF_PARAMS)
    hybrid_pred = hybrid_model.predict(X_test)
    hybrid_accuracy = accuracy_score(y_test, hybrid_pred)

    # Compare
    print("\n" + "=" * 60)
    print("RESULTS:")
    print("=" * 60)
    print(f"Baseline (price-only) accuracy: {baseline_accuracy:.4f}")
    print(f"Hybrid (price + sentiment) accuracy: {hybrid_accuracy:.4f}")

    improvement = hybrid_accuracy - baseline_accuracy
    if improvement > 0:
        print(f"\n✅ Hybrid model is better by {improvement:.4f} ({improvement * 100:.2f}%)")
    else:
        print(f"\n❌ Baseline model is better by {abs(improvement):.4f}")

    return {
        'baseline_accuracy': baseline_accuracy,
        'hybrid_accuracy': hybrid_accuracy,
        'improvement': improvement,
        'baseline_model': baseline_model,
        'hybrid_model': hybrid_model
    }


# ============================================================================
# COMPLETE PIPELINE
# ============================================================================

def train_and_evaluate_pipeline(X, y, feature_names):
    """
    Complete training and evaluation pipeline

    This function runs the entire ML workflow:
    1. Split data
    2. Train model
    3. Evaluate performance
    4. Generate visualizations

    Args:
        X: Features
        y: Target
        feature_names: List of feature names

    Returns:
        dict: Results including model and metrics
    """
    print("\n" + "=" * 70)
    print(" " * 20 + "ML TRAINING PIPELINE")
    print("=" * 70)

    # Step 1: Split data
    print("\nStep 1: Splitting data...")
    X_train, X_test, y_train, y_test = split_train_test(X, y)

    # Step 2: Train model
    print("\nStep 2: Training model...")
    model = train_random_forest(X_train, y_train)

    # Step 3: Evaluate
    print("\nStep 3: Evaluating performance...")
    metrics = evaluate_model(model, X_train, y_train, X_test, y_test)

    # Step 4: Visualizations
    print("\nStep 4: Generating visualizations...")

    # Confusion matrix
    test_pred = model.predict(X_test)
    plot_confusion_matrix(y_test, test_pred, "Test Set Confusion Matrix")

    # Feature importance
    plot_feature_importance(model, feature_names)

    print("\n" + "=" * 70)
    print(" " * 20 + "PIPELINE COMPLETE!")
    print("=" * 70)

    return {
        'model': model,
        'metrics': metrics,
        'X_train': X_train,
        'X_test': X_test,
        'y_train': y_train,
        'y_test': y_test,
        'predictions': test_pred
    }


# ============================================================================
# TESTING AND EXAMPLES
# ============================================================================

def run_example():
    """
    Example usage of models module
    """
    print("\n" + "=" * 60)
    print("MODELS MODULE - EXAMPLE")
    print("=" * 60 + "\n")

    # Import necessary modules
    import sys
    sys.path.append('..')
    from src.data_collection import load_stock_data
    from src.preprocessing import preprocess_stock_data
    from src.feature_engineering import engineer_all_features, prepare_ml_data

    # Load data
    ticker = 'AAPL'
    raw_data = load_stock_data(ticker, 'raw')

    if raw_data is not None:
        # Prepare data
        processed = preprocess_stock_data(raw_data)
        featured = engineer_all_features(processed)
        X, y, features = prepare_ml_data(featured)

        # Run complete pipeline
        results = train_and_evaluate_pipeline(X, y, features)

        print("\n📊 Final Model Performance:")
        print(f"Test Accuracy: {results['metrics']['test_accuracy']:.2%}")


if __name__ == '__main__':
    run_example()