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
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
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


class _ThresholdRF:
    """
    RandomForest wrapper that uses a calibrated decision threshold.
    Exposes the same interface (predict, predict_proba, feature_importances_)
    so downstream code needs no changes.
    """
    def __init__(self, model, threshold):
        self._model = model
        self.threshold = threshold
        self.feature_importances_ = model.feature_importances_
        self.classes_ = model.classes_

    def predict(self, X):
        proba = self._model.predict_proba(X)[:, 1]
        return (proba >= self.threshold).astype(int)

    def predict_proba(self, X):
        return self._model.predict_proba(X)


def train_random_forest(X_train, y_train, params=None, calibrate=True):
    """
    Train a Random Forest classifier with threshold calibration.

    RF with class_weight='balanced' produces systematically compressed
    probabilities — the raw 0.5 threshold is never crossed even when the
    model has genuine signal (at 0.40 threshold accuracy can be ~58%).
    We fix this by using the final 20% of training data (chronologically)
    to find the threshold that maximises accuracy, then bake it into the
    returned model. No test data is used so there is no data leakage.

    Args:
        X_train (pd.DataFrame): Training features
        y_train (pd.Series): Training target
        params (dict): Model hyperparameters (default from config)
        calibrate (bool): Whether to calibrate the decision threshold

    Returns:
        _ThresholdRF or RandomForestClassifier: Trained model
    """
    if params is None:
        params = config.RF_PARAMS

    if config.VERBOSE:
        print("\n" + "=" * 60)
        print("TRAINING RANDOM FOREST MODEL")
        print("=" * 60)
        print(f"Parameters: {params}")

    model = RandomForestClassifier(**params)

    if calibrate and len(X_train) >= 120:
        # Time-ordered split: train on first 80%, calibrate threshold on last 20%
        cal_split = int(len(X_train) * 0.8)
        X_fit = X_train.iloc[:cal_split]
        X_cal = X_train.iloc[cal_split:]
        y_fit = y_train.iloc[:cal_split]
        y_cal = y_train.iloc[cal_split:]

        model.fit(X_fit, y_fit)

        # Search for threshold that maximises accuracy on calibration set
        proba_cal = model.predict_proba(X_cal)[:, 1]
        best_thresh, best_acc = 0.5, 0.0
        for thresh in np.arange(0.30, 0.71, 0.01):
            pred = (proba_cal >= thresh).astype(int)
            acc = accuracy_score(y_cal, pred)
            if acc > best_acc:
                best_acc = acc
                best_thresh = float(thresh)

        if config.VERBOSE:
            print(f"✅ Calibrated threshold: {best_thresh:.2f} (val acc={best_acc:.4f})")

        return _ThresholdRF(model, best_thresh)

    model.fit(X_train, y_train)

    if config.VERBOSE:
        print("✅ Model training complete!")

    return model


def train_svm(X_train, y_train, params=None, calibrate=True):
    """
    Train a Support Vector Machine (SVM) classifier with threshold calibration.

    SVM requires StandardScaler (fit on training data only).
    Threshold calibration uses the last 20% of training data to find the
    probability cut-off that maximises accuracy without touching test data.

    Args:
        X_train (pd.DataFrame): Training features
        y_train (pd.Series): Training target
        params (dict): Model hyperparameters (default from config)
        calibrate (bool): Whether to calibrate the decision threshold

    Returns:
        tuple: (model, StandardScaler)
            model has a .threshold attribute and predict() uses it automatically
    """
    if params is None:
        params = config.SVM_PARAMS

    if config.VERBOSE:
        print("\n" + "=" * 60)
        print("TRAINING SVM MODEL")
        print("=" * 60)
        print(f"Parameters: {params}")

    scaler = StandardScaler()

    if calibrate and len(X_train) >= 120:
        cal_split = int(len(X_train) * 0.8)
        X_fit = X_train.iloc[:cal_split]
        X_cal = X_train.iloc[cal_split:]
        y_fit = y_train.iloc[:cal_split]
        y_cal = y_train.iloc[cal_split:]

        # Scaler must be fit only on training portion, not calibration portion
        X_fit_scaled = scaler.fit_transform(X_fit)
        X_cal_scaled = scaler.transform(X_cal)

        model = SVC(**params)
        model.fit(X_fit_scaled, y_fit)

        proba_cal = model.predict_proba(X_cal_scaled)[:, 1]
        best_thresh, best_acc = 0.5, 0.0
        for thresh in np.arange(0.30, 0.71, 0.01):
            pred = (proba_cal >= thresh).astype(int)
            acc = accuracy_score(y_cal, pred)
            if acc > best_acc:
                best_acc = acc
                best_thresh = float(thresh)

        model._threshold = best_thresh

        if config.VERBOSE:
            print(f"✅ Calibrated threshold: {best_thresh:.2f} (val acc={best_acc:.4f})")

        return model, scaler

    X_train_scaled = scaler.fit_transform(X_train)
    model = SVC(**params)
    model.fit(X_train_scaled, y_train)
    model._threshold = 0.5

    if config.VERBOSE:
        print("✅ Model training complete!")

    return model, scaler


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


def evaluate_svm(model, scaler, X_train, y_train, X_test, y_test):
    """
    Evaluate SVM model performance on both training and test sets

    SVM requires scaled data, so this function handles the scaling.

    Args:
        model: Trained SVM model
        scaler: Fitted StandardScaler
        X_train, y_train: Training data
        X_test, y_test: Test data

    Returns:
        dict: Dictionary with performance metrics
    """
    if config.VERBOSE:
        print("\n" + "=" * 60)
        print("SVM MODEL EVALUATION")
        print("=" * 60)

    # Scale using the scaler fit during training (never refit on test data)
    X_train_scaled = scaler.transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Use calibrated threshold if available, otherwise default SVC.predict
    thresh = getattr(model, '_threshold', 0.5)
    if thresh != 0.5:
        train_pred = (model.predict_proba(X_train_scaled)[:, 1] >= thresh).astype(int)
        test_pred = (model.predict_proba(X_test_scaled)[:, 1] >= thresh).astype(int)
    else:
        train_pred = model.predict(X_train_scaled)
        test_pred = model.predict(X_test_scaled)

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