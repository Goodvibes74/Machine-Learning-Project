"""Baseline evaluation script - runs all three models and prints metrics."""
import sys, os, warnings
warnings.filterwarnings('ignore')
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config
config.VERBOSE = False

from src.data_collection import load_stock_data
from src.preprocessing import preprocess_stock_data
from src.feature_engineering import engineer_all_features, prepare_ml_data
from src.models import train_random_forest, evaluate_model, train_svm, evaluate_svm
from src.xgboost_model import train_xgboost, evaluate_xgboost_model
from sklearn.metrics import roc_auc_score, classification_report
import numpy as np
import json

ticker = 'AAPL'
print('=== BASELINE EVALUATION ===')

raw = load_stock_data(ticker, 'raw')
processed = preprocess_stock_data(raw)
featured = engineer_all_features(processed)
X, y, features = prepare_ml_data(featured)

print(f'Dataset: {X.shape[0]} samples x {X.shape[1]} features')
print(f'Up days: {y.mean():.1%}  Down days: {(1-y.mean()):.1%}')

split = int(len(X) * 0.8)
X_train, X_test = X.iloc[:split], X.iloc[split:]
y_train, y_test = y.iloc[:split], y.iloc[split:]
print(f'Train: {len(X_train)}  Test: {len(X_test)}')
print()

print('--- Random Forest ---')
rf = train_random_forest(X_train, y_train)
rf_r = evaluate_model(rf, X_train, y_train, X_test, y_test)
rf_proba = rf.predict_proba(X_test)[:, 1]
rf_auc = roc_auc_score(y_test, rf_proba)
print(f'Train Acc={rf_r["train_accuracy"]:.4f}  Test Acc={rf_r["test_accuracy"]:.4f}  AUC={rf_auc:.4f}  F1={rf_r["test_f1"]:.4f}')

print()
print('--- XGBoost ---')
xgb = train_xgboost(X_train, y_train)
xgb_r, _ = evaluate_xgboost_model(xgb, X_train, y_train, X_test, y_test)
print(f'Train Acc={xgb_r["train_accuracy"]:.4f}  Test Acc={xgb_r["accuracy"]:.4f}  AUC={xgb_r["roc_auc"]:.4f}  F1={xgb_r["f1"]:.4f}')

print()
print('--- SVM ---')
svm_m, svm_s = train_svm(X_train, y_train)
svm_r = evaluate_svm(svm_m, svm_s, X_train, y_train, X_test, y_test)
svm_proba = svm_m.predict_proba(svm_s.transform(X_test))[:, 1]
svm_auc = roc_auc_score(y_test, svm_proba)
print(f'Train Acc={svm_r["train_accuracy"]:.4f}  Test Acc={svm_r["test_accuracy"]:.4f}  AUC={svm_auc:.4f}  F1={svm_r["test_f1"]:.4f}')

print()
print(f'Features ({len(features)}):')
for f in features:
    print(f'  {f}')

# Save
baseline = {
    'n_samples': int(X.shape[0]),
    'n_features': int(X.shape[1]),
    'RF':  {'train_acc': rf_r['train_accuracy'],  'test_acc': rf_r['test_accuracy'],  'auc': rf_auc,             'f1': rf_r['test_f1']},
    'XGB': {'train_acc': xgb_r['train_accuracy'], 'test_acc': xgb_r['accuracy'],      'auc': xgb_r['roc_auc'],   'f1': xgb_r['f1']},
    'SVM': {'train_acc': svm_r['train_accuracy'], 'test_acc': svm_r['test_accuracy'], 'auc': svm_auc,            'f1': svm_r['test_f1']},
}
os.makedirs('outputs', exist_ok=True)
with open('outputs/baseline_metrics.json', 'w') as fh:
    json.dump(baseline, fh, indent=2)
print()
print('Saved to outputs/baseline_metrics.json')
