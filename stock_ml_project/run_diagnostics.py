"""
Deep diagnostics: feature importances, walk-forward metrics, and ensemble test.
"""
import sys, os, warnings
warnings.filterwarnings('ignore')
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config
config.VERBOSE = False

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score
from sklearn.model_selection import TimeSeriesSplit
from src.data_collection import load_stock_data
from src.preprocessing import preprocess_stock_data
from src.feature_engineering import engineer_all_features, prepare_ml_data
from src.models import train_random_forest, train_svm, evaluate_svm
from src.xgboost_model import train_xgboost

# ─── Load data ────────────────────────────────────────────────────────────────
raw = load_stock_data('AAPL', 'raw')
processed = preprocess_stock_data(raw)
featured = engineer_all_features(processed)
X, y, features = prepare_ml_data(featured)
print(f"Dataset: {X.shape[0]} samples x {X.shape[1]} features\n")

# ─── 1. Feature importance (RF + XGBoost) ─────────────────────────────────────
print("=" * 60)
print("FEATURE IMPORTANCE (RF + XGBoost on full training set)")
print("=" * 60)

split = int(len(X) * 0.8)
X_train, X_test = X.iloc[:split], X.iloc[split:]
y_train, y_test = y.iloc[:split], y.iloc[split:]

rf = train_random_forest(X_train, y_train)
xgb = train_xgboost(X_train, y_train)

rf_imp = pd.Series(rf.feature_importances_, index=features)
xgb_imp = pd.Series(xgb.feature_importances_, index=features)
combined = (rf_imp + xgb_imp) / 2
combined = combined.sort_values(ascending=False)

print("\nRank  Feature                    RF-imp   XGB-imp  Combined")
print("-" * 60)
for rank, (feat, score) in enumerate(combined.items(), 1):
    print(f"  {rank:2d}  {feat:<26} {rf_imp[feat]:.4f}   {xgb_imp[feat]:.4f}   {score:.4f}")

# Features with combined importance < 0.5% → likely noise
weak = combined[combined < 0.005].index.tolist()
print(f"\nWeak features (combined < 0.5%): {weak}")

# ─── 2. Walk-forward (10-fold, RF only for speed) ─────────────────────────────
print("\n" + "=" * 60)
print("WALK-FORWARD (10-fold, RF + XGBoost)")
print("=" * 60)

tscv = TimeSeriesSplit(n_splits=10)

rf_accs, xgb_accs, rf_aucs, xgb_aucs = [], [], [], []

for fold, (tr_idx, te_idx) in enumerate(tscv.split(X), 1):
    Xtr, Xte = X.iloc[tr_idx], X.iloc[te_idx]
    ytr, yte = y.iloc[tr_idx], y.iloc[te_idx]

    if len(Xtr) < 60:   # skip folds that are too small
        continue

    rf_m = train_random_forest(Xtr, ytr)
    xgb_m = train_xgboost(Xtr, ytr)

    rf_pred = rf_m.predict(Xte)
    xgb_pred = xgb_m.predict(Xte)

    rf_acc  = accuracy_score(yte, rf_pred)
    xgb_acc = accuracy_score(yte, xgb_pred)
    rf_auc  = roc_auc_score(yte, rf_m.predict_proba(Xte)[:, 1])
    xgb_auc = roc_auc_score(yte, xgb_m.predict_proba(Xte)[:, 1])

    rf_accs.append(rf_acc);   xgb_accs.append(xgb_acc)
    rf_aucs.append(rf_auc);   xgb_aucs.append(xgb_auc)

    print(f"Fold {fold:2d} | n_train={len(Xtr):4d} n_test={len(Xte):3d} | "
          f"RF acc={rf_acc:.3f} auc={rf_auc:.3f} | "
          f"XGB acc={xgb_acc:.3f} auc={xgb_auc:.3f}")

print(f"\n{'':8} RF  mean acc={np.mean(rf_accs):.4f}  mean AUC={np.mean(rf_aucs):.4f}")
print(f"{'':8} XGB mean acc={np.mean(xgb_accs):.4f}  mean AUC={np.mean(xgb_aucs):.4f}")

# ─── 3. Ensemble test (RF + XGBoost majority probability) ─────────────────────
print("\n" + "=" * 60)
print("ENSEMBLE TEST (RF + XGBoost averaged probability)")
print("=" * 60)

rf_proba  = rf.predict_proba(X_test)[:, 1]
xgb_proba = xgb.predict_proba(X_test)[:, 1]
ens_proba = (rf_proba + xgb_proba) / 2
ens_pred  = (ens_proba >= 0.5).astype(int)

ens_acc = accuracy_score(y_test, ens_pred)
ens_f1  = f1_score(y_test, ens_pred)
ens_auc = roc_auc_score(y_test, ens_proba)
print(f"Ensemble  Test acc={ens_acc:.4f}  F1={ens_f1:.4f}  AUC={ens_auc:.4f}")

rf_auc_val  = roc_auc_score(y_test, rf_proba)
xgb_auc_val = roc_auc_score(y_test, xgb_proba)
print(f"RF alone  Test acc={accuracy_score(y_test, rf.predict(X_test)):.4f}  AUC={rf_auc_val:.4f}")
print(f"XGB alone Test acc={accuracy_score(y_test, xgb.predict(X_test)):.4f}  AUC={xgb_auc_val:.4f}")
