"""
Optimize XGBoost hyperparameters and perform feature selection on the clean German dataset.
Tuned parameters include max_depth, learning_rate, subsample, and colsample_bytree.
"""

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["OMP_NUM_THREADS"] = "1"

import pickle
import json
import time
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.metrics import (
    precision_score, recall_score, f1_score, roc_auc_score,
    confusion_matrix, accuracy_score
)
import xgboost as xgb

# Set paths
PROCESSED_DIR = Path("data/processed")
MODEL_DIR = Path("models")

# 1. Load Parquet feature sets
print("Loading feature sets...")
train_df = pd.read_parquet(PROCESSED_DIR / "train_clean_features.parquet")
val_df = pd.read_parquet(PROCESSED_DIR / "val_clean_features.parquet")
test_df = pd.read_parquet(PROCESSED_DIR / "test_clean_features.parquet")

# 2. Feature Selection: Remove 'closing_ratio' because it has 0 importance
all_feature_cols = [col for col in train_df.columns if col != 'label']
feature_cols = [col for col in all_feature_cols if col != 'closing_ratio']

print(f"Original features: {len(all_feature_cols)}, Selected features: {len(feature_cols)} (Dropped 'closing_ratio')")

X_train, y_train = train_df[feature_cols], train_df['label']
X_val, y_val = val_df[feature_cols], val_df['label']
X_test, y_test = test_df[feature_cols], test_df['label']

# 3. Grid Search parameters
grid_params = {
    'max_depth': [4, 6, 8],
    'learning_rate': [0.05, 0.1, 0.15],
    'subsample': [0.8, 0.9],
    'colsample_bytree': [0.8, 0.9]
}

best_score = 0.0
best_params = {}
best_model = None

# We will search a subset of combinations to make it run within a reasonable time
# Total combinations = 3 * 3 * 2 * 2 = 36. Let's do a Grid Search!
print("\n🔄 Running Grid Search for Hyperparameter Optimization...")
t_start = time.time()

# Let's keep scale_pos_weight = 1 as it was optimal in baseline training
weight = 1

import itertools
keys, values = zip(*grid_params.items())
experiments = [dict(zip(keys, v)) for v in itertools.product(*values)]

print(f"Evaluating {len(experiments)} hyperparameter configurations...")

for i, params in enumerate(experiments):
    model = xgb.XGBClassifier(
        n_estimators=120,  # standard estimators count for tuning speed
        scale_pos_weight=weight,
        random_state=42,
        eval_metric='logloss',
        max_depth=params['max_depth'],
        learning_rate=params['learning_rate'],
        subsample=params['subsample'],
        colsample_bytree=params['colsample_bytree']
    )
    
    # Fit model on training set
    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        verbose=False
    )
    
    # Predict on validation set
    y_pred = model.predict(X_val)
    f1 = f1_score(y_val, y_pred, zero_division=0)
    
    if f1 > best_score:
        best_score = f1
        best_params = params
        best_model = model
        print(f"  [New Best] Config {i+1}/{len(experiments)} - F1: {f1:.5f} | Params: {params}")

print(f"Grid search finished in {time.time() - t_start:.2f} seconds.")
print(f"Best Hyperparameters: {best_params} (Validation F1-Score: {best_score:.4f})")

# 4. Final Threshold Tuning on Validation set
print("\nTuning decision threshold on validation set...")
y_proba_val = best_model.predict_proba(X_val)[:, 1]
thresholds = np.arange(0.5, 0.99, 0.01)
threshold_results = []

for thresh in thresholds:
    y_pred_thresh = (y_proba_val >= thresh).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_val, y_pred_thresh).ravel()
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
    precision = precision_score(y_val, y_pred_thresh, zero_division=0)
    recall = recall_score(y_val, y_pred_thresh, zero_division=0)
    
    threshold_results.append({
        'threshold': thresh,
        'fpr': fpr,
        'precision': precision,
        'recall': recall
    })

threshold_df = pd.DataFrame(threshold_results)
candidates = threshold_df[threshold_df['fpr'] < 0.02]
if not candidates.empty:
    best_threshold = candidates.loc[candidates['recall'].idxmax(), 'threshold']
else:
    best_threshold = 0.90
print(f"Optimal Threshold (FPR < 2%): {best_threshold:.2f}")

# 5. Evaluate on Unseen Test Set
print("\n" + "=" * 60)
print("📊 EVALUATION: OPTIMIZED MODEL (Unseen Test Set)")
print("=" * 60)

y_proba_test = best_model.predict_proba(X_test)[:, 1]
y_pred_test = (y_proba_test >= best_threshold).astype(int)

test_accuracy = accuracy_score(y_test, y_pred_test)
test_precision = precision_score(y_test, y_pred_test, zero_division=0)
test_recall = recall_score(y_test, y_pred_test, zero_division=0)
test_f1 = f1_score(y_test, y_pred_test, zero_division=0)
test_roc_auc = roc_auc_score(y_test, y_proba_test)

tn, fp, fn, tp = confusion_matrix(y_test, y_pred_test).ravel()
test_fpr = fp / (fp + tn) if (fp + tn) > 0 else 0

print(f"Parameters:               {best_params}")
print(f"Optimal Threshold:        {best_threshold:.2f}")
print(f"Accuracy:                 {test_accuracy:.4f}")
print(f"Precision:                {test_precision:.4f}")
print(f"Recall:                   {test_recall:.4f}")
print(f"F1-Score:                 {test_f1:.4f}")
print(f"ROC-AUC:                  {test_roc_auc:.4f}")
print(f"False Positive Rate (FPR): {test_fpr:.4f}")
print(f"Confusion Matrix:         [[{tn} (TN), {fp} (FP)], [{fn} (FN), {tp} (TP)]]")

# Compare with baseline test accuracy
# Baseline test accuracy was 97.02%
with open(MODEL_DIR / "model_metadata_clean.json", "r") as f:
    baseline_meta = json.load(f)
baseline_acc = baseline_meta['performance']['accuracy']
print(f"\nBaseline Accuracy: {baseline_acc:.4f} vs Optimized Accuracy: {test_accuracy:.4f}")

# Save if optimized model matches or improves performance
if test_accuracy >= baseline_acc:
    print("\nSaving optimized model to: xgboost_model_optimized.pkl...")
    with open(MODEL_DIR / "xgboost_model_optimized.pkl", "wb") as f:
        pickle.dump(best_model, f)
        
    metadata = {
        'scale_pos_weight': int(weight),
        'threshold': float(best_threshold),
        'feature_names': feature_cols,
        'hyperparameters': {k: int(v) if isinstance(v, (int, np.integer)) else float(v) for k, v in best_params.items()},
        'performance': {
            'accuracy': float(test_accuracy),
            'precision': float(test_precision),
            'recall': float(test_recall),
            'f1': float(test_f1),
            'roc_auc': float(test_roc_auc),
            'fpr': float(test_fpr),
            'confusion_matrix': [[int(tn), int(fp)], [int(fn), int(tp)]]
        },
        'timestamp': pd.Timestamp.now().isoformat()
    }
    
    with open(MODEL_DIR / "model_metadata_optimized.json", "w") as f:
        json.dump(metadata, f, indent=2)
    print("Saved optimized model metadata.")
else:
    print("\nOptimized model did not outperform baseline. Skipping saving.")
