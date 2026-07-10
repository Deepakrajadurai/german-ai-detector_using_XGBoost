"""
Evaluate the trained XGBoost model on the entire clean dataset (Train, Val, Test, and combined).
Generate final metrics and confusion matrices for all partitions.
"""

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["OMP_NUM_THREADS"] = "1"

import pickle
import json
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.metrics import (
    precision_score, recall_score, f1_score, roc_auc_score,
    confusion_matrix, accuracy_score
)
import matplotlib.pyplot as plt
import seaborn as sns

# Set paths
PROCESSED_DIR = Path("data/processed")
MODEL_DIR = Path("models")
REPORTS_DIR = Path("reports/figures")
REPORT_FILE_DIR = Path("reports")

MODEL_DIR.mkdir(parents=True, exist_ok=True)
REPORTS_DIR.mkdir(parents=True, exist_ok=True)
REPORT_FILE_DIR.mkdir(parents=True, exist_ok=True)


def evaluate_set(y_true, y_proba, threshold):
    y_pred = (y_proba >= threshold).astype(int)
    
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    roc_auc = roc_auc_score(y_true, y_proba)
    
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
    
    return {
        'count': int(len(y_true)),
        'accuracy': float(accuracy),
        'precision': float(precision),
        'recall': float(recall),
        'f1': float(f1),
        'roc_auc': float(roc_auc),
        'fpr': float(fpr),
        'confusion_matrix': [[int(tn), int(fp)], [int(fn), int(tp)]]
    }


def main():
    print("=" * 60)
    print("XGBOOST EVALUATION: ENTIRE DATASET")
    print("=" * 60)
    
    # 1. Load model and metadata
    model_path = MODEL_DIR / "xgboost_model_clean.pkl"
    metadata_path = MODEL_DIR / "model_metadata_clean.json"
    
    if not model_path.exists() or not metadata_path.exists():
        print("Error: Trained model or metadata not found. Please run the training pipeline first.")
        return
        
    with open(model_path, "rb") as f:
        model = pickle.load(f)
        
    with open(metadata_path, "r") as f:
        metadata = json.load(f)
        
    threshold = metadata['threshold']
    feature_cols = metadata['feature_names']
    print(f"Loaded trained model with threshold {threshold:.4f} and {len(feature_cols)} features.")
    
    # 2. Load feature parquets
    print("\nLoading feature parquets...")
    train_features = pd.read_parquet(PROCESSED_DIR / "train_clean_features.parquet")
    val_features = pd.read_parquet(PROCESSED_DIR / "val_clean_features.parquet")
    test_features = pd.read_parquet(PROCESSED_DIR / "test_clean_features.parquet")
    
    # Separate X and y
    X_train, y_train = train_features[feature_cols], train_features['label']
    X_val, y_val = val_features[feature_cols], val_features['label']
    X_test, y_test = test_features[feature_cols], test_features['label']
    
    # Combine for entire dataset
    X_entire = pd.concat([X_train, X_val, X_test], ignore_index=True)
    y_entire = pd.concat([y_train, y_val, y_test], ignore_index=True)
    
    print(f"Loaded datasets:")
    print(f"  - Train: {len(X_train):,} rows")
    print(f"  - Val:   {len(X_val):,} rows")
    print(f"  - Test:  {len(X_test):,} rows")
    print(f"  - Entire Dataset: {len(X_entire):,} rows")
    
    # 3. Predict probabilities
    print("\nRunning predictions...")
    y_proba_train = model.predict_proba(X_train)[:, 1]
    y_proba_val = model.predict_proba(X_val)[:, 1]
    y_proba_test = model.predict_proba(X_test)[:, 1]
    y_proba_entire = model.predict_proba(X_entire)[:, 1]
    
    # 4. Evaluate each set
    print("Evaluating partitions...")
    train_metrics = evaluate_set(y_train, y_proba_train, threshold)
    val_metrics = evaluate_set(y_val, y_proba_val, threshold)
    test_metrics = evaluate_set(y_test, y_proba_test, threshold)
    entire_metrics = evaluate_set(y_entire, y_proba_entire, threshold)
    
    # Print results
    for name, m in [("Train", train_metrics), ("Val", val_metrics), ("Test", test_metrics), ("Entire Dataset", entire_metrics)]:
        print(f"\n--- {name} Performance ---")
        print(f"  Accuracy:                 {m['accuracy']:.4f}")
        print(f"  Precision:                {m['precision']:.4f}")
        print(f"  Recall:                   {m['recall']:.4f}")
        print(f"  F1-Score:                 {m['f1']:.4f}")
        print(f"  ROC-AUC:                  {m['roc_auc']:.4f}")
        print(f"  False Positive Rate (FPR): {m['fpr']:.4f}")
        tn, fp, fn, tp = m['confusion_matrix'][0][0], m['confusion_matrix'][0][1], m['confusion_matrix'][1][0], m['confusion_matrix'][1][1]
        print(f"  Confusion Matrix:         [[{tn} (TN), {fp} (FP)], [{fn} (FN), {tp} (TP)]]")
        
    # 5. Save combined metrics to JSON
    evaluation_results = {
        'threshold': threshold,
        'partitions': {
            'train': train_metrics,
            'val': val_metrics,
            'test': test_metrics,
            'entire_dataset': entire_metrics
        },
        'timestamp': pd.Timestamp.now().isoformat()
    }
    
    out_json_path = MODEL_DIR / "full_evaluation_metrics.json"
    with open(out_json_path, "w") as f:
        json.dump(evaluation_results, f, indent=2)
    print(f"\nSaved all metrics to: {out_json_path}")
    
    # 6. Plot Confusion Matrix for the ENTIRE dataset
    plt.figure(figsize=(6, 5))
    cm_entire = np.array(entire_metrics['confusion_matrix'])
    sns.heatmap(cm_entire, annot=True, fmt='d', cmap='Greens',
                xticklabels=['Human (0)', 'AI (1)'],
                yticklabels=['Human (0)', 'AI (1)'])
    plt.xlabel('Predicted Class')
    plt.ylabel('Actual Class')
    plt.title('Confusion Matrix (Entire Dataset)')
    plt.tight_layout()
    cm_plot_path = REPORTS_DIR / "confusion_matrix_entire.png"
    plt.savefig(cm_plot_path, dpi=300)
    plt.close()
    print(f"Saved confusion matrix plot to: {cm_plot_path}")
    
    print("\n" + "=" * 60)
    print("Full evaluation script completed successfully!")
    print("=" * 60)


if __name__ == '__main__':
    main()
