"""
Step 5: Train XGBoost with focus on minimizing False Positives.
"""

import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.metrics import (
    precision_score, recall_score, f1_score, roc_auc_score,
    confusion_matrix, classification_report
)
from sklearn.model_selection import GridSearchCV
import pickle
import json
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# Paths
FEATURES_DIR = Path("data/features")
MODEL_DIR = Path("models")
REPORTS_DIR = Path("reports/figures")
MODEL_DIR.mkdir(parents=True, exist_ok=True)
REPORTS_DIR.mkdir(parents=True, exist_ok=True)


def train_xgboost():
    """Train XGBoost with class weighting to minimize false positives."""
    
    print("=" * 60)
    print("STEP 5: Training XGBoost Classifier")
    print("=" * 60)
    
    # Load features
    train = pd.read_parquet(FEATURES_DIR / "train_features.parquet")
    val = pd.read_parquet(FEATURES_DIR / "val_features.parquet")
    
    # Separate features and labels
    feature_cols = [col for col in train.columns if col != 'label']
    
    X_train = train[feature_cols]
    y_train = train['label']
    X_val = val[feature_cols]
    y_val = val['label']
    
    print(f"Train: {len(X_train):,} samples ({sum(y_train==0):,} Human, {sum(y_train==1):,} AI)")
    print(f"Val:   {len(X_val):,} samples ({sum(y_val==0):,} Human, {sum(y_val==1):,} AI)")
    
    # ============================================
    # CLASS WEIGHT TUNING (Minimize False Positives)
    # ============================================
    weight_options = [1, 2, 3, 5, 8, 10]
    results = []
    
    print("\n🔄 Testing different class weights...")
    
    for weight in weight_options:
        print(f"\n  Testing scale_pos_weight={weight}")
        
        model = xgb.XGBClassifier(
            n_estimators=100,
            max_depth=6,
            learning_rate=0.1,
            scale_pos_weight=weight,  # Weight for AI class
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            use_label_encoder=False,
            eval_metric='logloss'
        )
        
        model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            verbose=False
        )
        
        # Predict with default threshold (0.5)
        y_pred = model.predict(X_val)
        y_proba = model.predict_proba(X_val)[:, 1]
        
        tn, fp, fn, tp = confusion_matrix(y_val, y_pred).ravel()
        fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
        precision = precision_score(y_val, y_pred, zero_division=0)
        recall = recall_score(y_val, y_pred, zero_division=0)
        
        results.append({
            'scale_pos_weight': weight,
            'precision': precision,
            'recall': recall,
            'f1': f1_score(y_val, y_pred, zero_division=0),
            'roc_auc': roc_auc_score(y_val, y_proba),
            'fpr': fpr,
            'tp': tp,
            'fp': fp,
            'tn': tn,
            'fn': fn
        })
        
        print(f"    Precision: {precision:.4f}, Recall: {recall:.4f}, FPR: {fpr:.4f}")
    
    # Select best: lowest FPR while maintaining reasonable recall
    results_df = pd.DataFrame(results)
    # Prioritize FPR minimization (primary goal)
    best_idx = results_df['fpr'].idxmin()
    best_weight = results_df.loc[best_idx, 'scale_pos_weight']
    
    print(f"\n✅ Best class weight: {best_weight} (lowest FPR: {results_df.loc[best_idx, 'fpr']:.4f})")
    
    # ============================================
    # TRAIN FINAL MODEL
    # ============================================
    final_model = xgb.XGBClassifier(
        n_estimators=100,
        max_depth=6,
        learning_rate=0.1,
        scale_pos_weight=best_weight,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        use_label_encoder=False,
        eval_metric='logloss'
    )
    
    final_model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        verbose=False
    )
    
    # ============================================
    # THRESHOLD TUNING (Further reduce FPs)
    # ============================================
    y_proba_val = final_model.predict_proba(X_val)[:, 1]
    
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
    
    # Choose threshold that gives FPR < 2% while maximizing recall
    candidates = threshold_df[threshold_df['fpr'] < 0.02]
    if not candidates.empty:
        best_threshold = candidates.loc[candidates['recall'].idxmax(), 'threshold']
    else:
        best_threshold = 0.90  # Default high threshold
    
    print(f"\n✅ Optimal threshold: {best_threshold:.2f}")
    
    # ============================================
    # FINAL EVALUATION ON VALIDATION SET
    # ============================================
    y_proba_val_final = final_model.predict_proba(X_val)[:, 1]
    y_pred_final = (y_proba_val_final >= best_threshold).astype(int)
    
    precision = precision_score(y_val, y_pred_final, zero_division=0)
    recall = recall_score(y_val, y_pred_final, zero_division=0)
    f1 = f1_score(y_val, y_pred_final, zero_division=0)
    roc_auc = roc_auc_score(y_val, y_proba_val_final)
    tn, fp, fn, tp = confusion_matrix(y_val, y_pred_final).ravel()
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
    
    print("\n" + "=" * 60)
    print("📊 FINAL MODEL PERFORMANCE (Validation Set)")
    print("=" * 60)
    print(f"scale_pos_weight: {best_weight}")
    print(f"Threshold: {best_threshold:.2f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1-Score: {f1:.4f}")
    print(f"ROC-AUC: {roc_auc:.4f}")
    print(f"False Positive Rate: {fpr:.4f}")
    print(f"Confusion Matrix: [[{tn} (TN), {fp} (FP)], [{fn} (FN), {tp} (TP)]]")
    
    # ============================================
    # SAVE MODEL AND METADATA
    # ============================================
    # Save model
    with open(MODEL_DIR / "xgboost_model.pkl", "wb") as f:
        pickle.dump(final_model, f)
    
    # Save metadata
    metadata = {
    'scale_pos_weight': int(best_weight),
    'threshold': float(best_threshold),
    'feature_names': feature_cols,
    'performance': {
        'precision': float(precision),
        'recall': float(recall),
        'f1': float(f1),
        'roc_auc': float(roc_auc),
        'fpr': float(fpr),
        'confusion_matrix': [[int(tn), int(fp)], [int(fn), int(tp)]]
    },
    'timestamp': pd.Timestamp.now().isoformat()
}
    
    with open(MODEL_DIR / "model_metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)
    
    # ============================================
    # FEATURE IMPORTANCE PLOT
    # ============================================
    importance = pd.DataFrame({
        'feature': feature_cols,
        'importance': final_model.feature_importances_
    }).sort_values('importance', ascending=True)
    
    plt.figure(figsize=(10, 8))
    plt.barh(importance['feature'][-15:], importance['importance'][-15:])
    plt.xlabel('Importance')
    plt.title('Top 15 Feature Importances')
    plt.tight_layout()
    plt.savefig(REPORTS_DIR / "feature_importance.png", dpi=300)
    print(f"\n✅ Feature importance plot saved to: {REPORTS_DIR}/feature_importance.png")
    
    # ============================================
    # CONFUSION MATRIX PLOT
    # ============================================
    plt.figure(figsize=(6, 5))
    cm = confusion_matrix(y_val, y_pred_final)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Human (0)', 'AI (1)'],
                yticklabels=['Human (0)', 'AI (1)'])
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix')
    plt.tight_layout()
    plt.savefig(REPORTS_DIR / "confusion_matrix.png", dpi=300)
    print(f"✅ Confusion matrix saved to: {REPORTS_DIR}/confusion_matrix.png")
    
    print("\n" + "=" * 60)
    print("🎉 Training complete!")
    print(f"📁 Model saved to: {MODEL_DIR}/xgboost_model.pkl")
    print(f"📁 Metadata saved to: {MODEL_DIR}/model_metadata.json")
    print("=" * 60)


if __name__ == "__main__":
    train_xgboost()