"""
Pipeline to train XGBoost on the full clean dataset (data/training_pair_v5_clean.csv).
Includes stratified splitting, parallel feature extraction with spaCy, weight & threshold tuning,
evaluation on test set, and metrics reporting.
"""

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["OMP_NUM_THREADS"] = "1"

import re
import time
import pickle
import json
import numpy as np
import pandas as pd
from collections import Counter
from pathlib import Path
from tqdm import tqdm
import spacy
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    precision_score, recall_score, f1_score, roc_auc_score,
    confusion_matrix, classification_report, accuracy_score
)
import matplotlib.pyplot as plt
import seaborn as sns
import xgboost as xgb

# Set paths
DATA_DIR = Path("data")
PROCESSED_DIR = Path("data/processed")
MODEL_DIR = Path("models")
REPORTS_DIR = Path("reports/figures")

PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
MODEL_DIR.mkdir(parents=True, exist_ok=True)
REPORTS_DIR.mkdir(parents=True, exist_ok=True)


class LegalFeatureExtractor:
    """Extract 18 features for legal/administrative German sentences."""
    
    def __init__(self):
        self.citation_pattern = re.compile(
            r'[§§]?\s*\d+\s*(?:[A-Z]+)?\s*(?:Abs\.\s*\d+)?\s*(?:S\.\s*\d+)?|'
            r'[Aa]rt\.\s*\d+|'
            r'Rn\.\s*\d+|'
            r'[Aa]\.?[Aa]\.?|'
            r'[Ii]\.?[Vv]\.?[Mm]\.?'
        )
        self.legal_terms = [
            'ermessen', 'verwaltungsakt', 'behörde', 'gericht', 'klage',
            'urteil', 'beschluss', 'verfahren', 'rechts', 'gesetz',
            'verordnung', 'satzung', 'bescheid', 'widerspruch'
        ]
        self.modal_particles = ['ja', 'doch', 'halt', 'eben', 'mal', 'schon', 'wohl']
        self.passive_indicators = ['wurde', 'wird', 'werden', 'worden', 'würde']
        self.function_words = ['der', 'die', 'das', 'und', 'oder', 'aber', 'den', 'dem']
        self.authority_terms = ['behörde', 'gericht', 'amt', 'ministerium']
        self.closings = ['mit freundlichen grüßen', 'im auftrag', 'hochachtungsvoll']

    def extract_features_batch(self, texts, nnlp, n_process=16):
        """Extract features for a batch of texts using nnlp.pipe."""
        features_list = []
        # Process in batches using spaCy's pipe with multiprocessing
        for doc in tqdm(nnlp.pipe(texts, batch_size=1000, n_process=n_process), total=len(texts)):
            features = self._extract_single(doc)
            features_list.append(features)
        
        return pd.DataFrame(features_list)
    
    def _extract_single(self, doc):
        """Extract features from a single spaCy Doc."""
        text = doc.text
        words = [token.text for token in doc]
        word_count = len(words)
        
        if word_count == 0:
            return self._default_features()
        
        text_lower = text.lower()
        
        features = {}
        
        # 1. Citation Density
        citations = self.citation_pattern.findall(text)
        features['citation_density'] = len(citations) / word_count
        
        # 2. Structure Entropy
        structure_markers = len(re.findall(r'\b[IVX]+\.|\b\d+\.|\([a-z]\)', text))
        features['structure_entropy'] = structure_markers / word_count
        
        # 3. Nominalization Ratio
        nominalizations = len(re.findall(r'\b\w+(?:ung|heit|keit|tion|sion)\b', text_lower))
        features['nominalization_ratio'] = nominalizations / word_count
        
        # 4. Passive Voice Ratio
        passive_count = sum(1 for token in doc if token.text in self.passive_indicators)
        features['passive_ratio'] = passive_count / word_count
        
        # 5. Modal Particle Ratio
        modal_count = sum(1 for token in doc if token.text in self.modal_particles)
        features['modal_particle_ratio'] = modal_count / word_count
        
        # 6. Jargon Consistency
        jargon_count = sum(1 for term in self.legal_terms if term in text_lower)
        features['jargon_consistency'] = jargon_count / word_count
        
        # 7. Authority Ratio
        authority_count = sum(1 for term in self.authority_terms if term in text_lower)
        features['authority_ratio'] = authority_count / word_count
        
        # 8. Closing Ratio
        closing_count = sum(1 for closing in self.closings if closing in text_lower)
        features['closing_ratio'] = closing_count / word_count
        
        # 9. Average Word Length
        word_lengths = [len(token.text) for token in doc if token.is_alpha]
        features['avg_word_length'] = np.mean(word_lengths) if word_lengths else 0
        
        # 10. Type-Token Ratio
        unique_words = len(set([token.text.lower() for token in doc if token.is_alpha]))
        features['type_token_ratio'] = unique_words / word_count if word_count > 0 else 0
        
        # 11. Function Word Ratio
        function_count = sum(1 for token in doc if token.text.lower() in self.function_words)
        features['function_word_ratio'] = function_count / word_count
        
        # 12. Capitalization Ratio
        capitalized = sum(1 for token in doc if token.text[0].isupper() if token.text)
        features['capitalization_ratio'] = capitalized / word_count if word_count > 0 else 0
        
        # 13. Abbreviation Density
        abbreviations = len(re.findall(r'\b[A-ZÄÖÜ]+\.\b|\b[A-ZÄÖÜ]{2,}\b', text))
        features['abbreviation_ratio'] = abbreviations / word_count
        
        # 14. Parenthetical Ratio
        parentheses = len(re.findall(r'\([^)]*\)', text))
        features['parenthetical_ratio'] = parentheses / word_count
        
        # 15. Punctuation Entropy
        punct_chars = re.findall(r'[.,:;!?]', text)
        if punct_chars:
            counts = Counter(punct_chars)
            total = len(punct_chars)
            probs = [count/total for count in counts.values()]
            features['punctuation_entropy'] = -sum(p * np.log(p) for p in probs)
        else:
            features['punctuation_entropy'] = 0
        
        # 16. Clause Density
        subjunctions = ['dass', 'weil', 'da', 'wenn', 'obwohl', 'während', 'bevor']
        clause_count = sum(1 for token in doc if token.text.lower() in subjunctions)
        features['clause_density'] = clause_count / word_count if word_count > 0 else 0
        
        # 17. Word Count
        features['word_count'] = word_count
        
        # 18. "man" Ratio
        man_count = sum(1 for token in doc if token.text.lower() == 'man')
        features['man_ratio'] = man_count / word_count if word_count > 0 else 0
        
        return features
    
    def _default_features(self):
        return {f: 0 for f in [
            'citation_density', 'structure_entropy', 'nominalization_ratio',
            'passive_ratio', 'modal_particle_ratio', 'jargon_consistency',
            'authority_ratio', 'closing_ratio', 'avg_word_length',
            'type_token_ratio', 'function_word_ratio', 'capitalization_ratio',
            'abbreviation_ratio', 'parenthetical_ratio', 'punctuation_entropy',
            'clause_density', 'word_count', 'man_ratio'
        ]}


def main():
    print("=" * 60)
    print("XGBOOST ML PIPELINE: german-ai-detector (Sentence-Level)")
    print("=" * 60)

    # 1. LOAD PRE-SPLIT SENTENCE DATASET
    print("Loading pre-split sentence datasets...")
    train_df = pd.read_parquet(PROCESSED_DIR / "train_sentences.parquet")
    val_df = pd.read_parquet(PROCESSED_DIR / "val_sentences.parquet")
    test_df = pd.read_parquet(PROCESSED_DIR / "test_sentences.parquet")
    
    print(f"   Train: {len(train_df):,} sentences (Human: {sum(train_df['label']==0):,}, AI: {sum(train_df['label']==1):,})")
    print(f"   Val:   {len(val_df):,} sentences (Human: {sum(val_df['label']==0):,}, AI: {sum(val_df['label']==1):,})")
    print(f"   Test:  {len(test_df):,} sentences (Human: {sum(test_df['label']==0):,}, AI: {sum(test_df['label']==1):,})")

    # 2. FEATURE EXTRACTION
    print("\nLoading spaCy German model (de_core_news_sm)...")
    nlp = spacy.load("de_core_news_sm")
    extractor = LegalFeatureExtractor()

    # Determine CPU cores for multiprocessing
    n_cores = os.cpu_count()
    n_process = max(1, min(16, n_cores - 2)) # Leave a couple of cores free
    print(f"Parallel feature extraction running on {n_process} processes.")

    print("\n[RUNNING] Extracting features for Train...")
    t0 = time.time()
    train_features = extractor.extract_features_batch(train_df['sentence'].tolist(), nlp, n_process=n_process)
    train_features['label'] = train_df['label'].values
    print(f"Train extraction took: {time.time()-t0:.2f} seconds")

    print("\n[RUNNING] Extracting features for Val...")
    t0 = time.time()
    val_features = extractor.extract_features_batch(val_df['sentence'].tolist(), nlp, n_process=n_process)
    val_features['label'] = val_df['label'].values
    print(f"Val extraction took: {time.time()-t0:.2f} seconds")

    print("\n[RUNNING] Extracting features for Test...")
    t0 = time.time()
    test_features = extractor.extract_features_batch(test_df['sentence'].tolist(), nlp, n_process=n_process)
    test_features['label'] = test_df['label'].values
    print(f"Test extraction took: {time.time()-t0:.2f} seconds")

    # Save features to parquet
    print("\nSaving extracted features to Parquet files...")
    train_features.to_parquet(PROCESSED_DIR / "train_clean_features.parquet")
    val_features.to_parquet(PROCESSED_DIR / "val_clean_features.parquet")
    test_features.to_parquet(PROCESSED_DIR / "test_clean_features.parquet")
    print(f"Features saved to: {PROCESSED_DIR}")

    # 4. PREPARE DATA FOR XGBOOST
    feature_cols = [col for col in train_features.columns if col != 'label']
    X_train, y_train = train_features[feature_cols], train_features['label']
    X_val, y_val = val_features[feature_cols], val_features['label']
    X_test, y_test = test_features[feature_cols], test_features['label']

    # 5. CLASS WEIGHT TUNING (Minimize False Positives on Val Set)
    weight_options = [1, 2, 3, 5, 8, 10]
    results = []
    print("\n[RUNNING] Tuning scale_pos_weight to minimize False Positives...")
    
    for weight in weight_options:
        print(f"  Testing scale_pos_weight={weight}...")
        model = xgb.XGBClassifier(
            n_estimators=100,
            max_depth=6,
            learning_rate=0.1,
            scale_pos_weight=weight,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            eval_metric='logloss'
        )
        
        model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            verbose=False
        )
        
        y_pred = model.predict(X_val)
        y_proba = model.predict_proba(X_val)[:, 1]
        
        tn, fp, fn, tp = confusion_matrix(y_val, y_pred).ravel()
        fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
        precision = precision_score(y_val, y_pred, zero_division=0)
        recall = recall_score(y_val, y_pred, zero_division=0)
        f1 = f1_score(y_val, y_pred, zero_division=0)
        
        results.append({
            'scale_pos_weight': weight,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'fpr': fpr
        })
        print(f"    Precision: {precision:.4f}, Recall: {recall:.4f}, FPR: {fpr:.4f}")

    results_df = pd.DataFrame(results)
    best_idx = results_df['fpr'].idxmin()
    best_weight = results_df.loc[best_idx, 'scale_pos_weight']
    print(f"\n[SUCCESS] Best class weight: {best_weight} (lowest FPR: {results_df.loc[best_idx, 'fpr']:.4f})")

    # 6. TRAIN FINAL MODEL WITH BEST WEIGHT
    print("\n[RUNNING] Training final XGBoost model...")
    final_model = xgb.XGBClassifier(
        n_estimators=150,  # Slightly higher for final model on full dataset
        max_depth=6,
        learning_rate=0.1,
        scale_pos_weight=best_weight,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        eval_metric='logloss'
    )
    
    final_model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        verbose=False
    )

    # 7. THRESHOLD TUNING (Val Set)
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
    
    # Target FPR < 2% (0.02)
    candidates = threshold_df[threshold_df['fpr'] < 0.02]
    if not candidates.empty:
        best_threshold = candidates.loc[candidates['recall'].idxmax(), 'threshold']
    else:
        best_threshold = 0.90
    print(f"[SUCCESS] Optimal threshold (FPR < 2%): {best_threshold:.2f}")

    # 8. FINAL EVALUATION (Test Set)
    print("\n" + "=" * 60)
    print("FINAL MODEL PERFORMANCE (Unseen Test Set)")
    print("=" * 60)
    
    y_proba_test = final_model.predict_proba(X_test)[:, 1]
    y_pred_test = (y_proba_test >= best_threshold).astype(int)
    
    test_accuracy = accuracy_score(y_test, y_pred_test)
    test_precision = precision_score(y_test, y_pred_test, zero_division=0)
    test_recall = recall_score(y_test, y_pred_test, zero_division=0)
    test_f1 = f1_score(y_test, y_pred_test, zero_division=0)
    test_roc_auc = roc_auc_score(y_test, y_proba_test)
    
    tn, fp, fn, tp = confusion_matrix(y_test, y_pred_test).ravel()
    test_fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
    
    print(f"Optimal scale_pos_weight: {best_weight}")
    print(f"Optimal Threshold:        {best_threshold:.2f}")
    print(f"Accuracy:                 {test_accuracy:.4f}")
    print(f"Precision:                {test_precision:.4f}")
    print(f"Recall:                   {test_recall:.4f}")
    print(f"F1-Score:                 {test_f1:.4f}")
    print(f"ROC-AUC:                  {test_roc_auc:.4f}")
    print(f"False Positive Rate (FPR): {test_fpr:.4f}")
    print(f"Confusion Matrix:         [[{tn} (TN), {fp} (FP)], [{fn} (FN), {tp} (TP)]]")

    # 9. SAVE MODEL AND METADATA
    model_path = MODEL_DIR / "xgboost_model_clean.pkl"
    print(f"\nSaving model to: {model_path}...")
    with open(model_path, "wb") as f:
        pickle.dump(final_model, f)
        
    metadata = {
        'scale_pos_weight': int(best_weight),
        'threshold': float(best_threshold),
        'feature_names': feature_cols,
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
    
    metadata_path = MODEL_DIR / "model_metadata_clean.json"
    print(f"Saving metadata to: {metadata_path}...")
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)

    # 10. GENERATE PLOTS
    # Feature Importance Plot
    importance = pd.DataFrame({
        'feature': feature_cols,
        'importance': final_model.feature_importances_
    }).sort_values('importance', ascending=True)
    
    plt.figure(figsize=(10, 8))
    plt.barh(importance['feature'][-15:], importance['importance'][-15:], color='skyblue')
    plt.xlabel('Importance Score')
    plt.title('Top 15 Feature Importances')
    plt.tight_layout()
    fi_plot_path = REPORTS_DIR / "feature_importance_clean.png"
    plt.savefig(fi_plot_path, dpi=300)
    plt.close()
    print(f"Saved feature importance plot to: {fi_plot_path}")

    # Confusion Matrix Plot
    plt.figure(figsize=(6, 5))
    cm = confusion_matrix(y_test, y_pred_test)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Human (0)', 'AI (1)'],
                yticklabels=['Human (0)', 'AI (1)'])
    plt.xlabel('Predicted Class')
    plt.ylabel('Actual Class')
    plt.title('Confusion Matrix (Test Set)')
    plt.tight_layout()
    cm_plot_path = REPORTS_DIR / "confusion_matrix_clean.png"
    plt.savefig(cm_plot_path, dpi=300)
    plt.close()
    print(f"Saved confusion matrix plot to: {cm_plot_path}")

    print("\n" + "=" * 60)
    print("Pipeline Run Successful and Completed!")
    print("=" * 60)


if __name__ == '__main__':
    main()
