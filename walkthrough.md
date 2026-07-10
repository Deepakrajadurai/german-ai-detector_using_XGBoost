# Walkthrough: XGBoost Training on Full Clean Dataset

I have successfully implemented, executed, and validated the XGBoost model training on the full dataset `data/training_pair_v5_clean.csv` containing 435,178 rows.

## Changes Made

1. **Created Pipeline Script**:
   - Added [train_clean_xgboost_pipeline.py](file:///c:/Users/vijayakr/Downloads/XGboost/german-ai-detector/src/train_clean_xgboost_pipeline.py) which handles splitting, feature extraction, training, hyperparameter tuning (class weighting and decision thresholds), testing, and output generation.
2. **Dependencies Installed**:
   - Installed `spacy` and `seaborn` in Python 3.14.
   - Downloaded spaCy German language model `de_core_news_sm`.
3. **Multiprocessing Optimization**:
   - Configured spaCy's `nlp.pipe` to run on 16 parallel processes, completing the feature extraction of the entire 435k rows in under 8 minutes.
4. **Output Directory Saving**:
   - Saved trained model to `models/xgboost_model_clean.pkl`
   - Saved execution metadata and performance metrics to `models/model_metadata_clean.json`
   - Saved visual reports: `confusion_matrix_clean.png` and `feature_importance_clean.png` in `reports/figures/`.

---

## Validation Results

The model was tested on an unseen **Test Set** containing **43,518 sentences** (balanced 50% Human, 50% AI). Below are the final performance metrics:

### Performance Metrics (Unseen Test Set)

| Metric | Value |
| :--- | :--- |
| **Accuracy** | 97.02% |
| **Precision** | 98.14% |
| **Recall** | 95.85% |
| **F1-Score** | 96.98% |
| **ROC-AUC** | 99.74% |
| **False Positive Rate (FPR)** | 1.82% |
| **Optimal scale_pos_weight** | 1 |
| **Optimal Decision Threshold** | 0.75 |

### Confusion Matrix

- **True Negatives (TN - Correctly identified Human)**: 21,363
- **True Positives (TP - Correctly identified AI)**: 20,856
- **False Positives (FP - Human flagged as AI)**: 396
- **False Negatives (FN - AI flagged as Human)**: 903

---

## Performance Visualizations

### Confusion Matrix (Test Set)

![Confusion Matrix (Test Set)](C:/Users/vijayakr/.gemini/antigravity-ide/brain/75b0cb4b-5f84-4200-bee7-66315ec3e38c/confusion_matrix_clean.png)

### Top 15 Feature Importances

![Top 15 Feature Importances](C:/Users/vijayakr/.gemini/antigravity-ide/brain/75b0cb4b-5f84-4200-bee7-66315ec3e38c/feature_importance_clean.png)

---

## Summary of Saved Files

- **Trained Model**: [xgboost_model_clean.pkl](file:///c:/Users/vijayakr/Downloads/XGboost/german-ai-detector/models/xgboost_model_clean.pkl)
- **Metadata and Performance JSON**: [model_metadata_clean.json](file:///c:/Users/vijayakr/Downloads/XGboost/german-ai-detector/models/model_metadata_clean.json)
- **Plots**:
  - [confusion_matrix_clean.png](file:///c:/Users/vijayakr/Downloads/XGboost/german-ai-detector/reports/figures/confusion_matrix_clean.png)
  - [feature_importance_clean.png](file:///c:/Users/vijayakr/Downloads/XGboost/german-ai-detector/reports/figures/feature_importance_clean.png)
- **Extracted Feature Sets (Parquet)**:
  - [train_clean_features.parquet](file:///c:/Users/vijayakr/Downloads/XGboost/german-ai-detector/data/processed/train_clean_features.parquet)
  - [val_clean_features.parquet](file:///c:/Users/vijayakr/Downloads/XGboost/german-ai-detector/data/processed/val_clean_features.parquet)
  - [test_clean_features.parquet](file:///c:/Users/vijayakr/Downloads/XGboost/german-ai-detector/data/processed/test_clean_features.parquet)
