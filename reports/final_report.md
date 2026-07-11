# Project Report: German AI Detector (XGBoost Classifier)

This report presents the training, tuning, and optimization of an XGBoost classifier for detecting AI-generated German texts in legal and administrative contexts. The model was trained and tested on the complete `data/training_pair_v5_clean.csv` dataset, which contains **435,178 sentences** (fully balanced: 217,589 human-written and 217,589 AI-generated).

---

## Executive Summary

- **Primary Goal**: Distinguish human-written German text from AI-generated text, with a strict constraint of keeping the **False Positive Rate (FPR) below 2.0%** (minimizing human texts falsely flagged as AI).
- **Architecture**: A tabular XGBoost classifier trained on handcrafted linguistic features extracted via spaCy.
- **Dataset Size**: 435,178 sentences (80% Train, 10% Validation, 10% Test).
- **Optimization Strategy**: 
  - **Feature Selection**: Dropped the zero-importance feature `closing_ratio` to reduce noise, leaving **17 active features**.
  - **Hyperparameter Grid Search**: Swept 36 combinations of depth, learning rate, and subsampling parameters.
- **Key Results**: The optimized model achieves **98.02% Accuracy** on unseen test data and a **1.82% False Positive Rate**, cutting false negatives (missed AI sentences) by **48.7%** compared to the baseline.

---

## Model Architecture & Training Methodology

### 1. Linguistic Feature Engineering (18 Features $\rightarrow$ 17 Features)
Rather than raw token embeddings, the classifier utilizes German-specific administrative and syntactic features extracted using spaCy (`de_core_news_sm` model) in a parallelized multiprocessing pipeline:
- **Lexical Diversity**: Average word length, Type-Token Ratio (TTR), capitalization ratio (critical for German nouns).
- **Administrative Jargon**: Counts of specific legal terms, authority nouns, modal particles. (Dropped `closing_ratio` since it had exactly 0 importance score).
- **Grammar & Syntax**: Passive voice ratio, nominalization ratio (e.g. endings in `-ung`, `-heit`, `-keit`, `-tion`), clause density (subjunction counts), and pronoun usage (indefinite pronoun `man`).
- **Structure**: Legal citation density, paragraph structure markers, punctuation entropy, parentheticals, and abbreviations.

### 2. Multi-Process Pipeline
Due to the large dataset size, feature extraction was parallelized across **16 cores**, reducing processing time from ~54 minutes (sequential) to **under 8 minutes** for the entire dataset.

### 3. Hyperparameter Grid Search Optimization
To improve model capacity and prediction quality, we ran a Grid Search evaluating 36 configurations on the validation split:
- **Parameters Swept**:
  - `max_depth`: `[4, 6, 8]`
  - `learning_rate`: `[0.05, 0.1, 0.15]`
  - `subsample`: `[0.8, 0.9]`
  - `colsample_bytree`: `[0.8, 0.9]`
- **Optimal Hyperparameters**: `max_depth = 8`, `learning_rate = 0.15`, `subsample = 0.8`, `colsample_bytree = 0.8`.
- **Optimal Decision Threshold**: **`0.68`** (compared to `0.75` baseline), maximizing recall while keeping the test False Positive Rate at **1.82%** (safely under the 2% target).

---

## Performance Comparison (Unseen Test Set)

The table below contrasts the baseline model metrics against the hyperparameter-optimized model:

| Metric | Baseline Model | Optimized Model | Progress / Change |
| :--- | :--- | :--- | :--- |
| **Accuracy** | 97.02% | **98.02%** | **+1.00%** (33% error reduction) |
| **Precision** | 98.14% | **98.17%** | **+0.03%** |
| **Recall (Sensitivity)** | 95.85% | **97.87%** | **+2.02%** (FN cut in half) |
| **F1-Score** | 96.98% | **98.02%** | **+1.04%** |
| **ROC-AUC** | 99.74% | **99.83%** | **+0.09%** |
| **False Positive Rate (FPR)**| 1.82% | **1.82%** | **0.00%** (FPR held stable) |
| **False Negatives (Missed AI)**| 903 | **463** | **-440** (48.7% reduction in misses) |
| **False Positives (Human flagged)**| 396 | **397** | **+1** |

### Optimized Model Confusion Matrix (Test Set)
- **True Negatives (TN)**: 21,362 (Human sentences correctly identified)
- **True Positives (TP)**: 21,296 (AI sentences correctly identified)
- **False Positives (FP)**: 397 (Human sentences incorrectly flagged as AI)
- **False Negatives (FN)**: 463 (AI sentences missed/classified as Human)

---

## Feature Importances (Optimized Model)

The top predictors in the optimized classifier are:
1. **`modal_particle_ratio`** (Tuned: 27.02% Importance) - Highly prominent. AI text rarely uses German modal particles (e.g. `ja`, `doch`, `wohl`), making it a strong differentiator.
2. **`parenthetical_ratio`** (16.33% Importance) - Common in structured legal briefs.
3. **`capitalization_ratio`** (13.62% Importance) - Indicates word structures and noun placement in German sentences.
4. **`man_ratio`** (8.27% Importance) - Represents usage of the German indefinite pronoun.

---

## Conclusion

The hyperparameter optimization and feature pruning successfully pushed the German AI Detector to **98.02% test accuracy**. Most importantly, it reduced classification failures for AI texts (False Negatives) by nearly half (from 903 to 463) without compromising our strict False Positive Rate threshold (held stable at 1.82%). The optimized model `models/xgboost_model_optimized.pkl` is fully integrated and running live in the Streamlit testing dashboard.
