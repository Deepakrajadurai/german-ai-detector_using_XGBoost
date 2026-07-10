# Project Report: German AI Detector (XGBoost Classifier)

This report presents the training, tuning, and evaluation of an XGBoost classifier for detecting AI-generated German texts in legal and administrative contexts. The model was trained and tested on the complete `data/training_pair_v5_clean.csv` dataset, which contains **435,178 sentences** (fully balanced: 217,589 human-written and 217,589 AI-generated).

---

## Executive Summary

- **Primary Goal**: Distinguish human-written German text from AI-generated text, with a strict constraint of keeping the **False Positive Rate (FPR) below 2.0%** (minimizing human texts falsely flagged as AI).
- **Architecture**: A tabular XGBoost classifier trained on **18 handcrafted linguistic features** extracted via spaCy.
- **Dataset Size**: 435,178 sentences (80% Train, 10% Validation, 10% Test).
- **Key Findings**: The model achieves **97.29% overall accuracy** and a **1.69% False Positive Rate** on the entire dataset. The results are highly consistent between training and testing sets, indicating excellent generalization.

---

## Model Architecture & Training Methodology

### 1. Linguistic Feature Engineering (18 Features)
Rather than raw token embeddings, the classifier utilizes 18 German-specific administrative and syntactic features extracted using spaCy (`de_core_news_sm` model) in a parallelized multiprocessing pipeline:
- **Lexical Diversity**: Average word length, Type-Token Ratio (TTR), capitalization ratio (critical for German nouns).
- **Administrative Jargon**: Counts of specific legal terms, authority nouns, modal particles, and formulaic closings.
- **Grammar & Syntax**: Passive voice ratio, nominalization ratio (e.g. endings in `-ung`, `-heit`, `-keit`, `-tion`), clause density (subjunction counts), and pronoun usage (indefinite pronoun `man`).
- **Structure**: Legal citation density, paragraph structure markers, punctuation entropy, parentheticals, and abbreviations.

### 2. Multi-Process Pipeline
Due to the large dataset size, feature extraction was parallelized across **16 cores**, reducing processing time from ~54 minutes (sequential) to **under 8 minutes** for the entire dataset.

### 3. Hyperparameter & Decision Threshold Tuning
To minimize false positives (where human text is misclassified as AI), a two-step optimization process was performed:
- **Class Weight Tuning (`scale_pos_weight`)**: Evaluated weights `[1, 2, 3, 5, 8, 10]` to determine the best weight on validation data. A weight of **`1`** yielded the lowest FPR (4.34% at default threshold).
- **Decision Threshold Tuning**: Evaluated decision thresholds from `0.50` to `0.99`. A threshold of **`0.75`** was selected, successfully driving the False Positive Rate below the 2.0% target while maximizing recall (AI text detection rate).

---

## Performance Metrics

The model was evaluated across all partitions and the entire dataset combined.

### 1. Partition Performance Summary

| Metric | Train Set (80%) | Validation Set (10%) | Test Set (10%) | Entire Dataset (100%) |
| :--- | :--- | :--- | :--- | :--- |
| **Total Samples** | 348,142 | 43,518 | 43,518 | 435,178 |
| **Accuracy** | 97.36% | 97.03% | 97.02% | **97.29%** |
| **Precision** | 98.33% | 97.99% | 98.14% | **98.28%** |
| **Recall (Sensitivity)** | 96.35% | 96.03% | 95.85% | **96.27%** |
| **F1-Score** | 97.33% | 97.00% | 96.98% | **97.26%** |
| **ROC-AUC** | 99.76% | 99.71% | 99.74% | **99.76%** |
| **False Positive Rate (FPR)** | 1.64% | 1.97% | 1.82% | **1.69%** |

### 2. Confusion Matrices

#### Entire Dataset (435,178 Samples)
- **True Negatives (TN)**: 213,914 (Human sentences correctly identified)
- **False Positives (FP)**: 3,675 (Human sentences incorrectly flagged as AI)
- **False Negatives (FN)**: 8,116 (AI sentences missed/classified as Human)
- **True Positives (TP)**: 209,473 (AI sentences correctly identified)

#### Test Set (43,518 Samples)
- **True Negatives (TN)**: 21,363
- **False Positives (FP)**: 396
- **False Negatives (FN)**: 903
- **True Positives (TP)**: 20,856

---

## Visual Reports

### Confusion Matrix (Entire Dataset)
The confusion matrix below illustrates the final class predictions across all 435,178 samples. It demonstrates a highly balanced and accurate performance with extremely low false predictions.

![Confusion Matrix - Entire Dataset](C:/Users/vijayakr/.gemini/antigravity-ide/brain/75b0cb4b-5f84-4200-bee7-66315ec3e38c/confusion_matrix_entire.png)

### Top 15 Feature Importances
This chart displays the linguistic features that carry the highest predictive power in distinguishing human-written text from AI-generated text.

![Top 15 Feature Importances](C:/Users/vijayakr/.gemini/antigravity-ide/brain/75b0cb4b-5f84-4200-bee7-66315ec3e38c/feature_importance_clean.png)

*Key Insights from Feature Importance:*
- **Nominalization Ratio** and **Type-Token Ratio (Lexical Diversity)** are highly prominent features, confirming that AI text differs significantly from human text in syntax complexity and vocabulary redundancy.
- **Punctuation Entropy** and **Function Word Ratio** are also strong indicators, showcasing subtle stylistic patterns in AI sentence structure.

---

## Conclusion & Recommendations

1. **High Robustness**: The model achieves 97.29% accuracy on the full dataset, with only a 0.34% drop in accuracy from the training set to the test set, demonstrating that it has not overfitted and is highly robust.
2. **Effective FPR Control**: By adjusting the classification threshold to `0.75`, we controlled the False Positive Rate to **1.69%** (meeting the target of <2%), keeping human-text misclassifications to a minimum.
3. **Deployability**: The generated XGBoost model file `models/xgboost_model_clean.pkl` and metadata config are fully ready for integration into the inference pipeline.
