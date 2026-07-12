# Model Complexity & Bias-Variance Analysis Report

This report presents a thorough analysis of the model complexity, underfitting/overfitting tendencies, and the bias-variance trade-off for all models trained within the **German AI Detector** project.

---

## 1. Executive Summary

We evaluated the behavior of our two primary model architectures:
1. **Baseline Sentence-Level Model** (18 handcrafted features, including `closing_ratio`)
2. **Optimized Sentence-Level Model** (17 handcrafted features, with `closing_ratio` pruned)

### Key Findings
* **No Overfitting (Low Variance)**: In both models, the gap between training and validation metrics is exceptionally narrow (less than **0.34%** at operational complexities), indicating that the models generalize extremely well.
* **Low Bias**: Training and validation accuracies reach **97.6% - 98.5%** at higher tree depths, demonstrating that the 17-18 handcrafted features capture the linguistic signature of administrative German with high fidelity.
* **Optimal Complexity**: The hyperparameter grid search correctly selected `max_depth = 8` for the optimized model, balancing representation power with decision boundaries.

---

## 2. Model Complexity & Validation Curves

To evaluate how model capacity affects generalization, we trained both architectures across a sweep of tree depths (`max_depth` from 2 to 12). 

![Model Complexity Curves](C:\Users\vijayakr\.gemini\antigravity-ide\brain\fd902ee2-9080-46d2-9a74-060bfd8e6b5d\model_complexity_curves.png)

### Numerical Results (Validation Curves)

| Model Configuration | Tree Depth (`max_depth`) | Train Loss | Val Loss | Train Acc | Val Acc | Generalization Gap |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Baseline (18 Features)** | 2 | 0.3790 | 0.3789 | 83.03% | 83.15% | -0.12% |
| | 4 | 0.2841 | 0.2849 | 88.69% | 88.62% | +0.07% |
| | **6 (Operational)** | **0.2098** | **0.2134** | **92.45%** | **92.11%** | **+0.34%** |
| | 8 | 0.1491 | 0.1534 | 95.36% | 95.08% | +0.28% |
| | 10 | 0.1062 | 0.1111 | 97.00% | 96.72% | +0.28% |
| | 12 | 0.0783 | 0.0838 | 97.93% | 97.63% | +0.30% |
| **Optimized (17 Features)** | 2 | 0.3525 | 0.3524 | 84.98% | 85.04% | -0.06% |
| | 4 | 0.2443 | 0.2462 | 90.40% | 90.19% | +0.21% |
| | 6 | 0.1776 | 0.1816 | 93.98% | 93.59% | +0.39% |
| | **8 (Operational)** | **0.1049** | **0.1099** | **96.91%** | **96.79%** | **+0.12%** |
| | 10 | 0.0696 | 0.0763 | 98.17% | 97.82% | +0.35% |
| | 12 | 0.0453 | 0.0523 | 98.83% | 98.57% | +0.26% |

> [!NOTE]
> *Operational* refers to the hyperparameters saved in the model pickle files (`xgboost_model_clean.pkl` and `xgboost_model_optimized.pkl`).

---

## 3. Analysis: Bias vs. Variance

### A. Underfitting (High Bias) Analysis
* **Definition**: A model underfits when it cannot learn the training patterns, leading to high training error and validation error.
* **Observation**: At `max_depth = 2`, both models exhibit underfitting (accuracy $\approx$ 83% - 85% and logloss $\approx$ 0.35 - 0.37). As the tree depth increases, the error drops steadily.
* **Operational Bias**:
  - The **Baseline Model** (`max_depth = 6`) has a validation loss of `0.2134` and accuracy of `92.11%` (prior to threshold shifting). It exhibits a slight bias due to lower depth.
  - The **Optimized Model** (`max_depth = 8`, higher learning rate of 0.15) has a validation loss of `0.1099` and accuracy of `96.79%` (prior to threshold shifting). Shifting to depth 8 successfully reduced bias by allowing the model to capture high-order feature interactions (e.g., combinations of modal particles, nominalization, and punctuation entropy).

### B. Overfitting (High Variance) Analysis
* **Definition**: A model overfits when it memorizes training noise instead of general patterns, leading to low training error but high validation/test error.
* **Observation**: Even at `max_depth = 12`, the generalization gap is incredibly small (only **0.30%** for baseline and **0.26%** for optimized). The validation curves follow the training curves closely and do not bend upwards (loss) or downwards (accuracy), which is the classic sign of overfitting.
* **Operational Variance**: Both models have **exceptionally low variance** (operational gaps of **0.34%** and **0.12%** respectively).

### C. Causes of the Low Variance
1. **Extensive Dataset Size**: Training is performed on **240,000 sentences** (out of 300,000 total sentences). Massive datasets naturally restrict tree models from splitting on sample-specific noise.
2. **Subsampling Regularization**: The models employ row subsampling (`subsample = 0.8 / 0.9`) and feature/column subsampling (`colsample_bytree = 0.8 / 0.9`). These configurations introduce random perturbations that prevent individual trees from co-adapting, acting as bagging regularizers.
3. **Domain-Specific Lightweight Feature Set**: Extracting 17 high-level stylistic features (e.g., POS ratios, jargon density) instead of fitting raw token embeddings restricts the model's capacity to memorize individual text strings.

---

## 4. Learning Curves: Impact of Data Volume

To evaluate if the model is data-limited or has reached its capacity, we analyzed the learning curves of the final **Optimized Model** by varying the training set size from 10% (24k sentences) to 100% (240k sentences).

![Model Learning Curves](C:\Users\vijayakr\.gemini\antigravity-ide\brain\fd902ee2-9080-46d2-9a74-060bfd8e6b5d\model_learning_curves.png)

### Numerical Results (Learning Curves)

| Training Samples | Train Loss | Val Loss | Train Acc | Val Acc | Generalization Gap |
| :---: | :---: | :---: | :---: | :---: | :---: |
| **24,000 (10%)** | 0.1005 | 0.1264 | 97.28% | 95.83% | +1.45% |
| **60,000 (25%)** | 0.1018 | 0.1155 | 97.12% | 96.33% | +0.79% |
| **120,000 (50%)** | 0.1014 | 0.1106 | 97.02% | 96.50% | +0.52% |
| **180,000 (75%)** | 0.1069 | 0.1134 | 96.88% | 96.42% | +0.46% |
| **240,000 (100%)** | **0.1049** | **0.1099** | **96.91%** | **96.79%** | **+0.12%** |

### Insights from Learning Curves
* **Convergence**: The training error and validation error converge as the number of samples increases. The generalization gap shrinks from **1.45%** (at 24k samples) to **0.12%** (at 240k samples).
* **Generalization Gains**: The validation accuracy increases from **95.83%** to **96.79%**. This demonstrates that the model leverages the large corpus to build stable decision boundaries.
* **Plateauing**: The curve starts to flatten between 180k and 240k samples. Adding more of the *same* style data will yield diminishing returns; further improvements require higher-capacity models or expanded feature definitions.
