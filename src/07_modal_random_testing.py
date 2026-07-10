# """
# Step 7: Randomly sample rows from training_pair_v5_clean.csv,
# run them through the trained detector, and save results as JSON.

# Usage:
#     python3 src/07_random_batch_test.py --n 20
#     python3 src/07_random_batch_test.py --n 50 --seed 42
#     python3 src/07_random_batch_test.py --n 100 --out results/my_test.json
# """

# import os
# os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
# os.environ["OMP_NUM_THREADS"] = "1"

# import argparse
# import json
# import random
# from pathlib import Path

# import pandas as pd
# import numpy as np
# from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# # Reuse everything from your existing test script
# import importlib.util

# _script_dir = Path(__file__).parent
# _inference_path = _script_dir / "06_test_inference.py"
# _spec = importlib.util.spec_from_file_location("inference", _inference_path)
# inference = importlib.util.module_from_spec(_spec)
# _spec.loader.exec_module(inference)


# def normalize_tag(value):
#     """Map various label encodings to 'ai' / 'human'."""
#     if value is None:
#         return "unknown"
#     v = str(value).strip().lower()
#     if v in ["ai", "1", "1.0", "machine", "generated", "ai-generated", "true"]:
#         return "ai"
#     if v in ["human", "0", "0.0", "real", "human-written", "false"]:
#         return "human"
#     return v  # fall back to whatever raw value it was


# def main():
#     parser = argparse.ArgumentParser(description="Randomly test the AI detector on CSV samples.")
#     parser.add_argument("--csv", default="data/training_pair_v5_clean.csv", 
#                         help="Path to CSV file (default: data/training_pair_v5_clean.csv)")
#     parser.add_argument("--text-col", default="text", 
#                         help="Column name containing the input text (default: text)")
#     parser.add_argument("--label-col", default="label", 
#                         help="Column name containing the ground-truth tag (default: label)")
#     parser.add_argument("--domain-col", default="domain", 
#                         help="Column name containing the domain (optional)")
#     parser.add_argument("--sep", default=",", help="Field delimiter")
#     parser.add_argument("--n", type=int, default=20, 
#                         help="Number of random samples to test (default: 20)")
#     parser.add_argument("--seed", type=int, default=None, 
#                         help="Random seed for reproducibility")
#     parser.add_argument("--out", default="results/random_test_results.json", 
#                         help="Output JSON path (default: results/random_test_results.json)")
#     args = parser.parse_args()

#     if args.seed is not None:
#         random.seed(args.seed)
#         np.random.seed(args.seed)

#     # ---- Load data ----
#     csv_path = Path(args.csv)
#     if not csv_path.exists():
#         print(f"❌ CSV not found: {csv_path}")
#         return

#     print("=" * 60)
#     print(f"📂 Loading CSV: {csv_path}")
#     print("=" * 60)
    
#     df = pd.read_csv(csv_path, sep=args.sep)

#     if args.text_col not in df.columns:
#         print(f"❌ Column '{args.text_col}' not found. Available columns: {list(df.columns)}")
#         return

#     has_labels = args.label_col in df.columns
#     if not has_labels:
#         print(f"⚠️  Column '{args.label_col}' not found — proceeding without ground-truth tags.")

#     # Drop empty/NaN text rows before sampling
#     df = df[df[args.text_col].notna()]
#     df = df[df[args.text_col].astype(str).str.strip() != ""]

#     # Show label distribution before sampling
#     if has_labels:
#         label_counts = df[args.label_col].value_counts()
#         print(f"\n📊 Label distribution in full dataset:")
#         for label, count in label_counts.items():
#             print(f"   {label}: {count:,} rows")

#     n = min(args.n, len(df))
#     sample_df = df.sample(n=n, random_state=args.seed)
    
#     print(f"\n🎯 Sampling {n} rows from {len(df):,} total rows...")

#     # ---- Load model ----
#     print("\n" + "=" * 60)
#     print("🧪 LOADING TRAINED MODEL")
#     print("=" * 60)
    
#     try:
#         model, metadata = inference.load_model()
#         extractor = inference.LegalFeatureExtractor()
#         print("✅ Model loaded successfully!")
#         print(f"   Threshold: {metadata.get('threshold', 0.5)*100:.1f}%")
#         print(f"   Validation Precision: {metadata['performance']['precision']:.2%}")
#         print(f"   Validation ROC-AUC: {metadata['performance']['roc_auc']:.2%}")
#     except FileNotFoundError as e:
#         print(f"❌ Error: {e}")
#         return

#     # ---- Run predictions ----
#     print("\n" + "=" * 60)
#     print(f"🔮 RUNNING PREDICTIONS ON {n} SAMPLES")
#     print("=" * 60)
    
#     test_cases = []
#     correct_count = 0
#     y_true = []
#     y_pred = []

#     for i, (_, row) in enumerate(sample_df.iterrows(), start=1):
#         text = str(row[args.text_col])
#         original_tag = normalize_tag(row[args.label_col]) if has_labels else "unknown"

#         result = inference.predict_text(text, model, metadata, extractor)
#         predicted_tag = "ai" if result["label"] == "AI-Generated" else "human"
        
#         is_correct = (predicted_tag == original_tag) if has_labels else None
#         if is_correct:
#             correct_count += 1

#         if has_labels:
#             y_true.append(1 if original_tag == "ai" else 0)
#             y_pred.append(1 if predicted_tag == "ai" else 0)

#         # Get domain if available
#         domain = None
#         if args.domain_col in df.columns:
#             domain = row.get(args.domain_col, None)

#         test_cases.append({
#             "id": i,
#             "source_id": row.get("id", None),
#             "domain": domain,
#             "input_text": text[:200] + "..." if len(text) > 200 else text,
#             "original_tag": original_tag,
#             "prediction": {
#             "predicted_label": result["label"],
#             "confidence": float(round(result["confidence"], 2)),
#             "probability_ai": float(round(result["probability_ai"], 2)),
#             "threshold": float(round(result["threshold"], 2)),
#         },
#             "correct": is_correct,
#         })

#         status = "✅" if is_correct else "❌" if is_correct is not None else "⚪"
#         print(f"[{i}/{n}] {status} {result['label']:<15} "
#               f"(conf: {result['confidence']:.1f}%) "
#               f"| true: {original_tag}")

#     # ---- Calculate metrics ----
#     summary = {
#         "total_samples": n,
#         "source_file": str(csv_path),
#         "seed": args.seed,
#     }
    
#     if has_labels:
#         accuracy = correct_count / n * 100 if n > 0 else 0
#         summary["accuracy"] = round(accuracy, 2)
#         summary["correct"] = correct_count
#         summary["incorrect"] = n - correct_count
        
#         # Confusion Matrix
#         cm = confusion_matrix(y_true, y_pred)
#         tn, fp, fn, tp = cm.ravel() if cm.size == 4 else (0, 0, 0, 0)
#         summary["confusion_matrix"] = {
#             "true_negatives": int(tn),
#             "false_positives": int(fp),
#             "false_negatives": int(fn),
#             "true_positives": int(tp)
#         }
        
#         # Classification Report
#         report = classification_report(y_true, y_pred, 
#                                        target_names=["Human", "AI"], 
#                                        output_dict=True)
#         summary["classification_report"] = report

#     # ---- Output ----
#     output = {
#         "summary": summary,
#         "test_cases": test_cases,
#     }

#     out_path = Path(args.out)
#     out_path.parent.mkdir(parents=True, exist_ok=True)
#     with open(out_path, "w", encoding="utf-8") as f:
#         json.dump(output, f, ensure_ascii=False, indent=2)

#     # ---- Print summary ----
#     print("\n" + "=" * 60)
#     print("📊 SUMMARY")
#     print("=" * 60)
    
#     if has_labels:
#         print(f"✅ Accuracy: {summary['accuracy']:.2f}% ({correct_count}/{n})")
#         print(f"\n📊 Confusion Matrix:")
#         print(f"   True Negatives (Human→Human): {summary['confusion_matrix']['true_negatives']}")
#         print(f"   False Positives (Human→AI):   {summary['confusion_matrix']['false_positives']}")
#         print(f"   False Negatives (AI→Human):   {summary['confusion_matrix']['false_negatives']}")
#         print(f"   True Positives (AI→AI):       {summary['confusion_matrix']['true_positives']}")
        
#         # Precision, Recall, F1 from report
#         report = summary["classification_report"]
#         print(f"\n📊 Classification Report:")
#         print(f"   Human - Precision: {report['Human']['precision']:.2%}, Recall: {report['Human']['recall']:.2%}, F1: {report['Human']['f1-score']:.2%}")
#         print(f"   AI    - Precision: {report['AI']['precision']:.2%}, Recall: {report['AI']['recall']:.2%}, F1: {report['AI']['f1-score']:.2%}")
#         print(f"   Macro Avg: F1: {report['macro avg']['f1-score']:.2%}")
#         print(f"   Weighted Avg: F1: {report['weighted avg']['f1-score']:.2%}")
#     else:
#         print("⚠️  No ground-truth labels available. Results are unvalidated.")

#     print(f"\n📁 Results saved to: {out_path}")
#     print("=" * 60)


# if __name__ == "__main__":
#     main()



"""
Step 7: Test the trained detector on specific slices of the CSV.

Usage:
    # Random 20 samples
    python3 src/07_modal_random_testing.py --n 20 --sep ','

    # First 500 texts
    python3 src/07_modal_random_testing.py --mode head --n 500 --sep ','

    # Last 500 texts
    python3 src/07_modal_random_testing.py --mode tail --n 500 --sep ','
"""

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["OMP_NUM_THREADS"] = "1"

import argparse
import json
import random
from pathlib import Path

import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# Reuse everything from your existing test script
import importlib.util

_script_dir = Path(__file__).parent
_inference_path = _script_dir / "06_test_inference.py"
_spec = importlib.util.spec_from_file_location("inference", _inference_path)
inference = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(inference)


def print_progress_bar(current, total, bar_length=40):
    """Print an in-place progress bar like: [████████████░░░░░░░░] 60% (300/500)"""
    fraction = current / total if total else 0
    filled = int(bar_length * fraction)
    bar = "█" * filled + "░" * (bar_length - filled)
    print(f"\r🔮 [{bar}] {fraction * 100:5.1f}% ({current}/{total})", end="", flush=True)
    if current == total:
        print()  # move to next line once done


def normalize_tag(value):
    """Map various label encodings to 'ai' / 'human'."""
    if value is None:
        return "unknown"
    v = str(value).strip().lower()
    if v in ["ai", "1", "1.0", "machine", "generated", "ai-generated", "true"]:
        return "ai"
    if v in ["human", "0", "0.0", "real", "human-written", "false"]:
        return "human"
    return v


def main():
    parser = argparse.ArgumentParser(description="Test the AI detector on CSV samples.")
    parser.add_argument("--csv", default="data/training_pair_v5_clean.csv",
                        help="Path to CSV file (default: data/training_pair_v5_clean.csv)")
    parser.add_argument("--text-col", default="text",
                        help="Column name containing the input text (default: text)")
    parser.add_argument("--label-col", default="label",
                        help="Column name containing the ground-truth tag (default: label)")
    parser.add_argument("--domain-col", default="domain",
                        help="Column name containing the domain (optional)")
    parser.add_argument("--sep", default=",",
                        help="Field delimiter (default: comma)")
    parser.add_argument("--n", type=int, default=20,
                        help="Number of samples to test (default: 20)")
    parser.add_argument("--mode", choices=["random", "head", "tail"], default="random",
                        help="Sampling mode: 'random' (default), 'head' (first N), or 'tail' (last N)")
    parser.add_argument("--seed", type=int, default=None,
                        help="Random seed for reproducibility (only used with --mode random)")
    parser.add_argument("--out", default="results/test_results.json",
                        help="Output JSON path (default: results/test_results.json)")
    args = parser.parse_args()

    if args.seed is not None and args.mode == "random":
        random.seed(args.seed)
        np.random.seed(args.seed)

    # ---- Load data ----
    csv_path = Path(args.csv)
    if not csv_path.exists():
        print(f"❌ CSV not found: {csv_path}")
        return

    print("=" * 60)
    print(f"📂 Loading CSV: {csv_path}")
    print("=" * 60)
    
    df = pd.read_csv(csv_path, sep=args.sep)

    if args.text_col not in df.columns:
        print(f"❌ Column '{args.text_col}' not found. Available columns: {list(df.columns)}")
        return

    has_labels = args.label_col in df.columns
    if not has_labels:
        print(f"⚠️  Column '{args.label_col}' not found — proceeding without ground-truth tags.")

    # Drop empty/NaN text rows
    df = df[df[args.text_col].notna()]
    df = df[df[args.text_col].astype(str).str.strip() != ""]

    # Show label distribution
    if has_labels:
        label_counts = df[args.label_col].value_counts()
        print(f"\n📊 Label distribution in full dataset:")
        for label, count in label_counts.items():
            print(f"   {label}: {count:,} rows")

    n = min(args.n, len(df))
    
    # ---- Sampling Logic ----
    if args.mode == "random":
        print(f"\n🎯 Sampling {n} random rows from {len(df):,} total rows...")
        sample_df = df.sample(n=n, random_state=args.seed)
    elif args.mode == "head":
        print(f"\n🎯 Taking the first {n} rows from {len(df):,} total rows...")
        sample_df = df.head(n)
    elif args.mode == "tail":
        print(f"\n🎯 Taking the last {n} rows from {len(df):,} total rows...")
        sample_df = df.tail(n)

    # ---- Load model ----
    print("\n" + "=" * 60)
    print("🧪 LOADING TRAINED MODEL")
    print("=" * 60)
    
    try:
        model, metadata = inference.load_model()
        extractor = inference.LegalFeatureExtractor()
        print("✅ Model loaded successfully!")
        print(f"   Threshold: {metadata.get('threshold', 0.5)*100:.1f}%")
        print(f"   Validation Precision: {metadata['performance']['precision']:.2%}")
        print(f"   Validation ROC-AUC: {metadata['performance']['roc_auc']:.2%}")
    except FileNotFoundError as e:
        print(f"❌ Error: {e}")
        return

    # ---- Run predictions ----
    print("\n" + "=" * 60)
    print(f"🔮 RUNNING PREDICTIONS ON {n} SAMPLES ({args.mode} mode)")
    print("=" * 60)
    
    test_cases = []
    correct_count = 0
    y_true = []
    y_pred = []

    for i, (_, row) in enumerate(sample_df.iterrows(), start=1):
        text = str(row[args.text_col])
        original_tag = normalize_tag(row[args.label_col]) if has_labels else "unknown"

        result = inference.predict_text(text, model, metadata, extractor)
        predicted_tag = "ai" if result["label"] == "AI-Generated" else "human"
        
        is_correct = (predicted_tag == original_tag) if has_labels else None
        if is_correct:
            correct_count += 1

        if has_labels:
            y_true.append(1 if original_tag == "ai" else 0)
            y_pred.append(1 if predicted_tag == "ai" else 0)

        domain = None
        if args.domain_col in df.columns:
            domain = row.get(args.domain_col, None)

        # Store the full text in the JSON, but truncate for display
        full_text = str(row[args.text_col])
        display_text = full_text[:200] + "..." if len(full_text) > 200 else full_text

        test_cases.append({
            "id": i,
            "source_id": row.get("id", None),
            "domain": domain,
            "input_text": full_text,  # Store full text in JSON
            "display_text": display_text,  # For quick preview
            "original_tag": original_tag,
            "prediction": {
                "predicted_label": result["label"],
                "confidence": float(round(result["confidence"], 2)),
                "probability_ai": float(round(result["probability_ai"], 2)),
                "threshold": float(round(result["threshold"], 2)),
            },
            "correct": is_correct,
        })

        print_progress_bar(i, n)

    # ---- Calculate metrics ----
    summary = {
        "total_samples": n,
        "mode": args.mode,
        "source_file": str(csv_path),
        "seed": args.seed if args.mode == "random" else None,
    }
    
    if has_labels:
        accuracy = correct_count / n * 100 if n > 0 else 0
        summary["accuracy"] = round(accuracy, 2)
        summary["correct"] = correct_count
        summary["incorrect"] = n - correct_count
        
        cm = confusion_matrix(y_true, y_pred)
        tn, fp, fn, tp = cm.ravel() if cm.size == 4 else (0, 0, 0, 0)
        summary["confusion_matrix"] = {
            "true_negatives": int(tn),
            "false_positives": int(fp),
            "false_negatives": int(fn),
            "true_positives": int(tp)
        }
        
        report = classification_report(y_true, y_pred, 
                                       target_names=["Human", "AI"], 
                                       output_dict=True)
        summary["classification_report"] = report

    # ---- Output ----
    output = {
        "summary": summary,
        "test_cases": test_cases,
    }

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)

    # ---- Print summary ----
    print("\n" + "=" * 60)
    print("📊 SUMMARY")
    print("=" * 60)
    
    if has_labels:
        print(f"✅ Accuracy: {summary['accuracy']:.2f}% ({correct_count}/{n})")
        print(f"\n📊 Confusion Matrix:")
        print(f"   True Negatives (Human→Human): {summary['confusion_matrix']['true_negatives']}")
        print(f"   False Positives (Human→AI):   {summary['confusion_matrix']['false_positives']}")
        print(f"   False Negatives (AI→Human):   {summary['confusion_matrix']['false_negatives']}")
        print(f"   True Positives (AI→AI):       {summary['confusion_matrix']['true_positives']}")
        
        report = summary["classification_report"]
        print(f"\n📊 Classification Report:")
        print(f"   Human - Precision: {report['Human']['precision']:.2%}, Recall: {report['Human']['recall']:.2%}, F1: {report['Human']['f1-score']:.2%}")
        print(f"   AI    - Precision: {report['AI']['precision']:.2%}, Recall: {report['AI']['recall']:.2%}, F1: {report['AI']['f1-score']:.2%}")
        print(f"   Weighted Avg F1: {report['weighted avg']['f1-score']:.2%}")
    else:
        print("⚠️  No ground-truth labels available. Results are unvalidated.")

    print(f"\n📁 Results saved to: {out_path}")
    print("=" * 60)


if __name__ == "__main__":
    main()