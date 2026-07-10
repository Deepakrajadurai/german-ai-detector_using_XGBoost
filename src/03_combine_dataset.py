''' ---------------------------- OLD SCRIPT -------------------------------
import pandas as pd
from pathlib import Path

INTERIM_DIR = Path("data/interim")
PROCESSED_DIR = Path("data/processed")
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

def main():
    print("=" * 60)
    print("STEP 3: Combining Human + AI Datasets")
    print("=" * 60)

    # --- Load Humans ---
    train_h = pd.read_parquet(INTERIM_DIR / "train_human.parquet")
    val_h = pd.read_parquet(INTERIM_DIR / "val_human.parquet")
    test_h = pd.read_parquet(INTERIM_DIR / "test_human.parquet")

    # --- Load AIs ---
    train_ai = pd.read_parquet(INTERIM_DIR / "ai_train.parquet")
    val_ai = pd.read_parquet(INTERIM_DIR / "ai_val.parquet")

    # --- Add label 0 to humans ---
    train_h['label'] = 0
    val_h['label'] = 0
    test_h['label'] = 0

    # --- Select only required columns ---
    train_h = train_h[['sentence', 'label']]
    train_ai = train_ai[['sentence', 'label']]
    val_h = val_h[['sentence', 'label']]
    val_ai = val_ai[['sentence', 'label']]
    test_h = test_h[['sentence', 'label']]  # Test set will only have humans for now

    # --- Combine and shuffle ---
    train = pd.concat([train_h, train_ai]).sample(frac=1, random_state=42)
    val = pd.concat([val_h, val_ai]).sample(frac=1, random_state=42)
    test = test_h.sample(frac=1, random_state=42)

    # --- Print Stats ---
    print("\n📊 FINAL DATASET STATISTICS:")
    print(f"Train: {len(train):,} total")
    print(f"  - Human: {len(train[train['label']==0]):,}")
    print(f"  - AI:    {len(train[train['label']==1]):,}")
    
    print(f"\nVal: {len(val):,} total")
    print(f"  - Human: {len(val[val['label']==0]):,}")
    print(f"  - AI:    {len(val[val['label']==1]):,}")
    
    print(f"\nTest: {len(test):,} total (Human-only for final evaluation)")

    # --- Save ---
    train.to_parquet(PROCESSED_DIR / "train_combined.parquet")
    val.to_parquet(PROCESSED_DIR / "val_combined.parquet")
    test.to_parquet(PROCESSED_DIR / "test_combined.parquet")

    print(f"\n✅ Combined datasets saved to: {PROCESSED_DIR}")
    print("   - train_combined.parquet")
    print("   - val_combined.parquet")
    print("   - test_combined.parquet")
    print("\n✅ Ready for Feature Extraction & Training!")

if __name__ == "__main__":
    main()
'''

# ------------------------------- New script for 50k sentences -----------------------


"""
Step 3: Merge Human and AI Datasets (Sampled to 50,000)
"""

import pandas as pd
from sklearn.model_selection import train_test_split
from pathlib import Path

# Paths
RAW_DIR = Path("data/raw")
PROCESSED_DIR = Path("data/processed")
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

# --- CONFIGURATION ---
TARGET_HUMAN = 25000
TARGET_AI = 25000
RANDOM_SEED = 42
TEST_SIZE = 0.2
VAL_SIZE = 0.1  # 10% of total for validation
# --------------------

def main():
    print("=" * 60)
    print(f"STEP 3: Merging Human + AI Datasets (Sampled to {TARGET_HUMAN + TARGET_AI:,} total)")
    print("=" * 60)
    
    # 1. Load datasets
    print("\n📂 Loading datasets...")
    human_df = pd.read_csv(RAW_DIR / "human_text.csv")
    ai_df = pd.read_csv(RAW_DIR / "ai_text_repaired.csv")
    
    print(f"   Human: {len(human_df):,} rows")
    print(f"   AI:    {len(ai_df):,} rows")
    
    # 2. Sample exactly 25,000 each
    print(f"\n🎯 Sampling {TARGET_HUMAN:,} Human + {TARGET_AI:,} AI sentences...")
    
    if len(human_df) > TARGET_HUMAN:
        human_sample = human_df.sample(n=TARGET_HUMAN, random_state=RANDOM_SEED)
    else:
        human_sample = human_df
        print(f"   ⚠️  Only {len(human_df):,} humans available. Using all.")
    
    if len(ai_df) > TARGET_AI:
        ai_sample = ai_df.sample(n=TARGET_AI, random_state=RANDOM_SEED)
    else:
        ai_sample = ai_df
        print(f"   ⚠️  Only {len(ai_df):,} AI texts available. Using all.")
    
    print(f"   Human sampled: {len(human_sample):,}")
    print(f"   AI sampled:    {len(ai_sample):,}")
    
    # 3. Add label column (if not already present)
    human_sample['label'] = 0
    ai_sample['label'] = 1
    
    # 4. Combine
    combined = pd.concat([human_sample, ai_sample], ignore_index=True)
    print(f"\n✅ Combined: {len(combined):,} total rows")
    
    # 5. Check if 'text' column exists, else use 'sentence'
    if 'text' in combined.columns:
        text_col = 'text'
    else:
        text_col = 'sentence'
    
    # 6. Keep only text + label for training
    combined_clean = combined[[text_col, 'label']].rename(columns={text_col: 'sentence'})
    
    # 7. RANDOM STRATIFIED SPLIT (preserves 25k/25k ratio)
    print("\n📊 Splitting data randomly (stratified by label)...")
    
    # First split: Train (80%) vs Temp (20%)
    train, temp = train_test_split(
        combined_clean,
        test_size=TEST_SIZE,
        random_state=RANDOM_SEED,
        stratify=combined_clean['label']  # Ensures balanced labels in each split
    )
    
    # Second split: Val (50% of Temp = 10% of total) vs Test (50% of Temp = 10% of total)
    val, test = train_test_split(
        temp,
        test_size=0.5,
        random_state=RANDOM_SEED,
        stratify=temp['label']  # Ensures balanced labels in val/test
    )
    
    # 8. Shuffle
    train = train.sample(frac=1, random_state=RANDOM_SEED).reset_index(drop=True)
    val = val.sample(frac=1, random_state=RANDOM_SEED).reset_index(drop=True)
    test = test.sample(frac=1, random_state=RANDOM_SEED).reset_index(drop=True)
    
    # 9. Print stats
    print(f"\n📊 Split Results:")
    print(f"   Train: {len(train):,} rows (Human: {len(train[train['label']==0]):,}, AI: {len(train[train['label']==1]):,})")
    print(f"   Val:   {len(val):,} rows (Human: {len(val[val['label']==0]):,}, AI: {len(val[val['label']==1]):,})")
    print(f"   Test:  {len(test):,} rows (Human: {len(test[test['label']==0]):,}, AI: {len(test[test['label']==1]):,})")
    
    # 10. Save
    train.to_parquet(PROCESSED_DIR / "train_combined.parquet")
    val.to_parquet(PROCESSED_DIR / "val_combined.parquet")
    test.to_parquet(PROCESSED_DIR / "test_combined.parquet")
    
    print(f"\n✅ Combined datasets saved to: {PROCESSED_DIR}")
    print("   - train_combined.parquet")
    print("   - val_combined.parquet")
    print("   - test_combined.parquet")
    print(f"\n✅ Ready for Feature Extraction!")

if __name__ == "__main__":
    main()