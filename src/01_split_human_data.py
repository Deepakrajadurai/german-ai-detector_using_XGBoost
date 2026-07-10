import pandas as pd
from sklearn.model_selection import train_test_split
from pathlib import Path

# Paths
RAW_DATA = Path("data/raw/human_data.csv")
INTERIM_DIR = Path("data/interim")

# Create interim folder if it doesn't exist
INTERIM_DIR.mkdir(parents=True, exist_ok=True)

def main():
    print("=" * 60)
    print("STEP 1: Splitting Human Data by Speaker")
    print("=" * 60)

    # 1. Load the raw CSV
    df = pd.read_csv(RAW_DATA)
    print(f"✅ Loaded {len(df):,} human sentences")

    # 2. Keep only necessary columns and rename 'text' to 'sentence'
    df = df[['text', 'speaker']].rename(columns={'text': 'sentence'})
    
    # 3. Drop any rows with missing speakers or empty sentences
    df = df.dropna(subset=['sentence', 'speaker'])
    df = df[df['sentence'].str.strip() != '']
    print(f"✅ Cleaned: {len(df):,} sentences remaining")

    # 4. Split by SPEAKER to prevent data leakage
    speakers = df['speaker'].unique().tolist()
    print(f"✅ Total unique speakers: {len(speakers)}")

    train_speakers, temp_speakers = train_test_split(
        speakers, test_size=0.2, random_state=42
    )
    val_speakers, test_speakers = train_test_split(
        temp_speakers, test_size=0.5, random_state=42
    )

    # 5. Assign sentences based on speaker
    train = df[df['speaker'].isin(train_speakers)]
    val = df[df['speaker'].isin(val_speakers)]
    test = df[df['speaker'].isin(test_speakers)]

    print(f"\n📊 Split Results:")
    print(f"   Train: {len(train):,} sentences ({len(train_speakers)} speakers)")
    print(f"   Val:   {len(val):,} sentences ({len(val_speakers)} speakers)")
    print(f"   Test:  {len(test):,} sentences ({len(test_speakers)} speakers)")

    # 6. Save to interim folder (only sentence + speaker, label=0 implicit)
    train[['sentence']].to_parquet(INTERIM_DIR / "train_human.parquet")
    val[['sentence']].to_parquet(INTERIM_DIR / "val_human.parquet")
    test[['sentence']].to_parquet(INTERIM_DIR / "test_human.parquet")

    print(f"\n✅ Human splits saved to: {INTERIM_DIR}")
    print("   - train_human.parquet")
    print("   - val_human.parquet")
    print("   - test_human.parquet")

if __name__ == "__main__":
    main()