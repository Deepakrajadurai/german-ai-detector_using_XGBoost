"""
Ingest data/training_pair_v5_clean.csv, split paragraphs into individual sentences,
filter out fragments, sample a balanced subset, and save train/val/test splits.
"""

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["OMP_NUM_THREADS"] = "1"

import pandas as pd
import numpy as np
import spacy
from sklearn.model_selection import train_test_split
from pathlib import Path
from tqdm import tqdm

DATA_DIR = Path("data")
PROCESSED_DIR = Path("data/processed")
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

TARGET_SENTENCES = 150000 # 150k human and 150k AI sentences

def split_paragraphs_to_sentences(texts, nlp):
    """Split a list of paragraph texts into individual clean sentences using spaCy."""
    sentences = []
    # Using nlp.pipe is fast for batch sentence splitting
    # Disable heavy pipeline components for speed
    for doc in tqdm(nlp.pipe(texts, disable=["tagger", "parser", "ner", "attribute_ruler", "lemmatizer"]), total=len(texts)):
        # Since parser is disabled, we can use sentencizer for fast sentence boundary detection
        for sent in doc.sents:
            sent_text = sent.text.strip()
            # Basic sanity check: sentence should have at least 5 words and not look like noise
            words = sent_text.split()
            if len(words) >= 5:
                sentences.append(sent_text)
    return sentences

def main():
    print("=" * 60)
    print("SENTENCE SPLITTING & BALANCING PIPELINE")
    print("=" * 60)
    
    csv_path = DATA_DIR / "training_pair_v5_clean.csv"
    print(f"Loading paragraph dataset from {csv_path}...")
    df = pd.read_csv(csv_path)
    
    # Drop rows with NaN texts
    df = df.dropna(subset=['text'])
    df = df[df['text'].astype(str).str.strip() != ""]
    
    print(f"Total paragraphs loaded: {len(df):,}")
    print(f"  Human paragraphs: {sum(df['label'] == 0):,}")
    print(f"  AI paragraphs:    {sum(df['label'] == 1):,}")
    
    # Load spaCy and add sentencizer
    print("Initializing spaCy sentencizer...")
    nlp = spacy.blank("de")
    nlp.add_pipe("sentencizer")
    
    # We sample a subset of paragraphs of each class to speed up sentence splitting
    # 45,000 paragraphs of each class will yield ~180k-220k sentences each, which is more than enough
    num_paras_to_sample = 45000
    
    print(f"Sampling {num_paras_to_sample:,} paragraphs of each class...")
    human_paras = df[df['label'] == 0]['text'].sample(n=num_paras_to_sample, random_state=42).tolist()
    ai_paras = df[df['label'] == 1]['text'].sample(n=num_paras_to_sample, random_state=42).tolist()
    
    print("\nSplitting human paragraphs into sentences...")
    human_sentences = split_paragraphs_to_sentences(human_paras, nlp)
    print(f"Extracted {len(human_sentences):,} clean human sentences.")
    
    print("\nSplitting AI paragraphs into sentences...")
    ai_sentences = split_paragraphs_to_sentences(ai_paras, nlp)
    print(f"Extracted {len(ai_sentences):,} clean AI sentences.")
    
    # Check if we have enough sentences
    if len(human_sentences) < TARGET_SENTENCES or len(ai_sentences) < TARGET_SENTENCES:
        raise ValueError(f"Not enough sentences extracted! Need at least {TARGET_SENTENCES:,} each.")
    
    # Sample exactly TARGET_SENTENCES of each class
    print(f"\nSampling exactly {TARGET_SENTENCES:,} sentences of each class for balancing...")
    np.random.seed(42)
    selected_human = np.random.choice(human_sentences, size=TARGET_SENTENCES, replace=False)
    selected_ai = np.random.choice(ai_sentences, size=TARGET_SENTENCES, replace=False)
    
    # Combine into a single DataFrame
    human_df = pd.DataFrame({'sentence': selected_human, 'label': 0})
    ai_df = pd.DataFrame({'sentence': selected_ai, 'label': 1})
    combined = pd.concat([human_df, ai_df], ignore_index=True)
    
    # Stratified Train/Val/Test Split
    print("\nPerforming stratified split...")
    train, temp = train_test_split(combined, test_size=0.2, random_state=42, stratify=combined['label'])
    val, test = train_test_split(temp, test_size=0.5, random_state=42, stratify=temp['label'])
    
    # Shuffle splits
    train = train.sample(frac=1, random_state=42).reset_index(drop=True)
    val = val.sample(frac=1, random_state=42).reset_index(drop=True)
    test = test.sample(frac=1, random_state=42).reset_index(drop=True)
    
    print(f"  Train set: {len(train):,} sentences")
    print(f"  Val set:   {len(val):,} sentences")
    print(f"  Test set:  {len(test):,} sentences")
    
    # Save to data/processed
    print("\nSaving sentence splits to data/processed...")
    train.to_parquet(PROCESSED_DIR / "train_sentences.parquet")
    val.to_parquet(PROCESSED_DIR / "val_sentences.parquet")
    test.to_parquet(PROCESSED_DIR / "test_sentences.parquet")
    
    print("✅ Sentence splitting completed successfully!")

if __name__ == "__main__":
    main()
