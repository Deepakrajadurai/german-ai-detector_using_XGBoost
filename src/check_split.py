"""
Check if the same source appears in Train and Test.
"""

import pandas as pd
from pathlib import Path

# Load the original human data
human = pd.read_csv("data/raw/human_text.csv")

# Load the final splits
train = pd.read_parquet("data/processed/train_combined.parquet")
val = pd.read_parquet("data/processed/val_combined.parquet")
test = pd.read_parquet("data/processed/test_combined.parquet")

print("=" * 60)
print("Checking Source Overlap Across Splits")
print("=" * 60)

# For human sentences, we need to know which source they came from
# Create a mapping of text -> source (only for humans)
human_map = {}
for idx, row in human.iterrows():
    text = row['text'].strip()
    source = row['source'] if 'source' in row else row.get('domain', 'unknown')
    human_map[text] = source

# Get sources from each split
train_sources = set()
val_sources = set()
test_sources = set()

for sent in train[train['label'] == 0]['sentence']:
    if sent in human_map:
        train_sources.add(human_map[sent])

for sent in val[val['label'] == 0]['sentence']:
    if sent in human_map:
        val_sources.add(human_map[sent])

for sent in test[test['label'] == 0]['sentence']:
    if sent in human_map:
        test_sources.add(human_map[sent])

print(f"\n📊 Unique sources per split:")
print(f"   Train: {len(train_sources)} sources")
print(f"   Val:   {len(val_sources)} sources")
print(f"   Test:  {len(test_sources)} sources")

print(f"\n📁 Sources:")
print(f"   Train: {train_sources}")
print(f"   Val:   {val_sources}")
print(f"   Test:  {test_sources}")

# Check overlap
overlap_train_test = train_sources & test_sources
overlap_train_val = train_sources & val_sources

print(f"\n🔄 Source overlap:")
print(f"   Train ∩ Val:  {len(overlap_train_val)} sources")
print(f"   Train ∩ Test: {len(overlap_train_test)} sources")

if len(overlap_train_test) > 0:
    print(f"\n⚠️  Same sources in Train and Test: {overlap_train_test}")
    print("   This is expected because we did a random split.")
    print("   For the full run, we can split by source to prevent leakage.")
else:
    print(f"\n✅ No source overlap! Good for generalization.")

print(f"\n📊 AI sentences distribution:")
print(f"   Train: {len(train[train['label']==1]):,}")
print(f"   Val:   {len(val[val['label']==1]):,}")
print(f"   Test:  {len(test[test['label']==1]):,}")