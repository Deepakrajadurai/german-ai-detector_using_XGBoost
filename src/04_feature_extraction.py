# ---------------------------- OLD Script for sample data of 2500 ---------------------------

# """
# Step 4: Extract 18 Linguistic Features for Law/Admin German Texts.
# """

# import pandas as pd
# import re
# import numpy as np
# from collections import Counter
# from pathlib import Path
# from tqdm import tqdm
# import spacy

# # Load German spaCy model
# nlp = spacy.load("de_core_news_sm")

# # Paths
# PROCESSED_DIR = Path("data/processed")
# FEATURES_DIR = Path("data/features")
# FEATURES_DIR.mkdir(parents=True, exist_ok=True)


# class LegalFeatureExtractor:
#     """Extract 18 features for legal/administrative German sentences."""
    
#     def __init__(self):
#         # 1. Legal citation patterns
#         self.citation_pattern = re.compile(
#             r'[§§]?\s*\d+\s*(?:[A-Z]+)?\s*(?:Abs\.\s*\d+)?\s*(?:S\.\s*\d+)?|'
#             r'[Aa]rt\.\s*\d+|'
#             r'Rn\.\s*\d+|'
#             r'[Aa]\.?[Aa]\.?|'
#             r'[Ii]\.?[Vv]\.?[Mm]\.?'
#         )
#         # 6. Legal jargon indicators
#         self.legal_terms = [
#             'ermessen', 'verwaltungsakt', 'behörde', 'gericht', 'klage',
#             'urteil', 'beschluss', 'verfahren', 'rechts', 'gesetz',
#             'verordnung', 'satzung', 'bescheid', 'widerspruch'
#         ]
#         # 5. Modal particles (German-specific)
#         self.modal_particles = ['ja', 'doch', 'halt', 'eben', 'mal', 'schon', 'wohl']
#         # 4. Passive voice indicators
#         self.passive_indicators = ['wurde', 'wird', 'werden', 'worden', 'würde']
#         # 11. Function words
#         self.function_words = ['der', 'die', 'das', 'und', 'oder', 'aber', 'den', 'dem']
#         # 7. Authority terms
#         self.authority_terms = ['behörde', 'gericht', 'amt', 'ministerium']
#         # 8. Formulaic closings
#         self.closings = ['mit freundlichen grüßen', 'im auftrag', 'hochachtungsvoll']

#     def extract_features(self, text):
#         """Extract all 18 features from a single sentence."""
#         if not text or len(text.strip()) < 3:
#             return self._default_features()
        
#         doc = nlp(text)
#         words = [token.text for token in doc]
#         word_count = len(words)
        
#         if word_count == 0:
#             return self._default_features()
        
#         # Convert to lowercase for matching
#         text_lower = text.lower()
        
#         features = {}
        
#         # 1. Citation Density
#         citations = self.citation_pattern.findall(text)
#         features['citation_density'] = len(citations) / word_count
        
#         # 2. Paragraph Structure Entropy
#         structure_markers = len(re.findall(r'\b[IVX]+\.|\b\d+\.|\([a-z]\)', text))
#         features['structure_entropy'] = structure_markers / word_count
        
#         # 3. Nominalization Ratio (words ending in -ung, -heit, -keit, -tion, -sion)
#         nominalizations = len(re.findall(r'\b\w+(?:ung|heit|keit|tion|sion)\b', text_lower))
#         features['nominalization_ratio'] = nominalizations / word_count
        
#         # 4. Passive Voice Ratio
#         passive_count = sum(1 for token in doc if token.text in self.passive_indicators)
#         features['passive_ratio'] = passive_count / word_count
        
#         # 5. Modal Particle Ratio
#         modal_count = sum(1 for token in doc if token.text in self.modal_particles)
#         features['modal_particle_ratio'] = modal_count / word_count
        
#         # 6. Jargon Consistency
#         jargon_count = sum(1 for term in self.legal_terms if term in text_lower)
#         features['jargon_consistency'] = jargon_count / word_count
        
#         # 7. Authority Ratio
#         authority_count = sum(1 for term in self.authority_terms if term in text_lower)
#         features['authority_ratio'] = authority_count / word_count
        
#         # 8. Closing Ratio
#         closing_count = sum(1 for closing in self.closings if closing in text_lower)
#         features['closing_ratio'] = closing_count / word_count
        
#         # 9. Average Word Length
#         word_lengths = [len(token.text) for token in doc if token.is_alpha]
#         features['avg_word_length'] = np.mean(word_lengths) if word_lengths else 0
        
#         # 10. Type-Token Ratio (Lexical Diversity)
#         unique_words = len(set([token.text.lower() for token in doc if token.is_alpha]))
#         features['type_token_ratio'] = unique_words / word_count if word_count > 0 else 0
        
#         # 11. Function Word Ratio
#         function_count = sum(1 for token in doc if token.text.lower() in self.function_words)
#         features['function_word_ratio'] = function_count / word_count
        
#         # 12. Capitalization Ratio (German-specific)
#         capitalized = sum(1 for token in doc if token.text[0].isupper() if token.text)
#         features['capitalization_ratio'] = capitalized / word_count if word_count > 0 else 0
        
#         # 13. Abbreviation Density
#         abbreviations = len(re.findall(r'\b[A-ZÄÖÜ]+\.\b|\b[A-ZÄÖÜ]{2,}\b', text))
#         features['abbreviation_ratio'] = abbreviations / word_count
        
#         # 14. Parenthetical Ratio
#         parentheses = len(re.findall(r'\([^)]*\)', text))
#         features['parenthetical_ratio'] = parentheses / word_count
        
#         # 15. Punctuation Entropy
#         punct_chars = re.findall(r'[.,:;!?]', text)
#         if punct_chars:
#             counts = Counter(punct_chars)
#             total = len(punct_chars)
#             probs = [count/total for count in counts.values()]
#             features['punctuation_entropy'] = -sum(p * np.log(p) for p in probs)
#         else:
#             features['punctuation_entropy'] = 0
        
#         # 16. Clause Density (subordinating conjunctions)
#         subjunctions = ['dass', 'weil', 'da', 'wenn', 'obwohl', 'während', 'bevor']
#         clause_count = sum(1 for token in doc if token.text.lower() in subjunctions)
#         features['clause_density'] = clause_count / word_count if word_count > 0 else 0
        
#         # 17. Word Count (sentence length)
#         features['word_count'] = word_count
        
#         # 18. "man" Ratio (German-specific indefinite pronoun)
#         man_count = sum(1 for token in doc if token.text.lower() == 'man')
#         features['man_ratio'] = man_count / word_count if word_count > 0 else 0
        
#         return features
    
#     def _default_features(self):
#         """Return default features for empty text."""
#         return {
#             'citation_density': 0,
#             'structure_entropy': 0,
#             'nominalization_ratio': 0,
#             'passive_ratio': 0,
#             'modal_particle_ratio': 0,
#             'jargon_consistency': 0,
#             'authority_ratio': 0,
#             'closing_ratio': 0,
#             'avg_word_length': 0,
#             'type_token_ratio': 0,
#             'function_word_ratio': 0,
#             'capitalization_ratio': 0,
#             'abbreviation_ratio': 0,
#             'parenthetical_ratio': 0,
#             'punctuation_entropy': 0,
#             'clause_density': 0,
#             'word_count': 0,
#             'man_ratio': 0
#         }


# def extract_features_for_dataset(df, split_name):
#     """Extract features for all sentences in a dataset."""
#     extractor = LegalFeatureExtractor()
    
#     features_list = []
#     for text in tqdm(df['sentence'], desc=f"Extracting {split_name}"):
#         features = extractor.extract_features(text)
#         features_list.append(features)
    
#     feature_df = pd.DataFrame(features_list)
#     feature_df['label'] = df['label'].values
    
#     return feature_df


# def main():
#     print("=" * 60)
#     print("STEP 4: Extracting 18 Linguistic Features")
#     print("=" * 60)
    
#     # Load combined datasets
#     train = pd.read_parquet(PROCESSED_DIR / "train_combined.parquet")
#     val = pd.read_parquet(PROCESSED_DIR / "val_combined.parquet")
    
#     print(f"Train: {len(train):,} sentences")
#     print(f"Val:   {len(val):,} sentences")
    
#     # Extract features
#     train_features = extract_features_for_dataset(train, "Train")
#     val_features = extract_features_for_dataset(val, "Val")
    
#     # Save
#     train_features.to_parquet(FEATURES_DIR / "train_features.parquet")
#     val_features.to_parquet(FEATURES_DIR / "val_features.parquet")
    
#     print(f"\n✅ Features saved to: {FEATURES_DIR}")
#     print(f"   - train_features.parquet ({len(train_features):,} rows, {len(train_features.columns)-1} features)")
#     print(f"   - val_features.parquet ({len(val_features):,} rows, {len(val_features.columns)-1} features)")
    
#     # Show feature statistics
#     print("\n📊 Feature Statistics (Train):")
#     feature_cols = [col for col in train_features.columns if col != 'label']
#     for col in feature_cols[:5]:  # Show first 5
#         print(f"   {col}: mean={train_features[col].mean():.4f}, std={train_features[col].std():.4f}")


# if __name__ == "__main__":
#     main()



# -------------------------------- NEW SCRIPT FOR 50K Sentences -----------------------


"""
Step 4: Extract 18 Linguistic Features (Optimized with nlp.pipe)
"""

import pandas as pd
import re
import numpy as np
from collections import Counter
from pathlib import Path
from tqdm import tqdm
import spacy

# Load German spaCy model
nlp = spacy.load("de_core_news_sm")

# Paths
PROCESSED_DIR = Path("data/processed")
FEATURES_DIR = Path("data/features")
FEATURES_DIR.mkdir(parents=True, exist_ok=True)


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

    def extract_features_batch(self, texts):
        """Extract features for a batch of texts using nlp.pipe."""
        features_list = []
        
        # Process in batches using spaCy's pipe (much faster)
        for doc in tqdm(nlp.pipe(texts, batch_size=500, n_process=1), total=len(texts)):
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
    print("STEP 4: Extracting 18 Linguistic Features (Optimized)")
    print("=" * 60)
    
    # Load combined datasets
    train = pd.read_parquet(PROCESSED_DIR / "train_combined.parquet")
    val = pd.read_parquet(PROCESSED_DIR / "val_combined.parquet")
    test = pd.read_parquet(PROCESSED_DIR / "test_combined.parquet")
    
    print(f"Train: {len(train):,} sentences")
    print(f"Val:   {len(val):,} sentences")
    print(f"Test:  {len(test):,} sentences")
    
    extractor = LegalFeatureExtractor()
    
    # Extract features in batch mode (much faster!)
    print("\n🔄 Extracting features for Train...")
    train_features = extractor.extract_features_batch(train['sentence'].tolist())
    train_features['label'] = train['label'].values
    
    print("\n🔄 Extracting features for Val...")
    val_features = extractor.extract_features_batch(val['sentence'].tolist())
    val_features['label'] = val['label'].values
    
    print("\n🔄 Extracting features for Test...")
    test_features = extractor.extract_features_batch(test['sentence'].tolist())
    test_features['label'] = test['label'].values
    
    # Save
    train_features.to_parquet(FEATURES_DIR / "train_features.parquet")
    val_features.to_parquet(FEATURES_DIR / "val_features.parquet")
    test_features.to_parquet(FEATURES_DIR / "test_features.parquet")
    
    print(f"\n✅ Features saved to: {FEATURES_DIR}")
    print(f"   - train_features.parquet ({len(train_features):,} rows, {len(train_features.columns)-1} features)")
    print(f"   - val_features.parquet ({len(val_features):,} rows, {len(val_features.columns)-1} features)")
    print(f"   - test_features.parquet ({len(test_features):,} rows, {len(test_features.columns)-1} features)")
    
    print("\n📊 Feature Statistics (Train):")
    feature_cols = [col for col in train_features.columns if col != 'label']
    for col in feature_cols[:5]:
        print(f"   {col}: mean={train_features[col].mean():.4f}, std={train_features[col].std():.4f}")

if __name__ == "__main__":
    main()