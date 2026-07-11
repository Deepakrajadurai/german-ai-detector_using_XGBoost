"""
Step 6: Test the Trained Model on New Text
"""


import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["OMP_NUM_THREADS"] = "1"
import re
import pickle
import json
import numpy as np
from collections import Counter
from pathlib import Path
import spacy

# Load German spaCy model
nlp = spacy.load("de_core_news_sm")


# ============================================================
# LEGAL FEATURE EXTRACTOR (Copy from 04_feature_extraction.py)
# ============================================================
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

    def extract_features(self, text):
        """Extract all 18 features from a single sentence."""
        if not text or len(text.strip()) < 3:
            return self._default_features()
        
        doc = nlp(text)
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


# ============================================================
# PREDICTION FUNCTION
# ============================================================
def load_model():
    """Load the trained XGBoost model and metadata."""
    with open("models/xgboost_model_optimized.pkl", "rb") as f:
        model = pickle.load(f)
    
    with open("models/model_metadata_optimized.json", "r") as f:
        metadata = json.load(f)
    
    return model, metadata


def predict_text(text, model, metadata, extractor):
    """
    Predict whether a German text is Human (0) or AI (1).
    
    Returns:
        dict with prediction, confidence, and probabilities
    """
    # Extract features
    features = extractor.extract_features(text)
    feature_names = metadata['feature_names']
    
    # Convert to DataFrame with proper feature order
    import pandas as pd
    feature_df = pd.DataFrame([features])[feature_names]
    
    # Get probability
    proba_ai = model.predict_proba(feature_df)[0, 1]  # Probability of being AI
    
    # Apply threshold
    threshold = metadata.get('threshold', 0.5)
    label = "AI-Generated" if proba_ai >= threshold else "Human-Written"
    confidence = proba_ai if label == "AI-Generated" else 1 - proba_ai
    
    return {
        'text': text[:200] + "..." if len(text) > 200 else text,
        'label': label,
        'confidence': confidence * 100,
        'probability_ai': proba_ai * 100,
        'threshold': threshold * 100
    }


# ============================================================
# MAIN (Interactive & Command-Line)
# ============================================================
def main():
    print("=" * 60)
    print("GERMAN AI DETECTOR - TEST MODE")
    print("=" * 60)
    
    # Load model
    try:
        model, metadata = load_model()
        extractor = LegalFeatureExtractor()
        print(f"Model loaded successfully!")
        print(f"   Threshold: {metadata.get('threshold', 0.5)*100:.1f}%")
        print(f"   Precision on validation: {metadata['performance']['precision']:.2%}")
        print(f"   ROC-AUC on validation: {metadata['performance']['roc_auc']:.2%}")
        print("=" * 60)
    except FileNotFoundError as e:
        print(f"Error: {e}")
        print("   Make sure you've run 05_train_xgboost.py first!")
        return
    
    # If command-line argument provided, predict it
    import sys
    if len(sys.argv) > 1:
        text = " ".join(sys.argv[1:])
        result = predict_text(text, model, metadata, extractor)
        print("\nINPUT TEXT:")
        print(f"   {result['text']}")
        print("\nPREDICTION:")
        print(f"   Label: {result['label']}")
        print(f"   Confidence: {result['confidence']:.1f}%")
        print(f"   AI Probability: {result['probability_ai']:.1f}%")
        print(f"   Threshold: {result['threshold']:.1f}%")
        return
    
    # Interactive mode
    print("\nEnter a German sentence from Law/Public Admin domain.")
    print("   Type 'quit' to exit.\n")
    
    while True:
        text = input("> ").strip()
        if text.lower() in ['quit', 'exit', 'q']:
            print("Goodbye!")
            break
        if not text:
            continue
        
        result = predict_text(text, model, metadata, extractor)
        print("\n" + "-" * 40)
        print(f"Text: {result['text']}")
        print(f"Prediction: {result['label']}")
        print(f"   Confidence: {result['confidence']:.1f}%")
        print(f"   AI Probability: {result['probability_ai']:.1f}%")
        print("-" * 40 + "\n")


if __name__ == "__main__":
    main()