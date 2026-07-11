"""
Streamlit Web App for German AI Detector (XGBoost).
Provides interactive user testing for administrative/legal German texts.
"""

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["OMP_NUM_THREADS"] = "1"

import streamlit as st
import pickle
import json
import numpy as np
import pandas as pd
import spacy
from collections import Counter
import re
import matplotlib.pyplot as plt

# Page config
st.set_page_config(
    page_title="German AI Detector - Interactive Testing",
    page_icon="🇩🇪",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Paths
MODEL_PATH = "models/xgboost_model_optimized.pkl"
METADATA_PATH = "models/model_metadata_optimized.json"

# Preset examples
EXAMPLES = {
    "--- Select an Example ---": "",
    "Example 1: Human-Written (VwVfG legal reference)": (
        "Ein Verwaltungsakt ist jede Verfügung, Entscheidung oder andere behördliche Maßnahme, "
        "die eine Behörde zur Regelung eines Einzelfalls auf dem Gebiet des öffentlichen Rechts "
        "trifft und die auf unmittelbare Rechtswirkung nach außen gerichtet ist. Dies ergibt sich aus "
        "§ 35 Satz 1 des Verwaltungsverfahrensgesetzes (VwVfG)."
    ),
    "Example 2: AI-Generated (Smooth flow, typical AI style)": (
        "Es ist von entscheidender Bedeutung zu betonen, dass der vorliegende Gesetzentwurf "
        "eine wesentliche Verbesserung für alle Bürgerinnen und Bürger darstellt. Durch die gezielte "
        "Förderung von zukunftsfähigen Infrastrukturprojekten schaffen wir die Grundlage für ein "
        "nachhaltiges Wirtschaftswachstum und sichern somit langfristig den Wohlstand unseres Landes."
    ),
    "Example 3: Human-Written (Administrative notification)": (
        "Ihrem Widerspruch vom 12. Februar gegen den Bescheid über die Festsetzung der Grundsteuer "
        "wird hiermit abgeholfen. Der angefochtene Bescheid wird aufgehoben. Ein neuer Bescheid "
        "geht Ihnen in den nächsten Tagen per Post zu. Die Kosten des Widerspruchsverfahrens trägt "
        "die Staatskasse. Mit freundlichen Grüßen, im Auftrag."
    ),
    "Example 4: AI-Generated (More clinical administrative text)": (
        "Im Rahmen der Prüfung der Zulässigkeit des Antrags wurde festgestellt, dass die gesetzlich "
        "vorgeschriebenen Voraussetzungen für eine positive Entscheidung derzeit nicht erfüllt sind. "
        "Daher wird dem Antragsteller dringend nahegelegt, die fehlenden Nachweise und Unterlagen "
        "innerhalb der gesetzten Frist von zwei Wochen nachzureichen, um eine Ablehnung abzuwenden."
    )
}


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

    def extract_features(self, text, nlp):
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
        
        features['citation_density'] = len(self.citation_pattern.findall(text)) / word_count
        features['structure_entropy'] = len(re.findall(r'\b[IVX]+\.|\b\d+\.|\([a-z]\)', text)) / word_count
        features['nominalization_ratio'] = len(re.findall(r'\b\w+(?:ung|heit|keit|tion|sion)\b', text_lower)) / word_count
        features['passive_ratio'] = sum(1 for token in doc if token.text in self.passive_indicators) / word_count
        features['modal_particle_ratio'] = sum(1 for token in doc if token.text in self.modal_particles) / word_count
        features['jargon_consistency'] = sum(1 for term in self.legal_terms if term in text_lower) / word_count
        features['authority_ratio'] = sum(1 for term in self.authority_terms if term in text_lower) / word_count
        features['closing_ratio'] = sum(1 for closing in self.closings if closing in text_lower) / word_count
        
        word_lengths = [len(token.text) for token in doc if token.is_alpha]
        features['avg_word_length'] = np.mean(word_lengths) if word_lengths else 0
        
        unique_words = len(set([token.text.lower() for token in doc if token.is_alpha]))
        features['type_token_ratio'] = unique_words / word_count if word_count > 0 else 0
        
        features['function_word_ratio'] = sum(1 for token in doc if token.text.lower() in self.function_words) / word_count
        
        capitalized = sum(1 for token in doc if token.text[0].isupper() if token.text)
        features['capitalization_ratio'] = capitalized / word_count if word_count > 0 else 0
        
        features['abbreviation_ratio'] = len(re.findall(r'\b[A-ZÄÖÜ]+\.\b|\b[A-ZÄÖÜ]{2,}\b', text)) / word_count
        features['parenthetical_ratio'] = len(re.findall(r'\([^)]*\)', text)) / word_count
        
        punct_chars = re.findall(r'[.,:;!?]', text)
        if punct_chars:
            counts = Counter(punct_chars)
            total = len(punct_chars)
            probs = [count/total for count in counts.values()]
            features['punctuation_entropy'] = -sum(p * np.log(p) for p in probs)
        else:
            features['punctuation_entropy'] = 0
        
        subjunctions = ['dass', 'weil', 'da', 'wenn', 'obwohl', 'während', 'bevor']
        features['clause_density'] = sum(1 for token in doc if token.text.lower() in subjunctions) / word_count
        features['word_count'] = word_count
        features['man_ratio'] = sum(1 for token in doc if token.text.lower() == 'man') / word_count
        
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


# Cache spaCy and model
@st.cache_resource
def load_nlp():
    return spacy.load("de_core_news_sm")

@st.cache_resource
def load_xgboost_model():
    if not os.path.exists(MODEL_PATH) or not os.path.exists(METADATA_PATH):
        return None, None
    with open(MODEL_PATH, "rb") as f:
        model = pickle.load(f)
    with open(METADATA_PATH, "r") as f:
        metadata = json.load(f)
    return model, metadata


# Initialize
nlp = load_nlp()
model, metadata = load_xgboost_model()
extractor = LegalFeatureExtractor()

# CSS adjustments for cleaner design
st.markdown("""
<style>
    .prediction-card {
        padding: 24px;
        border-radius: 12px;
        color: white;
        text-align: center;
        margin-bottom: 20px;
    }
    .human-card {
        background-color: #2e7d32;
        border: 2px solid #1b5e20;
    }
    .ai-card {
        background-color: #c62828;
        border: 2px solid #b71c1c;
    }
</style>
""", unsafe_allow_html=True)

# Main Title Layout
st.title("🇩🇪 German AI Detector (XGBoost)")
st.subheader("Interactive User Testing for Administrative & Legal Texts")
st.write(
    "Detect whether a German sentence is **Human-Written** or **AI-Generated** using an optimized XGBoost classifier "
    "trained on handcrafted linguistic and syntactic features."
)

# Sidebar
st.sidebar.header("📊 Model Specifications")
if metadata:
    st.sidebar.metric(label="Model Type", value="XGBoost Classifier")
    st.sidebar.metric(label="Total Dataset Size", value="435,178 rows")
    st.sidebar.metric(label="Test Set Accuracy", value=f"{metadata['performance']['accuracy']:.2%}")
    st.sidebar.metric(label="FPR (False Positive Rate)", value=f"{metadata['performance']['fpr']:.2%}")
    st.sidebar.metric(label="Detection Threshold", value=f"{metadata['threshold']:.2f}")
    
    st.sidebar.markdown("---")
    st.sidebar.write("**Extracted Features**:")
    for f in metadata['feature_names']:
        st.sidebar.markdown(f"- `{f}`")
else:
    st.sidebar.warning("Model and metadata files not found! Make sure you ran training and evaluation.")

# Main content splits into input and output
col1, col2 = st.columns([1, 1])

with col1:
    st.header("📝 User Input")
    
    # Preset Examples dropdown
    selected_preset = st.selectbox(
        "Load a preset example sentence:",
        options=list(EXAMPLES.keys())
    )
    
    # Text input
    default_text = EXAMPLES[selected_preset] if selected_preset != "--- Select an Example ---" else ""
    user_text = st.text_area(
        "Enter/paste your German text to test here:",
        value=default_text,
        height=200,
        placeholder="Schreiben Sie hier Ihren deutschen Text..."
    )
    
    detect_clicked = st.button("Detect AI Probability", type="primary", use_container_width=True)

with col2:
    st.header("🔮 Prediction & Explanation")
    
    if (detect_clicked or user_text) and user_text.strip():
        if model is None or metadata is None:
            st.error("Model files are not loaded correctly. Ensure training has completed.")
        else:
            with st.spinner("Extracting features and predicting..."):
                # Extract features
                features_dict = extractor.extract_features(user_text, nlp)
                features = pd.DataFrame([features_dict])[metadata['feature_names']]
                
                # Print requested checks immediately before prediction
                print(features.columns.tolist())
                print(features)
                print(features.dtypes)
                print(model.get_booster().feature_names)
                
                # Predict
                proba_ai = model.predict_proba(features)[0, 1]
                threshold = metadata['threshold']
                is_ai = proba_ai >= threshold
                confidence = proba_ai if is_ai else (1.0 - proba_ai)
                
                # Predict class card styling
                if is_ai:
                    st.markdown(
                        f'<div class="prediction-card ai-card">'
                        f'<h2>CLASSIFICATION: AI-GENERATED</h2>'
                        f'<h3>Probability: {proba_ai:.1%} (Threshold: {threshold:.1%})</h3>'
                        f'</div>',
                        unsafe_allow_html=True
                    )
                else:
                    st.markdown(
                        f'<div class="prediction-card human-card">'
                        f'<h2>CLASSIFICATION: HUMAN-WRITTEN</h2>'
                        f'<h3>Probability of AI: {proba_ai:.1%} (Threshold: {threshold:.1%})</h3>'
                        f'</div>',
                        unsafe_allow_html=True
                    )
                
                # Gauge representation using progress bar
                st.write("**AI Probability Gauge**:")
                st.progress(float(proba_ai))
                st.write(f"The model predicted a **{proba_ai:.1%}** probability that the text is AI-generated.")
                
                # Display feature details in expandable table
                with st.expander("📊 Show Extracted Features for this Text"):
                    feature_display = pd.DataFrame({
                        'Feature Name': list(features_dict.keys()),
                        'Extracted Value': list(features_dict.values())
                    })
                    st.dataframe(feature_display, use_container_width=True, hide_index=True)
                    
                    st.markdown(
                        "**Quick Guide on Key Features:**\n"
                        "- `nominalization_ratio`: Counts words ending in `-ung`, `-heit`, `-keit`, `-tion`, `-sion` (common in administrative German).\n"
                        "- `type_token_ratio`: Text vocabulary richness (lower values indicate repetitive/AI-like phrasing).\n"
                        "- `citation_density`: References to sections/laws (e.g. `§ 35 VwVfG`)."
                    )
    else:
        st.info("Enter German text on the left or select an example preset to analyze.")
