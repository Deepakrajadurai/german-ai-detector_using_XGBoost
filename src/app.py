"""
Streamlit Web App for German AI Detector (XGBoost).
Provides interactive sentence-by-sentence analysis for administrative/legal German texts.
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

# Page config
st.set_page_config(
    page_title="German AI Detector - Sentence Analysis",
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
        """Extract all features from a single sentence."""
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
    # Return spaCy German model with parser enabled for sentence splitting
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

# CSS adjustments for clean, premium design
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
        background-color: #1b5e20;
        border: 2px solid #2e7d32;
    }
    .ai-card {
        background-color: #b71c1c;
        border: 2px solid #c62828;
    }
    .warning-card {
        background-color: #e65100;
        border: 2px solid #ef6c00;
    }
    .sentence-highlight-ai {
        background-color: rgba(239, 83, 80, 0.3);
        border-bottom: 2px solid #ef5350;
        padding: 2px 4px;
        border-radius: 4px;
    }
    .sentence-highlight-human {
        background-color: rgba(102, 187, 106, 0.15);
        border-bottom: 2px dotted #66bb6a;
        padding: 2px 4px;
        border-radius: 4px;
    }
    .sentence-item {
        padding: 10px;
        border-radius: 6px;
        margin-bottom: 8px;
        background-color: #f1f3f5;
        border-left: 5px solid #ccc;
    }
    .sentence-item-ai {
        border-left-color: #ef5350;
    }
    .sentence-item-human {
        border-left-color: #66bb6a;
    }
</style>
""", unsafe_allow_html=True)

# Main Title Layout
st.title("🇩🇪 German AI Detector (Sentence-Level)")
st.subheader("Interactive Sentence-by-Sentence Analysis for Administrative & Legal German")
st.write(
    "Paste a paragraph or document below. The model splits the input into individual sentences and "
    "evaluates each one using a sentence-level XGBoost classifier."
)

# Sidebar
st.sidebar.header("📊 Model Specifications")
if metadata:
    st.sidebar.metric(label="Model Type", value="Sentence-Level XGBoost")
    st.sidebar.metric(label="Sentence Dataset Size", value="300,000 sentences")
    st.sidebar.metric(label="Test Set Accuracy", value=f"{metadata['performance']['accuracy']:.2%}")
    st.sidebar.metric(label="FPR (False Positive Rate)", value=f"{metadata['performance']['fpr']:.2%}")
    st.sidebar.metric(label="Sentence Detection Threshold", value=f"{metadata['threshold']:.2f}")
    
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
        height=220,
        placeholder="Schreiben Sie hier Ihren deutschen Text..."
    )
    
    detect_clicked = st.button("Analyze Text Block", type="primary", use_container_width=True)

with col2:
    st.header("🔮 Prediction & Explanation")
    
    if (detect_clicked or user_text) and user_text.strip():
        if model is None or metadata is None:
            st.error("Model files are not loaded correctly. Ensure training has completed.")
        else:
            with st.spinner("Splitting text into sentences and predicting..."):
                # Split paragraph into sentences using spaCy
                doc = nlp(user_text)
                sentences = [sent.text.strip() for sent in doc.sents if len(sent.text.strip()) > 3]
                
                if len(sentences) == 0:
                    st.warning("Please enter a valid text block with complete sentences.")
                else:
                    results = []
                    model_feats = model.get_booster().feature_names
                    threshold = metadata['threshold']
                    
                    for sent in sentences:
                        features_dict = extractor.extract_features(sent, nlp)
                        features_df = pd.DataFrame([features_dict])[model_feats]
                        proba_ai = model.predict_proba(features_df)[0, 1]
                        
                        results.append({
                            'sentence': sent,
                            'proba_ai': proba_ai,
                            'is_ai': proba_ai >= threshold,
                            'features': features_dict
                        })
                    
                    # Overall Document Metrics
                    total_sents = len(results)
                    ai_sents = sum(1 for r in results if r['is_ai'])
                    ai_ratio = ai_sents / total_sents
                    avg_proba = np.mean([r['proba_ai'] for r in results])
                    
                    # Display overall card
                    if ai_ratio >= 0.5:
                        st.markdown(
                            f'<div class="prediction-card ai-card">'
                            f'<h2>DOCUMENT CLASSIFICATION: AI-GENERATED</h2>'
                            f'<h3>{ai_sents}/{total_sents} sentences ({ai_ratio:.0%}) flagged as AI</h3>'
                            f'<h4>Average Sentence AI Prob: {avg_proba:.1%} (Threshold: {threshold:.1%})</h4>'
                            f'</div>',
                            unsafe_allow_html=True
                        )
                    elif ai_sents > 0:
                        st.markdown(
                            f'<div class="prediction-card warning-card">'
                            f'<h2>DOCUMENT CLASSIFICATION: CONTAINS AI SEGMENTS</h2>'
                            f'<h3>{ai_sents}/{total_sents} sentences ({ai_ratio:.0%}) flagged as AI</h3>'
                            f'<h4>Average Sentence AI Prob: {avg_proba:.1%} (Threshold: {threshold:.1%})</h4>'
                            f'</div>',
                            unsafe_allow_html=True
                        )
                    else:
                        st.markdown(
                            f'<div class="prediction-card human-card">'
                            f'<h2>DOCUMENT CLASSIFICATION: HUMAN-WRITTEN</h2>'
                            f'<h3>0/{total_sents} sentences flagged as AI</h3>'
                            f'<h4>Average Sentence AI Prob: {avg_proba:.1%} (Threshold: {threshold:.1%})</h4>'
                            f'</div>',
                            unsafe_allow_html=True
                        )
                    
                    # 1. Interactive Highlighting Representation
                    st.markdown("### 🔍 Highlighted Text View")
                    highlighted_html = ""
                    for r in results:
                        highlight_class = "sentence-highlight-ai" if r['is_ai'] else "sentence-highlight-human"
                        tooltip = f"AI Probability: {r['proba_ai']:.1%}"
                        highlighted_html += f'<span class="{highlight_class}" title="{tooltip}">{r["sentence"]}</span> '
                    
                    st.markdown(f'<div style="line-height: 2.0; font-size: 1.1rem; padding: 15px; border-radius: 8px; border: 1px solid #ddd; background-color: white;">{highlighted_html}</div>', unsafe_allow_html=True)
                    
                    st.markdown("---")
                    
                    # 2. Detailed Sentence-by-Sentence breakdown
                    st.markdown("### 📋 Sentence-by-Sentence Probability Breakdown")
                    for idx, r in enumerate(results, start=1):
                        style_class = "sentence-item-ai" if r['is_ai'] else "sentence-item-human"
                        color = "#d32f2f" if r['is_ai'] else "#388e3c"
                        
                        st.markdown(
                            f'<div class="sentence-item {style_class}">'
                            f'<strong>Satz {idx}:</strong> "{r["sentence"]}"<br/>'
                            f'<span style="color: {color};">AI Probability: {r["proba_ai"]:.1%}</span>'
                            f'</div>',
                            unsafe_allow_html=True
                        )
                        st.progress(float(r['proba_ai']))
                    
                    st.markdown("---")
                    
                    # 3. Expandable detailed feature inspection
                    with st.expander("📊 Inspect Extracted Feature Values"):
                        selected_sent_idx = st.selectbox(
                            "Select sentence to view exact features:",
                            options=list(range(1, len(results) + 1)),
                            format_func=lambda x: f"Satz {x}: {results[x-1]['sentence'][:50]}..."
                        )
                        
                        features_selected = results[selected_sent_idx - 1]['features']
                        feature_display = pd.DataFrame({
                            'Feature Name': list(features_selected.keys()),
                            'Extracted Value': list(features_selected.values())
                        })
                        st.dataframe(feature_display, use_container_width=True, hide_index=True)
    else:
        st.info("Enter German text on the left or select an example preset to analyze.")
