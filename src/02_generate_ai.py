
'''Data generation using openai api key'''
# import pandas as pd
# import openai
# from tqdm import tqdm
# import time
# import os
# from pathlib import Path
# from dotenv import load_dotenv

# # Load API key from .env file
# load_dotenv()
# openai.api_key = os.getenv("OPENAI_API_KEY")

# # Paths
# INTERIM_DIR = Path("data/interim")
# MODEL_NAME = "gpt-4o-mini"  # Super cheap and fast

# def generate_ai_batch(human_df, num_to_generate, split_name, output_path):
#     """
#     Generates AI paraphrases for a specific split.
#     Saves checkpoints every 100 rows (resume-safe).
#     """
#     print(f"\n🚀 Generating AI for split: {split_name}")
    
#     # Sample human sentences
#     if len(human_df) < num_to_generate:
#         print(f"⚠️  Warning: Only {len(human_df)} humans available. Using all.")
#         sample = human_df
#     else:
#         sample = human_df.sample(n=num_to_generate, random_state=42)

#     generated_data = []
#     start_index = 0

#     # --- Checkpoint Recovery ---
#     if output_path.exists():
#         existing = pd.read_parquet(output_path)
#         generated_data = existing.to_dict('records')
#         start_index = len(existing)
#         print(f"🔄 Resuming from checkpoint: Already generated {start_index} for {split_name}")
        
#         # Remove already processed originals to avoid duplicates
#         generated_originals = set(existing['original_sentence'])
#         sample = sample[~sample['sentence'].isin(generated_originals)]

#     # --- Generation Loop ---
#     for idx, row in tqdm(sample.iterrows(), total=len(sample), desc=f"Generating {split_name}"):
#         original_text = row['sentence']
        
#         prompt = f"""
# Du bist ein parlamentarischer Assistent. Formuliere den folgenden deutschen Bundestags-Redebeitrag im offiziellen, formalen Verwaltungsstil um.
# Behalte die genaue Bedeutung und alle Fachbegriffe bei, aber verändere die Satzstruktur und Wortwahl komplett.

# Original: {original_text}

# Offizielle Umformulierung:
# """
#         try:
#             response = openai.chat.completions.create(
#                 model=MODEL_NAME,
#                 messages=[
#                     {"role": "system", "content": "Du bist ein Experte für deutsche Verwaltungssprache."},
#                     {"role": "user", "content": prompt}
#                 ],
#                 temperature=0.7,
#                 max_tokens=150,
#                 seed=42 + idx  # Fixed seed for reproducibility
#             )
            
#             ai_text = response.choices[0].message.content.strip()
            
#             generated_data.append({
#                 "sentence": ai_text,
#                 "original_sentence": original_text,
#                 "split": split_name,
#                 "label": 1  # 1 = AI Generated
#             })
            
#             # --- Save Checkpoint every 100 rows ---
#             if len(generated_data) % 100 == 0:
#                 pd.DataFrame(generated_data).to_parquet(output_path)
                
#         except Exception as e:
#             print(f"\n❌ Error on index {idx}: {e}")
#             time.sleep(5)  # Wait longer if rate limited
        
#         time.sleep(0.05)  # Gentle rate limiting

#     # Final save
#     final_df = pd.DataFrame(generated_data)
#     final_df.to_parquet(output_path)
#     print(f"✅ Done generating for {split_name}. Total: {len(final_df)}")
#     return final_df

# def main():
#     print("=" * 60)
#     print("STEP 2: Generating AI Texts (5k Test Run)")
#     print("=" * 60)

#     # Load human splits
#     train_human = pd.read_parquet(INTERIM_DIR / "train_human.parquet")
#     val_human = pd.read_parquet(INTERIM_DIR / "val_human.parquet")

#     print(f"📊 Available human sentences:")
#     print(f"   Train: {len(train_human):,}")
#     print(f"   Val:   {len(val_human):,}")

#     # Generate 2,500 AI for Train
#     generate_ai_batch(
#         human_df=train_human,
#         num_to_generate=2500,
#         split_name="train",
#         output_path=INTERIM_DIR / "ai_train.parquet"
#     )

#     # Generate 2,500 AI for Val
#     generate_ai_batch(
#         human_df=val_human,
#         num_to_generate=2500,
#         split_name="val",
#         output_path=INTERIM_DIR / "ai_val.parquet"
#     )

#     print("\n🎉 AI generation complete!")
#     print(f"📁 Files saved to: {INTERIM_DIR}")

# if __name__ == "__main__":
#     main()


'''Data generation using the groq api key'''


import pandas as pd
import os
import time
from pathlib import Path
from tqdm import tqdm
from dotenv import load_dotenv
from groq import Groq

# Load API key from .env
load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

if not GROQ_API_KEY:
    raise ValueError("❌ GROQ_API_KEY not found in .env file!")

# Initialize Groq client
client = Groq(api_key=GROQ_API_KEY)

# ============================================================
# AUTO-DISCOVER ACTIVE MODELS (Replaces hardcoded MODEL_NAME)
# ============================================================
def get_active_models():
    """
    Return a list of active chat models from Groq, 
    restricted to the user's preferred list.
    """
    # Your preferred models (ONLY these will be used)
    preferred_order = [
        "llama-3.3-70b-versatile",
        "llama-3.1-8b-instant",
        "gemma2-9b-it",
        "deepseek-r1-distill-llama-70b"
    ]
    
    try:
        models = client.models.list()
        # Get all active model IDs
        all_active = [m.id for m in models.data if m.active]
        print(f"✅ Active Groq models: {all_active}")
        
        # Filter: Keep ONLY the models that are in preferred_order AND active
        available_preferred = [m for m in preferred_order if m in all_active]
        
        if not available_preferred:
            print(f"⚠️ None of your preferred models are active. Available: {all_active}")
            # Fallback to the first preferred model anyway (Groq will return error if inactive)
            return ["llama-3.3-70b-versatile"]
        
        print(f"✅ Available preferred models: {available_preferred}")
        return available_preferred
        
    except Exception as e:
        print(f"⚠️ Could not fetch models: {e}")
        return ["llama-3.3-70b-versatile"]  # fallback

ACTIVE_MODELS = get_active_models()
print(f"✅ Active Groq models: {ACTIVE_MODELS}")
MODEL_NAME = ACTIVE_MODELS[0]
print(f"🚀 Using model: {MODEL_NAME}")
# ============================================================

# Paths
INTERIM_DIR = Path("data/interim")

def generate_ai_batch(human_df, num_to_generate, split_name, output_path):
    """
    Generates AI paraphrases for a specific split using Groq API.
    Saves checkpoints every 100 rows (resume-safe).
    """
    print(f"\n🚀 Generating AI for split: {split_name} using {MODEL_NAME}")

    if len(human_df) < num_to_generate:
        print(f"⚠️  Warning: Only {len(human_df)} humans available. Using all.")
        sample = human_df
    else:
        sample = human_df.sample(n=num_to_generate, random_state=42)

    generated_data = []

    # Checkpoint Recovery
    if output_path.exists():
        existing = pd.read_parquet(output_path)
        generated_data = existing.to_dict('records')
        print(f"🔄 Resuming from checkpoint: Already generated {len(generated_data)} for {split_name}")
        generated_originals = set(existing['original_sentence'])
        sample = sample[~sample['sentence'].isin(generated_originals)]

    if len(sample) == 0:
        print(f"✅ All sentences already generated for {split_name}. Skipping.")
        return pd.DataFrame(generated_data)

    # Generation Loop
    for idx, row in tqdm(sample.iterrows(), total=len(sample), desc=f"Generating {split_name}"):
        original_text = row['sentence']

        prompt = f"""
Du bist ein parlamentarischer Assistent. Formuliere den folgenden deutschen Bundestags-Redebeitrag im offiziellen, formalen Verwaltungsstil um.
Behalte die genaue Bedeutung und alle Fachbegriffe bei, aber verändere die Satzstruktur und Wortwahl komplett.

Original: {original_text}

Offizielle Umformulierung:
"""

        try:
            response = client.chat.completions.create(
                model=MODEL_NAME,
                messages=[
                    {"role": "system", "content": "Du bist ein Experte für deutsche Verwaltungssprache."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.7,
                max_tokens=150,
                seed=42 + idx
            )

            ai_text = response.choices[0].message.content.strip()

            generated_data.append({
                "sentence": ai_text,
                "original_sentence": original_text,
                "split": split_name,
                "label": 1
            })

            if len(generated_data) % 100 == 0:
                pd.DataFrame(generated_data).to_parquet(output_path)

        except Exception as e:
            print(f"\n❌ Error on index {idx}: {e}")
            time.sleep(5)

        time.sleep(0.1)

    final_df = pd.DataFrame(generated_data)
    final_df.to_parquet(output_path)
    print(f"✅ Done generating for {split_name}. Total: {len(final_df)}")
    return final_df

def main():
    print("=" * 60)
    print("STEP 2: Generating AI Texts (5k Test Run) via Groq")
    print("=" * 60)

    train_human = pd.read_parquet(INTERIM_DIR / "train_human.parquet")
    val_human = pd.read_parquet(INTERIM_DIR / "val_human.parquet")

    print(f"📊 Available human sentences:")
    print(f"   Train: {len(train_human):,}")
    print(f"   Val:   {len(val_human):,}")

    generate_ai_batch(
        human_df=train_human,
        num_to_generate=0,
        split_name="train",
        output_path=INTERIM_DIR / "ai_train.parquet"
    )

    generate_ai_batch(
        human_df=val_human,
        num_to_generate=500,
        split_name="val",
        output_path=INTERIM_DIR / "ai_val.parquet"
    )

    print("\n🎉 AI generation complete!")
    print(f"📁 Files saved to: {INTERIM_DIR}")

if __name__ == "__main__":
    main()