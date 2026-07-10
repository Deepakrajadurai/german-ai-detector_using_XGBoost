import pandas as pd

human = pd.read_csv("data/raw/human_text.csv")
ai = pd.read_csv("data/raw/ai_text_repaired.csv")

print(f"Human: {len(human):,}")
print(f"AI: {len(ai):,}")
print(f"Total: {len(human) + len(ai):,}")