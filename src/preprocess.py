import re
import json
import pandas as pd
from sklearn.model_selection import train_test_split

def clean_text(s):
    s = s.lower()
    s = re.sub(r"[^a-z0-9\s']", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s

def load_intents_json(path):
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)

def build_training_csv_from_intents(intents_json, out_csv):
    rows = []
    for entry in intents_json['intents']:
        intent = entry['intent']
        for ex in entry.get('examples', []):
            rows.append({'text': clean_text(ex), 'intent': intent})
    df = pd.DataFrame(rows)
    df.to_csv(out_csv, index=False)
    return df

def load_csv(path):
    df = pd.read_csv(path)
    df['text'] = df['text'].astype(str).map(clean_text)
    return df

def train_test_split_df(df, test_size=0.2, random_state=42):
    return train_test_split(df['text'], df['intent'], test_size=test_size, random_state=random_state)
