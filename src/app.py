from flask import Flask, request, jsonify
import joblib
import random
import json
import csv
import os
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

# ========================
# File paths
# ========================
DATASET_PATH = "data/train_data.csv"
MODEL_PATH = "src/model.pkl"
FEEDBACK_FILE = "data/feedback.csv"
INTENTS_FILE = "data/intents.json"

# ========================
# 1. Collect Dataset
# ========================
def load_dataset():
    if not os.path.exists(DATASET_PATH):
        raise FileNotFoundError(f"{DATASET_PATH} not found!")
    return pd.read_csv(DATASET_PATH)

# ========================
# 2. Train / Retrain Model
# ========================
def train_model():
    df = load_dataset()
    X, y = df["text"], df["intent"]

    # Encode labels
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)

    # Split for evaluation (optional)
    X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)

    # Define pipeline
    pipeline = Pipeline([
        ("tfidf", TfidfVectorizer()),
        ("clf", LogisticRegression(max_iter=500))
    ])

    pipeline.fit(X_train, y_train)

    # Save model
    joblib.dump({"pipeline": pipeline, "label_encoder": le}, MODEL_PATH)
    print("✅ Model trained and saved.")

    return pipeline, le

# ========================
# 3. Load Model & Responses
# ========================
def load_model():
    if os.path.exists(MODEL_PATH):
        model_data = joblib.load(MODEL_PATH)
        return model_data["pipeline"], model_data["label_encoder"]
    else:
        return train_model()

pipeline, le = load_model()

with open(INTENTS_FILE, encoding="utf-8") as f:
    intents = json.load(f)

intent_to_responses = {item["intent"]: item["responses"] for item in intents}

fallback_responses = [
    "I'm sorry — I didn't understand that. Could you rephrase?",
    "Hmm, I'm not sure I follow. Could you clarify?",
]

# Feedback CSV
if not os.path.exists(FEEDBACK_FILE):
    with open(FEEDBACK_FILE, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["user_input", "predicted_intent", "response", "feedback"])

# ========================
# Flask API
# ========================
app = Flask(__name__)

@app.route("/chat", methods=["POST"])
def chat():
    """Handle chatbot query"""
    data = request.get_json()
    user_text = data.get("text", "")

    if not user_text:
        return jsonify({"error": "No input provided"}), 400

    # Predict
    probs = pipeline.predict_proba([user_text])[0]
    y_pred = pipeline.predict([user_text])[0]
    intent = le.inverse_transform([y_pred])[0]
    confidence = max(probs)

    # Apply threshold
    threshold = 0.30
    if confidence < threshold or intent not in intent_to_responses:
        response = random.choice(fallback_responses)
        intent = "fallback"
    else:
        response = random.choice(intent_to_responses[intent])

    return jsonify({
        "user_input": user_text,
        "intent": intent,
        "confidence": float(confidence),
        "response": response
    })

@app.route("/feedback", methods=["POST"])
def feedback():
    """Save feedback to CSV"""
    data = request.get_json()
    user_input = data.get("user_input", "")
    predicted_intent = data.get("predicted_intent", "")
    response = data.get("response", "")
    feedback = data.get("feedback", "").lower()

    if feedback not in ["yes", "no"]:
        return jsonify({"error": "Feedback must be 'yes' or 'no'"}), 400

    # Save feedback
    with open(FEEDBACK_FILE, "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([user_input, predicted_intent, response, feedback])

    return jsonify({"message": "Feedback recorded"})

@app.route("/retrain", methods=["POST"])
def retrain():
    """Retrain model using dataset (and feedback if merged)."""
    global pipeline, le
    pipeline, le = train_model()
    return jsonify({"message": "Model retrained successfully."})

if __name__ == "__main__":
    app.run(debug=True)
