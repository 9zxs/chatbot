from flask import Flask, request, jsonify
import joblib
import random
import json
import csv
import os

app = Flask(__name__)

print("Loading model...")
# Load trained model and label encoder
model_data = joblib.load("src/model.pkl")
pipeline = model_data["pipeline"]
le = model_data["label_encoder"]

# Load intents JSON
with open("data/intents.json", encoding="utf-8") as f:
    intents = json.load(f)

# Create dictionary mapping intent → responses
intent_to_responses = {item["intent"]: item["responses"] for item in intents}

# Define fallback responses
fallback_responses = [
    "I'm sorry — I didn't understand that. Could you rephrase?",
    "Hmm, I'm not sure I follow. Could you clarify?"
]

# File to save user feedback
FEEDBACK_FILE = "data/feedback.csv"
if not os.path.exists(FEEDBACK_FILE):
    with open(FEEDBACK_FILE, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        # Added confidence column
        writer.writerow(["user_input", "predicted_intent", "confidence", "response", "feedback"])


@app.route("/chat", methods=["POST"])
def chat():
    """Handle chatbot query"""
    data = request.get_json()
    user_text = data.get("text", "")

    if not user_text:
        return jsonify({"error": "No input provided"}), 400

    # Predict intent and probability
    proba = pipeline.predict_proba([user_text])[0]
    y_pred = proba.argmax()
    confidence = proba[y_pred]
    intent = le.inverse_transform([y_pred])[0]

    # Apply confidence threshold
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
    """Save user feedback to CSV"""
    data = request.get_json()
    user_input = data.get("user_input", "")
    predicted_intent = data.get("predicted_intent", "")
    confidence = data.get("confidence", 0.0)  # log confidence too
    response = data.get("response", "")
    feedback = data.get("feedback", "").lower()  # expect 'yes' or 'no'

    if feedback not in ["yes", "no"]:
        return jsonify({"error": "Feedback must be 'yes' or 'no'"}), 400

    # Append feedback to file
    with open(FEEDBACK_FILE, "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([user_input, predicted_intent, confidence, response, feedback])

    return jsonify({"message": "Feedback recorded"})


if __name__ == "__main__":
    app.run(debug=True)
