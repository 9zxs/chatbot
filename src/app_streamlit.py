import streamlit as st
import pandas as pd
import joblib
import random
import json
import csv
import os
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

# ========================
# Page Config
# ========================
st.set_page_config(page_title="🎓 University Chatbot", page_icon="🤖")

# ========================
# Load dataset & train model
# ========================
@st.cache_resource
def load_and_train():
    # Load dataset
    df = pd.read_csv("data/train_data.csv")

    # Encode labels
    le = LabelEncoder()
    df["label"] = le.fit_transform(df["intent"])

    # Split (train/test not really used here, but good practice)
    X_train, X_test, y_train, y_test = train_test_split(
        df["text"], df["label"], test_size=0.2, random_state=42
    )

    # Build pipeline
    pipeline = Pipeline([
        ("tfidf", TfidfVectorizer()),
        ("clf", LogisticRegression(max_iter=1000))
    ])

    # Train model
    pipeline.fit(X_train, y_train)

    return pipeline, le

pipeline, le = load_and_train()

# Load intents.json
with open("data/intents.json", encoding="utf-8") as f:
    intents = json.load(f)

intent_to_responses = {item["intent"]: item["responses"] for item in intents}

# Feedback file
FEEDBACK_FILE = "data/feedback.csv"
if not os.path.exists(FEEDBACK_FILE):
    with open(FEEDBACK_FILE, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["user_input", "predicted_intent", "response", "feedback"])

# ========================
# Custom CSS
# ========================
st.markdown(
    """
    <style>
    .chat-container {
        max-height: 400px;
        overflow-y: auto;
        padding: 10px;
        border: 1px solid #ddd;
        border-radius: 10px;
        background-color: #ffffff;
    }
    .user-bubble {
        background-color: #DCF8C6;
        padding: 10px;
        border-radius: 15px;
        margin: 5px 0;
        text-align: right;
    }
    .bot-bubble {
        background-color: #F1F0F0;
        padding: 10px;
        border-radius: 15px;
        margin: 5px 0;
        text-align: left;
    }
    /* Scrollbar */
    .chat-container::-webkit-scrollbar {
        width: 8px;
    }
    .chat-container::-webkit-scrollbar-track {
        background: #f1f1f1;
        border-radius: 10px;
    }
    .chat-container::-webkit-scrollbar-thumb {
        background: #888;
        border-radius: 10px;
    }
    .chat-container::-webkit-scrollbar-thumb:hover {
        background: #555;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ========================
# Streamlit UI
# ========================
st.title("🎓 University Chatbot")
st.markdown("Ask me about admissions, tuition, courses, and more.")

# Keep chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Input box
user_input = st.text_input("💬 Type your message here:")

if st.button("Send") and user_input.strip():
    # Predict intent + confidence
    proba = pipeline.predict_proba([user_input])[0]
    y_pred = pipeline.predict([user_input])[0]
    confidence = max(proba)
    intent = le.inverse_transform([y_pred])[0]

    # Apply confidence threshold
    threshold = 0.30
    if confidence < threshold or intent not in intent_to_responses:
        response = random.choice([
            "Sorry, I didn't quite get that.",
            "Can you please rephrase?",
            "I'm not sure I understand. Could you clarify?"
        ])
        intent = "fallback"
    else:
        response = random.choice(intent_to_responses[intent])

    # Save to history
    st.session_state.messages.append(
        {"user": user_input, "bot": response, "intent": intent}
    )

# ========================
# Display conversation
# ========================
st.markdown("<div class='chat-container'>", unsafe_allow_html=True)

for idx, chat in enumerate(st.session_state.messages):
    # User bubble
    st.markdown(f"<div class='user-bubble'>🙋‍♂️ {chat['user']}</div>", unsafe_allow_html=True)
    # Bot bubble
    st.markdown(f"<div class='bot-bubble'>🤖 {chat['bot']}</div>", unsafe_allow_html=True)

    # Feedback
    col1, col2 = st.columns(2)
    with col1:
        if st.button("👍 Helpful", key=f"yes_{idx}"):
            with open(FEEDBACK_FILE, "a", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow([chat["user"], chat["intent"], chat["bot"], "yes"])
            st.success("Feedback recorded: Yes")
    with col2:
        if st.button("👎 Not Helpful", key=f"no_{idx}"):
            with open(FEEDBACK_FILE, "a", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow([chat["user"], chat["intent"], chat["bot"], "no"])
            st.error("Feedback recorded: No")

st.markdown("</div>", unsafe_allow_html=True)
