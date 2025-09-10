import streamlit as st
import pandas as pd
import joblib
import random
import json
import csv
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

# ========================
# Page config
# ========================
st.set_page_config(page_title="University Chatbot", page_icon="🤖")

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

    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        df["text"], df["label"], test_size=0.2, random_state=42
    )

    # Build pipeline
    pipeline = Pipeline([
        ("tfidf", TfidfVectorizer()),
        ("clf", LogisticRegression(max_iter=1000))
    ])

    # Train
    pipeline.fit(X_train, y_train)

    # Accuracy scores
    train_acc = pipeline.score(X_train, y_train)
    test_acc = pipeline.score(X_test, y_test)

    return pipeline, le, train_acc, test_acc

pipeline, le, train_acc, test_acc = load_and_train()

# ========================
# Load intents
# ========================
with open("data/intents.json", encoding="utf-8") as f:
    intents = json.load(f)

intent_to_responses = {item["intent"]: item["responses"] for item in intents}
FEEDBACK_FILE = "data/feedback.csv"

if not os.path.exists(FEEDBACK_FILE):
    with open(FEEDBACK_FILE, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["user_input", "predicted_intent", "response", "feedback"])

# ========================
# Sidebar info panel
# ========================
st.sidebar.title("📊 Chatbot Info")
st.sidebar.write(f"✅ Training Accuracy: **{train_acc:.2f}**")
st.sidebar.write(f"✅ Testing Accuracy: **{test_acc:.2f}**")

# Feedback count
if os.path.exists(FEEDBACK_FILE):
    feedback_df = pd.read_csv(FEEDBACK_FILE)
    st.sidebar.write(f"📝 Feedback Collected: **{len(feedback_df)}** entries")
else:
    st.sidebar.write("📝 Feedback Collected: **0** entries")

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
    # Predict intent
    y_pred = pipeline.predict([user_input])[0]
    intent = le.inverse_transform([y_pred])[0]

    response = random.choice(intent_to_responses.get(intent, ["Sorry, I didn't understand that."]))

    # Add to session history
    st.session_state.messages.append(
        {"user": user_input, "bot": response, "intent": intent}
    )

# ========================
# Display conversation
# ========================

for idx, chat in enumerate(st.session_state.messages):
    # User bubble
    st.markdown(f"<div class='user-bubble'>🙋‍♂️ {chat['user']}</div>", unsafe_allow_html=True)
    # Bot bubble
    st.markdown(f"<div class='bot-bubble'>🤖 {chat['bot']}</div>", unsafe_allow_html=True)

    # Feedback buttons
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



