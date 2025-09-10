import streamlit as st
import joblib
import random
import json
import csv
import os

# ========================
# Load model and data
# ========================
st.set_page_config(page_title="University Chatbot", page_icon="🤖")

@st.cache_resource
def load_model():
    model_data = joblib.load("src/model.pkl")
    return model_data["pipeline"], model_data["label_encoder"]

pipeline, le = load_model()

with open("data/intents.json", encoding="utf-8") as f:
    intents = json.load(f)

intent_to_responses = {item["intent"]: item["responses"] for item in intents}
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
    /* Custom scrollbar */
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
# Display conversation in scrollable box
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
