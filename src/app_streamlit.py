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
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# ========================
# Page config
# ========================
st.set_page_config(page_title="University Chatbot", page_icon="🤖")

# ========================
# Paths
# ========================
TRAIN_FILE = "data/train_data.csv"
FEEDBACK_FILE = "data/feedback.csv"
INTENTS_FILE = "data/intents.json"

# ========================
# Load dataset & train model
# ========================
@st.cache_resource
def retrain_model():
    df = pd.read_csv(TRAIN_FILE)

    # Encode labels
    le = LabelEncoder()
    df["label"] = le.fit_transform(df["intent"])

    # ✅ Safe stratify check
    stratify = df["label"] if df["label"].value_counts().min() > 1 else None

    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        df["text"], df["label"], test_size=0.2, random_state=42, stratify=stratify
    )

    # Build pipeline
    pipeline = Pipeline([
        ("tfidf", TfidfVectorizer()),
        ("clf", LogisticRegression(max_iter=1000))
    ])

    pipeline.fit(X_train, y_train)

    # Accuracy scores
    train_acc = pipeline.score(X_train, y_train)
    test_acc = pipeline.score(X_test, y_test)

    return pipeline, le, train_acc, test_acc

# ========================
# Train / load model
# ========================
try:
    pipeline, le, train_acc, test_acc = retrain_model()
except Exception as e:
    st.error(f"⚠️ Model training failed: {e}")
    pipeline, le, train_acc, test_acc = None, None, 0, 0

# ========================
# Load intents
# ========================
with open(INTENTS_FILE, encoding="utf-8") as f:
    intents = json.load(f)

intent_to_responses = {item["intent"]: item["responses"] for item in intents}

# Create feedback file if not exists
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
    </style>
    """,
    unsafe_allow_html=True,
)

# ========================
# Tabs
# ========================
tab1, tab2, tab3, tab4 = st.tabs(["💬 Chatbot", "📊 Info", "📥 Feedback Data", "📈 Evaluation"])

# ------------------------
# Tab 1: Chatbot
# ------------------------
with tab1:
    st.title("🎓 University Chatbot")
    st.markdown("Ask me about admissions, tuition, courses, and more.")

    # Keep chat history
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Quick Questions
    st.markdown("#### 🔘 Quick Questions")
    quick_questions = [
        "When is the application deadline?",
        "How much is the tuition fee?",
        "What courses are available?",
        "Hello",
        "Thanks",
        "Goodbye"
    ]
    cols = st.columns(len(quick_questions))
    for i, q in enumerate(quick_questions):
        if cols[i].button(q):
            user_input = q
            # Predict intent
            y_pred = pipeline.predict([user_input])[0]
            intent = le.inverse_transform([y_pred])[0]
            response = random.choice(intent_to_responses.get(intent, ["Sorry, I didn't understand that."]))

            # Add to session history
            st.session_state.messages.append(
                {"user": user_input, "bot": response, "intent": intent}
            )

    # Text input
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

    # Display conversation
    for idx, chat in enumerate(st.session_state.messages):
        st.markdown(f"<div class='user-bubble'>🙋‍♂️ {chat['user']}</div>", unsafe_allow_html=True)
        st.markdown(f"<div class='bot-bubble'>🤖 {chat['bot']}</div>", unsafe_allow_html=True)

        # Feedback buttons
        col1, col2 = st.columns(2)
        with col1:
            if st.button("👍 Helpful", key=f"yes_{idx}"):
                with open(FEEDBACK_FILE, "a", newline="", encoding="utf-8") as f:
                    writer = csv.writer(f)
                    writer.writerow([chat["user"], chat["intent"], "N/A", chat["bot"], "yes"])
                st.success("Feedback recorded: Yes")
        with col2:
            if st.button("👎 Not Helpful", key=f"no_{idx}"):
                with open(FEEDBACK_FILE, "a", newline="", encoding="utf-8") as f:
                    writer = csv.writer(f)
                    writer.writerow([chat["user"], chat["intent"], "N/A", chat["bot"], "no"])
                st.error("Feedback recorded: No")

# ------------------------
# Tab 2: Info
# ------------------------
with tab2:
    st.subheader("📊 Chatbot Info")
    st.write(f"✅ Training Accuracy: **{train_acc:.2f}**")
    st.write(f"✅ Testing Accuracy: **{test_acc:.2f}**")

    if os.path.exists(FEEDBACK_FILE):
        feedback_df = pd.read_csv(FEEDBACK_FILE)
        st.write(f"📝 Feedback Collected: **{len(feedback_df)}** entries")
    else:
        st.write("📝 Feedback Collected: **0** entries")

# ------------------------
# Tab 3: Feedback Data
# ------------------------
with tab3:
    st.subheader("📥 Download Feedback Data")
    if os.path.exists(FEEDBACK_FILE):
        fb_data = pd.read_csv(FEEDBACK_FILE)
        st.dataframe(fb_data)
        csv_dl = fb_data.to_csv(index=False).encode("utf-8")
        st.download_button(
            label="⬇️ Download feedback.csv",
            data=csv_dl,
            file_name="feedback.csv",
            mime="text/csv",
        )
    else:
        st.info("No feedback data available yet.")

# ------------------------
# Tab 4: Evaluation
# ------------------------
with tab4:
    st.subheader("📈 Model Evaluation")

    # Retrain fresh model
    pipeline, le, train_acc, test_acc = retrain_model()

    # Reload dataset
    df_eval = pd.read_csv(TRAIN_FILE)
    y_true = le.transform(df_eval["intent"])
    y_pred_all = pipeline.predict(df_eval["text"])

    # Show classification report
    report = classification_report(y_true, y_pred_all, target_names=le.classes_, output_dict=True)
    st.write("### 📑 Classification Report")
    st.dataframe(pd.DataFrame(report).transpose())

    # Show confusion matrix
    cm = confusion_matrix(y_true, y_pred_all)
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=le.classes_, yticklabels=le.classes_, ax=ax)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title("Confusion Matrix")
    st.pyplot(fig)


