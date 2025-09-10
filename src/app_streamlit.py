import streamlit as st
import joblib
import random
import json
import csv
import os
from datetime import datetime

# ========================
# Page Configuration
# ========================
st.set_page_config(
    page_title="University Assistant", 
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ========================
# Load model and data
# ========================
@st.cache_resource
def load_model():
    try:
        model_data = joblib.load("src/model.pkl")
        return model_data["pipeline"], model_data["label_encoder"]
    except FileNotFoundError:
        st.error("Model file not found. Please ensure 'src/model.pkl' exists.")
        return None, None

@st.cache_data
def load_intents():
    try:
        with open("data/intents.json", encoding="utf-8") as f:
            intents = json.load(f)
        return {item["intent"]: item["responses"] for item in intents}
    except FileNotFoundError:
        st.error("Intents file not found. Please ensure 'data/intents.json' exists.")
        return {}

pipeline, le = load_model()
intent_to_responses = load_intents()

FEEDBACK_FILE = "data/feedback.csv"
if not os.path.exists(FEEDBACK_FILE):
    os.makedirs("data", exist_ok=True)
    with open(FEEDBACK_FILE, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["timestamp", "user_input", "predicted_intent", "response", "feedback"])

# ========================
# Enhanced CSS Styling
# ========================
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

.main {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    font-family: 'Inter', sans-serif;
}

.stApp {
    background: transparent;
}

.header-container {
    background: rgba(255, 255, 255, 0.95);
    backdrop-filter: blur(10px);
    padding: 2rem;
    border-radius: 20px;
    margin-bottom: 2rem;
    box-shadow: 0 8px 32px rgba(0, 0, 0, 0.1);
    text-align: center;
    border: 1px solid rgba(255, 255, 255, 0.2);
}

.header-title {
    font-size: 2.5rem;
    font-weight: 700;
    color: #2D3748;
    margin-bottom: 0.5rem;
    background: linear-gradient(135deg, #667eea, #764ba2);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
}

.header-subtitle {
    font-size: 1.1rem;
    color: #718096;
    font-weight: 400;
}

.chat-container {
    background: rgba(255, 255, 255, 0.95);
    backdrop-filter: blur(10px);
    border-radius: 20px;
    padding: 2rem;
    box-shadow: 0 8px 32px rgba(0, 0, 0, 0.1);
    border: 1px solid rgba(255, 255, 255, 0.2);
    min-height: 500px;
    max-height: 600px;
    overflow-y: auto;
    margin-bottom: 1rem;
}

.message-container {
    display: flex;
    margin: 1rem 0;
    animation: slideIn 0.3s ease-out;
}

.user-message {
    justify-content: flex-end;
}

.bot-message {
    justify-content: flex-start;
}

.message-bubble {
    max-width: 70%;
    padding: 1rem 1.25rem;
    border-radius: 18px;
    font-size: 0.95rem;
    line-height: 1.5;
    word-wrap: break-word;
    position: relative;
}

.user-bubble {
    background: linear-gradient(135deg, #667eea, #764ba2);
    color: white;
    margin-left: auto;
    box-shadow: 0 4px 12px rgba(102, 126, 234, 0.3);
}

.bot-bubble {
    background: linear-gradient(135deg, #f8f9fa, #e9ecef);
    color: #2D3748;
    margin-right: auto;
    box-shadow: 0 4px 12px rgba(0, 0, 0, 0.1);
    border-left: 4px solid #667eea;
}

.message-avatar {
    width: 40px;
    height: 40px;
    border-radius: 50%;
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 1.2rem;
    margin: 0 10px;
    flex-shrink: 0;
}

.user-avatar {
    background: linear-gradient(135deg, #667eea, #764ba2);
    color: white;
    order: 2;
}

.bot-avatar {
    background: linear-gradient(135deg, #4facfe, #00f2fe);
    color: white;
    order: 1;
}

.input-container {
    background: rgba(255, 255, 255, 0.95);
    backdrop-filter: blur(10px);
    border-radius: 20px;
    padding: 1.5rem;
    box-shadow: 0 8px 32px rgba(0, 0, 0, 0.1);
    border: 1px solid rgba(255, 255, 255, 0.2);
    margin-top: 1rem;
}

.feedback-container {
    display: flex;
    gap: 0.5rem;
    margin-top: 0.75rem;
    justify-content: flex-end;
}

.empty-state {
    text-align: center;
    padding: 3rem 1rem;
    color: #718096;
}

.empty-state-icon {
    font-size: 4rem;
    margin-bottom: 1rem;
    opacity: 0.5;
}

.chat-container::-webkit-scrollbar {
    width: 8px;
}

.chat-container::-webkit-scrollbar-track {
    background: rgba(0, 0, 0, 0.1);
    border-radius: 10px;
}

.chat-container::-webkit-scrollbar-thumb {
    background: linear-gradient(135deg, #667eea, #764ba2);
    border-radius: 10px;
}

.chat-container::-webkit-scrollbar-thumb:hover {
    background: linear-gradient(135deg, #5a67d8, #6b46c1);
}

@keyframes slideIn {
    from {
        opacity: 0;
        transform: translateY(20px);
    }
    to {
        opacity: 1;
        transform: translateY(0);
    }
}

.stDeployButton {display:none;}
footer {visibility: hidden;}
.stApp > header {visibility: hidden;}

@media (max-width: 768px) {
    .header-title {
        font-size: 2rem;
    }
    
    .message-bubble {
        max-width: 85%;
    }
    
    .chat-container {
        padding: 1rem;
        max-height: 400px;
    }
}
</style>
""", unsafe_allow_html=True)

# ========================
# Utility Functions
# ========================
def save_feedback(user_input, intent, response, feedback):
    """Save user feedback to CSV file"""
    try:
        with open(FEEDBACK_FILE, "a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow([
                datetime.now().isoformat(),
                user_input,
                intent,
                response,
                feedback
            ])
    except Exception as e:
        st.error(f"Error saving feedback: {e}")

def get_bot_response(user_input):
    """Get bot response for user input"""
    if pipeline is None or le is None:
        return "Sorry, the chatbot model is not available.", "error"
    
    try:
        y_pred = pipeline.predict([user_input])[0]
        intent = le.inverse_transform([y_pred])[0]
        response = random.choice(intent_to_responses.get(intent, ["Sorry, I didn't understand that. Could you please rephrase your question?"]))
        return response, intent
    except Exception as e:
        return "Sorry, I encountered an error processing your request.", "error"

# ========================
# Initialize Session State
# ========================
if "messages" not in st.session_state:
    st.session_state.messages = []

if "feedback_given" not in st.session_state:
    st.session_state.feedback_given = set()

# ========================
# Header Section
# ========================
st.markdown("""
    <div class="header-container">
        <div class="header-title">🎓 University Assistant</div>
        <div class="header-subtitle">Your smart companion for university information • Ask about admissions, courses, fees, and more!</div>
    </div>
    """, unsafe_allow_html=True)

# ========================
# Main Chat Interface
# ========================
col1, col2, col3 = st.columns([1, 8, 1])

with col2:
    # Chat container
    chat_html = '<div class="chat-container">'
    
    if not st.session_state.messages:
        chat_html += '''
        <div class="empty-state">
            <div class="empty-state-icon">💬</div>
            <h3>Welcome! How can I help you today?</h3>
            <p>Ask me about university admissions, courses, fees, campus life, and more.</p>
        </div>
        '''
    else:
        for idx, message in enumerate(st.session_state.messages):
            # User message
            chat_html += f'''
            <div class="message-container user-message">
                <div class="message-avatar user-avatar">👤</div>
                <div class="message-bubble user-bubble">{message["user"]}</div>
            </div>
            '''
            
            # Bot message
            chat_html += f'''
            <div class="message-container bot-message">
                <div class="message-avatar bot-avatar">🤖</div>
                <div class="message-bubble bot-bubble">
                    {message["bot"]}
                    <div class="feedback-container">
                        <span style="font-size: 0.8rem; color: #718096; margin-right: 0.5rem;">Was this helpful?</span>
                    </div>
                </div>
            </div>
            '''
    
    chat_html += '</div>'
    st.markdown(chat_html, unsafe_allow_html=True)
    
    # Feedback buttons for the latest message
    if st.session_state.messages:
        latest_idx = len(st.session_state.messages) - 1
        if latest_idx not in st.session_state.feedback_given:
            col_feedback1, col_feedback2, col_feedback3 = st.columns([6,
