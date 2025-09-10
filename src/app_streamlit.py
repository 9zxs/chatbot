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

st.markdown("</div>", unsafe_allow_html=True)at_html += f'''
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
            col_feedback1, col_feedback2, col_feedback3 = st.columns([6, 1, 1])
            
            with col_feedback2:
                if st.button("👍 Yes", key=f"helpful_{latest_idx}", help="This response was helpful"):
                    save_feedback(
                        st.session_state.messages[latest_idx]["user"],
                        st.session_state.messages[latest_idx]["intent"],
                        st.session_state.messages[latest_idx]["bot"],
                        "helpful"
                    )
                    st.session_state.feedback_given.add(latest_idx)
                    st.success("Thank you for your feedback! 😊", icon="✅")
                    st.rerun()
            
            with col_feedback3:
                if st.button("👎 No", key=f"not_helpful_{latest_idx}", help="This response was not helpful"):
                    save_feedback(
                        st.session_state.messages[latest_idx]["user"],
                        st.session_state.messages[latest_idx]["intent"],
                        st.session_state.messages[latest_idx]["bot"],
                        "not_helpful"
                    )
                    st.session_state.feedback_given.add(latest_idx)
                    st.info("Thanks for letting us know. We'll work on improving! 🔧", icon="💡")
                    st.rerun()

    # Input area
    st.markdown('<div class="input-container">', unsafe_allow_html=True)
    
    # Quick suggestions (only show when chat is empty)
    if not st.session_state.messages:
        st.markdown("**💡 Quick suggestions:**")
        suggestions = [
            "What are the admission requirements?",
            "Tell me about tuition fees",
            "What courses do you offer?",
            "How do I apply?",
            "Campus facilities"
        ]
        
        cols = st.columns(len(suggestions))
        for i, suggestion in enumerate(suggestions):
            with cols[i]:
                if st.button(suggestion, key=f"suggestion_{i}", help=f"Ask: {suggestion}"):
                    # Process the suggestion as user input
                    response, intent = get_bot_response(suggestion)
                    st.session_state.messages.append({
                        "user": suggestion,
                        "bot": response,
                        "intent": intent
                    })
                    st.rerun()
    
    # Text input and send button
    user_input_col, send_col = st.columns([8, 1])
    
    with user_input_col:
        user_input = st.text_input(
            "",
            placeholder="Type your question here... 💭",
            label_visibility="collapsed",
            key="user_input"
        )
    
    with send_col:
        send_button = st.button("Send", type="primary", use_container_width=True)
    
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Process user input
    if (send_button or user_input) and user_input and user_input.strip():
        if user_input.strip():
            # Get bot response
            response, intent = get_bot_response(user_input.strip())
            
            # Add to session history
            st.session_state.messages.append({
                "user": user_input.strip(),
                "bot": response,
                "intent": intent
            })
            
            # Clear input and rerun
            st.rerun()

# ========================
# Sidebar with Statistics (Optional)
# ========================



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



