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
import plotly.express as px
import plotly.graph_objects as go

# ========================
# Page config
# ========================
st.set_page_config(
    page_title="University Chatbot", 
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ========================
# Paths
# ========================
TRAIN_FILE = "data/train_data.csv"
FEEDBACK_FILE = "data/feedback.csv"
INTENTS_FILE = "data/intents.json"

# ========================
# Enhanced CSS Styling
# ========================
st.markdown("""
<style>
    /* Import Google Fonts */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    
    /* Global Styling */
    .stApp {
        font-family: 'Inter', sans-serif;
    }
    
    /* Header Styling */
    .main-header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2rem 1rem;
        border-radius: 15px;
        color: white;
        text-align: center;
        margin-bottom: 2rem;
        box-shadow: 0 8px 32px rgba(102, 126, 234, 0.3);
    }
    
    .main-header h1 {
        font-size: 2.5rem;
        font-weight: 700;
        margin: 0;
        text-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    
    .main-header p {
        font-size: 1.1rem;
        margin: 0.5rem 0 0 0;
        opacity: 0.9;
    }
    
    /* Chat Container */
    .chat-container {
        background: linear-gradient(145deg, #f8f9ff, #ffffff);
        border-radius: 20px;
        padding: 1.5rem;
        max-height: 500px;
        overflow-y: auto;
        box-shadow: 0 10px 40px rgba(0,0,0,0.1);
        border: 1px solid #e8eaff;
        margin-bottom: 1rem;
    }
    
    /* Chat Bubbles */
    .user-message {
        background: linear-gradient(135deg, #667eea, #764ba2);
        color: white;
        padding: 1rem 1.25rem;
        border-radius: 20px 20px 5px 20px;
        margin: 0.75rem 0 0.75rem 20%;
        box-shadow: 0 4px 15px rgba(102, 126, 234, 0.3);
        font-weight: 500;
        animation: slideInRight 0.3s ease-out;
    }
    
    .bot-message {
        background: linear-gradient(135deg, #ffffff, #f8f9ff);
        color: #2d3748;
        padding: 1rem 1.25rem;
        border-radius: 20px 20px 20px 5px;
        margin: 0.75rem 20% 0.75rem 0;
        border: 1px solid #e8eaff;
        box-shadow: 0 4px 15px rgba(0,0,0,0.08);
        animation: slideInLeft 0.3s ease-out;
    }
    
    .bot-message .metadata {
        font-size: 0.85rem;
        color: #718096;
        margin-top: 0.5rem;
        padding-top: 0.5rem;
        border-top: 1px solid #e8eaff;
    }
    
    /* Animations */
    @keyframes slideInRight {
        from { transform: translateX(100px); opacity: 0; }
        to { transform: translateX(0); opacity: 1; }
    }
    
    @keyframes slideInLeft {
        from { transform: translateX(-100px); opacity: 0; }
        to { transform: translateX(0); opacity: 1; }
    }
    
    /* Input Styling */
    .stTextInput > div > div > input {
        border-radius: 25px;
        border: 2px solid #e8eaff;
        padding: 0.75rem 1.25rem;
        font-size: 1rem;
        transition: all 0.3s ease;
    }
    
    .stTextInput > div > div > input:focus {
        border-color: #667eea;
        box-shadow: 0 0 0 3px rgba(102, 126, 234, 0.1);
    }
    
    /* Button Styling */
    .stButton > button {
        background: linear-gradient(135deg, #667eea, #764ba2);
        color: white;
        border: none;
        border-radius: 25px;
        padding: 0.75rem 2rem;
        font-weight: 600;
        transition: all 0.3s ease;
        box-shadow: 0 4px 15px rgba(102, 126, 234, 0.3);
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 25px rgba(102, 126, 234, 0.4);
    }
    
    /* Feedback Buttons */
    .feedback-container {
        display: flex;
        gap: 0.5rem;
        margin-top: 0.75rem;
    }
    
    .feedback-btn {
        padding: 0.5rem 1rem;
        border-radius: 20px;
        border: 1px solid #e8eaff;
        background: white;
        cursor: pointer;
        transition: all 0.3s ease;
        font-size: 0.9rem;
        font-weight: 500;
    }
    
    .feedback-btn.positive:hover {
        background: #48bb78;
        color: white;
        border-color: #48bb78;
    }
    
    .feedback-btn.negative:hover {
        background: #f56565;
        color: white;
        border-color: #f56565;
    }
    
    /* Stats Cards */
    .stats-card {
        background: linear-gradient(135deg, #ffffff, #f8f9ff);
        border-radius: 15px;
        padding: 1.5rem;
        border: 1px solid #e8eaff;
        box-shadow: 0 5px 20px rgba(0,0,0,0.08);
        text-align: center;
        transition: transform 0.3s ease;
    }
    
    .stats-card:hover {
        transform: translateY(-5px);
    }
    
    .stats-number {
        font-size: 2.5rem;
        font-weight: 700;
        background: linear-gradient(135deg, #667eea, #764ba2);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
    }
    
    .stats-label {
        color: #718096;
        font-weight: 500;
        margin-top: 0.5rem;
    }
    
    /* Sidebar Styling */
    .css-1d391kg {
        background: linear-gradient(180deg, #f8f9ff, #ffffff);
    }
    
    /* Tab Styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 2rem;
    }
    
    .stTabs [data-baseweb="tab"] {
        background: linear-gradient(135deg, #ffffff, #f8f9ff);
        border-radius: 10px;
        color: #4a5568;
        font-weight: 600;
        padding: 0.75rem 1.5rem;
        border: 1px solid #e8eaff;
    }
    
    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, #667eea, #764ba2);
        color: white;
    }
    
    /* Empty State */
    .empty-state {
        text-align: center;
        color: #718096;
        padding: 3rem 1rem;
    }
    
    .empty-state-icon {
        font-size: 4rem;
        margin-bottom: 1rem;
    }
    
    /* Success/Error Messages */
    .stSuccess, .stError {
        border-radius: 10px;
        border: none;
        box-shadow: 0 4px 15px rgba(0,0,0,0.1);
    }
</style>
""", unsafe_allow_html=True)

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
    model_loaded = True
except Exception as e:
    st.error(f"⚠️ Model training failed: {e}")
    pipeline, le, train_acc, test_acc = None, None, 0, 0
    model_loaded = False

# ========================
# Load intents
# ========================
try:
    with open(INTENTS_FILE, encoding="utf-8") as f:
        intents = json.load(f)
    intent_to_responses = {item["intent"]: item["responses"] for item in intents}
except:
    intent_to_responses = {"error": ["I'm having trouble accessing my knowledge base."]}

# Create feedback file if not exists
if not os.path.exists(FEEDBACK_FILE):
    os.makedirs(os.path.dirname(FEEDBACK_FILE), exist_ok=True)
    with open(FEEDBACK_FILE, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["user_input", "predicted_intent", "confidence", "response", "feedback"])

# ========================
# Sidebar
# ========================
with st.sidebar:
    st.markdown("### 🎓 Navigation")
    st.markdown("---")
    
    if model_loaded:
        st.success("✅ Model Ready")
    else:
        st.error("❌ Model Error")
    
    st.markdown("---")
    st.markdown("### 📊 Quick Stats")
    
    # Load feedback for stats
    if os.path.exists(FEEDBACK_FILE):
        try:
            feedback_df = pd.read_csv(FEEDBACK_FILE)
            total_conversations = len(feedback_df)
            positive_feedback = len(feedback_df[feedback_df['feedback'] == 'yes'])
            satisfaction_rate = (positive_feedback / total_conversations * 100) if total_conversations > 0 else 0
        except:
            total_conversations, satisfaction_rate = 0, 0
    else:
        total_conversations, satisfaction_rate = 0, 0
    
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Conversations", total_conversations)
    with col2:
        st.metric("Satisfaction", f"{satisfaction_rate:.0f}%")

# ========================
# Main Header
# ========================
st.markdown("""
<div class="main-header">
    <h1>🎓 University Assistant</h1>
    <p>Your intelligent guide to university life - ask about admissions, courses, tuition, and more!</p>
</div>
""", unsafe_allow_html=True)

# ========================
# Tabs
# ========================
tab1, tab2, tab3, tab4 = st.tabs(["💬 Chat", "📊 Analytics", "📥 Feedback Data", "⚙️ Settings"])

# ========================
# Tab 1: Enhanced Chatbot
# ========================
with tab1:
    if not model_loaded:
        st.error("🚫 Chatbot is currently unavailable due to model loading issues.")
        st.stop()
    
    # Keep chat history
    if "messages" not in st.session_state:
        st.session_state.messages = []
    
    # Chat interface
    col1, col2 = st.columns([4, 1])
    
    with col1:
        user_input = st.text_input(
            "Message", 
            placeholder="Ask me anything about the university...",
            key="user_input",
            label_visibility="collapsed"
        )
    
    with col2:
        send_button = st.button("Send 🚀", use_container_width=True)
    
    # Quick suggestions
    st.markdown("**💡 Quick suggestions:**")
    suggestion_cols = st.columns(4)
    suggestions = [
        "Admission requirements",
        "Tuition fees",
        "Available courses", 
        "Campus facilities"
    ]
    
    for i, suggestion in enumerate(suggestions):
        with suggestion_cols[i]:
            if st.button(suggestion, key=f"suggestion_{i}", use_container_width=True):
                user_input = suggestion
                send_button = True
    
    # Process input
    if (send_button and user_input.strip()) or user_input:
        if user_input.strip():
            # Predict intent + confidence
            try:
                y_pred = pipeline.predict([user_input])[0]
                intent = le.inverse_transform([y_pred])[0]
                proba = pipeline.predict_proba([user_input])[0]
                confidence = float(max(proba))
                
                response = random.choice(intent_to_responses.get(intent, ["I'm sorry, I didn't quite understand that. Could you please rephrase your question?"]))
                
                # Add to session history
                st.session_state.messages.append({
                    "user": user_input, 
                    "bot": response, 
                    "intent": intent, 
                    "confidence": confidence
                })
                
                # Clear input
                st.session_state.user_input = ""
                st.rerun()
                
            except Exception as e:
                st.error(f"Error processing your message: {str(e)}")
    
        # Display conversation
        # Display conversation
if st.session_state.messages:
    chat_html = '<div class="chat-container">'

    for idx, chat in enumerate(st.session_state.messages):
        # User bubble
        chat_html += f"""
        <div class="user-message">
            👤 {chat['user']}
        </div>
        """

        # Bot bubble
        confidence_color = "🟢" if chat['confidence'] > 0.7 else "🟡" if chat['confidence'] > 0.5 else "🔴"
        chat_html += f"""
        <div class="bot-message">
            🤖 {chat['bot']}
            <div class="metadata">
                🎯 <strong>Intent:</strong> {chat['intent']} | 
                {confidence_color} <strong>Confidence:</strong> {chat['confidence']:.2f}
            </div>
        </div>
        """

        # Feedback buttons (Streamlit)
        col1, col2, col3 = st.columns([1, 1, 4])
        with col1:
            if st.button("👍 Helpful", key=f"yes_{idx}", use_container_width=True):
                with open(FEEDBACK_FILE, "a", newline="", encoding="utf-8") as f:
                    writer = csv.writer(f)
                    writer.writerow([chat["user"], chat["intent"], f"{chat['confidence']:.2f}", chat["bot"], "yes"])
                st.success("✅ Thanks for your feedback!")
        with col2:
            if st.button("👎 Not Helpful", key=f"no_{idx}", use_container_width=True):
                with open(FEEDBACK_FILE, "a", newline="", encoding="utf-8") as f:
                    writer = csv.writer(f)
                    writer.writerow([chat["user"], chat["intent"], f"{chat['confidence']:.2f}", chat["bot"], "no"])
                st.error("📝 Feedback recorded. We'll improve!")

    chat_html += '</div>'
    st.markdown(chat_html, unsafe_allow_html=True)

    # Clear chat button
    if st.button("🗑️ Clear Chat History"):
        st.session_state.messages = []
        st.rerun()

else:
    st.markdown("""
    <div class="empty-state">
        <div class="empty-state-icon">💬</div>
        <h3>Start a conversation!</h3>
        <p>Ask me anything about the university and I'll do my best to help you.</p>
    </div>
    """, unsafe_allow_html=True)


# ========================
# Tab 2: Enhanced Analytics
# ========================
with tab2:
    st.subheader("📊 Chatbot Performance Analytics")
    
    # Model performance cards
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown(f"""
        <div class="stats-card">
            <div class="stats-number">{train_acc:.0%}</div>
            <div class="stats-label">Training Accuracy</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown(f"""
        <div class="stats-card">
            <div class="stats-number">{test_acc:.0%}</div>
            <div class="stats-label">Testing Accuracy</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown(f"""
        <div class="stats-card">
            <div class="stats-number">{total_conversations}</div>
            <div class="stats-label">Total Interactions</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        st.markdown(f"""
        <div class="stats-card">
            <div class="stats-number">{satisfaction_rate:.0f}%</div>
            <div class="stats-label">User Satisfaction</div>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Analytics charts
    if os.path.exists(FEEDBACK_FILE) and total_conversations > 0:
        try:
            feedback_df = pd.read_csv(FEEDBACK_FILE)
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("📈 Feedback Distribution")
                feedback_counts = feedback_df['feedback'].value_counts()
                
                fig_pie = px.pie(
                    values=feedback_counts.values,
                    names=['Positive' if x == 'yes' else 'Negative' for x in feedback_counts.index],
                    color_discrete_map={'Positive': '#48bb78', 'Negative': '#f56565'}
                )
                fig_pie.update_layout(showlegend=True, height=300)
                st.plotly_chart(fig_pie, use_container_width=True)
            
            with col2:
                st.subheader("🎯 Top Intents")
                intent_counts = feedback_df['predicted_intent'].value_counts().head(5)
                
                fig_bar = px.bar(
                    x=intent_counts.values,
                    y=intent_counts.index,
                    orientation='h',
                    color=intent_counts.values,
                    color_continuous_scale='Blues'
                )
                fig_bar.update_layout(showlegend=False, height=300)
                st.plotly_chart(fig_bar, use_container_width=True)
            
            # Confidence distribution
            st.subheader("📊 Confidence Score Distribution")
            fig_hist = px.histogram(
                feedback_df,
                x='confidence',
                nbins=20,
                color='feedback',
                color_discrete_map={'yes': '#48bb78', 'no': '#f56565'}
            )
            fig_hist.update_layout(height=300)
            st.plotly_chart(fig_hist, use_container_width=True)
            
        except Exception as e:
            st.error(f"Error loading analytics: {str(e)}")
    else:
        st.info("📊 No feedback data available yet. Start chatting to see analytics!")

# ========================
# Tab 3: Enhanced Feedback Data
# ========================
with tab3:
    st.subheader("📥 Feedback Data Management")
    
    if os.path.exists(FEEDBACK_FILE):
        try:
            fb_data = pd.read_csv(FEEDBACK_FILE)
            
            if len(fb_data) > 0:
                col1, col2, col3 = st.columns([2, 1, 1])
                
                with col1:
                    st.metric("Total Entries", len(fb_data))
                with col2:
                    positive_count = len(fb_data[fb_data['feedback'] == 'yes'])
                    st.metric("Positive", positive_count)
                with col3:
                    negative_count = len(fb_data[fb_data['feedback'] == 'no'])
                    st.metric("Negative", negative_count)
                
                st.markdown("---")
                
                # Filters
                col1, col2 = st.columns(2)
                with col1:
                    feedback_filter = st.selectbox(
                        "Filter by Feedback",
                        options=["All", "Positive", "Negative"]
                    )
                
                with col2:
                    intent_filter = st.selectbox(
                        "Filter by Intent",
                        options=["All"] + list(fb_data['predicted_intent'].unique())
                    )
                
                # Apply filters
                filtered_data = fb_data.copy()
                if feedback_filter != "All":
                    filter_value = "yes" if feedback_filter == "Positive" else "no"
                    filtered_data = filtered_data[filtered_data['feedback'] == filter_value]
                
                if intent_filter != "All":
                    filtered_data = filtered_data[filtered_data['predicted_intent'] == intent_filter]
                
                st.dataframe(
                    filtered_data,
                    use_container_width=True,
                    height=400
                )
                
                # Download options
                col1, col2 = st.columns(2)
                
                with col1:
                    csv_data = fb_data.to_csv(index=False).encode("utf-8")
                    st.download_button(
                        label="⬇️ Download All Data (CSV)",
                        data=csv_data,
                        file_name="feedback_data.csv",
                        mime="text/csv",
                        use_container_width=True
                    )
                
                with col2:
                    if len(filtered_data) > 0:
                        filtered_csv = filtered_data.to_csv(index=False).encode("utf-8")
                        st.download_button(
                            label="⬇️ Download Filtered Data",
                            data=filtered_csv,
                            file_name="filtered_feedback_data.csv",
                            mime="text/csv",
                            use_container_width=True
                        )
            else:
                st.info("📊 No feedback data collected yet.")
                
        except Exception as e:
            st.error(f"Error loading feedback data: {str(e)}")
    else:
        st.info("📊 No feedback data available yet.")

# ========================
# Tab 4: Settings
# ========================
with tab4:
    st.subheader("⚙️ Chatbot Settings")
    
    # Model information
    st.markdown("### 🤖 Model Information")
    
    col1, col2 = st.columns(2)
    with col1:
        st.info(f"**Model Type:** Logistic Regression with TF-IDF")
        st.info(f"**Training Accuracy:** {train_acc:.2%}")
    
    with col2:
        st.info(f"**Testing Accuracy:** {test_acc:.2%}")
        st.info(f"**Status:** {'✅ Active' if model_loaded else '❌ Error'}")
    
    st.markdown("---")
    
    # Data management
    st.markdown("### 📁 Data Management")
    
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("🔄 Retrain Model", use_container_width=True):
            with st.spinner("Retraining model..."):
                try:
                    pipeline, le, train_acc, test_acc = retrain_model()
                    st.success("✅ Model retrained successfully!")
                    st.rerun()
                except Exception as e:
                    st.error(f"❌ Retraining failed: {str(e)}")
    
    with col2:
        if st.button("🗑️ Clear All Data", use_container_width=True):
            if st.checkbox("⚠️ I understand this will delete all feedback data"):
                try:
                    if os.path.exists(FEEDBACK_FILE):
                        os.remove(FEEDBACK_FILE)
                    st.session_state.messages = []
                    st.success("✅ All data cleared!")
                    st.rerun()
                except Exception as e:
                    st.error(f"❌ Error clearing data: {str(e)}")
    
    st.markdown("---")
    
    # About section
    st.markdown("### ℹ️ About")
    st.markdown("""
    This University Chatbot is built using:
    - **Streamlit** for the web interface
    - **Scikit-learn** for machine learning
    - **TF-IDF Vectorization** for text processing
    - **Logistic Regression** for intent classification
    - **Plotly** for interactive charts
    
    The chatbot learns from user feedback to improve its responses over time.
    """)


