import streamlit as st
import pickle
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

st.set_page_config(
    page_title="Spam Email Detector",
    page_icon="🛡️",
    layout="centered",
    initial_sidebar_state="collapsed"
)

# Custom CSS
st.markdown("""
    <style>
        [data-testid="stSidebar"] { display: none; }
        .main-header { font-size: 2.8rem; color: #1E88E5; text-align: center; margin-bottom: 0.5rem; }
        .sub-header { text-align: center; color: #555; font-size: 1.1rem; margin-bottom: 2rem; }
        .result-card {
            padding: 30px; border-radius: 15px; margin: 25px 0;
            text-align: center; font-size: 1.8rem; font-weight: bold;
            box-shadow: 0 4px 15px rgba(0,0,0,0.1);
        }
        .spam { background-color: #ff4d4d !important; color: white !important; border: 5px solid #d32f2f; }
        .ham { background-color: #4caf50 !important; color: white !important; border: 5px solid #2e7d32; }
    </style>
""", unsafe_allow_html=True)

nltk.download('stopwords', quiet=True)

st.markdown("<h1 class='main-header'>🛡️ Spam Email Detector</h1>", unsafe_allow_html=True)
st.markdown("<p class='sub-header'>Advanced Email Spam Detection</p>", unsafe_allow_html=True)

# Load Model
@st.cache_resource
def load_model():
    try:
        with open("MultinomialNaiveBayes.pkl", 'rb') as f:
            model = pickle.load(f)
        with open("vectorizer.pkl", 'rb') as f:
            cv = pickle.load(f)
        return model, cv
    except Exception:
        st.error("❌ Model files not found.")
        st.stop()

model, cv = load_model()

# Initialize Session State
if 'text_input' not in st.session_state:
    st.session_state.text_input = ""

# Callback Function
def set_example(text):
    st.session_state.text_input = text

# Text Area
user_input = st.text_area(
    "Enter your email or message:",
    value=st.session_state.text_input,
    height=220,
    placeholder="Paste the email content here...",
    key="main_text_area"
)

# Quick Examples
st.markdown("**Quick Examples:**")
col1, col2, col3 = st.columns(3)

with col1:
    if st.button("Spam Example 1", use_container_width=True):
        set_example("Congratulations! You've won a $1000 Walmart gift card. Click here to claim now! Limited time offer.")
        st.rerun()

with col2:
    if st.button("Spam Example 2", use_container_width=True):
        set_example("Your account has been suspended. Verify your details immediately or it will be permanently closed.")
        st.rerun()

with col3:
    if st.button("Safe Example", use_container_width=True):
        set_example("Hey John, can we reschedule our meeting to 4 PM tomorrow? Let me know if that works.")
        st.rerun()

# Predict Button
if st.button("🔍 Check for Spam", type="primary", use_container_width=True):
    if not user_input.strip():
        st.warning("⚠️ Please enter a message.")
    else:
        with st.spinner("Analyzing..."):
            ps = PorterStemmer()
            review = re.sub('[^a-zA-Z]', ' ', user_input).lower().split()
            review = [ps.stem(word) for word in review if word not in stopwords.words('english')]
            review = ' '.join(review)
            
            input_vector = cv.transform([review]).toarray()
            prediction = model.predict(input_vector)[0]
            prob = model.predict_proba(input_vector)[0]
            
            if prediction == 1:
                st.markdown('<div class="result-card spam">🚨 THIS IS SPAM</div>', unsafe_allow_html=True)
                st.metric("Spam Confidence", f"{prob[1]*100:.2f}%")
            else:
                st.markdown('<div class="result-card ham">✅ THIS IS SAFE (Ham)</div>', unsafe_allow_html=True)
                st.metric("Safe Confidence", f"{prob[0]*100:.2f}%")
            
            confidence = prob[1] if prediction == 1 else prob[0]
            st.progress(confidence)

# Footer
st.markdown("---")
st.markdown("""
    <p style='text-align: center; color: #777; font-size: 0.95rem; margin-top: 10px;'>
        ADVANCED SPAM EMAIL DETECTOR
    </p>
""", unsafe_allow_html=True)
