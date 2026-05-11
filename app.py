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

# Session State
if 'message' not in st.session_state:
    st.session_state.message = ""

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
        st.error("❌ Model files not found. Please upload `MultinomialNaiveBayes.pkl` and `vectorizer.pkl` to your repository.")
        st.stop()

model, cv = load_model()

# Input
message = st.text_area(
    "Enter your email or message:",
    value=st.session_state.message,
    height=220,
    placeholder="Paste the email content here...",
    key="message_input"          # ← This line was missing (Important)
)

# Examples
st.markdown("**Quick Examples:**")
c1, c2, c3 = st.columns(3)
with c1:
    if st.button("Spam Example 1", use_container_width=True):
        st.session_state.message = "Congratulations! You've won a $1000 Walmart gift card. Click here to claim now!"
        st.rerun()
with c2:
    if st.button("Spam Example 2", use_container_width=True):
        st.session_state.message = "Your account has been suspended. Verify your details immediately."
        st.rerun()
with c3:
    if st.button("Safe Example", use_container_width=True):
        st.session_state.message = "Hey, are we still meeting tomorrow at 3 PM?"
        st.rerun()

# Predict Button
if st.button("🔍 Check for Spam", type="primary", use_container_width=True):
    if not message.strip():
        st.warning("Please enter a message.")
    else:
        with st.spinner("Analyzing..."):
            ps = PorterStemmer()
            review = re.sub('[^a-zA-Z]', ' ', message).lower().split()
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

st.markdown("---")
st.markdown("""
    <p style='text-align: center; color: #777; font-size: 0.95rem; margin-top: 10px;'>
        ADVANCED SPAM EMAIL DETECTOR
    </p>
""", unsafe_allow_html=True)