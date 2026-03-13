import streamlit as st
import pickle
import numpy as np
import time

from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

# -----------------------------
# Page Config
# -----------------------------
st.set_page_config(
    page_title="Next Word Predictor",
    page_icon="🧠",
    layout="centered",
    initial_sidebar_state="collapsed"
)

# -----------------------------
# Hide Streamlit defaults & Inject Full Custom CSS
# -----------------------------
st.markdown("""
<style>
/* ===== HIDE STREAMLIT DEFAULTS ===== */
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
header {visibility: hidden;}
.stDeployButton {display: none;}
div[data-testid="stToolbar"] {display: none;}
div[data-testid="stDecoration"] {display: none;}
div[data-testid="stStatusWidget"] {display: none;}

/* ===== IMPORT FONTS ===== */
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800;900&family=JetBrains+Mono:wght@400;500;600&display=swap');

/* ===== GLOBAL BACKGROUND ===== */
.stApp {
    background: linear-gradient(135deg, #0a0a0f 0%, #0d1117 25%, #101820 50%, #0d1117 75%, #0a0a0f 100%);
    font-family: 'Inter', sans-serif;
    color: #e0e0e0;
}

/* Animated background particles effect */
.stApp::before {
    content: '';
    position: fixed;
    top: 0; left: 0; right: 0; bottom: 0;
    background:
        radial-gradient(ellipse 800px 600px at 20% 30%, rgba(88, 28, 255, 0.08), transparent),
        radial-gradient(ellipse 600px 500px at 80% 70%, rgba(0, 198, 255, 0.06), transparent),
        radial-gradient(ellipse 500px 400px at 50% 50%, rgba(168, 85, 247, 0.04), transparent);
    pointer-events: none;
    z-index: 0;
}

/* ===== REMOVE PADDING ===== */
.block-container {
    padding-top: 0rem !important;
    padding-bottom: 2rem !important;
    max-width: 820px !important;
}

/* ===== TOP NAVIGATION BAR ===== */
.top-nav {
    display: flex;
    align-items: center;
    justify-content: space-between;
    padding: 14px 30px;
    background: rgba(13, 17, 23, 0.85);
    backdrop-filter: blur(20px);
    -webkit-backdrop-filter: blur(20px);
    border-bottom: 1px solid rgba(255, 255, 255, 0.06);
    margin: -1rem -4rem 0 -4rem;
    position: sticky;
    top: 0;
    z-index: 999;
}

.nav-brand {
    display: flex;
    align-items: center;
    gap: 10px;
    font-weight: 700;
    font-size: 1.15rem;
    color: #ffffff;
    letter-spacing: -0.3px;
}

.nav-brand-icon {
    width: 32px;
    height: 32px;
    border-radius: 10px;
    background: linear-gradient(135deg, #7c3aed, #2563eb);
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 16px;
}

.nav-links {
    display: flex;
    gap: 6px;
}

.nav-link {
    padding: 7px 18px;
    border-radius: 8px;
    font-size: 0.82rem;
    font-weight: 500;
    color: #9ca3af;
    text-decoration: none;
    transition: all 0.25s ease;
    cursor: pointer;
    letter-spacing: 0.2px;
}

.nav-link:hover, .nav-link-active {
    background: rgba(124, 58, 237, 0.15);
    color: #c4b5fd;
}

.nav-status {
    display: flex;
    align-items: center;
    gap: 8px;
    font-size: 0.78rem;
    color: #6ee7b7;
    font-weight: 500;
}

.status-dot {
    width: 8px;
    height: 8px;
    background: #34d399;
    border-radius: 50%;
    animation: statusPulse 2s ease-in-out infinite;
    box-shadow: 0 0 8px rgba(52, 211, 153, 0.5);
}

@keyframes statusPulse {
    0%, 100% { opacity: 1; transform: scale(1); }
    50% { opacity: 0.5; transform: scale(0.85); }
}

/* ===== HERO SECTION ===== */
.hero-section {
    text-align: center;
    padding: 48px 20px 12px 20px;
    position: relative;
}

.hero-badge {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    padding: 5px 16px;
    background: rgba(124, 58, 237, 0.12);
    border: 1px solid rgba(124, 58, 237, 0.25);
    border-radius: 50px;
    font-size: 0.72rem;
    font-weight: 600;
    color: #c4b5fd;
    text-transform: uppercase;
    letter-spacing: 1.2px;
    margin-bottom: 20px;
}

.hero-title {
    font-size: 2.8rem;
    font-weight: 800;
    letter-spacing: -1.5px;
    line-height: 1.15;
    margin-bottom: 14px;
    background: linear-gradient(135deg, #ffffff 0%, #c4b5fd 40%, #818cf8 60%, #60a5fa 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    animation: titleShimmer 4s ease-in-out infinite;
    background-size: 200% auto;
}

@keyframes titleShimmer {
    0%, 100% { background-position: 0% center; }
    50% { background-position: 100% center; }
}

.hero-subtitle {
    font-size: 1rem;
    color: #6b7280;
    font-weight: 400;
    max-width: 500px;
    margin: 0 auto;
    line-height: 1.6;
}

.hero-subtitle strong {
    color: #a78bfa;
    font-weight: 600;
}

/* ===== GLASSMORPHISM CARD ===== */
.glass-card {
    background: rgba(255, 255, 255, 0.03);
    backdrop-filter: blur(24px);
    -webkit-backdrop-filter: blur(24px);
    border: 1px solid rgba(255, 255, 255, 0.07);
    border-radius: 20px;
    padding: 32px 30px;
    margin: 24px auto 20px auto;
    position: relative;
    overflow: hidden;
    box-shadow:
        0 8px 32px rgba(0, 0, 0, 0.3),
        inset 0 1px 0 rgba(255, 255, 255, 0.05);
}

.glass-card::before {
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0;
    height: 1px;
    background: linear-gradient(90deg, transparent, rgba(124, 58, 237, 0.4), rgba(59, 130, 246, 0.4), transparent);
}

.card-header {
    display: flex;
    align-items: center;
    gap: 10px;
    margin-bottom: 22px;
    font-size: 0.88rem;
    font-weight: 600;
    color: #d1d5db;
}

.card-header-icon {
    width: 28px;
    height: 28px;
    border-radius: 8px;
    background: linear-gradient(135deg, rgba(124, 58, 237, 0.2), rgba(59, 130, 246, 0.2));
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 14px;
    border: 1px solid rgba(124, 58, 237, 0.2);
}

/* ===== STREAMLIT INPUT OVERRIDES ===== */
.stTextInput > div > div > input {
    background: rgba(255, 255, 255, 0.04) !important;
    border: 1px solid rgba(255, 255, 255, 0.1) !important;
    border-radius: 14px !important;
    color: #f0f0f0 !important;
    font-family: 'Inter', sans-serif !important;
    font-size: 0.95rem !important;
    padding: 14px 18px !important;
    transition: all 0.3s ease !important;
    caret-color: #a78bfa !important;
}

.stTextInput > div > div > input:focus {
    border-color: rgba(124, 58, 237, 0.5) !important;
    box-shadow: 0 0 0 3px rgba(124, 58, 237, 0.1), 0 0 20px rgba(124, 58, 237, 0.08) !important;
    background: rgba(255, 255, 255, 0.06) !important;
}

.stTextInput > div > div > input::placeholder {
    color: #4b5563 !important;
}

.stTextInput label {
    color: #9ca3af !important;
    font-size: 0.82rem !important;
    font-weight: 500 !important;
    letter-spacing: 0.3px !important;
}

/* ===== PROMPT CHIPS ===== */
.prompt-chips-section {
    margin: 14px 0 6px 0;
}

.prompt-chips-label {
    font-size: 0.72rem;
    color: #6b7280;
    font-weight: 500;
    text-transform: uppercase;
    letter-spacing: 1px;
    margin-bottom: 10px;
}

/* ===== STREAMLIT BUTTON OVERRIDES ===== */
.stButton > button {
    background: linear-gradient(135deg, #7c3aed 0%, #4f46e5 50%, #2563eb 100%) !important;
    color: white !important;
    border: none !important;
    border-radius: 12px !important;
    padding: 11px 28px !important;
    font-weight: 600 !important;
    font-size: 0.85rem !important;
    font-family: 'Inter', sans-serif !important;
    letter-spacing: 0.2px !important;
    transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1) !important;
    box-shadow: 0 4px 15px rgba(124, 58, 237, 0.25) !important;
    width: 100% !important;
}

.stButton > button:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 8px 25px rgba(124, 58, 237, 0.35) !important;
    filter: brightness(1.1) !important;
}

.stButton > button:active {
    transform: translateY(0px) !important;
}

/* Chip buttons - specific columns */
div[data-testid="stHorizontalBlock"]:has(.chip-row) .stButton > button {
    background: rgba(255, 255, 255, 0.05) !important;
    border: 1px solid rgba(255, 255, 255, 0.1) !important;
    color: #c4b5fd !important;
    padding: 8px 16px !important;
    font-size: 0.78rem !important;
    border-radius: 50px !important;
    box-shadow: none !important;
    font-weight: 500 !important;
}

div[data-testid="stHorizontalBlock"]:has(.chip-row) .stButton > button:hover {
    background: rgba(124, 58, 237, 0.15) !important;
    border-color: rgba(124, 58, 237, 0.35) !important;
    transform: translateY(-1px) !important;
    box-shadow: 0 4px 12px rgba(124, 58, 237, 0.15) !important;
}

/* ===== RESULT BOX ===== */
.result-container {
    margin-top: 20px;
    animation: resultSlideIn 0.5s cubic-bezier(0.4, 0, 0.2, 1);
}

@keyframes resultSlideIn {
    from { opacity: 0; transform: translateY(15px); }
    to { opacity: 1; transform: translateY(0); }
}

.result-box {
    background: rgba(255, 255, 255, 0.03);
    border: 1px solid rgba(255, 255, 255, 0.08);
    border-radius: 16px;
    padding: 24px 26px;
    position: relative;
    overflow: hidden;
}

.result-box::before {
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0;
    height: 2px;
    background: linear-gradient(90deg, #7c3aed, #3b82f6, #06b6d4);
    background-size: 200% auto;
    animation: gradientFlow 3s linear infinite;
}

@keyframes gradientFlow {
    0% { background-position: 0% center; }
    100% { background-position: 200% center; }
}

.result-label {
    display: flex;
    align-items: center;
    gap: 8px;
    font-size: 0.72rem;
    color: #6b7280;
    text-transform: uppercase;
    letter-spacing: 1px;
    font-weight: 600;
    margin-bottom: 12px;
}

.result-text {
    font-size: 1.2rem;
    font-weight: 600;
    color: #f0f0f0;
    line-height: 1.7;
    letter-spacing: -0.2px;
}

.result-text .highlight-word {
    background: linear-gradient(135deg, #7c3aed, #3b82f6);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    font-weight: 700;
}

/* ===== THINKING INDICATOR ===== */
.thinking-indicator {
    display: flex;
    align-items: center;
    gap: 12px;
    padding: 16px 20px;
    background: rgba(124, 58, 237, 0.06);
    border: 1px solid rgba(124, 58, 237, 0.12);
    border-radius: 14px;
    margin-top: 16px;
    animation: thinkingFadeIn 0.3s ease;
}

@keyframes thinkingFadeIn {
    from { opacity: 0; }
    to { opacity: 1; }
}

.thinking-dots {
    display: flex;
    gap: 5px;
}

.thinking-dot {
    width: 7px;
    height: 7px;
    background: #a78bfa;
    border-radius: 50%;
    animation: thinkingBounce 1.4s ease-in-out infinite;
}

.thinking-dot:nth-child(2) { animation-delay: 0.2s; }
.thinking-dot:nth-child(3) { animation-delay: 0.4s; }

@keyframes thinkingBounce {
    0%, 80%, 100% { transform: scale(0.6); opacity: 0.4; }
    40% { transform: scale(1); opacity: 1; }
}

.thinking-text {
    font-size: 0.82rem;
    color: #a78bfa;
    font-weight: 500;
}

/* ===== WARNING / ERROR ===== */
.warning-box {
    display: flex;
    align-items: center;
    gap: 10px;
    padding: 14px 20px;
    background: rgba(245, 158, 11, 0.08);
    border: 1px solid rgba(245, 158, 11, 0.2);
    border-radius: 12px;
    margin-top: 16px;
    font-size: 0.85rem;
    color: #fbbf24;
    font-weight: 500;
    animation: resultSlideIn 0.3s ease;
}

/* ===== STATS BAR ===== */
.stats-bar {
    display: flex;
    justify-content: center;
    gap: 40px;
    padding: 20px;
    margin-top: 12px;
}

.stat-item {
    text-align: center;
}

.stat-value {
    font-size: 1.3rem;
    font-weight: 700;
    background: linear-gradient(135deg, #c4b5fd, #818cf8);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
}

.stat-label {
    font-size: 0.68rem;
    color: #6b7280;
    text-transform: uppercase;
    letter-spacing: 0.8px;
    font-weight: 500;
    margin-top: 3px;
}

/* ===== FOOTER ===== */
.custom-footer {
    text-align: center;
    padding: 28px 20px 10px 20px;
    color: #374151;
    font-size: 0.75rem;
    font-weight: 400;
    letter-spacing: 0.3px;
    border-top: 1px solid rgba(255, 255, 255, 0.04);
    margin-top: 30px;
}

.custom-footer a {
    color: #7c3aed;
    text-decoration: none;
    font-weight: 600;
}

/* ===== TYPING CURSOR ===== */
.typing-cursor {
    display: inline-block;
    width: 2px;
    height: 1.1em;
    background: #a78bfa;
    margin-left: 3px;
    vertical-align: text-bottom;
    animation: cursorBlink 1s step-end infinite;
}

@keyframes cursorBlink {
    0%, 100% { opacity: 1; }
    50% { opacity: 0; }
}

/* ===== HIDE DEFAULT STREAMLIT ALERTS (we use custom) ===== */
.stAlert { display: none !important; }

/* ===== SCROLLBAR ===== */
::-webkit-scrollbar { width: 6px; }
::-webkit-scrollbar-track { background: transparent; }
::-webkit-scrollbar-thumb { background: rgba(124, 58, 237, 0.3); border-radius: 10px; }
::-webkit-scrollbar-thumb:hover { background: rgba(124, 58, 237, 0.5); }

/* ===== RESPONSIVE ===== */
@media (max-width: 640px) {
    .hero-title { font-size: 2rem; }
    .top-nav { padding: 10px 16px; }
    .glass-card { padding: 22px 18px; }
    .stats-bar { gap: 24px; }
}
</style>
""", unsafe_allow_html=True)

# -----------------------------
# Top Navigation Bar
# -----------------------------
st.markdown("""
<div class="top-nav">
    <div class="nav-brand">
        <div class="nav-brand-icon">🧠</div>
        NeuralText
    </div>
    <div class="nav-links">
        <span class="nav-link nav-link-active">⌂ Home</span>
        <span class="nav-link">◈ Model</span>
        <span class="nav-link">▷ Prediction</span>
        <span class="nav-link">ⓘ About</span>
    </div>
    <div class="nav-status">
        <div class="status-dot"></div>
        LSTM Online
    </div>
</div>
""", unsafe_allow_html=True)

# -----------------------------
# Hero Section
# -----------------------------
st.markdown("""
<div class="hero-section">
    <div class="hero-badge">✦ Powered by Deep Learning</div>
    <div class="hero-title">Next Word Predictor</div>
    <div class="hero-subtitle">
        Generate intelligent text predictions using an <strong>LSTM neural network</strong> trained on real-world data.
    </div>
</div>
""", unsafe_allow_html=True)

# -----------------------------
# Stats Bar
# -----------------------------
st.markdown("""
<div class="stats-bar">
    <div class="stat-item">
        <div class="stat-value">LSTM</div>
        <div class="stat-label">Architecture</div>
    </div>
    <div class="stat-item">
        <div class="stat-value">Keras</div>
        <div class="stat-label">Framework</div>
    </div>
    <div class="stat-item">
        <div class="stat-value">10</div>
        <div class="stat-label">Max Words</div>
    </div>
    <div class="stat-item">
        <div class="stat-value">~0.3s</div>
        <div class="stat-label">Latency</div>
    </div>
</div>
""", unsafe_allow_html=True)

# -----------------------------
# Load Model
# -----------------------------
@st.cache_resource
def load_all():
    model = load_model("next_word_model.h5", compile=False)
    with open("tokenizer.pkl", "rb") as f:
        tokenizer = pickle.load(f)

    with open("max_len.pkl", "rb") as f:
        max_len = pickle.load(f)

    return model, tokenizer, max_len

model, tokenizer, max_len = load_all()

# -----------------------------
# Prediction Function
# -----------------------------
def predict_next_word(text):

    seq = tokenizer.texts_to_sequences([text])[0]
    seq = pad_sequences([seq], maxlen=max_len-1, padding='pre')

    pred = model.predict(seq, verbose=0)[0]

    temperature = 0.7
    pred = np.log(pred + 1e-8) / temperature
    pred = np.exp(pred) / np.sum(np.exp(pred))

    top_k = 5
    top_indices = np.argsort(pred)[-top_k:]
    top_probs = pred[top_indices]
    top_probs = top_probs / np.sum(top_probs)

    pred_index = np.random.choice(top_indices, p=top_probs)

    next_word = ""

    for word, index in tokenizer.word_index.items():
        if index == pred_index:
            next_word = word
            break

    return next_word

# -----------------------------
# Sentence Generator
# -----------------------------
def generate_sentence(seed, n_words):

    sentence = seed

    for _ in range(n_words):

        seq = tokenizer.texts_to_sequences([sentence])[0]
        seq = pad_sequences([seq], maxlen=max_len-1, padding='pre')

        pred = model.predict(seq, verbose=0)[0]
        pred_index = np.argmax(pred)

        next_word = ""

        for word, index in tokenizer.word_index.items():
            if index == pred_index:
                next_word = word
                break

        if next_word == "":
            break

        sentence = sentence + " " + next_word

    return sentence

# -----------------------------
# Main Card UI
# -----------------------------
st.markdown("""
<div class="glass-card">
    <div class="card-header">
        <div class="card-header-icon">✦</div>
        Text Input — Enter your seed phrase below
    </div>
""", unsafe_allow_html=True)

# Initialize session state
if "user_input" not in st.session_state:
    st.session_state.user_input = ""

# Text input
user_input = st.text_input(
    "Your prompt",
    value=st.session_state.user_input,
    placeholder="Type a phrase like 'the future of'...",
    label_visibility="collapsed",
    key="text_input_main"
)

# Example Prompt Chips
st.markdown('<div class="prompt-chips-label">💡 Try an example</div>', unsafe_allow_html=True)

chip_cols = st.columns([1, 1, 1, 1, 1])
chip_labels = ["life is", "who are", "success comes", "the world", "i want to"]
chip_clicked = None

st.markdown('<div class="chip-row"></div>', unsafe_allow_html=True)
for i, label in enumerate(chip_labels):
    with chip_cols[i]:
        if st.button(label, key=f"chip_{i}"):
            chip_clicked = label

# Handle chip click
if chip_clicked:
    st.session_state.user_input = chip_clicked
    st.rerun()

st.markdown('</div>', unsafe_allow_html=True)  # close glass-card

# Action Buttons
btn_cols = st.columns(2, gap="medium")

with btn_cols[0]:
    predict_btn = st.button("⚡  Predict Next Word", key="predict_btn", use_container_width=True)

with btn_cols[1]:
    generate_btn = st.button("✨  Generate Sentence", key="generate_btn", use_container_width=True)

# Use current input
current_input = user_input if user_input else st.session_state.user_input

# -----------------------------
# Prediction Output
# -----------------------------
if predict_btn:
    if current_input.strip() == "":
        st.markdown("""
        <div class="warning-box">
            ⚠️ Please enter a text prompt to get a prediction.
        </div>
        """, unsafe_allow_html=True)
    else:
        # Thinking indicator
        thinking_placeholder = st.empty()
        thinking_placeholder.markdown("""
        <div class="thinking-indicator">
            <div class="thinking-dots">
                <div class="thinking-dot"></div>
                <div class="thinking-dot"></div>
                <div class="thinking-dot"></div>
            </div>
            <span class="thinking-text">🤖 AI is thinking...</span>
        </div>
        """, unsafe_allow_html=True)

        word = predict_next_word(current_input)
        time.sleep(0.6)  # slight delay for effect
        thinking_placeholder.empty()

        st.markdown(f"""
        <div class="result-container">
            <div class="result-box">
                <div class="result-label">✦ Predicted Next Word</div>
                <div class="result-text">
                    {current_input} <span class="highlight-word">{word}</span><span class="typing-cursor"></span>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)

# -----------------------------
# Sentence Output
# -----------------------------
if generate_btn:
    if current_input.strip() == "":
        st.markdown("""
        <div class="warning-box">
            ⚠️ Please enter a text prompt to generate a sentence.
        </div>
        """, unsafe_allow_html=True)
    else:
        # Thinking indicator
        thinking_placeholder = st.empty()
        thinking_placeholder.markdown("""
        <div class="thinking-indicator">
            <div class="thinking-dots">
                <div class="thinking-dot"></div>
                <div class="thinking-dot"></div>
                <div class="thinking-dot"></div>
            </div>
            <span class="thinking-text">🤖 AI is generating text...</span>
        </div>
        """, unsafe_allow_html=True)

        sentence = generate_sentence(current_input, 10)
        time.sleep(0.8)  # slight delay for effect
        thinking_placeholder.empty()

        # Split to highlight generated words
        original_words = current_input.strip().split()
        all_words = sentence.strip().split()
        generated_words = all_words[len(original_words):]

        original_part = " ".join(original_words)
        generated_part = " ".join(generated_words)

        st.markdown(f"""
        <div class="result-container">
            <div class="result-box">
                <div class="result-label">✦ Generated Sentence</div>
                <div class="result-text">
                    {original_part} <span class="highlight-word">{generated_part}</span><span class="typing-cursor"></span>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)

# -----------------------------
# Footer
# -----------------------------
st.markdown("""
<div class="custom-footer">
    Built with ❤️ by <a href="#">Anmol</a> · Deep Learning · LSTM · TensorFlow/Keras · Streamlit
</div>
""", unsafe_allow_html=True)
