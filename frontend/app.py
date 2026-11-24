
import streamlit as st
import subprocess
import sys
import os
from pathlib import Path

# Project paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent

SCRIPTS = {
    "ASL (American Sign Language)": "src/predict.py",
    "ISL (Indian Sign Language)": "src_isl/predict_isl_webcam.py"
}

# Page configuration
st.set_page_config(
    page_title="SignConnect",
    page_icon="🤝",
    layout="wide"
)

# Styling
st.markdown("""
<style>
.main {
    background-color: #f5f7fb;
}
.title {
    text-align: center;
    color: #24476b;
    font-size: 48px;
    font-weight: bold;
}
.subtitle {
    text-align: center;
    color: #52677d;
    font-size: 18px;
}
.info-card {
    background-color: #e8f0fa;
    padding: 20px;
    border-radius: 12px;
    margin: 10px 0;
}
</style>
""", unsafe_allow_html=True)

# Session state
if "process" not in st.session_state:
    st.session_state.process = None

# Header
st.markdown(
    '<div class="title">SignConnect</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">'
    'Bridging Communication Through Sign Language Recognition'
    '</div>',
    unsafe_allow_html=True
)

st.divider()

# Introduction
st.subheader("Welcome to SignConnect")

st.write(
    "SignConnect is an AI-powered sign language recognition "
    "system that recognises hand signs and converts them "
    "into English alphabets."
)

# Language selection
st.subheader("Select Sign Language")

language = st.selectbox(
    "Choose your preferred language:",
    list(SCRIPTS.keys())
)

# Instructions
if language.startswith("ASL"):
    st.info(
        "ASL uses the American Sign Language recognition model. "
        "The webcam window also supports basic sentence formation."
    )
else:
    st.info(
        "ISL uses the Indian Sign Language recognition model "
        "for real-time alphabet recognition."
    )

st.divider()

# Recognition controls
st.subheader("Live Sign Recognition")

col1, col2 = st.columns(2)

with col1:
    start = st.button(
        "Start Recognition",
        type="primary",
        use_container_width=True
    )

with col2:
    stop = st.button(
        "Stop Recognition",
        use_container_width=True
    )

process = st.session_state.process

if start:
    if process is not None and process.poll() is None:
        st.warning("Recognition is already running.")
    else:
        script = PROJECT_ROOT / SCRIPTS[language]

        if not script.exists():
            st.error(f"Prediction script not found: {script}")
        else:
            try:
                process = subprocess.Popen(
                    [sys.executable, str(script)],
                    cwd=str(PROJECT_ROOT)
                )
                st.session_state.process = process
                st.success(
                    f"{language} recognition started. "
                    "Check the webcam window."
                )
            except Exception as error:
                st.error(f"Could not start recognition: {error}")

if stop:
    process = st.session_state.process

    if process is not None and process.poll() is None:
        process.terminate()
        try:
            process.wait(timeout=3)
        except subprocess.TimeoutExpired:
            process.kill()

        st.session_state.process = None
        st.success("Recognition stopped.")
    else:
        st.info("No recognition session is currently running.")

# Status
process = st.session_state.process

if process is not None and process.poll() is None:
    st.success("Recognition is currently running.")
else:
    st.caption("Recognition is currently stopped.")

st.divider()

# Footer
st.markdown(
    "<center>SignConnect | Technology for Inclusive Communication</center>",
    unsafe_allow_html=True
)
