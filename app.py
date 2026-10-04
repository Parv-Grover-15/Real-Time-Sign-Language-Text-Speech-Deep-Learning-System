"""
Real-Time Sign Language -> Text -> Speech AI
Streamlit Main Application
"""
import sys
import time
from pathlib import Path
import cv2
import numpy as np
import streamlit as st

# Add project root to sys.path
ROOT_DIR = Path(__file__).resolve().parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from config.settings import (
    CONFIDENCE_THRESHOLD,
    STABILIZATION_WINDOW,
    RELEASE_COOLDOWN
)
from src.detection.landmark_detector import LandmarkDetector
from src.recognition.landmark_classifier import LandmarkClassifier
from src.recognition.word_classifier import WordClassifier
from src.recognition.lstm_classifier import LSTMAttentionClassifier
from src.processing.stabilizer import PredictionStabilizer
from src.processing.sentence_builder import SentenceBuilder
from src.processing.grammar_corrector import GrammarCorrector
from src.speech.tts_engine import TTSEngine

# Streamlit Page Config
st.set_page_config(
    page_title="Sign Language AI",
    page_icon="🤟",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS styling
st.markdown("""
    <style>
    .main-title {
        font-size: 2.2rem;
        font-weight: 700;
        color: #1E88E5;
        text-align: center;
        margin-bottom: 0.5rem;
    }
    .sub-title {
        font-size: 1.1rem;
        color: #555555;
        text-align: center;
        margin-bottom: 1.5rem;
    }
    .metric-card {
        background-color: #F8F9FA;
        border-radius: 10px;
        padding: 15px;
        border-left: 5px solid #1E88E5;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
    }
    .sentence-box {
        background-color: #E3F2FD;
        border-radius: 8px;
        padding: 15px;
        font-size: 1.3rem;
        font-weight: 600;
        color: #0D47A1;
        min-height: 60px;
    }
    </style>
""", unsafe_allow_html=True)

# Initialize Session State
if "sentence_builder" not in st.session_state:
    st.session_state.sentence_builder = SentenceBuilder()
if "tts_engine" not in st.session_state:
    st.session_state.tts_engine = TTSEngine()
if "grammar_corrector" not in st.session_state:
    st.session_state.grammar_corrector = GrammarCorrector(use_llm=False)
if "corrected_text" not in st.session_state:
    st.session_state.corrected_text = ""
if "last_gesture" not in st.session_state:
    st.session_state.last_gesture = "None"
if "last_confidence" not in st.session_state:
    st.session_state.last_confidence = 0.0

@st.cache_resource
def load_landmark_detector():
    return LandmarkDetector()

@st.cache_resource
def load_classifiers():
    alphabet_model = LandmarkClassifier()
    word_model = WordClassifier()
    lstm_model = LSTMAttentionClassifier()
    return alphabet_model, word_model, lstm_model

def main():
    st.markdown('<div class="main-title">🤟 Real-Time Sign Language → Text → Speech AI</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-title">Multi-modal Deep Learning Framework for Accessible Communication</div>', unsafe_allow_html=True)

    detector = load_landmark_detector()
    alphabet_model, word_model, lstm_model = load_classifiers()

    # Sidebar Configuration Controls
    st.sidebar.header("⚙️ System Configuration")

    mode_selection = st.sidebar.radio(
        "Target Recognition Mode",
        ["📖 Full Word Sign Recognition", "🔤 Alphabet Fingerspelling (A-Z)", "🔄 PyTorch LSTM Sequence Motion"]
    )

    conf_threshold = st.sidebar.slider(
        "Confidence Threshold",
        min_value=0.50,
        max_value=0.95,
        value=CONFIDENCE_THRESHOLD,
        step=0.05
    )

    window_size = st.sidebar.slider(
        "Stabilization Window (Frames)",
        min_value=3,
        max_value=15,
        value=STABILIZATION_WINDOW,
        step=1
    )

    enable_grammar = st.sidebar.checkbox("Enable Grammar Correction", value=True)
    cam_index = st.sidebar.number_input("Camera Index", min_value=0, max_value=5, value=0)

    # Select active classifier based on mode
    if "Full Word" in mode_selection:
        active_classifier = word_model
    elif "Alphabet" in mode_selection:
        active_classifier = alphabet_model
    else:
        active_classifier = lstm_model

    # Instantiate Stabilizer
    stabilizer = PredictionStabilizer(
        window_size=window_size,
        min_confidence=conf_threshold,
        release_cooldown=RELEASE_COOLDOWN
    )

    # Main Layout Columns
    col_cam, col_ctrl = st.columns([1.2, 1.0])

    with col_ctrl:
        st.subheader("📝 Live Transcription")

        # Current Recognized Sign Card
        sign_placeholder = st.empty()
        sign_placeholder.markdown(
            f"""
            <div class="metric-card">
                <span style="font-size: 0.9rem; color: #666;">Current Recognized Sign</span><br>
                <span style="font-size: 1.8rem; font-weight: bold; color: #1565C0;">
                    {st.session_state.last_gesture}
                </span>
                <span style="font-size: 1.0rem; color: #2E7D32; margin-left: 15px;">
                    ({st.session_state.last_confidence * 100:.1f}%)
                </span>
            </div>
            """,
            unsafe_allow_html=True
        )

        st.markdown("#### Constructed Sentence")
        sentence_placeholder = st.empty()
        sentence_placeholder.markdown(
            f'<div class="sentence-box">{st.session_state.sentence_builder.get_text() or "..."}</div>',
            unsafe_allow_html=True
        )

        if enable_grammar:
            st.markdown("#### Grammar Corrected Output")
            grammar_placeholder = st.empty()
            grammar_placeholder.info(st.session_state.corrected_text or "Corrected sentence will appear here...")

        st.markdown("---")
        st.subheader("🎮 Real-Time Controls")
        btn_col1, btn_col2, btn_col3, btn_col4 = st.columns(4)

        with btn_col1:
            if st.button("🔊 Speak", use_container_width=True):
                target_text = st.session_state.corrected_text if (enable_grammar and st.session_state.corrected_text) else st.session_state.sentence_builder.get_text()
                if target_text:
                    st.session_state.tts_engine.speak(target_text)
                    st.toast(f"Speaking: '{target_text}'", icon="🔊")

        with btn_col2:
            if st.button("⌫ Delete", use_container_width=True):
                st.session_state.sentence_builder.delete_last()
                if enable_grammar:
                    st.session_state.corrected_text = st.session_state.grammar_corrector.correct_sentence(st.session_state.sentence_builder.get_text())
                st.rerun()

        with btn_col3:
            if st.button("🗑️ Clear", use_container_width=True):
                st.session_state.sentence_builder.clear()
                st.session_state.corrected_text = ""
                st.rerun()

        with btn_col4:
            if st.button("✨ Fix Grammar", use_container_width=True):
                current_raw = st.session_state.sentence_builder.get_text()
                st.session_state.corrected_text = st.session_state.grammar_corrector.correct_sentence(current_raw)
                st.rerun()

        with st.expander("ℹ️ Supported Signs & Mode Reference"):
            if "Full Word" in mode_selection:
                st.write("**Supported Word Signs (20 Words):** Hello, Thank You, Yes, No, Please, Help, Water, Love, Good, Bad, Stop, More, Book, Friend, Family, Home, School, Eat, Drink, Happy")
            else:
                st.write("**Supported Alphabet Signs (26 Letters):** A, B, C, D, E, F, G, H, I, J, K, L, M, N, O, P, Q, R, S, T, U, V, W, X, Y, Z")
            st.write("**Control Gestures:** Space, Delete, Clear")

    with col_cam:
        st.subheader("📹 Live Video Feed")
        run_camera = st.checkbox("Start Camera", value=False)
        frame_window = st.image([])
        fps_placeholder = st.empty()

        if run_camera:
            cap = cv2.VideoCapture(int(cam_index))
            if not cap.isOpened():
                st.error(f"Cannot access camera index {cam_index}. Please check device permissions.")
                return

            prev_time = time.time()

            while run_camera:
                ret, frame = cap.read()
                if not ret:
                    st.warning("Failed to grab camera frame.")
                    break

                # Flip horizontally for natural mirror view
                frame = cv2.flip(frame, 1)

                # MediaPipe Landmark Detection
                annotated_frame, raw_landmarks = detector.process_frame(frame)

                current_sign = "None"
                confidence = 0.0

                if raw_landmarks is not None:
                    # Model Inference
                    predicted_sign, conf, _ = active_classifier.predict(raw_landmarks)
                    if predicted_sign and conf >= conf_threshold:
                        current_sign = predicted_sign
                        confidence = conf

                        # Prediction Stabilization
                        accepted_sign = stabilizer.update(current_sign, confidence)
                        if accepted_sign:
                            st.session_state.sentence_builder.process_sign(accepted_sign)
                            if enable_grammar:
                                st.session_state.corrected_text = st.session_state.grammar_corrector.correct_sentence(
                                    st.session_state.sentence_builder.get_text()
                                )

                # Calculate FPS
                curr_time = time.time()
                fps = 1.0 / max(curr_time - prev_time, 1e-5)
                prev_time = curr_time

                # Update UI elements
                st.session_state.last_gesture = current_sign
                st.session_state.last_confidence = confidence

                # Overlay status text on video
                cv2.putText(annotated_frame, f"Mode: {mode_selection.split()[1]}", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                cv2.putText(annotated_frame, f"Sign: {current_sign} ({confidence*100:.1f}%)", (10, 65),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                cv2.putText(annotated_frame, f"FPS: {fps:.1f}", (10, 100),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 165, 0), 2)

                frame_window.image(annotated_frame, channels="BGR", use_container_width=True)
                fps_placeholder.caption(f"⚡ Processing Speed: {fps:.1f} FPS | Resolution: {frame.shape[1]}x{frame.shape[0]}")

                # Update transcription text box dynamically
                sentence_placeholder.markdown(
                    f'<div class="sentence-box">{st.session_state.sentence_builder.get_text() or "..."}</div>',
                    unsafe_allow_html=True
                )
                sign_placeholder.markdown(
                    f"""
                    <div class="metric-card">
                        <span style="font-size: 0.9rem; color: #666;">Current Recognized Sign</span><br>
                        <span style="font-size: 1.8rem; font-weight: bold; color: #1565C0;">
                            {current_sign}
                        </span>
                        <span style="font-size: 1.0rem; color: #2E7D32; margin-left: 15px;">
                            ({confidence * 100:.1f}%)
                        </span>
                    </div>
                    """,
                    unsafe_allow_html=True
                )
                if enable_grammar:
                    grammar_placeholder.info(st.session_state.corrected_text or "Corrected sentence will appear here...")

            cap.release()

if __name__ == "__main__":
    main()
