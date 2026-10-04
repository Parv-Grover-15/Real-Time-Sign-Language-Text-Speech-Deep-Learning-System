# 🤟 Real-Time Sign Language → Text → Speech AI

An end-to-end multi-modal deep learning application that converts live webcam sign language gestures into text sentences, applies grammar correction, and reads them aloud using text-to-speech.

---

## 🌟 Target Pipeline

```text
Webcam 
  │
  ▼
Hand / Landmark Detection (MediaPipe Hands)
  │
  ▼
Feature Extraction & Normalization (Wrist Origin + Scale Normalization)
  │
  ▼
Deep Learning Sign Classifier (PyTorch MLP / LSTM + Attention)
  │
  ▼
Confidence Filtering (Thresholding >= 0.70)
  │
  ▼
Prediction Stabilization (Hysteresis Windowing & Release Cooldown)
  │
  ▼
Sentence Builder (Letter append, Space, Delete, Clear)
  │
  ▼
Grammar Correction (T5 Transformer / Rule-Based Corrector)
  │
  ▼
Local Text-to-Speech Engine (Non-blocking pyttsx3)
```

---

## 🚀 Key Features

* **Multi-modal DL Architectures**:
  - **PyTorch Landmark MLP**: Lightweight, ultra-fast 63-dim hand feature classifier (26 ASL alphabet signs A-Z).
  - **PyTorch LSTM + Self-Attention**: Sequence classifier for temporal motion features with velocity deltas.
* **Hand & Landmark Detection**: MediaPipe Hands extracting 21 3D hand coordinates.
* **Prediction Stabilization**: Windowed voting and release debouncing to prevent repeated characters when holding a sign.
* **Real-time Sentence Builder**: Supports gesture composition with Space, Delete, Clear, and letter appending.
* **Grammar Correction**: Transforms raw signed word sequences into fluent English sentences.
* **Local Non-Blocking TTS**: Uses `pyttsx3` in background threads so webcam video stays smooth.
* **Streamlit UI**: Responsive dashboard with live video, confidence bars, FPS counter, and interactive controls.

---

## 📁 Repository Structure

```text
sign-language-ai/
├── app.py                      # Main Streamlit web application
├── models/                     # Trained PyTorch models & metadata
│   ├── landmark_model.pt       # PyTorch Landmark Model weights
│   ├── landmark_meta.json      # Model metadata (classes, scaler parameters)
│   └── lstm_attention.pt       # PyTorch LSTM + Attention model weights
├── config/
│   ├── settings.py             # System parameters & thresholds
│   └── class_map.json          # Sign class mapping
├── src/
│   ├── detection/
│   │   └── landmark_detector.py # MediaPipe hand detector & skeleton visualizer
│   ├── preprocessing/
│   │   └── feature_extractor.py # Origin translation, scale normalization, velocity
│   ├── recognition/
│   │   ├── landmark_classifier.py # PyTorch MLP model
│   │   └── lstm_classifier.py     # PyTorch LSTM + Attention model
│   ├── processing/
│   │   ├── stabilizer.py          # Prediction stabilizer & debouncer
│   │   ├── sentence_builder.py    # Sentence builder
│   │   └── grammar_corrector.py   # T5 / Rule-based grammar corrector
│   └── speech/
│       └── tts_engine.py          # Threaded local pyttsx3 engine
├── training/
│   ├── train_mlp.py            # MLP model training script
│   └── train_lstm_attention.py # PyTorch LSTM + Attention training script
├── tests/
│   ├── test_components.py      # Unit tests for all modules
│   └── test_pipeline.py        # End-to-end integration test
├── requirements.txt            # Dependencies
├── README.md                   # Documentation
└── THIRD_PARTY_NOTICES.md      # License notices for 4 source repositories
```

---

## 🛠️ How to Run

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Run the Streamlit UI
Run via Python module (recommended on Windows):
```bash
python -m streamlit run app.py
```
Or directly:
```bash
streamlit run app.py
```

### 3. Run Automated Tests
```bash
python -m pytest tests/ -v
```

---

## 🏷️ Supported Signs

* **Alphabet (26 ASL Signs)**: `A`, `B`, `C`, `D`, `E`, `F`, `G`, `H`, `I`, `J`, `K`, `L`, `M`, `N`, `O`, `P`, `Q`, `R`, `S`, `T`, `U`, `V`, `W`, `X`, `Y`, `Z`
* **Control Gestures / UI Buttons**: `Space`, `Delete`, `Clear`, `Speak`, `Fix Grammar`

---

## ⚠️ Important Limitations

1. **Lighting & Background**: Detection accuracy depends on clear lighting and an unobstructed camera view of the hands.
2. **Single Hand Focus**: Primary landmark model targets 1 hand at a time for optimal real-time performance on standard laptops.
3. **Static vs. Dynamic Signs**: Letters with motion (like `J` and `Z`) benefit from the sequence LSTM model, while static postures are handled by the MLP classifier.
