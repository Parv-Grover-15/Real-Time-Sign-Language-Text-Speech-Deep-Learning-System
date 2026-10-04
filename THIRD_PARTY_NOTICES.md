# Third-Party Notices & License Acknowledgments

This project, **Real-Time Sign Language → Text → Speech AI**, incorporates code, ideas, model architectures, and design concepts adapted from the following open-source repositories:

---

## 1. Signify
* **Source Repository:** `Signify`
* **Author / Copyright:** Usf132
* **License:** MIT License
* **Reused / Adapted Components:**
  - PyTorch Deep Neural Network landmark model (`models/landmark.pt`) and landmark metadata schema (`models/landmark_meta.json`).
  - Hand landmark extraction and wrist origin-distance feature scaling logic (`signlens/landmark.py`).
  - Prediction stabilization logic (`PredictionStabilizer` in `signlens/stabilizer.py`).
  - Sentence construction buffer (`SentenceBuilder` in `signlens/sentence.py`).

```text
MIT License

Copyright (c) 2026 Usf132

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.
```

---

## 2. Sign-Bridge
* **Source Repository:** `Sign-Bridge`
* **Reused / Adapted Components:**
  - LSTM + Attention sequence model architecture (`Training_the_model.py`).
  - HuggingFace T5 / Rule-based grammar correction module (`llm.py`).
  - Non-blocking threaded local text-to-speech synthesis using `pyttsx3` (`Realtime_test.py`).

---

## 3. sign-language-dl
* **Source Repository:** `sign-language-dl`
* **Author / Copyright:** Sagar
* **License:** MIT License
* **Reused / Adapted Components:**
  - MediaPipe hand bounding box detection algorithms and CNN feature preprocessing utilities (`models/mediapipe_utils.py`, `models/cnn_model.py`).

```text
MIT License

Copyright (c) 2026 Sagar

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.
```

---

## 4. sign-language-recognition
* **Source Repository:** `sign-language-recognition`
* **Reused / Adapted Components:**
  - MediaPipe landmark visualization techniques and Streamlit live webcam interface layout concepts (`streamlit/app.py`).
