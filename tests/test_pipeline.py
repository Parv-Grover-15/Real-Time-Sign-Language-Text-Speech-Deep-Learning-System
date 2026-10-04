"""
End-to-End Pipeline Integration Test
"""
import pytest
import numpy as np
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.recognition.landmark_classifier import LandmarkClassifier
from src.processing.stabilizer import PredictionStabilizer
from src.processing.sentence_builder import SentenceBuilder
from src.processing.grammar_corrector import GrammarCorrector

def test_full_pipeline_flow():
    # Instantiate full pipeline components
    classifier = LandmarkClassifier()
    stabilizer = PredictionStabilizer(window_size=3, min_confidence=0.50, release_cooldown=3)
    sentence_builder = SentenceBuilder()
    grammar_corrector = GrammarCorrector(use_llm=False)

    # Simulate sequence of raw landmark frames
    dummy_landmarks = np.random.randn(21, 3).astype(np.float32)

    # Run 10 mock frames through recognition and stabilization
    confirmed_signs = []
    for _ in range(10):
        predicted_sign, conf, _ = classifier.predict(dummy_landmarks)
        accepted = stabilizer.update(predicted_sign, conf)
        if accepted:
            sentence_builder.process_sign(accepted)
            confirmed_signs.append(accepted)

    raw_text = sentence_builder.get_text()
    corrected_text = grammar_corrector.correct_sentence(raw_text)

    # Verify pipeline outputs
    assert isinstance(raw_text, str)
    assert isinstance(corrected_text, str)
    print(f"Pipeline Test Success! Raw: '{raw_text}', Corrected: '{corrected_text}'")

if __name__ == "__main__":
    test_full_pipeline_flow()
