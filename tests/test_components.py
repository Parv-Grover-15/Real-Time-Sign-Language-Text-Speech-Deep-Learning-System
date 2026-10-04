"""
Unit tests for individual pipeline components
"""
import pytest
import numpy as np
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.preprocessing.feature_extractor import FeatureExtractor
from src.recognition.landmark_classifier import LandmarkClassifier
from src.recognition.word_classifier import WordClassifier
from src.recognition.lstm_classifier import LSTMAttentionClassifier
from src.processing.stabilizer import PredictionStabilizer
from src.processing.sentence_builder import SentenceBuilder
from src.processing.grammar_corrector import GrammarCorrector

def test_feature_extractor():
    extractor = FeatureExtractor()
    dummy_landmarks = np.random.randn(21, 3).astype(np.float32)
    features = extractor.normalize_landmarks(dummy_landmarks)
    assert features is not None
    assert len(features) == 63
    assert isinstance(features, np.ndarray)

def test_landmark_classifier():
    classifier = LandmarkClassifier()
    dummy_landmarks = np.random.randn(21, 3).astype(np.float32)
    top_class, conf, all_probs = classifier.predict(dummy_landmarks)
    assert top_class in classifier.class_names
    assert 0.0 <= conf <= 1.0
    assert len(all_probs) == classifier.num_classes

def test_word_classifier():
    classifier = WordClassifier()
    dummy_landmarks = np.random.randn(21, 3).astype(np.float32)
    top_word, conf, all_probs = classifier.predict(dummy_landmarks)
    assert top_word in classifier.class_names
    assert 0.0 <= conf <= 1.0
    assert len(all_probs) == classifier.num_classes

def test_lstm_classifier():
    classifier = LSTMAttentionClassifier()
    seq = [np.random.randn(63).astype(np.float32) for _ in range(10)]
    top_class, conf = classifier.predict_sequence(seq)
    assert top_class is not None
    assert 0.0 <= conf <= 1.0

def test_prediction_stabilizer():
    stabilizer = PredictionStabilizer(window_size=3, min_confidence=0.70, release_cooldown=5)

    assert stabilizer.update("A", 0.90) is None
    assert stabilizer.update("A", 0.90) is None
    assert stabilizer.update("A", 0.90) == "A"
    assert stabilizer.update("A", 0.90) is None

def test_sentence_builder():
    sb = SentenceBuilder()
    sb.process_sign("Hello")
    sb.process_sign("World")
    assert sb.get_text() == "Hello World"

    sb.process_sign("Delete")
    assert sb.get_text() == "Hello"

    sb.process_sign("Clear")
    assert sb.get_text() == ""

def test_grammar_corrector():
    corrector = GrammarCorrector(use_llm=False)
    out = corrector.correct_sentence("i go market")
    assert isinstance(out, str)
    assert len(out) > 0
    assert out.endswith(".")
