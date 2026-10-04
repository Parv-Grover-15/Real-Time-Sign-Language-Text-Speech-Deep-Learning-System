"""
Feature Extraction and Normalization module
"""
import numpy as np
from config.settings import WRIST_IDX, MIDDLE_MCP_IDX

class FeatureExtractor:
    def __init__(self, mean: np.ndarray = None, scale: np.ndarray = None):
        self.mean = np.array(mean, dtype=np.float32) if mean is not None else None
        self.scale = np.array(scale, dtype=np.float32) if scale is not None else None

    def normalize_landmarks(self, landmarks: np.ndarray) -> np.ndarray:
        """
        Normalizes 21 (x, y, z) hand landmarks by centering at wrist (index 0)
        and scaling by wrist-to-middle-MCP (index 9) distance.
        Returns a 63-dimensional flattened numpy array.
        """
        if landmarks is None or len(landmarks) != 21:
            return None

        landmarks = np.asarray(landmarks, dtype=np.float32)

        # 1. Translate wrist to (0, 0, 0)
        origin = landmarks[WRIST_IDX]
        centered = landmarks - origin

        # 2. Scale by Euclidean distance from wrist to middle MCP
        scale_ref = np.linalg.norm(centered[MIDDLE_MCP_IDX])
        scale_ref = max(scale_ref, 1e-6)
        normalized = centered / scale_ref

        features = normalized.flatten().astype(np.float32)

        # 3. Standard scaling if scaler parameters are loaded
        if self.mean is not None and self.scale is not None:
            features = (features - self.mean) / self.scale

        return features

    @staticmethod
    def compute_velocity(current_features: np.ndarray, previous_features: np.ndarray) -> np.ndarray:
        """
        Computes velocity features (frame delta) for sequence analysis.
        """
        if current_features is None:
            return None
        if previous_features is None:
            return np.zeros_like(current_features)
        return current_features - previous_features
