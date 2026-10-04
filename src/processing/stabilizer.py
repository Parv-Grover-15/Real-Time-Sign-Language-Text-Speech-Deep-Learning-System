"""
Prediction Stabilization and Debouncing module (from Signify contribution)
"""
from collections import deque

class PredictionStabilizer:
    def __init__(self, window_size=5, min_confidence=0.70, release_cooldown=10):
        self.window_size = window_size
        self.min_confidence = min_confidence
        self.release_cooldown = release_cooldown

        self.predictions = deque(maxlen=window_size)
        self.last_accepted_sign = None
        self.cooldown_counter = 0

    def update(self, sign: str, confidence: float):
        """
        Updates the stabilizer with a frame prediction.
        Returns accepted sign string if a stable gesture is confirmed, otherwise None.
        """
        # Decrement cooldown counter if active
        if self.cooldown_counter > 0:
            self.cooldown_counter -= 1

        # Reject low-confidence predictions
        if confidence < self.min_confidence or sign is None:
            self.predictions.clear()
            return None

        # Add current frame prediction
        self.predictions.append(sign)

        # Wait until window is full of consecutive predictions
        if len(self.predictions) < self.window_size:
            return None

        # Check if all predictions in window are identical
        first = self.predictions[0]
        if all(s == first for s in self.predictions):
            # Check if this sign was already accepted and user is still holding it
            if first == self.last_accepted_sign and self.cooldown_counter > 0:
                return None

            # Confirm and accept new sign!
            self.last_accepted_sign = first
            self.cooldown_counter = self.release_cooldown
            self.predictions.clear()
            return first

        return None

    def reset(self):
        self.predictions.clear()
        self.last_accepted_sign = None
        self.cooldown_counter = 0
