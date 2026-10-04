"""
Landmark Detector module using MediaPipe Hands
"""
import cv2
import mediapipe as mp
import numpy as np

class LandmarkDetector:
    def __init__(self, static_image_mode=False, max_num_hands=1, min_detection_confidence=0.5, min_tracking_confidence=0.5):
        self.mp_hands = mp.solutions.hands
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles
        self.hands = self.mp_hands.Hands(
            static_image_mode=static_image_mode,
            max_num_hands=max_num_hands,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence
        )

    def process_frame(self, frame: np.ndarray):
        """
        Processes a BGR image frame and returns:
        - annotated_frame: Frame with hand landmarks drawn
        - landmarks: Nx3 numpy array of (x, y, z) coordinates or None
        """
        if frame is None:
            return None, None

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.hands.process(frame_rgb)

        annotated_frame = frame.copy()

        if not results.multi_hand_landmarks:
            return annotated_frame, None

        hand_landmarks = results.multi_hand_landmarks[0]

        # Draw landmark skeleton on frame
        self.mp_drawing.draw_landmarks(
            annotated_frame,
            hand_landmarks,
            self.mp_hands.HAND_CONNECTIONS,
            self.mp_drawing_styles.get_default_hand_landmarks_style(),
            self.mp_drawing_styles.get_default_hand_connections_style()
        )

        landmarks = np.array(
            [[lm.x, lm.y, lm.z] for lm in hand_landmarks.landmark],
            dtype=np.float32
        )

        return annotated_frame, landmarks

    def close(self):
        self.hands.close()
