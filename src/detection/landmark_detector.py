"""
Landmark Detector module supporting both MediaPipe 1.0+ Tasks API and legacy solutions API.
"""
import os
import urllib.request
import cv2
import numpy as np
import mediapipe as mp
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent.parent
MODEL_TASK_PATH = ROOT_DIR / "models" / "hand_landmarker.task"
TASK_URL = "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"

# Standard 21 Hand Landmark Connection pairs
HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),        # Thumb
    (0, 5), (5, 6), (6, 7), (7, 8),        # Index
    (5, 9), (9, 10), (10, 11), (11, 12),   # Middle
    (9, 13), (13, 14), (14, 15), (15, 16), # Ring
    (13, 17), (0, 17), (17, 18), (18, 19), (19, 20) # Pinky
]

def ensure_task_model_downloaded():
    if not MODEL_TASK_PATH.exists():
        MODEL_TASK_PATH.parent.mkdir(parents=True, exist_ok=True)
        print(f"[LandmarkDetector] Downloading hand_landmarker.task to {MODEL_TASK_PATH}...")
        try:
            urllib.request.urlretrieve(TASK_URL, str(MODEL_TASK_PATH))
            print("[LandmarkDetector] Download completed successfully!")
        except Exception as e:
            print(f"[LandmarkDetector] Failed to download task model: {e}")

class LandmarkDetector:
    def __init__(self, static_image_mode=False, max_num_hands=1, min_detection_confidence=0.5, min_tracking_confidence=0.5):
        self.use_tasks_api = False
        self.tasks_detector = None
        self.legacy_hands = None

        # 1. Try Legacy MediaPipe Solutions API first if available
        if hasattr(mp, "solutions") and hasattr(mp.solutions, "hands"):
            try:
                self.mp_hands = mp.solutions.hands
                self.mp_drawing = mp.solutions.drawing_utils
                self.mp_drawing_styles = mp.solutions.drawing_styles
                self.legacy_hands = self.mp_hands.Hands(
                    static_image_mode=static_image_mode,
                    max_num_hands=max_num_hands,
                    min_detection_confidence=min_detection_confidence,
                    min_tracking_confidence=min_tracking_confidence
                )
                print("[LandmarkDetector] Using MediaPipe Solutions API.")
                return
            except Exception as e:
                print(f"[LandmarkDetector] Legacy API init fallback: {e}")

        # 2. Try MediaPipe 1.0+ Tasks API
        try:
            ensure_task_model_downloaded()
            from mediapipe.tasks import python
            from mediapipe.tasks.python import vision

            base_options = python.BaseOptions(model_asset_path=str(MODEL_TASK_PATH))
            options = vision.HandLandmarkerOptions(
                base_options=base_options,
                running_mode=vision.RunningMode.IMAGE,
                num_hands=max_num_hands,
                min_hand_detection_confidence=min_detection_confidence,
                min_tracking_confidence=min_tracking_confidence
            )
            self.tasks_detector = vision.HandLandmarker.create_from_options(options)
            self.use_tasks_api = True
            print("[LandmarkDetector] Using MediaPipe 1.0+ Tasks API (HandLandmarker).")
        except Exception as e:
            print(f"[LandmarkDetector] Tasks API init failed: {e}")

    def process_frame(self, frame: np.ndarray):
        """
        Processes a BGR image frame and returns:
        - annotated_frame: Frame with drawn hand landmarks
        - landmarks: (21, 3) numpy array of (x, y, z) coordinates or None
        """
        if frame is None:
            return None, None

        annotated_frame = frame.copy()

        # Handle MediaPipe Solutions API
        if self.legacy_hands is not None:
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self.legacy_hands.process(frame_rgb)
            if not results.multi_hand_landmarks:
                return annotated_frame, None

            hand_landmarks = results.multi_hand_landmarks[0]
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

        # Handle MediaPipe Tasks API
        if self.tasks_detector is not None:
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
            results = self.tasks_detector.detect(mp_image)

            if not results.hand_landmarks or len(results.hand_landmarks) == 0:
                return annotated_frame, None

            first_hand = results.hand_landmarks[0]
            h, w = frame.shape[:2]

            # Draw Skeleton Connections
            pixel_points = []
            for lm in first_hand:
                cx, cy = int(lm.x * w), int(lm.y * h)
                pixel_points.append((cx, cy))
                cv2.circle(annotated_frame, (cx, cy), 5, (0, 255, 0), -1)

            for p1, p2 in HAND_CONNECTIONS:
                if p1 < len(pixel_points) and p2 < len(pixel_points):
                    cv2.line(annotated_frame, pixel_points[p1], pixel_points[p2], (255, 0, 0), 2)

            landmarks = np.array(
                [[lm.x, lm.y, lm.z] for lm in first_hand],
                dtype=np.float32
            )
            return annotated_frame, landmarks

        return annotated_frame, None

    def close(self):
        if self.legacy_hands:
            self.legacy_hands.close()
        if self.tasks_detector:
            self.tasks_detector.close()
