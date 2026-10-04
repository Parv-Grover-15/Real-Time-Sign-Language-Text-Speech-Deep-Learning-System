"""
Configuration settings for Real-Time Sign Language -> Text -> Speech AI
"""
from pathlib import Path

# Base Paths
BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_DIR = BASE_DIR / "models"
CONFIG_DIR = BASE_DIR / "config"

# Model Paths
MLP_MODEL_PATH = MODEL_DIR / "landmark_model.pt"
METADATA_PATH = MODEL_DIR / "landmark_meta.json"
CLASS_MAP_PATH = CONFIG_DIR / "class_map.json"

# Recognition Parameters
CONFIDENCE_THRESHOLD = 0.70    # Minimum softmax probability to accept a prediction
STABILIZATION_WINDOW = 5       # Number of consecutive identical predictions needed
RELEASE_COOLDOWN = 15          # Frames to wait before accepting the same gesture again

# Landmark Normalization Parameters
WRIST_IDX = 0
MIDDLE_MCP_IDX = 9

# Sequence Modeling Parameters (for LSTM + Attention)
SEQUENCE_LENGTH = 20
FEATURE_DIM = 63

# Speech Settings
TTS_RATE = 150                 # Speech rate (words per min)
TTS_VOLUME = 1.0

# UI Settings
CAMERA_WIDTH = 640
CAMERA_HEIGHT = 480
FPS_TARGET = 30
