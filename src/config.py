"""Configuration for MTG card recognition system."""

import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

BASE_DIR = Path(__file__).parent.parent

# Paths
DATABASE_PATH = Path(os.getenv("DATABASE_PATH", "data/mtg_recognition.db"))
DATABASE_PATH = BASE_DIR / DATABASE_PATH if not DATABASE_PATH.is_absolute() else DATABASE_PATH

SCRYFALL_IMAGES_DIR = Path(os.getenv("SCRYFALL_IMAGES_DIR", "data/scryfall"))
SCRYFALL_IMAGES_DIR = BASE_DIR / SCRYFALL_IMAGES_DIR if not SCRYFALL_IMAGES_DIR.is_absolute() else SCRYFALL_IMAGES_DIR

INDEXES_DIR = Path(os.getenv("INDEXES_DIR", "data/indexes"))
INDEXES_DIR = BASE_DIR / INDEXES_DIR if not INDEXES_DIR.is_absolute() else INDEXES_DIR

MODELS_DIR = Path(os.getenv("MODELS_DIR", "data/models"))
MODELS_DIR = BASE_DIR / MODELS_DIR if not MODELS_DIR.is_absolute() else MODELS_DIR

LOG_DIR = Path(os.getenv("LOG_DIR", "logs"))
LOG_DIR = BASE_DIR / LOG_DIR if not LOG_DIR.is_absolute() else LOG_DIR

# Image dimensions (MTG aspect ratio ~0.72)
CANONICAL_WIDTH = 363
CANONICAL_HEIGHT = 504
CANONICAL_IMAGE_SIZE = 512

# Recognition
FAISS_TOP_K = int(os.getenv("FAISS_TOP_K", "20"))
ACCEPT_THRESHOLD = float(os.getenv("ACCEPT_THRESHOLD", "0.85"))
MANUAL_THRESHOLD = float(os.getenv("MANUAL_THRESHOLD", "0.65"))
CLARITY_THRESHOLD = float(os.getenv("CLARITY_THRESHOLD", "0.10"))

# Composite embedding weights
COMPOSITE_WEIGHT_FULL = 0.70
COMPOSITE_WEIGHT_COLLECTOR = 0.20
COMPOSITE_WEIGHT_NAME = 0.10

# Indexing
INDEXING_BATCH_SIZE = int(os.getenv("INDEXING_BATCH_SIZE", "100"))

# API
API_HOST = os.getenv("API_HOST", "0.0.0.0")
API_PORT = int(os.getenv("API_PORT", "8000"))

# Logging
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")

# Create directories
for dir_path in [SCRYFALL_IMAGES_DIR, INDEXES_DIR, MODELS_DIR, LOG_DIR]:
    dir_path.mkdir(parents=True, exist_ok=True)
