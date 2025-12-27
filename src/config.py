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

# OCR settings (for disambiguation)
# NOTE: Warped images include ~5% extra border due to margin_reduction=-0.05 in card_detector.py
# Coordinates are adjusted to account for this expansion

# Title region coordinates (adjusted for 5% border expansion)
# Original card title is at ~5% from top; with expansion, it's at ~7-12% in warped image
OCR_TITLE_X_START = 0.06
OCR_TITLE_X_END = 0.78
OCR_TITLE_Y_START = 0.06  # Was 0.045, adjusted for border expansion
OCR_TITLE_Y_END = 0.12    # Was 0.10, adjusted for border expansion

# Collector region (adjusted for 5% border expansion)
# Contains set code, collector number, rarity, language, artist name
# Original is at bottom ~4% of card; with expansion, it's at ~90-96% in warped image
OCR_COLLECTOR_X_START = 0.0
OCR_COLLECTOR_X_END = 0.55
OCR_COLLECTOR_Y_START = 0.90  # Was 0.955, adjusted for border expansion
OCR_COLLECTOR_Y_END = 0.96    # Was 0.99, adjusted for border expansion

# Tesseract settings
OCR_PSM_MODE = int(os.getenv("OCR_PSM_MODE", "7"))  # 7 = single line mode
OCR_CONFIDENCE_THRESHOLD = float(os.getenv("OCR_CONFIDENCE_THRESHOLD", "0.8"))
OCR_BOOST_WEIGHT = float(os.getenv("OCR_BOOST_WEIGHT", "0.15"))  # Score boost for OCR matches

# Minimum OCR confidence to use results (below this, treat as garbage)
OCR_MIN_CONFIDENCE = float(os.getenv("OCR_MIN_CONFIDENCE", "0.45"))

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
