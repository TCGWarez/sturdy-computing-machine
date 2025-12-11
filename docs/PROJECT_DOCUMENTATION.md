# MTG Card Recognition System - Complete Documentation

A system that identifies Magic: The Gathering cards from scanned or photographed images using computer vision and machine learning.

---

## Table of Contents

1. [What This Project Does](#what-this-project-does)
2. [How It Works (Simple Explanation)](#how-it-works-simple-explanation)
3. [Project Structure](#project-structure)
4. [Database Schema](#database-schema)
5. [The Recognition Pipeline (Step by Step)](#the-recognition-pipeline-step-by-step)
6. [How Scoring Works](#how-scoring-works)
7. [API Endpoints](#api-endpoints)
8. [Command Line Tools](#command-line-tools)
9. [Configuration](#configuration)
10. [Dependencies](#dependencies)
11. [File Naming Rules](#file-naming-rules)
12. [Getting Started](#getting-started)
13. [Important Technical Details](#important-technical-details)
14. [Glossary](#glossary)

---

## What This Project Does

This system takes a photo or scan of a Magic: The Gathering card and tells you:
- What card it is (name)
- What set it belongs to (like "Double Masters Remastered" or "Dominaria United")
- What collector number it has (the number at the bottom of the card)
- What finish it has (regular, foil, or etched)
- How confident the system is about this match (a score from 0 to 1)

**Accuracy Goal:** 92% or higher on sets with many similar-looking cards.

---

## How It Works (Simple Explanation)

Think of it like this:

1. **You give the system a photo of a card**
2. **The system finds the card in the photo** (even if the photo is tilted or has a messy background)
3. **The system creates a "fingerprint" of the card** (a list of 512 numbers that describe what the card looks like)
4. **The system compares this fingerprint to all known cards** in its database
5. **The system returns the best match** along with how confident it is

The "fingerprint" is called an **embedding**. It is created using a model called **CLIP** (Contrastive Language-Image Pre-training) which was trained by OpenAI to understand images.

---

## Project Structure

```
mtg_image_recog/
│
├── api/                              # WEB SERVER CODE
│   ├── main.py                       # Starts the web server
│   ├── database.py                   # Database tables for batch uploads
│   ├── models.py                     # Data shapes for requests and responses
│   ├── routes/
│   │   └── batch.py                  # Handles uploading and recognizing cards
│   └── services/
│       └── recognition.py            # Connects web server to card matching logic
│
├── src/                              # MAIN APPLICATION CODE
│   ├── config.py                     # All settings and file paths
│   │
│   ├── detection/
│   │   └── card_detector.py          # Finds card edges in a photo
│   │
│   ├── embeddings/
│   │   ├── base_embedder.py          # Template for embedding creators
│   │   ├── embedder.py               # Creates fingerprints using CLIP model
│   │   ├── region_extractor.py       # Cuts out parts of the card (name, collector number)
│   │   └── train.py                  # Optional: train a custom CLIP model
│   │
│   ├── indexing/
│   │   ├── indexer.py                # Main code for processing and storing cards
│   │   ├── phash.py                  # Creates simple visual fingerprints (perceptual hashing)
│   │   └── image_processor.py        # Image manipulation helpers
│   │
│   ├── ann/
│   │   ├── faiss_index.py            # Fast similarity search (finds similar fingerprints)
│   │   └── hnsw_index.py             # Alternative similarity search method
│   │
│   ├── recognition/
│   │   ├── matcher.py                # THE MAIN LOGIC - matches your card to known cards
│   │   └── orb_utils.py              # Visual verification using keypoints
│   │
│   ├── database/
│   │   ├── schema.py                 # Database table definitions (cards, embeddings, etc.)
│   │   └── db.py                     # Functions to read/write database
│   │
│   ├── utils/
│   │   └── scryfall.py               # Reads card information from Scryfall files
│   │
│   └── cli/
│       └── main.py                   # Command line interface
│
├── scripts/                          # HELPER SCRIPTS
│   ├── download_default_cards.py     # Downloads card images from Scryfall
│   ├── index_set.py                  # Indexes a set of cards (one command does everything)
│   └── run_regression.py             # Tests accuracy of the system
│
├── web/                              # FRONTEND (HTML/CSS)
│   ├── static/
│   │   └── css/style.css             # Styling for web pages
│   └── templates/
│       ├── upload.html               # Page to upload cards
│       └── results.html              # Page to see results
│
├── data/                             # DATA STORAGE (created when you run the system)
│   ├── scryfall/                     # Downloaded card images
│   │   └── {SET_CODE}/               # One folder per set (like "DMR", "ONE")
│   │       ├── nonfoil/              # Non-foil card images
│   │       └── foil/                 # Foil card images
│   ├── indexes/                      # Search indexes (FAISS files)
│   └── mtg_recognition.db            # SQLite database with all card data
│
├── uploads/                          # Temporary storage for uploaded images
│
├── requirements.txt                  # Python packages needed
├── .env                              # Your local settings (not in git)
└── .env.example                      # Template for settings
```

---

## Database Schema

The system uses SQLite (a simple file-based database). There are 5 tables:

### Table 1: `cards`
Stores basic information about each card.

| Column | Type | Description |
|--------|------|-------------|
| id | TEXT | Unique identifier (like "357_Mindskinner_e5a967_nonfoil") |
| scryfall_id | TEXT | UUID from Scryfall website |
| name | TEXT | Card name (like "The Mindskinner") |
| set_code | TEXT | Three-letter set code (like "DSK", "DMR", "ONE") |
| collector_number | TEXT | Number at bottom of card (like "357") |
| finish | TEXT | Either "nonfoil", "foil", or "etched" |
| variant_type | TEXT | Card style: "normal", "extended_art", "showcase", "borderless" |
| image_path | TEXT | Where the reference image is stored on disk |
| created_at | DATETIME | When this card was added to the database |

### Table 2: `phash_variants`
Stores three simple visual fingerprints for each card.

| Column | Type | Description |
|--------|------|-------------|
| id | INTEGER | Unique row number |
| card_id | TEXT | Links to cards table |
| variant_type | TEXT | Which part: "full", "name", or "collector" |
| phash | TEXT | 64-bit fingerprint as hexadecimal text |

**Why three fingerprints?**
- **full**: The entire card image
- **name**: Just the top 15% (where the card name is)
- **collector**: Just the bottom 10% (where the collector number is)

This helps match cards even when they have similar art but different numbers.

### Table 3: `composite_embeddings`
Stores the main fingerprint (512 numbers from CLIP model).

| Column | Type | Description |
|--------|------|-------------|
| card_id | TEXT | Links to cards table (primary key) |
| embedding | BLOB | The main fingerprint used for searching (512 numbers as bytes) |
| full_embedding | BLOB | Fingerprint of full card (for debugging) |
| collector_embedding | BLOB | Fingerprint of collector area (for debugging) |
| name_embedding | BLOB | Fingerprint of name area (for debugging) |
| weight_full | FLOAT | How much the full card counts (default 0.70) |
| weight_collector | FLOAT | How much the collector area counts (default 0.20) |
| weight_name | FLOAT | How much the name area counts (default 0.10) |

**What is "composite"?**
The main fingerprint is a weighted mix of three areas:
- 70% from the full card
- 20% from the collector number area (very important for telling similar cards apart!)
- 10% from the name area

### Table 4: `batches`
Tracks batch upload jobs.

| Column | Type | Description |
|--------|------|-------------|
| id | TEXT | Unique batch identifier (UUID) |
| user_id | TEXT | For future user accounts (not used yet) |
| status | TEXT | "uploading", "processing", "completed", or "failed" |
| total_cards | INTEGER | How many cards were uploaded |
| processed_cards | INTEGER | How many have been processed so far |
| set_code | TEXT | Optional filter for what set to search |
| created_at | DATETIME | When batch was created |
| completed_at | DATETIME | When batch finished |

### Table 5: `batch_results`
Stores recognition results for each uploaded image.

| Column | Type | Description |
|--------|------|-------------|
| id | INTEGER | Unique row number |
| batch_id | TEXT | Links to batches table |
| image_id | TEXT | Unique ID for this specific image |
| image_filename | TEXT | Original filename |
| image_path | TEXT | Where the uploaded file is stored |
| matched_card_id | TEXT | What card the system thinks it is |
| confidence | FLOAT | How confident (0.0 to 1.0) |
| clarity_score | FLOAT | Gap between best and second-best match |
| is_ambiguous | BOOLEAN | True if multiple cards look very similar |
| candidates_json | TEXT | Top 20 possible matches as JSON |
| is_corrected | BOOLEAN | True if user fixed the match |
| corrected_card_id | TEXT | The correct card (if user fixed it) |
| correction_reason | TEXT | Why user made the correction |
| corrected_at | DATETIME | When correction was made |

---

## The Recognition Pipeline (Step by Step)

When you upload a photo of a card, here is exactly what happens:

### Step 1: Card Detection and Perspective Fix

**File:** `src/detection/card_detector.py`

**What it does:**
- Finds the card edges in your photo
- Fixes any tilt or rotation
- Crops out just the card
- Resizes to exactly 363 x 504 pixels (the correct aspect ratio for MTG cards)

**How it works:**
1. Convert image to grayscale
2. Apply blur to reduce noise
3. Find edges using Canny edge detection
4. Look for a rectangle shape (the card)
5. Check that the shape has the right aspect ratio (0.714 = width/height for MTG cards)
6. Apply perspective transformation to straighten the card

**If detection fails:** The system uses the entire image as-is.

### Step 2: Region Extraction

**File:** `src/embeddings/region_extractor.py`

**What it does:**
Cuts out three areas from the card:

1. **Full Card**: The entire 363x504 image
2. **Name Region**: Top 15% of the card (where the card name is printed)
3. **Collector Region**: Bottom 10% of the card (where the collector number and set code are printed)

**Why this matters:**
Many cards have similar art. The collector number is UNIQUE within a set, so it is the best way to tell cards apart.

### Step 3: Generate CLIP Embedding

**File:** `src/embeddings/embedder.py`

**What it does:**
Turns each region into a list of 512 numbers (called an "embedding" or "fingerprint").

**How it works:**
1. Load the CLIP model (ViT-B/32 version)
2. Preprocess each image region (resize, normalize colors)
3. Run through the neural network
4. Normalize the output so all embeddings have length 1.0

**Then combine into one "composite" embedding:**
```
composite = 0.70 × full_embedding + 0.20 × collector_embedding + 0.10 × name_embedding
```

This single composite embedding is used for searching.

### Step 4: FAISS Search

**File:** `src/ann/faiss_index.py`

**What it does:**
Finds the 20 most similar cards from the database.

**How it works:**
1. Load the pre-built FAISS index (a special data structure for fast similarity search)
2. Search using inner product (dot product) on normalized vectors
3. Return top 20 matches with similarity scores

**Speed:** This takes about 1 millisecond, even with thousands of cards.

### Step 5: Compute pHash Distances

**File:** `src/indexing/phash.py`

**What it does:**
For each of the 20 candidates, compare simple visual fingerprints.

**How pHash works:**
1. Resize image to 8x8 pixels
2. Convert to grayscale
3. Calculate average brightness
4. For each pixel: 1 if brighter than average, 0 if darker
5. Result: 64-bit fingerprint (stored as hex string like "c3e7e7c3c3e7e7c3")

**Hamming distance:** Count how many bits are different between two fingerprints. Lower = more similar.

### Step 6: Compute Combined Score

**File:** `src/recognition/matcher.py`

**What it does:**
For each candidate, compute a final score:

```
embedding_score = similarity from FAISS (0.0 to 1.0)

phash_score = 0.60 × (1 - full_distance/64)
            + 0.30 × (1 - name_distance/64)
            + 0.10 × (1 - collector_distance/64)

combined_score = 0.85 × embedding_score + 0.15 × phash_score
```

Why 85% embedding / 15% pHash?
- CLIP embeddings are very accurate after proper preprocessing
- pHash is sensitive to scanner artifacts and compression
- But pHash helps catch cases where CLIP might miss

### Step 7: ORB Verification

**File:** `src/recognition/orb_utils.py`

**What it does:**
Visual double-check using keypoint matching (ORB = Oriented FAST and Rotated BRIEF).

**How it works:**
1. Find distinctive points in both images (corners, blobs)
2. Describe each point with a fingerprint
3. Match points between images
4. Use "Lowe's ratio test" to filter bad matches
5. Count good matches

**Score adjustment:**
- If ORB score > 0.5: Boost combined score by up to 50%
- If ORB score > 0.3: Boost combined score by up to 30%
- If ORB score < 0.3: Reduce combined score by 5%

This helps catch mistakes where the CLIP embedding was fooled.

### Step 8: Compute Clarity and Determine Match Method

**What it does:**
Decide if the match is trustworthy.

**Clarity score:** The gap between the #1 and #2 candidates.
- If gap > 0.10: Clear winner
- If gap < 0.10: Ambiguous (multiple similar candidates)

**Match method:**
- `auto_accept`: Confidence >= 0.85 AND clear winner. System is sure.
- `ambiguous_high`: Confidence >= 0.85 BUT multiple similar candidates. Needs human review.
- `manual_review`: Confidence < 0.85. Needs human review.

### Step 9: Return Results

The system returns:
- Best matching card (ID, name, set, collector number, finish)
- Confidence score (0.0 to 1.0)
- Match method (auto_accept, ambiguous_high, manual_review)
- Clarity score
- Top 20 alternative candidates (for review interface)
- Detected card boundary corners (for drawing boxes on the original image)

---

## How Scoring Works

### The Math

```
FINAL_SCORE = (0.85 × EMBEDDING_SCORE + 0.15 × PHASH_SCORE) × ORB_MULTIPLIER
```

Where:
- EMBEDDING_SCORE: From FAISS, range 0.0 to 1.0 (inner product of normalized vectors)
- PHASH_SCORE: Weighted combination of three hash distances
- ORB_MULTIPLIER: Between 0.95 and 1.5 depending on keypoint matches

### Score Interpretation

| Score Range | Meaning |
|-------------|---------|
| 0.95 - 1.00 | Near-perfect match (very confident) |
| 0.90 - 0.95 | Good match (confident) |
| 0.85 - 0.90 | Acceptable match (threshold for auto-accept) |
| 0.65 - 0.85 | Uncertain (needs manual review) |
| Below 0.65 | Poor match (likely wrong card) |

### Thresholds

| Threshold | Default Value | Meaning |
|-----------|---------------|---------|
| ACCEPT_THRESHOLD | 0.85 | Scores above this are auto-accepted |
| MANUAL_THRESHOLD | 0.65 | Scores below this need careful review |
| CLARITY_THRESHOLD | 0.10 | Minimum gap between top 2 candidates |

---

## API Endpoints

The web server runs on `http://localhost:8000`.

### Upload Cards for Recognition

```
POST /api/batch/upload
```

**What to send:**
- `files`: One or more image files (or a ZIP file)
- `set_code` (optional): Which set(s) to search, like "DMR" or "DMR,SLD"
  - If not provided, the system searches ALL indexed sets
  - **Limit:** When no set_code is provided, batch size is limited to 50 images (slower search)
- `finish` (optional): "nonfoil" or "foil"
- `prefer_foil` (optional): Set to true if cards are foil

**What you get back:**
```json
{
  "batch_id": "abc123...",
  "status": "processing",
  "total_cards": 5,
  "processed_cards": 0,
  "created_at": "2024-01-15T10:30:00"
}
```

**Error when exceeding batch limit (no set_code):**
```json
{
  "detail": "When no set_code is specified, batch size is limited to 50 images. You uploaded 75 images. Please specify a set_code or reduce batch size."
}
```

### Check Processing Status

```
GET /api/batch/{batch_id}/status
```

**What you get back:**
```json
{
  "batch_id": "abc123...",
  "status": "completed",
  "total_cards": 5,
  "processed_cards": 5,
  "created_at": "2024-01-15T10:30:00"
}
```

Status can be: `uploading`, `processing`, `completed`, or `failed`.

### Get Results

```
GET /api/batch/{batch_id}/results
```

**What you get back:**
```json
{
  "batch_info": { ... },
  "results": [
    {
      "image_id": "xyz789...",
      "image_filename": "card1.jpg",
      "card_id": "357_The_Mindskinner_e5a967_nonfoil",
      "card_name": "The Mindskinner",
      "set_code": "DSK",
      "collector_number": "357",
      "finish": "nonfoil",
      "confidence": 0.92,
      "clarity_score": 0.15,
      "is_ambiguous": false,
      "candidates": [ ... ]
    }
  ]
}
```

### Correct a Match

```
POST /api/batch/{batch_id}/correct
```

**What to send:**
```json
{
  "image_id": "xyz789...",
  "correct_card_id": "358_Other_Card_abc123_nonfoil",
  "reason": "Wrong collector number"
}
```

### Export Results

```
GET /api/batch/{batch_id}/export?format=csv
```

Returns a CSV file with all results.

```
GET /api/batch/{batch_id}/export?format=json
```

Returns a JSON file with all results.

### View Images

```
GET /api/batch/{batch_id}/image/{image_id}
```
Returns the uploaded image.

```
GET /api/batch/reference/{card_id}
```
Returns the reference image from Scryfall.

```
GET /api/batch/{batch_id}/image/{image_id}/debug
```
Returns the uploaded image with the detected card boundary drawn on it.

### Search Cards

```
GET /api/batch/search?query=mindskinner&set_code=DSK&limit=20
```

Search for cards by name (useful for correction modal).

---

## Command Line Tools

### Download Card Images

```bash
# Download a single set
uv run python scripts/download_default_cards.py --sets DMR

# Download multiple sets
uv run python scripts/download_default_cards.py --sets DMR SLD ONE

# Download only nonfoil cards
uv run python scripts/download_default_cards.py --sets DMR --finishes nonfoil
```

This downloads images from Scryfall and creates JSON files with card metadata.

### Index a Set

```bash
# Index one set (processes all finishes: nonfoil + foil)
uv run python scripts/index_set.py DMR

# Index all sets you have downloaded
uv run python scripts/index_set.py --all

# Use GPU for faster processing
uv run python scripts/index_set.py DMR --device cuda

# Force re-index even if already done
uv run python scripts/index_set.py DMR --force

# Preview what would be indexed (dry run)
uv run python scripts/index_set.py --all --dry-run
```

This command does everything:
1. Reads all images for the set
2. Normalizes them to 363x504
3. Extracts regions (full, name, collector)
4. Computes CLIP embeddings
5. Computes pHash fingerprints
6. Stores everything in database
7. Builds FAISS search index

### Start the Web Server

```bash
# Simple way
uv run python api/main.py

# With auto-reload for development
uv run uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload
```

Then open `http://localhost:8000` in your browser.

### Initialize the Database

```bash
# Create indexing tables
uv run python -c "from src.database.schema import init_db; init_db()"

# Create batch processing tables
uv run python -c "from api.database import init_batch_tables; init_batch_tables()"
```

---

## Configuration

All settings are in a `.env` file. Copy `.env.example` to `.env` and edit:

```bash
# Where to store everything
DATABASE_PATH=data/mtg_recognition.db
SCRYFALL_IMAGES_DIR=./data/scryfall
INDEXES_DIR=data/indexes
MODELS_DIR=data/models
LOG_DIR=logs

# How much each region matters for the composite embedding
COMPOSITE_WEIGHT_FULL=0.70
COMPOSITE_WEIGHT_COLLECTOR=0.20
COMPOSITE_WEIGHT_NAME=0.10

# When to auto-accept vs request manual review
ACCEPT_THRESHOLD=0.85
MANUAL_THRESHOLD=0.65
CLARITY_THRESHOLD=0.10

# How many candidates to retrieve from FAISS
FAISS_TOP_K=20

# Batch size for indexing
INDEXING_BATCH_SIZE=100

# Web server settings
API_HOST=0.0.0.0
API_PORT=8000

# Logging level
LOG_LEVEL=INFO
```

---

## Dependencies

### Machine Learning and Computer Vision

| Package | Version | What it does |
|---------|---------|--------------|
| torch | >= 2.0.0 | Deep learning framework |
| torchvision | >= 0.15.0 | Image transformations |
| open-clip-torch | >= 2.20.0 | The CLIP model for embeddings |
| opencv-python | >= 4.8.0 | Image processing, card detection, ORB |
| Pillow | >= 10.0.0 | Reading and writing images |
| imagehash | >= 4.3.1 | Perceptual hashing |
| numpy | >= 1.24.0 | Number arrays |
| scipy | >= 1.11.0 | Scientific computing |

### Vector Search

| Package | Version | What it does |
|---------|---------|--------------|
| faiss-cpu | >= 1.7.4 | Fast similarity search |
| hnswlib | >= 0.7.0 | Alternative search method |

### Web Server

| Package | Version | What it does |
|---------|---------|--------------|
| fastapi | >= 0.104.0 | Web API framework |
| uvicorn | >= 0.24.0 | Runs the web server |
| python-multipart | >= 0.0.6 | File upload handling |
| jinja2 | >= 3.1.2 | HTML templates |

### Data and Networking

| Package | Version | What it does |
|---------|---------|--------------|
| aiohttp | >= 3.8.0 | Download images from internet |
| aiofiles | >= 23.0.0 | Write files asynchronously |
| requests | >= 2.31.0 | HTTP requests |
| ijson | >= 3.2.0 | Parse large JSON files without loading all into memory |

### Database

| Package | Version | What it does |
|---------|---------|--------------|
| sqlalchemy | >= 2.0.0 | Database operations |

### Command Line

| Package | Version | What it does |
|---------|---------|--------------|
| click | >= 8.1.7 | Build command line tools |
| tqdm | >= 4.66.0 | Progress bars |
| python-dotenv | >= 1.0.0 | Read .env files |

---

## File Naming Rules

### Card Images (from Scryfall)

```
{COLLECTOR_NUMBER}_{CARD_NAME}_{SCRYFALL_ID_FIRST_12_CHARS}.jpg
```

Example: `357_The_Mindskinner_e5a967e3b23b.jpg`

Special characters in card names are replaced with underscores.

### JSON Metadata Files

Same name as image but with `.json` extension:
```
357_The_Mindskinner_e5a967e3b23b.json
```

Contains full Scryfall card data including UUID, frame effects, and all other metadata.

### Card IDs in Database

```
{COLLECTOR_NUMBER}_{CARD_NAME}_{SCRYFALL_ID}_{FINISH}
```

Example: `357_The_Mindskinner_e5a967e3b23b_nonfoil`

### FAISS Index Files

```
{SET_CODE}_{FINISH}_composite.faiss
{SET_CODE}_{FINISH}_composite.meta
```

Example:
- `DMR_nonfoil_composite.faiss` - The actual search index
- `DMR_nonfoil_composite.meta` - Metadata (list of card IDs, settings)

---

## Getting Started

### 1. Install Python Dependencies

```bash
# Install uv (a fast Python package manager) if you don't have it
pip install uv

# Install all dependencies
uv pip install -r requirements.txt
```

### 2. Set Up Configuration

```bash
# Copy the example configuration
cp .env.example .env

# Edit .env to set your paths (especially SCRYFALL_IMAGES_DIR if you have space elsewhere)
```

### 3. Initialize the Database

```bash
uv run python -c "from src.database.schema import init_db; init_db()"
uv run python -c "from api.database import init_batch_tables; init_batch_tables()"
```

### 4. Download Card Images

```bash
# Download one set to test (takes about 5-10 minutes)
uv run python scripts/download_default_cards.py --sets DMR
```

### 5. Index the Set

```bash
# This takes about 5-15 minutes per set depending on your computer
uv run python scripts/index_set.py DMR
```

### 6. Start the Server

```bash
uv run python api/main.py
```

### 7. Open the Web Interface

Go to `http://localhost:8000` in your browser and upload some card images!

---

## Important Technical Details

### Why 363 x 504 Pixels?

Magic cards have an aspect ratio of approximately 0.714 (63mm wide x 88mm tall). Using 363 x 504 maintains this ratio while keeping the height close to 512 (a common size in image processing).

**Critical:** Both the indexed Scryfall images AND the scanned images MUST be resized to exactly 363 x 504 before generating CLIP embeddings. If sizes differ, the embeddings will not match properly.

### Why Composite Embeddings?

Originally, the system used only the full card image for FAISS search. This caused problems:
- Cards with similar art (like basic lands) would rank higher than the correct card
- The collector number (most discriminative feature) was being ignored during search

The fix: Create a single "composite" embedding that weights different regions:
- 70% full card (captures overall appearance)
- 20% collector region (captures the unique number at bottom)
- 10% name region (helps with variant discrimination)

This ensures the right card is in the top 20 candidates.

### Why pHash AND CLIP?

They complement each other:

**CLIP embeddings:**
- Understand semantic content (what the card looks like)
- Robust to color shifts and lighting changes
- Sometimes confused by similar art styles

**pHash (Perceptual Hash):**
- Simple pixel-level comparison
- Very fast
- Good at exact matches
- Sensitive to scanner artifacts

Using both together gives better accuracy than either alone.

### Why ORB Verification?

ORB (Oriented FAST and Rotated BRIEF) finds distinctive keypoints in images and matches them. This provides a final visual check:
- If a card matches many keypoints, it's probably correct
- If a card matches few keypoints despite high embedding similarity, something might be wrong

This catches edge cases where CLIP was fooled.

### Multi-Set Search

You can search multiple sets at once by passing comma-separated codes:
```
set_code=DMR,SLD,ONE
```

The system will search each set's index separately and return the best match across all sets.

### All-Sets Search (No Set Code)

If you don't specify a set code at all, the system will automatically search ALL indexed sets:
- Discovers all available FAISS index files
- Searches each one and returns the best match
- **Slower** than specifying a set (has to load and search multiple indexes)
- **Limited to 50 images per batch** when no set code is specified

---

## Glossary

| Term | Meaning |
|------|---------|
| **Embedding** | A list of numbers (vector) that represents an image. Similar images have similar embeddings. |
| **CLIP** | A neural network model trained to understand images and text. We use it to create embeddings. |
| **FAISS** | A library for fast similarity search among millions of vectors. Made by Facebook/Meta. |
| **pHash** | Perceptual hash. A simple fingerprint of an image based on brightness patterns. |
| **Hamming Distance** | How many bits differ between two binary strings. Used to compare pHash values. |
| **ORB** | A keypoint detection algorithm. Finds distinctive corners and blobs in images. |
| **Cosine Similarity** | A way to measure how similar two vectors are (0 = completely different, 1 = identical). |
| **L2 Normalization** | Making a vector have length 1.0. Required for cosine similarity to work. |
| **Composite Embedding** | A weighted combination of embeddings from different card regions. |
| **Set Code** | Three-letter code for a Magic set (like "DMR" for Double Masters Remastered). |
| **Collector Number** | The number at the bottom of a Magic card (unique within each set). |
| **Finish** | Whether a card is regular (nonfoil), foil, or etched. |
| **Clarity Score** | The gap between the #1 and #2 match. Higher = more certain. |
| **Ambiguous Match** | When multiple cards have very similar scores, making it hard to choose. |
| **Scryfall** | A website and API that provides Magic card data and images. |
| **Batch** | A group of images uploaded together for recognition. |
| **UUID** | Universally Unique Identifier. A long string that uniquely identifies each card in Scryfall. |

---

## Security Notes

- This system does NOT require authentication by default
- Uploaded images are stored in the `uploads/` folder and NOT automatically deleted
- The SQLite database contains no sensitive data
- All external network requests go only to Scryfall (api.scryfall.com)

---

## Troubleshooting

### "No cards with composite embeddings found"

**Cause:** The set hasn't been indexed, or the set code case doesn't match.

**Fix:**
1. Run `uv run python scripts/index_set.py YOUR_SET`
2. Make sure set codes are uppercase in the database

### "Index not found"

**Cause:** FAISS index file doesn't exist.

**Fix:** Run the indexing script for that set.

### Low confidence scores

**Possible causes:**
1. Image quality is poor (blurry, dark, wrong colors)
2. Card is tilted too much for detection to work
3. Set hasn't been indexed

**Fix:** Take a clearer photo, try different lighting, or manually crop the card before uploading.

### Cards matching wrong set

**Cause:** Not specifying which set to search.

**Fix:** Always provide the `set_code` parameter when you know which set the cards are from.

---

*This documentation was generated to be a complete reference for the MTG Card Recognition System. If you find errors or have questions, please open an issue on GitHub.*
