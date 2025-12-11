# MTG Card Recognition System

Recognize Magic: The Gathering cards from scanned images using CLIP embeddings, perceptual hashing, and ORB keypoint matching.

## Recognition Pipeline

```
Card Detection (OpenCV) → CLIP Embedding → FAISS Search → pHash Rerank → ORB Verify → Match
```

**Scoring:** 85% CLIP embedding + 15% pHash, with ORB verification for top candidates.

## Quick Start

### 1. Install

```bash
pip install uv
uv pip install -r requirements.txt
```

### 2. Initialize Database

```bash
uv run python -c "from src.database.schema import init_db; init_db()"
uv run python -c "from api.database import init_batch_tables; init_batch_tables()"
```

### 3. Download Card Images

```bash
uv run python scripts/download_default_cards.py --sets DMR
uv run python scripts/download_default_cards.py --sets DMR CLB SLD
```

### 4. Index the Set

```bash
uv run python scripts/index_set.py DMR              # Single set
uv run python scripts/index_set.py DMR SLD NEO      # Multiple sets
uv run python scripts/index_set.py --all            # All downloaded sets
uv run python scripts/index_set.py DMR --device cuda  # GPU acceleration
uv run python scripts/index_set.py --all --dry-run  # Preview what would be indexed
```

Creates per-set FAISS indexes: `{SET}_nonfoil_composite.faiss` and `{SET}_foil_composite.faiss`

### 5. Build Unified Index (Optional)

Merge all per-set indexes into a single "ALL" index for cross-set search:

```bash
uv run python scripts/build_unified_index.py --finish nonfoil
uv run python scripts/build_unified_index.py --finish foil
uv run python scripts/build_unified_index.py --list-only  # See available indexes
```

Creates `ALL_nonfoil_composite.faiss` and `ALL_foil_composite.faiss`. Re-run after indexing new sets.

**NOTE:** A pre-computed index, unified index and db is provided on a cdn. These files can be requested be contacting me on  [![Discord](https://img.shields.io/badge/My-Discord-%235865F2.svg)](https://discord.gg/tfbznkTnXN)

### 6. Test Recognition

```bash
uv run python scripts/recognize_card.py path/to/image.jpg --set DMR
```

## API Server

```bash
uv run uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload
                        or
pip install uvicorn
python -m uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload
```


Open http://localhost:8000 for batch recognition.

### Endpoints

- `POST /api/batch/upload` - Upload images
- `GET /api/batch/{batch_id}/status` - Check progress
- `GET /api/batch/{batch_id}/results` - Get results
- `POST /api/batch/{batch_id}/correct` - Submit corrections
- `GET /api/batch/{batch_id}/export?format=csv` - Export (CSV/JSON)

## Project Structure

```
mtg_image_recog/
├── api/                      # FastAPI web service
│   ├── main.py
│   ├── database.py
│   ├── routes/batch.py
│   └── services/
├── src/
│   ├── detection/            # Card detection & warp
│   ├── embeddings/           # CLIP embedder
│   ├── indexing/             # Indexing & pHash
│   ├── recognition/          # Matcher & ORB
│   ├── ann/                  # FAISS indices
│   └── database/             # SQLite schema
├── scripts/
│   ├── download_default_cards.py
│   ├── index_set.py
│   └── recognize_card.py
├── web/                      # Frontend
└── data/
    ├── scryfall/             # Card images
    ├── indexes/              # FAISS files
    └── mtg_recognition.db
```

## Configuration

Copy `.env.example` to `.env`:

```bash
DATABASE_PATH=data/mtg_recognition.db
SCRYFALL_IMAGES_DIR=./data/scryfall
INDEXES_DIR=./data/indexes
ACCEPT_THRESHOLD=0.85
MANUAL_THRESHOLD=0.65
```

## Docker

```bash
docker-compose build
docker-compose run --rm mtg-recognition python scripts/index_set.py DMR
docker-compose up
```

## Technical Details

### Indexing
- Composite CLIP embedding: 70% full card + 20% collector region + 10% name region
- 3 pHash variants per card (full/name/collector)
- Auto-detects variant types (normal/extended_art/showcase/borderless)
- Separate indexes for nonfoil and foil/etched finishes

### ORB Verification
- Keypoint matching with Lowe's ratio test
- Boosts/penalizes based on visual similarity

### FAISS Index
- Inner product on L2-normalized embeddings
- ~1ms retrieval for top 20 candidates

## Output

- `card_name`, `set_code`, `collector_number`, `finish`
- `scryfall_id` - Scryfall UUID
- `confidence` - Match score
- `candidates` - Alternative matches

## License

MIT

## Acknowledgments

- [Scryfall](https://scryfall.com/) for card data
- [OpenCLIP](https://github.com/mlfoundations/open_clip) for embeddings
