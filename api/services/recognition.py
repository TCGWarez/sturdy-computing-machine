"""
Recognition service wrapper
Uses CardMatcher for consistent recognition across API and CLI
"""

from pathlib import Path
import sys
import logging
from typing import List, Optional

logger = logging.getLogger(__name__)

project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.recognition.matcher import CardMatcher
from src.config import INDEXES_DIR
from src.utils.device import resolve_device

# Cache matchers by set_code to avoid reloading indices
_matcher_cache = {}


def get_available_indexes(finish: str = 'nonfoil') -> List[str]:
    """
    Discover all available FAISS indexes for a given finish.

    Returns:
        List of set codes that have indexes available
    """
    if not INDEXES_DIR.exists():
        return []

    # Look for index files matching pattern: {SET}_{finish}_composite.faiss
    pattern = f"*_{finish}_composite.faiss"
    index_files = list(INDEXES_DIR.glob(pattern))

    set_codes = []
    for index_file in index_files:
        # Extract set code from filename like "DMR_nonfoil_composite.faiss"
        parts = index_file.stem.split('_')
        if len(parts) >= 3:
            set_code = parts[0]
            set_codes.append(set_code)

    return sorted(set_codes)


def get_matcher(set_code: str, finish: str = 'nonfoil', device: Optional[str] = None) -> CardMatcher:
    """Get or create a cached CardMatcher instance"""
    resolved_device = resolve_device(device)
    cache_key = f"{set_code.upper()}_{finish}_{resolved_device}"

    if cache_key not in _matcher_cache:
        _matcher_cache[cache_key] = CardMatcher(
            set_code=set_code,
            finish=finish,
            device=resolved_device
        )

    return _matcher_cache[cache_key]


def recognize_card(
    image_path: Path,
    set_code: str = None,
    finish: str = None,
    prefer_foil: bool = False,
    device: Optional[str] = None
) -> dict:
    """
    Recognize a single card using CardMatcher

    Supports comma-separated set codes (e.g., "DMR,SLD") for multi-set searches.
    If no set_code is provided, searches ALL available indexed sets.

    Args:
        image_path: Path to card image
        set_code: Set code filter (optional). Can be comma-separated for multi-set search.
                  If None, searches all available indexes.
        finish: Finish filter ('nonfoil' or 'foil')
        prefer_foil: If True, prefer foil matches; if False, prefer nonfoil
        device: Device to use ('cpu', 'cuda', or None for auto-detection)

    Returns:
        dict with recognition result including boundary_corners
    """
    if not finish:
        finish = 'nonfoil'

    # Determine which sets to search
    if set_code:
        # User specified set codes (comma-separated)
        set_codes = [s.strip().upper() for s in set_code.split(',') if s.strip()]
        if not set_codes:
            return {"error": "No valid set codes provided"}
    else:
        # No set specified - try to use unified "ALL" index first
        all_index_path = INDEXES_DIR / f"ALL_{finish}_composite.faiss"
        if all_index_path.exists():
            # Use unified index for fast all-sets search
            set_codes = ['ALL']
            logger.info(f"No set_code specified, using unified ALL index")
        else:
            pass
            # # Fallback to sequential search of all indexes (slow)
            # set_codes = get_available_indexes(finish)
            # if not set_codes:
            #     return {"error": f"No indexed sets found for finish '{finish}'. Please index some sets first."}
            # logger.warning(f"No unified ALL index found. Falling back to sequential search of {len(set_codes)} sets. "
            #               f"Run 'python scripts/build_unified_index.py --finish {finish}' to build a unified index for faster searches.")

    # Track best result across all sets
    best_result = None
    best_score = -1
    errors = []

    for code in set_codes:
        try:
            matcher = get_matcher(code, finish, device)
            result = matcher.match_scanned(image_path, use_ocr=False, debug=False)

            if result.match_card_id and result.candidates:
                top_score = result.candidates[0].combined_score
                if top_score > best_score:
                    best_score = top_score
                    best_result = result

        except FileNotFoundError as e:
            errors.append(f"{code}: Index not found")
            logger.warning(f"Index not found for {code}/{finish}, skipping")
            continue
        except Exception as e:
            errors.append(f"{code}: {str(e)}")
            logger.error(f"Recognition failed for {code}: {e}")
            continue

    # Return best result if found
    if best_result and best_result.candidates:
        top_candidate = best_result.candidates[0]

        # Get top 5 candidates for review mode (excluding the top match)
        other_candidates = []
        for c in best_result.candidates[1:6]:
            other_candidates.append({
                'card_id': c.card_id,
                'card_name': c.card_name,
                'set_code': c.set_code,
                'collector_number': c.collector_number,
                'finish': c.finish,
                'combined_score': c.combined_score,
                'image_path': c.image_path
            })

        return {
            'card_id': top_candidate.card_id,
            'name': top_candidate.card_name,
            'set_code': top_candidate.set_code,
            'collector_number': top_candidate.collector_number,
            'finish': top_candidate.finish,
            'scryfall_id': top_candidate.scryfall_id,
            'combined_score': top_candidate.combined_score,
            'composite_score': top_candidate.embedding_score,
            'phash_score': top_candidate.phash_score,
            'orb_score': 0.0,
            'image_path': top_candidate.image_path,
            'boundary_corners': best_result.boundary_corners,
            'processing_time': best_result.processing_time,
            'clarity_score': best_result.clarity_score,
            'is_ambiguous': best_result.is_ambiguous,
            'match_method': best_result.match_method,
            'candidates': other_candidates
        }

    # No results found
    if errors:
        return {"error": f"No matches found. Issues: {'; '.join(errors)}"}
    return {"error": "No matches found in any provided set"}
