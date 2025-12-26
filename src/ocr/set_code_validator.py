"""
Fuzzy set code validation for OCR correction.

MTG set codes are 3-5 character codes (e.g., "TDM", "NEO", "DMR").
OCR often makes single-character errors that can be corrected by
fuzzy matching against the known list of valid set codes.

Common OCR errors:
- O/0 confusion: "NE0" -> "NEO"
- D/O confusion: "TOM" -> "TDM"
- I/1/l confusion: "M1C" -> "MIC"
- S/5/8 confusion: "BR5" -> "BRS"
"""

import logging
from difflib import SequenceMatcher
from typing import Optional, Set, List, Tuple

logger = logging.getLogger(__name__)

# Global cache for valid set codes
_valid_set_codes: Optional[Set[str]] = None


# Common OCR character substitutions (original -> possible corrections)
OCR_SUBSTITUTIONS = {
    '0': ['O', 'D', 'Q'],
    'O': ['0', 'D', 'Q'],
    'D': ['O', '0'],
    '1': ['I', 'L', 'l'],
    'I': ['1', 'L', 'l'],
    'L': ['1', 'I'],
    'l': ['1', 'I', 'L'],
    '5': ['S'],
    'S': ['5'],
    '8': ['B'],
    'B': ['8'],
    'G': ['6'],
    '6': ['G'],
    'Z': ['2'],
    '2': ['Z'],
}


def load_valid_set_codes(db_session=None) -> Set[str]:
    """
    Load all valid set codes from database.

    Args:
        db_session: SQLAlchemy session. If None, creates a new session.

    Returns:
        Set of valid uppercase set codes.
    """
    global _valid_set_codes

    if _valid_set_codes is not None:
        return _valid_set_codes

    try:
        if db_session is None:
            from src.database.db import SessionLocal
            db_session = SessionLocal()
            should_close = True
        else:
            should_close = False

        from src.database.schema import Card
        codes = db_session.query(Card.set_code).distinct().all()
        _valid_set_codes = {code[0].upper() for code in codes if code[0]}

        if should_close:
            db_session.close()

        logger.info(f"Loaded {len(_valid_set_codes)} valid set codes from database")
        return _valid_set_codes

    except Exception as e:
        logger.warning(f"Could not load set codes from database: {e}")
        # Return empty set - fuzzy matching will return None
        return set()


def get_valid_set_codes() -> Set[str]:
    """Get cached valid set codes, loading from DB if needed."""
    global _valid_set_codes
    if _valid_set_codes is None:
        return load_valid_set_codes()
    return _valid_set_codes


def set_valid_set_codes(codes: Set[str]) -> None:
    """Manually set valid codes (for testing)."""
    global _valid_set_codes
    _valid_set_codes = {c.upper() for c in codes}


def _levenshtein_distance(s1: str, s2: str) -> int:
    """Calculate Levenshtein edit distance between two strings."""
    if len(s1) < len(s2):
        return _levenshtein_distance(s2, s1)

    if len(s2) == 0:
        return len(s1)

    previous_row = range(len(s2) + 1)
    for i, c1 in enumerate(s1):
        current_row = [i + 1]
        for j, c2 in enumerate(s2):
            insertions = previous_row[j + 1] + 1
            deletions = current_row[j] + 1
            substitutions = previous_row[j] + (c1 != c2)
            current_row.append(min(insertions, deletions, substitutions))
        previous_row = current_row

    return previous_row[-1]


def _generate_substitutions(text: str) -> List[str]:
    """Generate possible OCR corrections using character substitution table."""
    results = [text]

    for i, char in enumerate(text):
        if char in OCR_SUBSTITUTIONS:
            for replacement in OCR_SUBSTITUTIONS[char]:
                variant = text[:i] + replacement + text[i+1:]
                results.append(variant)

    return results


def fuzzy_match_set_code(
    raw_ocr: str,
    valid_codes: Optional[Set[str]] = None,
    max_distance: int = 1
) -> Optional[str]:
    """
    Match OCR'd text against valid set codes with fuzzy matching.

    Tries multiple strategies:
    1. Exact match
    2. OCR character substitutions (O/0, I/1, etc.)
    3. Levenshtein distance matching

    Args:
        raw_ocr: Raw OCR text (e.g., "TOM", "NE0")
        valid_codes: Set of valid codes. If None, loads from database.
        max_distance: Maximum Levenshtein distance for fuzzy match (default: 1)

    Returns:
        Matched set code or None if no match found.

    Examples:
        >>> fuzzy_match_set_code("TOM", {"TDM", "NEO", "DMR"})
        'TDM'
        >>> fuzzy_match_set_code("NE0", {"TDM", "NEO", "DMR"})
        'NEO'
        >>> fuzzy_match_set_code("XYZ", {"TDM", "NEO", "DMR"})
        None
    """
    if not raw_ocr:
        return None

    raw = raw_ocr.upper().strip()

    if valid_codes is None:
        valid_codes = get_valid_set_codes()

    if not valid_codes:
        logger.warning("No valid set codes available for matching")
        return None

    # Strategy 1: Exact match
    if raw in valid_codes:
        logger.debug(f"Set code exact match: {raw}")
        return raw

    # Strategy 2: Try OCR substitutions
    for variant in _generate_substitutions(raw):
        if variant in valid_codes:
            logger.debug(f"Set code OCR substitution: {raw} -> {variant}")
            return variant

    # Strategy 3: Levenshtein distance matching
    best_match = None
    best_distance = max_distance + 1

    for valid in valid_codes:
        # Only consider codes of similar length
        if abs(len(valid) - len(raw)) > max_distance:
            continue

        distance = _levenshtein_distance(raw, valid)
        if distance < best_distance:
            best_distance = distance
            best_match = valid

    if best_match and best_distance <= max_distance:
        logger.debug(f"Set code Levenshtein match: {raw} -> {best_match} (distance={best_distance})")
        return best_match

    logger.debug(f"No set code match found for: {raw}")
    return None


def extract_set_code_from_text(text: str, valid_codes: Optional[Set[str]] = None) -> Optional[str]:
    """
    Extract and fuzzy-match a set code from OCR text.

    Looks for 2-5 character uppercase sequences that could be set codes.

    Args:
        text: Full OCR text that may contain a set code
        valid_codes: Set of valid codes. If None, loads from database.

    Returns:
        Matched set code or None.

    Examples:
        >>> extract_set_code_from_text("TOM EN FILIP", {"TDM", "NEO"})
        'TDM'
    """
    import re

    if not text:
        return None

    if valid_codes is None:
        valid_codes = get_valid_set_codes()

    # Find all potential set codes (2-5 uppercase letters/numbers)
    potential_codes = re.findall(r'\b([A-Z0-9]{2,5})\b', text.upper())

    for code in potential_codes:
        match = fuzzy_match_set_code(code, valid_codes)
        if match:
            return match

    return None


# Quick test
if __name__ == "__main__":
    # Test with mock data
    test_codes = {"TDM", "NEO", "DMR", "ONE", "BRO", "MOM", "WOE", "LCI", "MKM", "OTJ"}
    set_valid_set_codes(test_codes)

    test_cases = [
        ("TDM", "TDM"),   # Exact match
        ("TOM", "TDM"),   # D/O confusion
        ("NE0", "NEO"),   # 0/O confusion
        ("0NE", "ONE"),   # 0/O at start
        ("BR0", "BRO"),   # 0/O at end
        ("M0M", "MOM"),   # 0/O in middle
        ("W0E", "WOE"),   # 0/O
        ("LCl", "LCI"),   # l/I confusion
        ("MKN", "MKM"),   # Close Levenshtein
        ("XYZ", None),    # No match
        ("AB", None),     # Too short, no match
    ]

    print("Testing fuzzy set code matching:")
    for raw, expected in test_cases:
        result = fuzzy_match_set_code(raw, test_codes)
        status = "PASS" if result == expected else "FAIL"
        print(f"  {status}: '{raw}' -> '{result}' (expected: '{expected}')")
