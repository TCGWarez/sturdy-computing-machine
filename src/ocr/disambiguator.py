"""
OCR-based disambiguation for ambiguous card matches.

When the CLIP-based matcher returns multiple high-confidence candidates,
this module uses OCR to extract text from the card and validate against
the database to determine the correct match.
"""

import re
import logging
from dataclasses import dataclass
from typing import List, Optional, TYPE_CHECKING

import numpy as np

from src.ocr.base_ocr import BaseOCRService, OCRResult
from src.ocr.region_crops import crop_title_for_ocr, crop_collector_for_ocr
from src.database.schema import Card

if TYPE_CHECKING:
    from sqlalchemy.orm import Session
    from src.recognition.matcher import MatchCandidate

logger = logging.getLogger(__name__)


@dataclass
class CollectorInfo:
    """Parsed collector information from OCR."""

    collector_number: Optional[str] = None
    """Collector number (e.g., '123' from '123/280')."""

    set_code: Optional[str] = None
    """Set code (e.g., 'NEO' from '123/280 * NEO')."""

    rarity: Optional[str] = None
    """Rarity letter: C=Common, U=Uncommon, R=Rare, M=Mythic, S=Special, etc."""

    raw_text: str = ""
    """Original OCR text before parsing."""

    pattern_matched: Optional[str] = None
    """Which regex pattern matched (for debugging)."""

    def __bool__(self) -> bool:
        """Return True if any useful info was extracted."""
        return bool(self.collector_number or self.set_code)


@dataclass
class OCRDisambiguationResult:
    """Result of OCR disambiguation attempt."""

    success: bool
    """True if OCR successfully identified a card."""

    card_id: Optional[str] = None
    """Database card_id if a match was found."""

    title_text: Optional[str] = None
    """OCR-extracted title text."""

    collector_info: Optional[CollectorInfo] = None
    """Parsed collector information."""

    title_confidence: float = 0.0
    """OCR confidence for title extraction."""

    collector_confidence: float = 0.0
    """OCR confidence for collector extraction."""

    method: str = "none"
    """How the match was found: 'collector_match', 'title_match', 'failed'."""

    matched_candidates: List[str] = None
    """List of candidate card_ids that matched OCR results."""

    def __post_init__(self):
        if self.matched_candidates is None:
            self.matched_candidates = []


class OCRDisambiguator:
    """
    Uses OCR to disambiguate between similar card candidates.

    The disambiguation process:
    1. Extract title and collector regions from the warped card image
    2. Run OCR on both regions using region-specific preprocessing
    3. Parse collector text to extract set_code, collector_number, rarity, and foil status
    4. Query database to find matching cards
    5. Boost candidates that match OCR results

    Usage:
        disambiguator = OCRDisambiguator(
            ocr_service=TesseractOCRService(),
            db=session
        )
        result = disambiguator.disambiguate(warped_image, candidates)
        if result.success:
            print(f"OCR match: {result.card_id}")
    """

    # Rarity indicators: C=Common, U=Uncommon, R=Rare, M=Mythic, S=Special, P=Promo, T=Timeshifted, B=Bonus, L=Land
    RARITY_LETTERS = r'[CURMSPTBL]'

    # Pattern 1: Multi-line format - "158/244 R ... UNF" (set code after rarity, possibly on next line)
    # The .*? allows for misread characters (like star read as "4") between rarity and set code
    # Requires 2+ digit collector number to avoid false matches
    PATTERN_MULTILINE = re.compile(
        rf'(\d{{2,}}[a-z]?)\s*/\s*\d+\s+{RARITY_LETTERS}\b.*?([A-Z]{{3}})\b',
        re.IGNORECASE
    )

    # Pattern 2: Format with rarity inline - "188/244 R UNF", "0270 R TDM"
    # Matches: collector[/total] [separator] [rarity] set_code
    PATTERN_WITH_RARITY = re.compile(
        rf'(\d{{2,}}[a-z]?)\s*(?:/\s*\d+)?\s*[•·.*\s]*{RARITY_LETTERS}\s+([A-Z]{{2,5}})\b',
        re.IGNORECASE
    )

    # Pattern 3: Format with separator - "357/291 * DSK", "123 • NEO"
    # Matches: collector[/total] separator set_code (no rarity)
    PATTERN_WITH_SEPARATOR = re.compile(
        r'(\d{2,}[a-z]?)\s*(?:/\s*\d+)?\s*[•·.*]+\s*([A-Z][A-Z0-9]{1,4})\b',
        re.IGNORECASE
    )

    # Pattern 4: Format with space - "123/280 NEO", "0270 TDM"
    # Matches: collector[/total] space set_code (2+ chars to avoid single rarity letter)
    PATTERN_WITH_SPACE = re.compile(
        r'(\d{2,}[a-z]?)\s*(?:/\s*\d+)?\s+([A-Z][A-Z0-9]{1,4})\b',
        re.IGNORECASE
    )

    # Pattern 5: Just collector number - "123/280" or "123"
    PATTERN_NUMBER_ONLY = re.compile(
        r'(\d+[a-z]?)(?:\s*/\s*\d+)?'
    )

    def __init__(self, ocr_service: BaseOCRService, db: "Session"):
        """
        Initialize the OCR disambiguator.

        Args:
            ocr_service: OCR service for text extraction
            db: SQLAlchemy database session
        """
        self.ocr_service = ocr_service
        self.db = db

    def disambiguate(
        self,
        warped_image: np.ndarray,
        candidates: List["MatchCandidate"],
        boost_weight: float = 0.15
    ) -> OCRDisambiguationResult:
        """
        Attempt to disambiguate candidates using OCR.

        Args:
            warped_image: Warped card image (RGB, 363x504)
            candidates: List of match candidates from CLIP matcher
            boost_weight: Score boost for OCR-validated candidates

        Returns:
            OCRDisambiguationResult with match info
        """
        logger.info("Starting OCR disambiguation...")

        # Extract and OCR title region
        title_result = self._ocr_title(warped_image)
        logger.info(f"Title OCR: '{title_result.text}' (conf: {title_result.confidence:.2f})")

        # Extract and OCR collector region
        collector_result = self._ocr_collector(warped_image)
        logger.info(f"Collector OCR: '{collector_result.text}' (conf: {collector_result.confidence:.2f})")

        # Parse collector info
        collector_info = self._parse_collector_text(collector_result.text)

        # If no set code found via patterns, try fuzzy matching against valid set codes
        if not collector_info.set_code:
            from src.ocr.set_code_validator import extract_set_code_from_text
            fuzzy_set = extract_set_code_from_text(collector_result.text)
            if fuzzy_set:
                collector_info.set_code = fuzzy_set
                logger.info(f"Fuzzy matched set code: '{fuzzy_set}' from '{collector_result.text}'")

        logger.info(
            f"Parsed collector: number={collector_info.collector_number}, "
            f"set={collector_info.set_code}, rarity={collector_info.rarity}"
        )

        # Try to validate against database
        matched_card_ids = self._validate_against_candidates(
            title_text=title_result.text,
            collector_info=collector_info,
            candidates=candidates
        )

        if matched_card_ids:
            logger.info(f"OCR matched {len(matched_card_ids)} candidates: {matched_card_ids}")

            # Boost matching candidates
            self._boost_candidates(candidates, matched_card_ids, boost_weight)

            return OCRDisambiguationResult(
                success=True,
                card_id=matched_card_ids[0] if len(matched_card_ids) == 1 else None,
                title_text=title_result.text,
                collector_info=collector_info,
                title_confidence=title_result.confidence,
                collector_confidence=collector_result.confidence,
                method='collector_match' if collector_info else 'title_match',
                matched_candidates=matched_card_ids
            )

        # Fallback: Try triangulation with title + set_code
        if title_result.text and collector_info.set_code:
            # Determine finish from candidates
            finish = candidates[0].finish if candidates else 'nonfoil'
            triangulated_card = self.triangulate_card(
                title_text=title_result.text,
                set_code=collector_info.set_code,
                collector_number=collector_info.collector_number,
                finish=finish
            )
            if triangulated_card:
                logger.info(f"Triangulation found: {triangulated_card.name} ({triangulated_card.set_code} #{triangulated_card.collector_number})")
                # Boost the triangulated card if it's in candidates
                for candidate in candidates:
                    if candidate.card_id == triangulated_card.id:
                        original_score = candidate.combined_score
                        candidate.combined_score = min(1.0, original_score + boost_weight * 2)  # Strong boost
                        logger.info(f"Triangulation boost: {candidate.card_name} {original_score:.3f} -> {candidate.combined_score:.3f}")
                        break

                return OCRDisambiguationResult(
                    success=True,
                    card_id=triangulated_card.id,
                    title_text=title_result.text,
                    collector_info=collector_info,
                    title_confidence=title_result.confidence,
                    collector_confidence=collector_result.confidence,
                    method='triangulation',
                    matched_candidates=[triangulated_card.id]
                )

        logger.info("OCR disambiguation did not find a match")
        return OCRDisambiguationResult(
            success=False,
            title_text=title_result.text,
            collector_info=collector_info,
            title_confidence=title_result.confidence,
            collector_confidence=collector_result.confidence,
            method='failed'
        )

    def _ocr_title(self, image: np.ndarray) -> OCRResult:
        """Extract and OCR the title region using title-optimized preprocessing."""
        try:
            title_crop = crop_title_for_ocr(image)
            return self.ocr_service.extract_title_text(title_crop)
        except Exception as e:
            logger.warning(f"Title OCR failed: {e}")
            return OCRResult(text="", confidence=0.0)

    def _ocr_collector(self, image: np.ndarray) -> OCRResult:
        """Extract and OCR the collector region using collector-optimized preprocessing."""
        try:
            collector_crop = crop_collector_for_ocr(image)
            return self.ocr_service.extract_collector_text(collector_crop)
        except Exception as e:
            logger.warning(f"Collector OCR failed: {e}")
            return OCRResult(text="", confidence=0.0)

    @classmethod
    def _parse_collector_text(cls, text: str) -> CollectorInfo:
        """
        Parse collector text to extract set_code, collector_number, rarity, and foil status.

        Handles various formats:
        - "158/244 R 4 TNE+EN" -> number=158, set=TNE (multiline with OCR errors)
        - "188/244 R UNF + EN" -> number=188, set=UNF, rarity=R
        - "357/291 * DSK" -> number=357, set=DSK
        - "0270 TDM" -> number=0270, set=TDM
        - "123/280 NEO" -> number=123, set=NEO
        - "158/244 R UNF XEN" -> foil version (has star merged with EN)

        Args:
            text: Raw OCR text from collector region

        Returns:
            CollectorInfo with parsed data
        """
        result = CollectorInfo(raw_text=text)

        if not text:
            return result

        clean_text = text.strip().upper()

        # Try multiline pattern first - "158/244 R ... UNF" (handles OCR noise between rarity and set)
        match = cls.PATTERN_MULTILINE.search(clean_text)
        if match:
            result.collector_number = match.group(1).lower()
            result.set_code = match.group(2).upper()
            # Extract rarity
            rarity_match = re.search(rf'/\s*\d+\s+({cls.RARITY_LETTERS})\b', clean_text, re.IGNORECASE)
            if rarity_match:
                result.rarity = rarity_match.group(1).upper()
            result.pattern_matched = 'PATTERN_MULTILINE'
            return result

        # Try pattern with rarity inline - "188/244 R UNF"
        match = cls.PATTERN_WITH_RARITY.search(clean_text)
        if match:
            result.collector_number = match.group(1).lower()
            result.set_code = match.group(2).upper()
            # Extract rarity from the match region
            rarity_match = re.search(rf'({cls.RARITY_LETTERS})\s+{re.escape(match.group(2))}', clean_text, re.IGNORECASE)
            if rarity_match:
                result.rarity = rarity_match.group(1).upper()
            result.pattern_matched = 'PATTERN_WITH_RARITY'
            return result

        # Try pattern with separator - "357/291 * DSK"
        match = cls.PATTERN_WITH_SEPARATOR.search(clean_text)
        if match:
            result.collector_number = match.group(1).lower()
            result.set_code = match.group(2).upper()
            result.pattern_matched = 'PATTERN_WITH_SEPARATOR'
            return result

        # Try pattern with just space - "123/280 NEO"
        match = cls.PATTERN_WITH_SPACE.search(clean_text)
        if match:
            # Make sure we didn't just capture a rarity letter
            potential_set = match.group(2).upper()
            if len(potential_set) >= 2:  # Set codes are at least 2 chars
                result.collector_number = match.group(1).lower()
                result.set_code = potential_set
                result.pattern_matched = 'PATTERN_WITH_SPACE'
                return result

        # Fallback to number only
        match = cls.PATTERN_NUMBER_ONLY.search(clean_text)
        if match:
            result.collector_number = match.group(1).lower()
            result.pattern_matched = 'PATTERN_NUMBER_ONLY'

        return result

    def _validate_against_candidates(
        self,
        title_text: str,
        collector_info: CollectorInfo,
        candidates: List["MatchCandidate"]
    ) -> List[str]:
        """
        Validate OCR results against the candidate list.

        Priority:
        1. Set code + collector number match (highest confidence)
        2. Collector number only match
        3. Title fuzzy match

        Args:
            title_text: OCR-extracted title
            collector_info: Parsed collector info
            candidates: List of match candidates

        Returns:
            List of matching card_ids
        """
        matched_ids = []

        # First, try set_code + collector_number match
        if collector_info.set_code and collector_info.collector_number:
            for candidate in candidates:
                if (candidate.set_code.upper() == collector_info.set_code.upper() and
                    candidate.collector_number.lower() == collector_info.collector_number.lower()):
                    matched_ids.append(candidate.card_id)
                    logger.debug(f"Exact collector match: {candidate.card_name} ({candidate.set_code} #{candidate.collector_number})")

            if matched_ids:
                return matched_ids

        # Second, try collector_number only match
        if collector_info.collector_number:
            for candidate in candidates:
                if candidate.collector_number.lower() == collector_info.collector_number.lower():
                    matched_ids.append(candidate.card_id)
                    logger.debug(f"Collector number match: {candidate.card_name} #{candidate.collector_number}")

            if matched_ids:
                return matched_ids

        # Third, try title fuzzy match
        if title_text and len(title_text) >= 3:
            title_clean = title_text.lower().strip()
            for candidate in candidates:
                candidate_name = candidate.card_name.lower()
                # Check if OCR title is contained in card name or vice versa
                if title_clean in candidate_name or candidate_name in title_clean:
                    matched_ids.append(candidate.card_id)
                    logger.debug(f"Title match: '{title_text}' -> {candidate.card_name}")

        return matched_ids

    def _boost_candidates(
        self,
        candidates: List["MatchCandidate"],
        matched_ids: List[str],
        boost_weight: float
    ) -> None:
        """
        Boost the scores of candidates that match OCR results.

        Args:
            candidates: List of candidates to modify (in-place)
            matched_ids: IDs of candidates to boost
            boost_weight: Amount to add to combined_score
        """
        for candidate in candidates:
            if candidate.card_id in matched_ids:
                original_score = candidate.combined_score
                candidate.combined_score = min(1.0, original_score + boost_weight)
                logger.info(
                    f"OCR boost: {candidate.card_name} "
                    f"{original_score:.3f} -> {candidate.combined_score:.3f}"
                )

    def validate_full_database(
        self,
        collector_info: CollectorInfo,
        finish: str = 'nonfoil'
    ) -> Optional[str]:
        """
        Validate collector info against full database (not just candidates).

        Use this when you want to find a card even if it's not in the
        current candidate list.

        Args:
            collector_info: Parsed collector info
            finish: Card finish to filter by

        Returns:
            card_id if found, None otherwise
        """
        if not collector_info.set_code or not collector_info.collector_number:
            return None

        try:
            card = self.db.query(Card).filter(
                Card.set_code == collector_info.set_code.upper(),
                Card.collector_number == collector_info.collector_number,
                Card.finish == finish.lower()
            ).first()

            if card:
                logger.info(f"Database match: {card.name} ({card.set_code} #{card.collector_number})")
                return card.id

        except Exception as e:
            logger.error(f"Database lookup failed: {e}")

        return None

    def triangulate_card(
        self,
        title_text: Optional[str],
        set_code: Optional[str],
        collector_number: Optional[str],
        finish: str = 'nonfoil'
    ) -> Optional[Card]:
        """
        Use triangle of information to find exact card match.

        With any 2 of {title, set_code, collector_number}, we can usually
        identify the exact card. Priority order:
        1. set_code + collector_number (EXACT match, always unique)
        2. title + set_code (usually unique per set)
        3. title + collector_number (filter by collector, match title)

        Args:
            title_text: OCR'd card title (e.g., "Wind-Scarred Crag")
            set_code: Set code (e.g., "TDM") - should be fuzzy-matched first
            collector_number: Collector number (e.g., "027")
            finish: Card finish ('foil' or 'nonfoil')

        Returns:
            Card if found, None otherwise
        """
        try:
            # Strategy 1: set_code + collector_number = EXACT match
            if set_code and collector_number:
                card = self.db.query(Card).filter(
                    Card.set_code == set_code.upper(),
                    Card.collector_number == collector_number,
                    Card.finish == finish.lower()
                ).first()
                if card:
                    logger.info(f"Triangle match (set+collector): {card.name} ({card.set_code} #{card.collector_number})")
                    return card

            # Strategy 2: title + set_code (usually unique per set)
            if title_text and set_code:
                # Normalize title for comparison
                title_normalized = title_text.lower().strip()
                cards = self.db.query(Card).filter(
                    Card.set_code == set_code.upper(),
                    Card.finish == finish.lower()
                ).all()

                # Find exact or close title match
                for card in cards:
                    if card.name.lower() == title_normalized:
                        logger.info(f"Triangle match (title+set): {card.name} ({card.set_code})")
                        return card

                # Try fuzzy title match (first word match for multi-word titles)
                title_first_word = title_normalized.split()[0] if title_normalized else ""
                if title_first_word and len(title_first_word) > 3:
                    for card in cards:
                        card_first_word = card.name.lower().split()[0]
                        if card_first_word == title_first_word:
                            logger.info(f"Triangle match (title-fuzzy+set): {card.name} ({card.set_code})")
                            return card

            # Strategy 3: title + collector_number (cross-set search)
            if title_text and collector_number:
                title_normalized = title_text.lower().strip()
                cards = self.db.query(Card).filter(
                    Card.collector_number == collector_number,
                    Card.finish == finish.lower()
                ).all()

                for card in cards:
                    if card.name.lower() == title_normalized:
                        logger.info(f"Triangle match (title+collector): {card.name} ({card.set_code} #{card.collector_number})")
                        return card

        except Exception as e:
            logger.error(f"Triangle lookup failed: {e}")

        logger.debug(f"No triangle match found for title='{title_text}', set='{set_code}', collector='{collector_number}'")
        return None
