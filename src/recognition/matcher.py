"""
src/recognition/matcher.py: Main card matching pipeline

1. Input: scanned image path
2. Detect and warp to canonical size
3. Compute CLIP embedding
4. ANN search for top K=20 candidates
5. Score candidates: embedding (0.6) + pHash (0.3) + OCR (0.1)
6. Return match if score > threshold, else flag for review
"""

import cv2
import numpy as np
from pathlib import Path
from typing import Union, List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime
import json
import logging

from src.config import (
    CANONICAL_WIDTH,
    CANONICAL_HEIGHT,
    ACCEPT_THRESHOLD,
    MANUAL_THRESHOLD,
    CLARITY_THRESHOLD
)
from src.detection.card_detector import detect_and_warp
from src.embeddings.embedder import CLIPEmbedder
from src.ann.faiss_index import FAISSIndex
from src.ann.hnsw_index import HNSWIndex
from src.indexing.phash import (
    compute_phash_variants,
    phash_to_int,
    batch_hamming_distance,
    int_to_phash
)
from src.recognition.orb_utils import compute_orb_similarity
from src.database.schema import SessionLocal, Card, PhashVariant, CompositeEmbedding
from src.database.db import get_card_full_data
from src.indexing.indexer import deserialize_embedding
from src.config import OCR_BOOST_WEIGHT

logger = logging.getLogger(__name__)


@dataclass
class MatchCandidate:
    """Single candidate match with scoring details"""
    card_id: str
    card_name: str
    set_code: str
    collector_number: str
    finish: str
    scryfall_id: str
    image_path: str

    # Individual scores
    embedding_score: float
    phash_score: float

    # Combined score
    combined_score: float

    # Distance details
    embedding_distance: float
    phash_full_distance: int
    phash_name_distance: int
    phash_collector_distance: int

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization"""
        return {
            'card_id': self.card_id,
            'card_name': self.card_name,
            'set_code': self.set_code,
            'collector_number': self.collector_number,
            'finish': self.finish,
            'scryfall_id': self.scryfall_id,
            'image_path': self.image_path,
            'embedding_score': float(self.embedding_score),
            'phash_score': float(self.phash_score),
            'combined_score': float(self.combined_score),
            'embedding_distance': float(self.embedding_distance),
            'phash_full_distance': int(self.phash_full_distance),
            'phash_name_distance': int(self.phash_name_distance),
            'phash_collector_distance': int(self.phash_collector_distance)
        }


@dataclass
class MatchResult:
    """Final match result with top candidate and alternatives"""
    scanned_path: str
    match_card_id: Optional[str]
    confidence: float
    match_method: str  # 'auto_accept', 'ambiguous_high', 'manual_review'
    candidates: List[MatchCandidate]
    processing_time: float

    # Clarity scoring (search-based matching)
    clarity_score: float = 1.0  # Gap between top 2 candidates (1.0 = only one candidate)
    is_ambiguous: bool = False  # True if multiple high-confidence candidates

    # Boundary detection (4 corners of detected card in original image)
    # Format: [[x1,y1], [x2,y2], [x3,y3], [x4,y4]] - top-left, top-right, bottom-right, bottom-left
    boundary_corners: Optional[List[List[float]]] = None

    # OCR disambiguation info (for card name/set/collector matching, NOT finish detection)
    ocr_used: bool = False
    ocr_title: Optional[str] = None
    ocr_collector: Optional[str] = None
    ocr_set_code: Optional[str] = None
    ocr_collector_number: Optional[str] = None
    ocr_title_confidence: Optional[float] = None
    ocr_collector_confidence: Optional[float] = None

    # Debug info
    debug_info: Optional[Dict[str, Any]] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON/CSV serialization"""
        return {
            'scanned_path': self.scanned_path,
            'match_card_id': self.match_card_id,
            'confidence': float(self.confidence),
            'match_method': self.match_method,
            'clarity_score': float(self.clarity_score),
            'is_ambiguous': self.is_ambiguous,
            'candidates': [c.to_dict() for c in self.candidates],
            'processing_time': float(self.processing_time),
            'boundary_corners': self.boundary_corners,
            'ocr_used': self.ocr_used,
            'ocr_title': self.ocr_title,
            'ocr_collector': self.ocr_collector,
            'ocr_set_code': self.ocr_set_code,
            'ocr_collector_number': self.ocr_collector_number,
            'ocr_title_confidence': self.ocr_title_confidence,
            'ocr_collector_confidence': self.ocr_collector_confidence,
            'debug_info': self.debug_info
        }

    def to_json(self) -> str:
        """Convert to JSON string"""
        return json.dumps(self.to_dict(), indent=2)


def compute_clarity(candidates: List[MatchCandidate], threshold: float = CLARITY_THRESHOLD) -> Tuple[float, bool]:
    """
    Compute clarity score to determine if match is ambiguous.

    Search-based matching: instead of just checking "is top score high enough?",
    also check "does top score stand out from alternatives?"

    Args:
        candidates: List of match candidates, sorted by combined_score descending
        threshold: Minimum gap between #1 and #2 for clear match (default 0.10)

    Returns:
        (clarity_score, is_ambiguous):
            - clarity_score: gap between top 2 candidates (1.0 if only one candidate)
            - is_ambiguous: True if multiple candidates have similar high scores
    """
    if len(candidates) < 2:
        return (1.0, False)  # Only one candidate = clear

    scores = [c.combined_score for c in candidates[:3]]
    gap_1_2 = scores[0] - scores[1]  # Gap between #1 and #2

    # If #3 exists and is also close, more ambiguous
    # Cluster detection: if top 3 are all within threshold, definitely ambiguous
    if len(scores) >= 3:
        if (scores[0] - scores[2]) < threshold:
            return (gap_1_2, True)  # Tight cluster of top 3

    # Primary clarity check: gap between #1 and #2
    is_ambiguous = gap_1_2 < threshold
    return (gap_1_2, is_ambiguous)


class CardMatcher:
    """Main card matching pipeline."""

    def __init__(
        self,
        set_code: str,
        finish: str = 'nonfoil',
        index_path: Optional[Path] = None,
        index_type: str = 'faiss',
        device: Optional[str] = None,
        embedder_checkpoint: Optional[Path] = None,
        top_k: int = 20,
        accept_threshold: float = ACCEPT_THRESHOLD,
        manual_threshold: float = MANUAL_THRESHOLD,
        scoring_weights: Optional[Dict[str, float]] = None
    ):
        """
        Initialize card matcher

        Args:
            set_code: Set code to match against
            finish: Finish type ('foil', 'nonfoil', 'etched')
            index_path: Path to pre-built ANN index (optional, will auto-detect if None)
            index_type: Type of ANN index ('faiss' or 'hnsw')
            device: Device for embedder ('cpu', 'cuda', or None for auto)
            embedder_checkpoint: Optional fine-tuned embedder checkpoint
            top_k: Number of candidates to retrieve from ANN search
            accept_threshold: Confidence threshold for automatic acceptance
            manual_threshold: Confidence threshold below which manual review is needed
            scoring_weights: Custom scoring weights dict with keys 'embedding', 'phash', 'ocr'
        """
        self.set_code = set_code.upper()
        self.finish = finish.lower()
        self.top_k = top_k
        self.accept_threshold = accept_threshold
        self.manual_threshold = manual_threshold
        self.index_type = index_type

        # Scoring weights (Updated after preprocessing fix)
        # With consistent 363x504 preprocessing, embedding scores are now reliable (0.90+)
        # pHash is sensitive to scanner artifacts so weight it lower
        if scoring_weights is None:
            self.weights = {
                'embedding': 0.85,  # High weight - embeddings are now reliable
                'phash': 0.15       # Low weight - sensitive to scanner artifacts
            }
        else:
            self.weights = scoring_weights

        logger.info(f"Initializing CardMatcher for {self.set_code}/{self.finish}")
        logger.info(f"Scoring weights: {self.weights}")
        logger.info(f"Thresholds - Accept: {self.accept_threshold}, Manual: {self.manual_threshold}")

        # Initialize embedder
        logger.info("Loading CLIP embedder...")
        self.embedder = CLIPEmbedder(device=device, checkpoint_path=embedder_checkpoint)

        # Load ANN index
        logger.info(f"Loading {index_type.upper()} index...")
        if index_type == 'faiss':
            self.ann_index = FAISSIndex()
        elif index_type == 'hnsw':
            self.ann_index = HNSWIndex()
        else:
            raise ValueError(f"Unknown index type: {index_type}")

        # Auto-detect index path if not provided
        if index_path is None:
            from src.config import INDEXES_DIR
            ext = 'faiss' if index_type == 'faiss' else 'hnsw'
            # Default to composite index as we use composite embeddings
            index_path = INDEXES_DIR / f"{self.set_code}_{self.finish}_composite.{ext}"

        if not index_path.exists():
            raise FileNotFoundError(
                f"Index not found: {index_path}\n"
                f"Please build the index first using:\n"
                f"  python -m src.ann.{index_type}_index --set {self.set_code} --finish {self.finish}"
            )

        self.ann_index.load(index_path)
        logger.info(f"Loaded index with {self.ann_index.get_num_vectors()} vectors")

        # Initialize database connection
        self.db = SessionLocal()

        # Initialize OCR disambiguator (lazy load to avoid import errors if pytesseract not installed)
        self._ocr_disambiguator = None

        logger.info("CardMatcher ready")

    def _get_ocr_disambiguator(self):
        """Lazy-load OCR disambiguator to avoid import errors if pytesseract not installed."""
        if self._ocr_disambiguator is None:
            try:
                from src.ocr.tesseract_service import TesseractOCRService
                from src.ocr.disambiguator import OCRDisambiguator
                ocr_service = TesseractOCRService()
                if ocr_service.is_available():
                    self._ocr_disambiguator = OCRDisambiguator(ocr_service, self.db)
                    logger.info("OCR disambiguator initialized")
                else:
                    logger.warning("Tesseract not available, OCR disambiguation disabled")
            except ImportError as e:
                logger.warning(f"OCR modules not available: {e}")
        return self._ocr_disambiguator

    def match_scanned(
        self,
        image_path: Union[str, Path],
        use_ocr: bool = False,  # Deprecated, kept for API compatibility
        debug: bool = False
    ) -> MatchResult:
        """
        Match a scanned card image

        Pipeline:
        1. Detect and warp card to canonical 363x504
        2. Compute CLIP composite embedding
        3. FAISS search for top K candidates
        4. pHash reranking
        5. ORB keypoint verification

        Args:
            image_path: Path to scanned image
            use_ocr: Deprecated, ignored (kept for API compatibility)
            debug: Enable debug output and visualizations

        Returns:
            MatchResult with top match and candidates
        """
        start_time = datetime.now()
        image_path = Path(image_path)

        logger.info(f"Matching scanned image: {image_path}")

        # Step 1: Detect and warp to canonical size
        logger.info("Step 1: Detecting and warping card...")
        image = cv2.imread(str(image_path))
        if image is None:
            raise ValueError(f"Could not load image: {image_path}")

        warped, mask, corners = detect_and_warp(image, debug=debug)

        # Store boundary corners for API response (convert numpy to list)
        boundary_corners = corners.tolist() if corners is not None else None

        # Convert BGR to RGB for embedder
        warped_rgb = cv2.cvtColor(warped, cv2.COLOR_BGR2RGB)

        # Step 2: Compute Composite Embedding
        logger.info("Step 2: Computing composite embedding...")
        
        # Import dependencies for composite embedding
        from src.embeddings.region_extractor import RegionExtractor
        from src.config import COMPOSITE_WEIGHT_FULL, COMPOSITE_WEIGHT_COLLECTOR, COMPOSITE_WEIGHT_NAME
        
        # Extract regions
        regions = RegionExtractor.extract_all_regions(warped_rgb)
        
        # Compute composite embedding
        embedding_result = self.embedder.get_composite_embedding(
            full_image=regions['full'],
            regions={'collector': regions['collector'], 'name': regions['name']},
            weights={
                'full': COMPOSITE_WEIGHT_FULL,
                'collector': COMPOSITE_WEIGHT_COLLECTOR,
                'name': COMPOSITE_WEIGHT_NAME
            }
        )
        query_embedding = embedding_result['composite']

        # Step 3: ANN search for top K candidates
        logger.info(f"Step 3: ANN search (K={self.top_k})...")
        ann_candidates = self.ann_index.query(query_embedding, top_k=self.top_k)

        if not ann_candidates:
            logger.warning("No candidates found from ANN search")
            return MatchResult(
                scanned_path=str(image_path),
                match_card_id=None,
                confidence=0.0,
                match_method='no_candidates',
                candidates=[],
                processing_time=(datetime.now() - start_time).total_seconds()
            )

        logger.info(f"Found {len(ann_candidates)} candidates from ANN search")

        # Step 4: Compute pHash for scanned image (3 variants)
        logger.info("Step 4: Computing pHash variants for scanned image...")
        from PIL import Image
        scanned_pil = Image.fromarray(warped_rgb)
        scanned_phash_variants = compute_phash_variants(scanned_pil)
        scanned_phash_ints = {
            variant: phash_to_int(phash_hex)
            for variant, phash_hex in scanned_phash_variants.items()
        }

        # Step 5: For top K, compute pHash distances and combine scores
        logger.info("Step 5: Computing combined scores for candidates...")
        candidates = []

        for card_id, ann_distance in ann_candidates:
            # Get card data with all variants
            card_data = get_card_full_data(self.db, card_id)
            if not card_data:
                logger.warning(f"Card {card_id} not found in database")
                continue

            card = card_data['card']
            phash_variants = card_data['phash_variants']
            embedding_record = card_data['embedding']

            # Get reference embedding
            ref_embedding = deserialize_embedding(embedding_record.embedding)

            # Compute embedding score
            # For cosine similarity (used by FAISS with IP), higher is better
            # ann_distance is already similarity score from ANN index
            embedding_score = ann_distance if self.index_type == 'faiss' else ann_distance
            embedding_distance = 1.0 - embedding_score  # Convert to distance for storage

            # Compute pHash distances for all 3 variants
            phash_distances = {}
            for variant_type in ['full', 'name', 'collector']:
                if variant_type in phash_variants:
                    ref_phash_hex = phash_variants[variant_type].phash
                    scanned_phash_int = scanned_phash_ints.get(variant_type, 0)
                    scanned_phash_hex = int_to_phash(scanned_phash_int)

                    distances = batch_hamming_distance(scanned_phash_hex, [ref_phash_hex])
                    phash_distances[variant_type] = int(distances[0])
                else:
                    phash_distances[variant_type] = 64  # Max distance if variant missing

            # Compute pHash score (weighted: 0.6*full + 0.3*name + 0.1*collector)
            max_dist = 64.0
            phash_full_score = 1.0 - min(phash_distances['full'] / max_dist, 1.0)
            phash_name_score = 1.0 - min(phash_distances['name'] / max_dist, 1.0)
            phash_collector_score = 1.0 - min(phash_distances['collector'] / max_dist, 1.0)

            phash_score = (
                0.6 * phash_full_score +
                0.3 * phash_name_score +
                0.1 * phash_collector_score
            )

            # Combined score: 85% embedding + 15% pHash
            combined_score = (
                self.weights['embedding'] * embedding_score +
                self.weights['phash'] * phash_score
            )

            # Create candidate
            candidate = MatchCandidate(
                card_id=card.id,
                card_name=card.name,
                set_code=card.set_code,
                collector_number=card.collector_number or '',
                finish=card.finish,
                scryfall_id=card.scryfall_id,
                image_path=card.image_path or '',
                embedding_score=embedding_score,
                phash_score=phash_score,
                combined_score=combined_score,
                embedding_distance=embedding_distance,
                phash_full_distance=phash_distances['full'],
                phash_name_distance=phash_distances['name'],
                phash_collector_distance=phash_distances['collector']
            )

            candidates.append(candidate)

        # Sort candidates by combined score (descending)
        candidates.sort(key=lambda c: c.combined_score, reverse=True)

        if not candidates:
            logger.warning("No valid candidates after scoring")
            return MatchResult(
                scanned_path=str(image_path),
                match_card_id=None,
                confidence=0.0,
                match_method='no_valid_candidates',
                candidates=[],
                processing_time=(datetime.now() - start_time).total_seconds()
            )
        
        # Step 5b: ORB Keypoint Verification (Reranking)
        # Research-backed: ORB provides strong visual confirmation for borderline cases
        # Only run on top candidates to save time
        logger.info("Step 5b: Running ORB keypoint verification on top candidates...")
        
        # Take top 5 candidates for ORB verification
        orb_candidates = candidates[:5]
        orb_applied = False
        
        for candidate in orb_candidates:
            if candidate.image_path and Path(candidate.image_path).exists():
                try:
                    # Compute ORB similarity
                    orb_score = compute_orb_similarity(warped_rgb, candidate.image_path)
                    
                    # Apply multiplicative scoring (reranking)
                    # Boost strong matches, penalize weak ones slightly
                    original_score = candidate.combined_score
                    
                    if orb_score > 0.5:
                        # Excellent match - up to 50% boost
                        multiplier = 1.0 + (0.5 * orb_score)
                        candidate.combined_score = min(1.0, original_score * multiplier)
                        logger.info(f"  ORB boost for {candidate.card_name}: {original_score:.3f} -> {candidate.combined_score:.3f} (score: {orb_score:.3f})")
                        orb_applied = True
                    elif orb_score > 0.3:
                        # Good match - up to 30% boost
                        multiplier = 1.0 + (0.3 * orb_score)
                        candidate.combined_score = min(1.0, original_score * multiplier)
                        logger.info(f"  ORB boost for {candidate.card_name}: {original_score:.3f} -> {candidate.combined_score:.3f} (score: {orb_score:.3f})")
                        orb_applied = True
                    else:
                        # Weak match - 5% penalty
                        multiplier = 0.95
                        candidate.combined_score = original_score * multiplier
                        logger.info(f"  ORB penalty for {candidate.card_name}: {original_score:.3f} -> {candidate.combined_score:.3f} (score: {orb_score:.3f})")
                        
                except Exception as e:
                    logger.warning(f"ORB verification failed for {candidate.card_name}: {e}")
            else:
                logger.warning(f"Image path not found for {candidate.card_name}: {candidate.image_path}")
        
        if orb_applied:
            # Re-sort candidates after ORB updates
            candidates.sort(key=lambda c: c.combined_score, reverse=True)
            logger.info(f"New top candidate after ORB: {candidates[0].card_name}")

        # Step 5c: Compute initial clarity to decide if OCR is needed
        clarity_score, is_ambiguous = compute_clarity(candidates)
        initial_confidence = candidates[0].combined_score

        # Step 5d: OCR Disambiguation (only if ambiguous or low confidence)
        ocr_result = None
        triangulated_match = None  # Direct match from triangulation (not in candidates)
        if is_ambiguous or initial_confidence < self.accept_threshold:
            logger.info("Step 5d: Running OCR disambiguation...")
            ocr_disambiguator = self._get_ocr_disambiguator()
            if ocr_disambiguator:
                try:
                    ocr_result = ocr_disambiguator.disambiguate(
                        warped_rgb,
                        candidates,
                        boost_weight=OCR_BOOST_WEIGHT
                    )
                    if ocr_result.success:
                        # Check if triangulation found a card not in candidates
                        if ocr_result.method == 'triangulation' and ocr_result.card_id:
                            # Check if this card is in our candidates
                            card_in_candidates = any(c.card_id == ocr_result.card_id for c in candidates)
                            if not card_in_candidates:
                                # Triangulation found a card NOT in candidates - use it directly
                                logger.info(f"Triangulation found card not in candidates: {ocr_result.card_id}")
                                triangulated_card = self.db.query(Card).filter(Card.id == ocr_result.card_id).first()
                                if triangulated_card:
                                    # Create a high-confidence candidate for the triangulated card
                                    triangulated_match = MatchCandidate(
                                        card_id=triangulated_card.id,
                                        card_name=triangulated_card.name,
                                        set_code=triangulated_card.set_code,
                                        collector_number=triangulated_card.collector_number or '',
                                        finish=triangulated_card.finish,
                                        scryfall_id=triangulated_card.scryfall_id,
                                        image_path=triangulated_card.image_path or '',
                                        embedding_score=0.95,  # High confidence from OCR
                                        phash_score=0.95,
                                        combined_score=0.95,  # Triangulation is high confidence
                                        embedding_distance=0.05,
                                        phash_full_distance=0,
                                        phash_name_distance=0,
                                        phash_collector_distance=0
                                    )
                                    # Insert as top candidate
                                    candidates.insert(0, triangulated_match)
                                    logger.info(f"Inserted triangulated card as top candidate: {triangulated_card.name} ({triangulated_card.set_code} #{triangulated_card.collector_number})")

                        # Re-sort candidates after OCR boost
                        candidates.sort(key=lambda c: c.combined_score, reverse=True)
                        logger.info(f"New top candidate after OCR: {candidates[0].card_name}")
                        # Recompute clarity after OCR
                        clarity_score, is_ambiguous = compute_clarity(candidates)
                except Exception as e:
                    logger.warning(f"OCR disambiguation failed: {e}")
            else:
                logger.debug("OCR disambiguator not available, skipping")

        # Step 6: Determine match confidence and clarity
        top_candidate = candidates[0]
        match_card_id = top_candidate.card_id
        confidence = top_candidate.combined_score

        # Determine match method based on confidence AND clarity
        # Triangulation is highest priority - it's an exact database match
        if triangulated_match is not None and top_candidate.card_id == triangulated_match.card_id:
            match_method = 'ocr_triangulation'
            logger.info(f"🎯 OCR Triangulation match: {top_candidate.card_name} ({top_candidate.set_code} #{top_candidate.collector_number})")
        elif confidence >= self.accept_threshold:
            if is_ambiguous:
                # High confidence but multiple similar candidates
                match_method = 'ambiguous_high'
                logger.info(f"⚠ Ambiguous high confidence: {top_candidate.card_name} ({confidence:.1%}), "
                           f"clarity={clarity_score:.3f}")
            else:
                # Clear winner
                match_method = 'auto_accept'
                logger.info(f"✓ Auto-accept: {top_candidate.card_name} ({confidence:.1%}), "
                           f"clarity={clarity_score:.3f}")
        else:
            match_method = 'manual_review'
            logger.info(f"? Manual review needed: {top_candidate.card_name} ({confidence:.1%})")

        # Create result
        processing_time = (datetime.now() - start_time).total_seconds()

        # Build OCR info for result (disambiguation only, not finish detection)
        ocr_used = ocr_result is not None and ocr_result.success
        ocr_title = ocr_result.title_text if ocr_result else None
        ocr_collector = ocr_result.collector_info.raw_text if ocr_result and ocr_result.collector_info else None
        ocr_set_code = ocr_result.collector_info.set_code if ocr_result and ocr_result.collector_info else None
        ocr_collector_number = ocr_result.collector_info.collector_number if ocr_result and ocr_result.collector_info else None
        ocr_title_confidence = ocr_result.title_confidence if ocr_result else None
        ocr_collector_confidence = ocr_result.collector_confidence if ocr_result else None

        result = MatchResult(
            scanned_path=str(image_path),
            match_card_id=match_card_id,
            confidence=confidence,
            match_method=match_method,
            candidates=candidates,
            processing_time=processing_time,
            clarity_score=clarity_score,
            is_ambiguous=is_ambiguous,
            boundary_corners=boundary_corners,
            ocr_used=ocr_used,
            ocr_title=ocr_title,
            ocr_collector=ocr_collector,
            ocr_set_code=ocr_set_code,
            ocr_collector_number=ocr_collector_number,
            ocr_title_confidence=ocr_title_confidence,
            ocr_collector_confidence=ocr_collector_confidence
        )

        logger.info(f"Matching completed in {processing_time:.2f}s")

        # Debug output if requested
        if debug:
            debug_output_path = Path(f"debug_match_{image_path.stem}.json")
            with open(debug_output_path, 'w') as f:
                json.dump(result.to_dict(), f, indent=2)
            logger.info(f"Debug output saved to {debug_output_path}")

        return result

    def match_batch(
        self,
        image_paths: List[Union[str, Path]],
        debug: bool = False
    ) -> List[MatchResult]:
        """
        Match multiple scanned images

        Args:
            image_paths: List of image paths
            debug: Enable debug output

        Returns:
            List of MatchResult objects
        """
        results = []

        for image_path in image_paths:
            try:
                result = self.match_scanned(image_path, debug=debug)
                results.append(result)
            except Exception as e:
                logger.error(f"Error matching {image_path}: {e}")
                import traceback
                traceback.print_exc()

                # Create error result
                results.append(MatchResult(
                    scanned_path=str(image_path),
                    match_card_id=None,
                    confidence=0.0,
                    match_method='error',
                    candidates=[],
                    processing_time=0.0,
                    debug_info={'error': str(e)}
                ))

        return results

    def close(self):
        """Close database connection"""
        self.db.close()


def main():
    """CLI entry point for matcher"""
    import argparse

    parser = argparse.ArgumentParser(description='Match scanned card images')
    parser.add_argument('image_path', help='Path to scanned image')
    parser.add_argument('--set', required=True, help='Set code (e.g., M21)')
    parser.add_argument('--finish', default='nonfoil', choices=['foil', 'nonfoil', 'etched'],
                        help='Finish type')
    parser.add_argument('--index-type', default='faiss', choices=['faiss', 'hnsw'],
                        help='ANN index type')
    parser.add_argument('--top-k', type=int, default=20,
                        help='Number of candidates from ANN search')
    parser.add_argument('--no-ocr', action='store_true',
                        help='Disable OCR disambiguation')
    parser.add_argument('--debug', action='store_true',
                        help='Enable debug output')
    parser.add_argument('--output', type=str, default=None,
                        help='Output JSON file path')

    args = parser.parse_args()

    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    # Create matcher
    matcher = CardMatcher(
        set_code=args.set,
        finish=args.finish,
        index_type=args.index_type,
        top_k=args.top_k
    )

    try:
        # Match image
        result = matcher.match_scanned(
            args.image_path,
            use_ocr=not args.no_ocr,
            debug=args.debug
        )

        # Print results
        print("\n" + "="*80)
        print("MATCH RESULT")
        print("="*80)
        print(f"Scanned: {result.scanned_path}")
        print(f"Match: {result.match_card_id}")
        print(f"Confidence: {result.confidence:.3f}")
        print(f"Method: {result.match_method}")
        print(f"Processing Time: {result.processing_time:.2f}s")

        if result.ocr_used:
            print(f"\nOCR Used: Yes")
            if result.ocr_title:
                print(f"OCR Title: {result.ocr_title}")
            if result.ocr_collector:
                print(f"OCR Collector: {result.ocr_collector}")
            if result.ocr_set_code:
                print(f"OCR Set Code: {result.ocr_set_code}")
            if result.ocr_collector_number:
                print(f"OCR Collector #: {result.ocr_collector_number}")

        print(f"\nTop 5 Candidates:")
        print("-" * 80)
        for i, candidate in enumerate(result.candidates[:5], 1):
            print(f"{i}. {candidate.card_name} ({candidate.set_code} #{candidate.collector_number})")
            print(f"   Score: {candidate.combined_score:.3f} (emb: {candidate.embedding_score:.3f}, "
                  f"phash: {candidate.phash_score:.3f})")

        # Save to file if requested
        if args.output:
            output_path = Path(args.output)
            with open(output_path, 'w') as f:
                f.write(result.to_json())
            print(f"\nResults saved to: {output_path}")

    finally:
        matcher.close()


if __name__ == '__main__':
    main()
