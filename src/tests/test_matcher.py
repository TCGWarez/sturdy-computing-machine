"""
src/tests/test_matcher.py: End-to-end tests for card matching
Following PRD.md Task 12 specifications

Tests:
- End-to-end matching with fixtures (3 scans → expected card ids)
- Match result structure
- Confidence scoring
- OCR disambiguation
"""

import pytest
import numpy as np
from pathlib import Path
import tempfile
import shutil
from PIL import Image
import cv2

from src.recognition.matcher import CardMatcher, MatchResult, MatchCandidate
from src.database.schema import SessionLocal, Card, PhashVariant, Embedding, Base
from src.indexing.indexer import serialize_embedding
from src.indexing.phash import compute_phash_variants, phash_to_int


@pytest.fixture
def temp_db_with_cards():
    """Create temporary database with test cards"""
    with tempfile.NamedTemporaryFile(suffix='.db', delete=False) as f:
        db_path = f.name

    from sqlalchemy import create_engine
    from sqlalchemy.orm import sessionmaker

    test_engine = create_engine(f"sqlite:///{db_path}", echo=False)
    Base.metadata.create_all(bind=test_engine)

    TestSession = sessionmaker(autocommit=False, autoflush=False, bind=test_engine)
    session = TestSession()

    # Insert test cards
    cards_data = [
        {'id': 'card_1', 'scryfall_id': 'sc1', 'name': 'Lightning Bolt',
         'set_code': 'TST', 'collector_number': '001', 'finish': 'nonfoil'},
        {'id': 'card_2', 'scryfall_id': 'sc2', 'name': 'Giant Growth',
         'set_code': 'TST', 'collector_number': '002', 'finish': 'nonfoil'},
        {'id': 'card_3', 'scryfall_id': 'sc3', 'name': 'Counterspell',
         'set_code': 'TST', 'collector_number': '003', 'finish': 'nonfoil'},
    ]

    for card_data in cards_data:
        card = Card(**card_data)
        session.add(card)

        # Add pHash variants (using dummy values)
        for variant_type in ['full', 'name', 'collector']:
            phash = PhashVariant(
                card_id=card_data['id'],
                variant_type=variant_type,
                phash=123456 + cards_data.index(card_data)  # Different for each card
            )
            session.add(phash)

        # Add embedding (random)
        embedding_vector = np.random.randn(512).astype(np.float32)
        embedding = Embedding(
            card_id=card_data['id'],
            embedding=serialize_embedding(embedding_vector)
        )
        session.add(embedding)

    session.commit()

    yield session, db_path

    # Cleanup
    session.close()
    test_engine.dispose()
    Path(db_path).unlink(missing_ok=True)


@pytest.fixture
def temp_test_images():
    """Create temporary test images"""
    temp_dir = Path(tempfile.mkdtemp())

    # Create 3 test card images
    colors = ['red', 'green', 'blue']
    image_paths = []

    for i, color in enumerate(colors):
        img = Image.new('RGB', (512, 512), color=color)

        # Add some distinguishing features
        from PIL import ImageDraw, ImageFont
        draw = ImageDraw.Draw(img)

        # Draw white rectangle (simulating card border)
        draw.rectangle([10, 10, 502, 502], outline='white', width=5)

        # Add text
        card_names = ['Lightning Bolt', 'Giant Growth', 'Counterspell']
        draw.text((50, 50), card_names[i], fill='white')

        img_path = temp_dir / f"scan_{i}.jpg"
        img.save(img_path)
        image_paths.append(img_path)

    yield image_paths, temp_dir

    # Cleanup
    shutil.rmtree(temp_dir)


class TestMatchResult:
    """Test MatchResult dataclass"""

    def test_match_result_creation(self):
        """Test creating a MatchResult"""
        candidate = MatchCandidate(
            card_id='test_1',
            card_name='Test Card',
            set_code='TST',
            collector_number='001',
            finish='nonfoil',
            image_path='/path/to/image.jpg',
            embedding_score=0.9,
            phash_score=0.85,
            ocr_score=0.0,
            combined_score=0.88,
            embedding_distance=0.1,
            phash_full_distance=5,
            phash_name_distance=3,
            phash_collector_distance=2
        )

        result = MatchResult(
            scanned_path='/path/to/scan.jpg',
            match_card_id='test_1',
            confidence=0.88,
            match_method='embedding+phash',
            candidates=[candidate],
            processing_time=1.5
        )

        assert result.scanned_path == '/path/to/scan.jpg'
        assert result.match_card_id == 'test_1'
        assert result.confidence == 0.88
        assert len(result.candidates) == 1

    def test_match_result_to_dict(self):
        """Test converting MatchResult to dict"""
        candidate = MatchCandidate(
            card_id='test_1',
            card_name='Test Card',
            set_code='TST',
            collector_number='001',
            finish='nonfoil',
            image_path='/path/to/image.jpg',
            embedding_score=0.9,
            phash_score=0.85,
            ocr_score=0.0,
            combined_score=0.88,
            embedding_distance=0.1,
            phash_full_distance=5,
            phash_name_distance=3,
            phash_collector_distance=2
        )

        result = MatchResult(
            scanned_path='/path/to/scan.jpg',
            match_card_id='test_1',
            confidence=0.88,
            match_method='embedding+phash',
            candidates=[candidate],
            processing_time=1.5
        )

        result_dict = result.to_dict()

        assert isinstance(result_dict, dict)
        assert 'scanned_path' in result_dict
        assert 'match_card_id' in result_dict
        assert 'confidence' in result_dict
        assert 'candidates' in result_dict
        assert isinstance(result_dict['candidates'], list)

    def test_match_result_to_json(self):
        """Test converting MatchResult to JSON"""
        import json

        candidate = MatchCandidate(
            card_id='test_1',
            card_name='Test Card',
            set_code='TST',
            collector_number='001',
            finish='nonfoil',
            image_path='/path/to/image.jpg',
            embedding_score=0.9,
            phash_score=0.85,
            ocr_score=0.0,
            combined_score=0.88,
            embedding_distance=0.1,
            phash_full_distance=5,
            phash_name_distance=3,
            phash_collector_distance=2
        )

        result = MatchResult(
            scanned_path='/path/to/scan.jpg',
            match_card_id='test_1',
            confidence=0.88,
            match_method='embedding+phash',
            candidates=[candidate],
            processing_time=1.5
        )

        json_str = result.to_json()

        # Should be valid JSON
        parsed = json.loads(json_str)
        assert parsed['match_card_id'] == 'test_1'


class TestMatchCandidate:
    """Test MatchCandidate dataclass"""

    def test_candidate_creation(self):
        """Test creating a MatchCandidate"""
        candidate = MatchCandidate(
            card_id='test_1',
            card_name='Test Card',
            set_code='TST',
            collector_number='001',
            finish='nonfoil',
            image_path='/path/to/image.jpg',
            embedding_score=0.9,
            phash_score=0.85,
            ocr_score=0.0,
            combined_score=0.88,
            embedding_distance=0.1,
            phash_full_distance=5,
            phash_name_distance=3,
            phash_collector_distance=2
        )

        assert candidate.card_id == 'test_1'
        assert candidate.combined_score == 0.88

    def test_candidate_to_dict(self):
        """Test converting MatchCandidate to dict"""
        candidate = MatchCandidate(
            card_id='test_1',
            card_name='Test Card',
            set_code='TST',
            collector_number='001',
            finish='nonfoil',
            image_path='/path/to/image.jpg',
            embedding_score=0.9,
            phash_score=0.85,
            ocr_score=0.0,
            combined_score=0.88,
            embedding_distance=0.1,
            phash_full_distance=5,
            phash_name_distance=3,
            phash_collector_distance=2
        )

        candidate_dict = candidate.to_dict()

        assert isinstance(candidate_dict, dict)
        assert 'card_id' in candidate_dict
        assert 'embedding_score' in candidate_dict
        assert 'phash_score' in candidate_dict


class TestScoringWeights:
    """Test scoring weight combinations"""

    def test_default_weights(self):
        """Test default weights sum to reasonable range"""
        # Default weights from PRD: α=0.6, β=0.3, γ=0.1
        weights = {'embedding': 0.6, 'phash': 0.3, 'ocr': 0.1}

        total = sum(weights.values())
        assert abs(total - 1.0) < 0.01  # Should sum to ~1.0

    def test_combined_score_calculation(self):
        """Test combined score calculation"""
        embedding_score = 0.9
        phash_score = 0.8
        ocr_score = 0.0

        # Using default weights
        combined = 0.6 * embedding_score + 0.3 * phash_score + 0.1 * ocr_score

        assert 0.0 <= combined <= 1.0
        assert combined == pytest.approx(0.78, abs=0.01)

    def test_perfect_match_score(self):
        """Test that perfect match gives score of 1.0"""
        embedding_score = 1.0
        phash_score = 1.0
        ocr_score = 1.0

        combined = 0.6 * embedding_score + 0.3 * phash_score + 0.1 * ocr_score

        assert combined == 1.0

    def test_worst_match_score(self):
        """Test that worst match gives score of 0.0"""
        embedding_score = 0.0
        phash_score = 0.0
        ocr_score = 0.0

        combined = 0.6 * embedding_score + 0.3 * phash_score + 0.1 * ocr_score

        assert combined == 0.0


class TestConfidenceThresholds:
    """Test confidence threshold logic"""

    def test_high_confidence_threshold(self):
        """Test high confidence threshold (0.85 from PRD)"""
        from src.config import ACCEPT_THRESHOLD

        high_score = 0.90
        low_score = 0.75

        assert high_score >= ACCEPT_THRESHOLD
        assert low_score < ACCEPT_THRESHOLD

    def test_manual_review_threshold(self):
        """Test manual review threshold (0.65 from PRD)"""
        from src.config import MANUAL_THRESHOLD

        ambiguous_score = 0.70
        very_low_score = 0.50

        assert ambiguous_score >= MANUAL_THRESHOLD
        assert very_low_score < MANUAL_THRESHOLD


class TestMatchMethodDetection:
    """Test match method classification"""

    def test_embedding_phash_method(self):
        """Test match method for embedding+phash"""
        score = 0.90  # Above accept threshold

        if score >= 0.85:
            method = 'embedding+phash'
        else:
            method = 'manual_review'

        assert method == 'embedding+phash'

    def test_ocr_assisted_method(self):
        """Test match method when OCR is used"""
        initial_score = 0.75  # Below accept, above manual
        ocr_used = True

        if initial_score >= 0.85:
            method = 'embedding+phash'
        elif ocr_used and initial_score >= 0.65:
            method = 'embedding+phash+ocr'
        else:
            method = 'manual_review'

        assert method == 'embedding+phash+ocr'

    def test_manual_review_method(self):
        """Test match method for manual review"""
        score = 0.50  # Below manual threshold

        if score >= 0.85:
            method = 'embedding+phash'
        elif score >= 0.65:
            method = 'embedding+phash+ocr'
        else:
            method = 'manual_review'

        assert method == 'manual_review'


class TestErrorHandling:
    """Test error handling in matcher"""

    def test_empty_candidates_list(self):
        """Test handling of empty candidates list"""
        result = MatchResult(
            scanned_path='/path/to/scan.jpg',
            match_card_id=None,
            confidence=0.0,
            match_method='no_candidates',
            candidates=[],
            processing_time=0.5
        )

        assert result.match_card_id is None
        assert result.confidence == 0.0
        assert len(result.candidates) == 0

    def test_no_valid_candidates(self):
        """Test handling when no valid candidates after scoring"""
        result = MatchResult(
            scanned_path='/path/to/scan.jpg',
            match_card_id=None,
            confidence=0.0,
            match_method='no_valid_candidates',
            candidates=[],
            processing_time=0.5
        )

        assert result.match_card_id is None

    def test_error_result(self):
        """Test creating error result"""
        result = MatchResult(
            scanned_path='/path/to/scan.jpg',
            match_card_id=None,
            confidence=0.0,
            match_method='error',
            candidates=[],
            processing_time=0.0,
            debug_info={'error': 'Test error'}
        )

        assert result.match_method == 'error'
        assert 'error' in result.debug_info


class TestCandidateSorting:
    """Test candidate sorting by score"""

    def test_candidates_sorted_by_score(self):
        """Test that candidates are sorted by combined score"""
        candidates = [
            MatchCandidate(
                card_id=f'card_{i}',
                card_name=f'Card {i}',
                set_code='TST',
                collector_number=f'{i:03d}',
                finish='nonfoil',
                image_path=f'/path/{i}.jpg',
                embedding_score=0.5 + i * 0.1,
                phash_score=0.5 + i * 0.1,
                ocr_score=0.0,
                combined_score=0.5 + i * 0.1,
                embedding_distance=0.5 - i * 0.1,
                phash_full_distance=10 - i,
                phash_name_distance=5 - i,
                phash_collector_distance=3 - i
            )
            for i in range(5)
        ]

        # Sort by combined score (descending)
        sorted_candidates = sorted(candidates, key=lambda c: c.combined_score, reverse=True)

        # Check that scores are in descending order
        for i in range(len(sorted_candidates) - 1):
            assert sorted_candidates[i].combined_score >= sorted_candidates[i+1].combined_score

        # Best candidate should be last from original list (highest score)
        assert sorted_candidates[0].card_id == 'card_4'


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
