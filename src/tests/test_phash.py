"""
src/tests/test_phash.py: Unit tests for pHash utilities
Following PRD.md Task 12 specifications

Tests:
- Identical images should have Hamming distance 0
- Similar images should have small Hamming distance
- Different images should have large Hamming distance
- Batch Hamming distance computation
"""

import pytest
import numpy as np
from PIL import Image
from pathlib import Path
import tempfile

from src.indexing.phash import (
    compute_phash,
    compute_dhash,
    hamming_distance,
    phash_to_int,
    int_to_phash,
    batch_hamming_distance,
    filter_by_hamming_distance,
    compute_phash_variants,
    combine_phash_scores
)


@pytest.fixture
def temp_image():
    """Create a temporary test image"""
    img = Image.new('RGB', (512, 512), color='red')
    with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as f:
        img.save(f.name)
        yield f.name
    # Cleanup
    Path(f.name).unlink(missing_ok=True)


@pytest.fixture
def identical_images():
    """Create two identical images"""
    img = Image.new('RGB', (512, 512), color='blue')

    with tempfile.NamedTemporaryFile(suffix='_1.png', delete=False) as f1:
        img.save(f1.name)
        path1 = f1.name

    with tempfile.NamedTemporaryFile(suffix='_2.png', delete=False) as f2:
        img.save(f2.name)
        path2 = f2.name

    yield path1, path2

    # Cleanup
    Path(path1).unlink(missing_ok=True)
    Path(path2).unlink(missing_ok=True)


@pytest.fixture
def similar_images():
    """Create two similar images (same color, slightly different pattern)"""
    img1 = Image.new('RGB', (512, 512), color='green')
    img2 = Image.new('RGB', (512, 512), color='green')

    # Add a small white square to img2 to make it slightly different
    from PIL import ImageDraw
    draw = ImageDraw.Draw(img2)
    draw.rectangle([250, 250, 260, 260], fill='white')

    with tempfile.NamedTemporaryFile(suffix='_1.png', delete=False) as f1:
        img1.save(f1.name)
        path1 = f1.name

    with tempfile.NamedTemporaryFile(suffix='_2.png', delete=False) as f2:
        img2.save(f2.name)
        path2 = f2.name

    yield path1, path2

    # Cleanup
    Path(path1).unlink(missing_ok=True)
    Path(path2).unlink(missing_ok=True)


@pytest.fixture
def different_images():
    """Create two completely different images"""
    img1 = Image.new('RGB', (512, 512), color='red')
    img2 = Image.new('RGB', (512, 512), color='blue')

    with tempfile.NamedTemporaryFile(suffix='_1.png', delete=False) as f1:
        img1.save(f1.name)
        path1 = f1.name

    with tempfile.NamedTemporaryFile(suffix='_2.png', delete=False) as f2:
        img2.save(f2.name)
        path2 = f2.name

    yield path1, path2

    # Cleanup
    Path(path1).unlink(missing_ok=True)
    Path(path2).unlink(missing_ok=True)


class TestPHashBasics:
    """Test basic pHash functionality"""

    def test_compute_phash_returns_hex_string(self, temp_image):
        """Test that compute_phash returns a hex string"""
        phash = compute_phash(temp_image)
        assert isinstance(phash, str)
        # Should be hex string (16 characters for 64-bit hash)
        assert len(phash) == 16
        # Should be valid hex
        int(phash, 16)

    def test_compute_dhash_returns_hex_string(self, temp_image):
        """Test that compute_dhash returns a hex string"""
        dhash = compute_dhash(temp_image)
        assert isinstance(dhash, str)
        assert len(dhash) == 16
        int(dhash, 16)

    def test_identical_images_have_zero_distance(self, identical_images):
        """Test that identical images have Hamming distance 0"""
        path1, path2 = identical_images

        phash1 = compute_phash(path1)
        phash2 = compute_phash(path2)

        distance = hamming_distance(phash1, phash2)
        assert distance == 0, f"Identical images should have distance 0, got {distance}"

    def test_similar_images_have_small_distance(self, similar_images):
        """Test that similar images have small Hamming distance"""
        path1, path2 = similar_images

        phash1 = compute_phash(path1)
        phash2 = compute_phash(path2)

        distance = hamming_distance(phash1, phash2)
        # Similar images should have distance < 10 (PRD threshold)
        assert distance < 10, f"Similar images should have distance < 10, got {distance}"

    def test_different_images_have_large_distance(self, different_images):
        """Test that different images have large Hamming distance"""
        path1, path2 = different_images

        phash1 = compute_phash(path1)
        phash2 = compute_phash(path2)

        distance = hamming_distance(phash1, phash2)
        # Different images should have distance > 10
        assert distance > 10, f"Different images should have distance > 10, got {distance}"


class TestPHashConversions:
    """Test pHash conversion functions"""

    def test_phash_to_int_conversion(self):
        """Test converting hex hash to integer"""
        hex_hash = "a1b2c3d4e5f60718"
        int_hash = phash_to_int(hex_hash)
        assert isinstance(int_hash, int)
        assert int_hash > 0

    def test_int_to_phash_conversion(self):
        """Test converting integer hash back to hex"""
        int_hash = 0xa1b2c3d4e5f60718
        hex_hash = int_to_phash(int_hash)
        assert isinstance(hex_hash, str)
        assert hex_hash == "a1b2c3d4e5f60718"

    def test_round_trip_conversion(self):
        """Test round-trip conversion hex -> int -> hex"""
        original = "1234567890abcdef"
        int_val = phash_to_int(original)
        result = int_to_phash(int_val)
        assert result == original


class TestBatchHammingDistance:
    """Test batch Hamming distance computation"""

    def test_batch_hamming_distance_empty_list(self):
        """Test batch Hamming distance with empty candidate list"""
        query_hash = "1234567890abcdef"
        candidates = []

        distances = batch_hamming_distance(query_hash, candidates)
        assert len(distances) == 0

    def test_batch_hamming_distance_single_candidate(self):
        """Test batch Hamming distance with single candidate"""
        query_hash = "1234567890abcdef"
        candidates = ["1234567890abcdef"]  # Identical

        distances = batch_hamming_distance(query_hash, candidates)
        assert len(distances) == 1
        assert distances[0] == 0

    def test_batch_hamming_distance_multiple_candidates(self):
        """Test batch Hamming distance with multiple candidates"""
        query_hash = "1234567890abcdef"
        candidates = [
            "1234567890abcdef",  # Identical (distance 0)
            "1234567890abcdff",  # 1 bit different
            "0000000000000000",  # Many bits different
        ]

        distances = batch_hamming_distance(query_hash, candidates)
        assert len(distances) == 3
        assert distances[0] == 0
        assert distances[1] < distances[2]

    def test_filter_by_hamming_distance(self):
        """Test filtering candidates by Hamming distance threshold"""
        query_hash = "1234567890abcdef"
        candidates = [
            "1234567890abcdef",  # Distance 0
            "1234567890abcdff",  # Distance 1
            "0000000000000000",  # Distance > 10
        ]

        # Filter with threshold 5
        filtered = filter_by_hamming_distance(query_hash, candidates, max_hamming=5)

        # Should return first 2 candidates
        assert len(filtered) == 2
        # Should be sorted by distance
        assert filtered[0][1] <= filtered[1][1]

    def test_filter_by_hamming_distance_with_top_n(self):
        """Test filtering with top_n limit"""
        query_hash = "1234567890abcdef"
        candidates = [
            "1234567890abcdef",  # Distance 0
            "1234567890abcdff",  # Distance 1
            "1234567890abcdee",  # Distance 2
        ]

        # Filter with threshold 10 but limit to top 2
        filtered = filter_by_hamming_distance(query_hash, candidates, max_hamming=10, top_n=2)

        assert len(filtered) <= 2


class TestPHashVariants:
    """Test pHash variant computation (full, name, collector)"""

    def test_compute_phash_variants(self, temp_image):
        """Test computing 3 pHash variants"""
        variants = compute_phash_variants(temp_image)

        # Should return dict with 3 keys
        assert isinstance(variants, dict)
        assert 'full' in variants
        assert 'name' in variants
        assert 'collector' in variants

        # All should be hex strings
        for variant_type, phash in variants.items():
            assert isinstance(phash, str)
            assert len(phash) == 16

    def test_variants_are_different(self):
        """Test that variants for different regions are different"""
        # Create an image with distinct regions
        img = Image.new('RGB', (512, 512))
        from PIL import ImageDraw
        draw = ImageDraw.Draw(img)

        # Top region (name) - red
        draw.rectangle([0, 0, 512, 77], fill='red')  # 15% of 512 = 77

        # Bottom-left region (collector) - blue
        draw.rectangle([0, 451, 128, 512], fill='blue')  # 12% height, 25% width

        # Rest - green
        draw.rectangle([128, 451, 512, 512], fill='green')
        draw.rectangle([0, 77, 512, 451], fill='green')

        variants = compute_phash_variants(img)

        # Variants should be different
        assert variants['full'] != variants['name']
        assert variants['full'] != variants['collector']
        assert variants['name'] != variants['collector']

    def test_combine_phash_scores(self):
        """Test combining pHash scores with weights"""
        # Perfect matches (all distances 0)
        score = combine_phash_scores(0, 0, 0)
        assert score == 1.0

        # Maximum distances (all 64)
        score = combine_phash_scores(64, 64, 64)
        assert score == 0.0

        # Mixed distances
        score = combine_phash_scores(10, 5, 2)
        assert 0.0 < score < 1.0

        # Test with custom weights
        score = combine_phash_scores(10, 5, 2, weights=(0.5, 0.3, 0.2))
        assert 0.0 < score < 1.0


class TestPHashReproducibility:
    """Test that pHash is reproducible"""

    def test_same_image_same_hash(self, temp_image):
        """Test that computing hash twice gives same result"""
        phash1 = compute_phash(temp_image)
        phash2 = compute_phash(temp_image)
        assert phash1 == phash2

    def test_pil_image_vs_path(self, temp_image):
        """Test that hash is same whether computed from path or PIL Image"""
        phash_from_path = compute_phash(temp_image)

        img = Image.open(temp_image)
        phash_from_pil = compute_phash(img)

        assert phash_from_path == phash_from_pil


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
