"""
src/tests/test_detector.py: Unit tests for card detection and warping

Tests:
- Synthetic warp tests with known transforms
- Card detection on rotated images
- Output dimensions match canonical size
- Corner detection accuracy
"""

import pytest
import numpy as np
import cv2
from pathlib import Path
import tempfile

from src.detection.card_detector import (
    detect_and_warp,
    detect_card_from_file,
    _order_corners,
    _perspective_warp,
    CANONICAL_SIZE
)


@pytest.fixture
def synthetic_card_image():
    """Create a synthetic card image (white rectangle on black background)"""
    # Create black background
    img = np.zeros((800, 600, 3), dtype=np.uint8)

    # Draw white rectangle (simulating a card)
    # Card is 400x280 (roughly MTG card aspect ratio ~1.4:1)
    cv2.rectangle(img, (100, 200), (500, 600), (255, 255, 255), -1)

    return img


@pytest.fixture
def rotated_card_image():
    """Create a synthetic card image rotated 45 degrees"""
    # Create larger canvas to fit rotated card
    img = np.zeros((1000, 1000, 3), dtype=np.uint8)

    # Draw white rectangle at center
    card_width, card_height = 400, 280
    center_x, center_y = 500, 500

    # Define card corners before rotation
    corners = np.array([
        [-card_width//2, -card_height//2],
        [card_width//2, -card_height//2],
        [card_width//2, card_height//2],
        [-card_width//2, card_height//2]
    ], dtype=np.float32)

    # Rotate 45 degrees
    angle = 45
    theta = np.radians(angle)
    rotation_matrix = np.array([
        [np.cos(theta), -np.sin(theta)],
        [np.sin(theta), np.cos(theta)]
    ])

    rotated_corners = corners @ rotation_matrix.T
    rotated_corners += [center_x, center_y]

    # Draw rotated rectangle
    cv2.fillPoly(img, [rotated_corners.astype(np.int32)], (255, 255, 255))

    return img, rotated_corners


@pytest.fixture
def temp_card_image(synthetic_card_image):
    """Save synthetic card image to temporary file"""
    with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as f:
        cv2.imwrite(f.name, synthetic_card_image)
        yield f.name
    Path(f.name).unlink(missing_ok=True)


class TestCardDetection:
    """Test card detection functionality"""

    def test_detect_and_warp_returns_correct_size(self, synthetic_card_image):
        """Test that warped output has canonical size"""
        warped, mask, corners = detect_and_warp(synthetic_card_image)

        # Check warped image size
        assert warped.shape[0] == CANONICAL_SIZE
        assert warped.shape[1] == CANONICAL_SIZE

    def test_detect_and_warp_with_custom_size(self, synthetic_card_image):
        """Test detection with custom canonical size"""
        custom_size = 256
        warped, mask, corners = detect_and_warp(synthetic_card_image, canonical_size=custom_size)

        assert warped.shape[0] == custom_size
        assert warped.shape[1] == custom_size

    def test_detect_and_warp_returns_4_corners(self, synthetic_card_image):
        """Test that detection returns 4 corner points"""
        warped, mask, corners = detect_and_warp(synthetic_card_image)

        assert corners is not None
        assert corners.shape == (4, 2)

    def test_detect_and_warp_on_rotated_card(self):
        """Test detection on rotated card"""
        img, true_corners = rotated_card_image.__wrapped__()

        warped, mask, corners = detect_and_warp(img)

        # Should still return canonical size
        assert warped.shape[0] == CANONICAL_SIZE
        assert warped.shape[1] == CANONICAL_SIZE

        # Should detect 4 corners
        assert corners is not None
        assert corners.shape == (4, 2)

    def test_detect_card_from_file(self, temp_card_image):
        """Test convenience function for file-based detection"""
        warped = detect_card_from_file(temp_card_image)

        assert warped.shape[0] == CANONICAL_SIZE
        assert warped.shape[1] == CANONICAL_SIZE

    def test_detect_card_from_file_with_output(self, temp_card_image):
        """Test saving warped image to file"""
        with tempfile.NamedTemporaryFile(suffix='_warped.png', delete=False) as f:
            output_path = Path(f.name)

        try:
            warped = detect_card_from_file(temp_card_image, output_path=output_path)

            # Check output file exists
            assert output_path.exists()

            # Check output file has correct dimensions
            loaded = cv2.imread(str(output_path))
            assert loaded.shape[0] == CANONICAL_SIZE
            assert loaded.shape[1] == CANONICAL_SIZE

        finally:
            output_path.unlink(missing_ok=True)


class TestDetectionFailureModes:
    """Test detection behavior on edge cases"""

    def test_empty_image_raises_error(self):
        """Test that empty image raises ValueError"""
        empty_image = np.array([])

        with pytest.raises(ValueError, match="Empty input image"):
            detect_and_warp(empty_image)

    def test_pure_black_image_uses_full_image(self):
        """Test that pure black image (no edges) uses full image as fallback"""
        black_image = np.zeros((600, 400, 3), dtype=np.uint8)

        warped, mask, corners = detect_and_warp(black_image)

        # Should still return valid output
        assert warped.shape[0] == CANONICAL_SIZE
        assert warped.shape[1] == CANONICAL_SIZE

        # Corners should be image boundaries
        h, w = black_image.shape[:2]
        expected_corners = np.array([
            [0, 0],
            [w-1, 0],
            [w-1, h-1],
            [0, h-1]
        ], dtype=np.float32)

        # Check corners are approximately at image boundaries
        assert np.allclose(corners, expected_corners, atol=1.0)

    def test_grayscale_image_works(self):
        """Test that grayscale images work"""
        # Create grayscale card
        gray_img = np.zeros((800, 600), dtype=np.uint8)
        cv2.rectangle(gray_img, (100, 200), (500, 600), 255, -1)

        warped, mask, corners = detect_and_warp(gray_img)

        assert warped.shape[0] == CANONICAL_SIZE
        assert warped.shape[1] == CANONICAL_SIZE


class TestCornerOrdering:
    """Test corner ordering functionality"""

    def test_order_corners_returns_tl_tr_br_bl(self):
        """Test that corners are ordered correctly: TL, TR, BR, BL"""
        # Create corners in random order
        corners = np.array([
            [400, 400],  # BR
            [100, 100],  # TL
            [400, 100],  # TR
            [100, 400],  # BL
        ], dtype=np.float32)

        ordered = _order_corners(corners)

        # Check order: TL, TR, BR, BL
        # TL should have minimum sum of coordinates
        # TR should have maximum x, minimum y
        # BR should have maximum sum
        # BL should have minimum x, maximum y

        tl, tr, br, bl = ordered

        # TL should be top-left
        assert tl[0] < tr[0]  # TL.x < TR.x
        assert tl[1] < bl[1]  # TL.y < BL.y

        # TR should be top-right
        assert tr[0] > tl[0]  # TR.x > TL.x
        assert tr[1] < br[1]  # TR.y < BR.y

        # BR should be bottom-right
        assert br[0] > bl[0]  # BR.x > BL.x
        assert br[1] > tr[1]  # BR.y > TR.y

        # BL should be bottom-left
        assert bl[0] < br[0]  # BL.x < BR.x
        assert bl[1] > tl[1]  # BL.y > TL.y


class TestPerspectiveWarp:
    """Test perspective transformation"""

    def test_perspective_warp_output_size(self):
        """Test that perspective warp produces correct output size"""
        img = np.ones((600, 400, 3), dtype=np.uint8) * 128

        # Define source corners
        corners = np.array([
            [50, 50],
            [350, 50],
            [350, 550],
            [50, 550]
        ], dtype=np.float32)

        warped = _perspective_warp(img, corners, size=CANONICAL_SIZE)

        assert warped.shape[0] == CANONICAL_SIZE
        assert warped.shape[1] == CANONICAL_SIZE
        assert warped.shape[2] == 3

    def test_perspective_warp_preserves_channels(self):
        """Test that warp preserves color channels"""
        # Create colored image
        img = np.ones((600, 400, 3), dtype=np.uint8)
        img[:, :, 0] = 255  # Red channel
        img[:, :, 1] = 128  # Green channel
        img[:, :, 2] = 64   # Blue channel

        corners = np.array([
            [0, 0],
            [399, 0],
            [399, 599],
            [0, 599]
        ], dtype=np.float32)

        warped = _perspective_warp(img, corners, size=256)

        # Check that warped image still has color channels
        assert warped.shape[2] == 3
        # Colors should be approximately preserved
        assert np.mean(warped[:, :, 0]) > np.mean(warped[:, :, 1])
        assert np.mean(warped[:, :, 1]) > np.mean(warped[:, :, 2])


class TestDebugMode:
    """Test debug visualization functionality"""

    def test_debug_mode_creates_visualization(self, synthetic_card_image, tmp_path):
        """Test that debug mode creates visualization file"""
        import os

        # Change to temp directory
        old_cwd = os.getcwd()
        os.chdir(tmp_path)

        try:
            warped, mask, corners = detect_and_warp(synthetic_card_image, debug=True)

            # Check that debug file was created
            debug_file = Path("debug_card_detection.jpg")
            assert debug_file.exists()

        finally:
            os.chdir(old_cwd)

    def test_debug_disabled_no_file(self, synthetic_card_image, tmp_path):
        """Test that debug=False doesn't create visualization"""
        import os

        old_cwd = os.getcwd()
        os.chdir(tmp_path)

        try:
            warped, mask, corners = detect_and_warp(synthetic_card_image, debug=False)

            # Check that debug file was NOT created
            debug_file = Path("debug_card_detection.jpg")
            assert not debug_file.exists()

        finally:
            os.chdir(old_cwd)


class TestSyntheticTransforms:
    """Test with known synthetic transforms"""

    def test_known_rotation_90_degrees(self):
        """Test detection on card rotated exactly 90 degrees"""
        # Create card image
        img = np.zeros((800, 600, 3), dtype=np.uint8)

        # Original card dimensions
        card_w, card_h = 280, 400

        # Rotate 90 degrees by swapping dimensions and flipping
        # For 90-degree rotation, we create rotated shape directly
        center_x, center_y = 300, 400
        offset_x, offset_y = card_h//2, card_w//2

        corners_rotated = np.array([
            [center_x - offset_x, center_y - offset_y],
            [center_x + offset_x, center_y - offset_y],
            [center_x + offset_x, center_y + offset_y],
            [center_x - offset_x, center_y + offset_y]
        ], dtype=np.int32)

        cv2.fillPoly(img, [corners_rotated], (255, 255, 255))

        warped, mask, corners = detect_and_warp(img)

        # Should produce canonical size output
        assert warped.shape[0] == CANONICAL_SIZE
        assert warped.shape[1] == CANONICAL_SIZE

    def test_known_scaling(self):
        """Test detection on scaled card"""
        # Create small card
        img = np.zeros((400, 300, 3), dtype=np.uint8)
        cv2.rectangle(img, (50, 50), (250, 350), (255, 255, 255), -1)

        warped, mask, corners = detect_and_warp(img)

        # Should scale to canonical size
        assert warped.shape[0] == CANONICAL_SIZE
        assert warped.shape[1] == CANONICAL_SIZE


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
