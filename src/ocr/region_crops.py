"""
OCR-specific region cropping utilities.

Provides crop functions for OCR text extraction:
- Title: 8-75% width x 4.5-10% height (card name)
- Collector: 0-50% width x 95.5-99% height (set code, collector number)

Note: These coordinates differ from the embedding regions in RegionExtractor,
which are optimized for CLIP embeddings rather than OCR text extraction.
"""

import numpy as np
import logging

from src.config import (
    CANONICAL_WIDTH,
    CANONICAL_HEIGHT,
    OCR_TITLE_X_START,
    OCR_TITLE_X_END,
    OCR_TITLE_Y_START,
    OCR_TITLE_Y_END,
    OCR_COLLECTOR_X_START,
    OCR_COLLECTOR_X_END,
    OCR_COLLECTOR_Y_START,
    OCR_COLLECTOR_Y_END,
)

logger = logging.getLogger(__name__)


def _crop_region(
    image: np.ndarray,
    x_start_pct: float,
    x_end_pct: float,
    y_start_pct: float,
    y_end_pct: float,
    region_name: str
) -> np.ndarray:
    """
    Common cropping logic with validation.

    Args:
        image: Input image as numpy array
        x_start_pct: Left boundary as percentage (0.0-1.0)
        x_end_pct: Right boundary as percentage (0.0-1.0)
        y_start_pct: Top boundary as percentage (0.0-1.0)
        y_end_pct: Bottom boundary as percentage (0.0-1.0)
        region_name: Name of region for error messages

    Returns:
        Cropped region as numpy array

    Raises:
        ValueError: If image is invalid or crop results in empty image
    """
    if image is None or image.size == 0:
        raise ValueError(f"Empty input image for {region_name} crop")

    h, w = image.shape[:2]

    # Warn if dimensions differ from canonical size
    if w != CANONICAL_WIDTH or h != CANONICAL_HEIGHT:
        logger.warning(
            f"Image dimensions {w}x{h} differ from canonical {CANONICAL_WIDTH}x{CANONICAL_HEIGHT}. "
            "OCR accuracy may be affected."
        )

    # Calculate crop coordinates
    x_start = max(0, int(w * x_start_pct))
    x_end = min(w, int(w * x_end_pct))
    y_start = max(0, int(h * y_start_pct))
    y_end = min(h, int(h * y_end_pct))

    # Validate bounds
    if x_end <= x_start or y_end <= y_start:
        raise ValueError(
            f"{region_name} crop bounds invalid: "
            f"x=[{x_start}:{x_end}], y=[{y_start}:{y_end}]"
        )

    # Extract crop
    crop = image[y_start:y_end, x_start:x_end]

    if crop.size == 0:
        raise ValueError(
            f"{region_name} crop resulted in empty image. "
            f"Bounds: [{y_start}:{y_end}, {x_start}:{x_end}]"
        )

    logger.debug(f"{region_name} crop: [{y_start}:{y_end}, {x_start}:{x_end}] -> {crop.shape}")
    return crop


def crop_title_for_ocr(image: np.ndarray) -> np.ndarray:
    """
    Crop the title region from a warped card image for OCR.

    Coordinates:
    - X: 8-75% of width
    - Y: 4.5-10% of height

    Args:
        image: Warped card image (should be 363x504 for best results)

    Returns:
        Cropped title region as numpy array

    Raises:
        ValueError: If image dimensions are invalid
    """
    return _crop_region(
        image,
        OCR_TITLE_X_START,
        OCR_TITLE_X_END,
        OCR_TITLE_Y_START,
        OCR_TITLE_Y_END,
        "Title"
    )


def crop_collector_for_ocr(image: np.ndarray) -> np.ndarray:
    """
    Crop the collector region from a warped card image for OCR.

    Coordinates:
    - X: 0-50% of width
    - Y: 95.5-99% of height

    The collector region contains:
    - Set code (e.g., "EOE")
    - Language code (e.g., "EN")
    - Artist name

    Args:
        image: Warped card image (should be 363x504 for best results)

    Returns:
        Cropped collector region as numpy array

    Raises:
        ValueError: If image dimensions are invalid
    """
    return _crop_region(
        image,
        OCR_COLLECTOR_X_START,
        OCR_COLLECTOR_X_END,
        OCR_COLLECTOR_Y_START,
        OCR_COLLECTOR_Y_END,
        "Collector"
    )
