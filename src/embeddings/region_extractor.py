"""
Region extraction utilities for composite embeddings

Extracts specific regions from MTG card images:
- Name region (top 15% for variant discrimination)
- Collector region (bottom 10% for set code + collector number)

CRITICAL: Both indexed and scanned images MUST be normalized to 363x504
before region extraction to ensure consistent CLIP embeddings.
"""

import cv2
import numpy as np
from PIL import Image
from typing import Dict, Union
import logging

from src.config import CANONICAL_WIDTH, CANONICAL_HEIGHT

logger = logging.getLogger(__name__)


class RegionExtractor:
    """
    Extracts specific regions from card images for composite embeddings
    """

    @staticmethod
    def crop_name_region(
        image: np.ndarray,
        top_ratio: float = 0.15,
        left_margin_ratio: float = 0.10,
        right_margin_ratio: float = 0.25
    ) -> np.ndarray:
        """
        Crop name region from top of card

        The name region is the top 15% of the card, used for:
        - Variant discrimination (showcase vs normal art)
        - Card name matching

        Args:
            image: Card image as numpy array
            top_ratio: Height ratio to crop from top (default 0.15 = 15%)
            left_margin_ratio: Left margin to exclude (default 0.10 = 10%)
            right_margin_ratio: Right margin to exclude (default 0.25 = 25%)

        Returns:
            Cropped name region as numpy array
        """
        h, w = image.shape[:2]

        # Crop dimensions
        crop_h = int(h * top_ratio)  # Top 15% height
        left_margin = int(w * left_margin_ratio)  # 10% from left
        right_margin = int(w * right_margin_ratio)  # 25% from right

        # Extract region (skip top 5px to avoid border)
        return image[5:crop_h, left_margin:w - right_margin]

    @staticmethod
    def crop_collector_region(
        image: np.ndarray,
        bottom_ratio: float = 0.10,
        left_margin_ratio: float = 0.02,
        right_margin_ratio: float = 0.02
    ) -> np.ndarray:
        """
        Crop collector region from bottom of card

        The collector region is the bottom 10% of the card, containing:
        - Collector number (most discriminative feature!)
        - Set code
        - Artist name
        - Copyright info

        This region is highly discriminative because collector numbers
        are unique within a set.

        Args:
            image: Card image as numpy array
            bottom_ratio: Height ratio from bottom (default 0.10 = 10%)
            left_margin_ratio: Left margin to exclude (default 0.02 = 2%)
            right_margin_ratio: Right margin to exclude (default 0.02 = 2%)

        Returns:
            Cropped collector region as numpy array
        """
        h, w = image.shape[:2]

        # Crop dimensions
        crop_h = int(h * bottom_ratio)  # Bottom 10% height
        left_margin = int(w * left_margin_ratio)  # 2% from left
        right_margin = int(w * right_margin_ratio)  # 2% from right

        # Extract region from bottom
        return image[h - crop_h:h, left_margin:w - right_margin]

    @staticmethod
    def extract_all_regions(image: Union[Image.Image, np.ndarray], validate_dimensions: bool = True) -> Dict[str, Image.Image]:
        """
        Extract all regions needed for composite embedding

        This is the main entry point for region extraction. It extracts:
        - Full card image
        - Name region (top 15%)
        - Collector region (bottom 10%)

        These regions are then passed to the embedder's get_composite_embedding()
        method to create the pre-weighted composite embedding.

        CRITICAL: Both indexed and scanned images MUST be 363x504 for consistent
        region extraction and CLIP embeddings.

        Args:
            image: PIL Image or numpy array (must be 363x504)
            validate_dimensions: If True, warn on dimension mismatch (default True)

        Returns:
            Dict with 'full', 'name', 'collector' PIL Images
        """
        # Convert to numpy if needed
        if isinstance(image, Image.Image):
            img_array = np.array(image)
            full_pil = image
        else:
            img_array = image
            # Convert BGR to RGB if needed (OpenCV uses BGR)
            if len(img_array.shape) == 3 and img_array.shape[2] == 3:
                img_rgb = cv2.cvtColor(img_array, cv2.COLOR_BGR2RGB)
            else:
                img_rgb = img_array
            full_pil = Image.fromarray(img_rgb)

        # Validate dimensions (critical for consistent embeddings)
        h, w = img_array.shape[:2]
        if validate_dimensions and (w != CANONICAL_WIDTH or h != CANONICAL_HEIGHT):
            logger.warning(
                f"Dimension mismatch! Expected {CANONICAL_WIDTH}x{CANONICAL_HEIGHT}, "
                f"got {w}x{h}. This may cause inconsistent embeddings."
            )

        # Extract regions
        name_region = RegionExtractor.crop_name_region(img_array)
        collector_region = RegionExtractor.crop_collector_region(img_array)

        # Convert regions to RGB PIL Images
        def to_pil_rgb(arr):
            if len(arr.shape) == 2:  # Grayscale
                return Image.fromarray(arr).convert('RGB')
            elif arr.shape[2] == 4:  # RGBA
                return Image.fromarray(arr).convert('RGB')
            else:  # Assume BGR from OpenCV
                arr_rgb = cv2.cvtColor(arr, cv2.COLOR_BGR2RGB)
                return Image.fromarray(arr_rgb)

        return {
            'full': full_pil,
            'name': to_pil_rgb(name_region),
            'collector': to_pil_rgb(collector_region)
        }
