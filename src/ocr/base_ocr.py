"""
Base OCR service interface.

Defines abstract interface for OCR engines, allowing for
swappable implementations (Tesseract, EasyOCR, PaddleOCR, etc.)
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional
import numpy as np


@dataclass
class OCRResult:
    """Result from OCR text extraction."""

    text: str
    """Extracted text, stripped of leading/trailing whitespace."""

    confidence: float
    """Confidence score from 0.0 to 1.0. Higher is better."""

    raw_text: Optional[str] = None
    """Raw text before any post-processing."""

    def __bool__(self) -> bool:
        """Return True if text was successfully extracted."""
        return bool(self.text.strip())


class BaseOCRService(ABC):
    """
    Abstract base class for OCR services.

    Implementations should handle:
    - Image preprocessing (grayscale, thresholding, etc.)
    - Text extraction
    - Confidence scoring

    Example usage:
        ocr = TesseractOCRService()
        result = ocr.extract_text(image)
        if result:
            print(f"Found: {result.text} (confidence: {result.confidence:.2f})")
    """

    @abstractmethod
    def extract_text(
        self,
        image: np.ndarray,
        psm: int = 7,
        lang: str = 'eng'
    ) -> OCRResult:
        """
        Extract text from an image region.

        Args:
            image: Image as numpy array (RGB or grayscale)
            psm: Page segmentation mode (Tesseract-style, default 7 = single line)
            lang: Language code (default 'eng' for English)

        Returns:
            OCRResult with extracted text and confidence
        """
        pass

    @abstractmethod
    def is_available(self) -> bool:
        """
        Check if the OCR service is properly installed and available.

        Returns:
            True if the service can be used, False otherwise
        """
        pass
