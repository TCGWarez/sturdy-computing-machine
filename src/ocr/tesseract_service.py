"""
Tesseract OCR service implementation.

Uses pytesseract to extract text from card regions.
Region-specific preprocessing optimized for MTG card text.
"""

import cv2
import numpy as np
import logging
import platform
from pathlib import Path
from typing import Optional

from src.ocr.base_ocr import BaseOCRService, OCRResult
from src.config import OCR_PSM_MODE

logger = logging.getLogger(__name__)

# Lazy import pytesseract to avoid import errors if not installed
_pytesseract = None

# Common Tesseract install locations on Windows
WINDOWS_TESSERACT_PATHS = [
    r"C:\Program Files\Tesseract-OCR\tesseract.exe",
    r"C:\Program Files (x86)\Tesseract-OCR\tesseract.exe",
    r"C:\ProgramData\chocolatey\bin\tesseract.exe",
]


def _find_tesseract_windows() -> Optional[str]:
    """Find Tesseract executable on Windows."""
    for path in WINDOWS_TESSERACT_PATHS:
        if Path(path).exists():
            logger.info(f"Found Tesseract at: {path}")
            return path
    return None


def _get_pytesseract():
    """Lazy load pytesseract module and configure path if needed."""
    global _pytesseract
    if _pytesseract is None:
        try:
            import pytesseract

            # On Windows, auto-configure Tesseract path if not in PATH
            if platform.system() == "Windows":
                tesseract_path = _find_tesseract_windows()
                if tesseract_path:
                    pytesseract.pytesseract.tesseract_cmd = tesseract_path
                    logger.info(f"Configured pytesseract to use: {tesseract_path}")

            _pytesseract = pytesseract
        except ImportError:
            raise ImportError(
                "pytesseract is required for OCR. Install with: pip install pytesseract\n"
                "Also ensure Tesseract is installed on your system:\n"
                "  Windows: choco install tesseract or download from GitHub\n"
                "  Linux: apt-get install tesseract-ocr\n"
                "  macOS: brew install tesseract"
            )
    return _pytesseract


class TesseractOCRService(BaseOCRService):
    """
    Tesseract-based OCR service for extracting text from MTG cards.

    Provides region-specific OCR methods optimized for different card areas:
    - extract_title_text(): For card titles (raw grayscale + PSM 7 with fallbacks)
    - extract_collector_text(): For collector info (CLAHE + invert + PSM 6)

    Usage:
        ocr = TesseractOCRService()
        title_result = ocr.extract_title_text(title_crop)
        collector_result = ocr.extract_collector_text(collector_crop)
    """

    def __init__(self, tesseract_cmd: Optional[str] = None):
        """
        Initialize the Tesseract OCR service.

        Args:
            tesseract_cmd: Optional path to tesseract executable.
                          If not provided, uses system PATH.
        """
        self._tesseract_cmd = tesseract_cmd

        # Verify Tesseract is available on initialization
        if tesseract_cmd:
            pytesseract = _get_pytesseract()
            pytesseract.pytesseract.tesseract_cmd = tesseract_cmd

    def is_available(self) -> bool:
        """Check if Tesseract is properly installed."""
        try:
            pytesseract = _get_pytesseract()
            version = pytesseract.get_tesseract_version()
            logger.info(f"Tesseract version: {version}")
            return True
        except Exception as e:
            logger.warning(f"Tesseract not available: {e}")
            return False

    # Valid Tesseract PSM modes (0-13)
    VALID_PSM_MODES = range(0, 14)

    def _to_grayscale(self, image: np.ndarray) -> np.ndarray:
        """Convert image to grayscale."""
        if len(image.shape) == 3:
            if image.shape[2] == 4:  # RGBA
                return cv2.cvtColor(image, cv2.COLOR_RGBA2GRAY)
            elif image.shape[2] == 3:  # RGB or BGR
                return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        return image.copy()

    def _run_tesseract(
        self,
        image: np.ndarray,
        config: str,
        lang: str = 'eng'
    ) -> OCRResult:
        """
        Run Tesseract OCR on preprocessed image.

        Args:
            image: Preprocessed grayscale/binary image
            config: Tesseract config string (e.g., '--psm 7 --oem 3')
            lang: Language code

        Returns:
            OCRResult with extracted text and confidence
        """
        try:
            pytesseract = _get_pytesseract()

            data = pytesseract.image_to_data(
                image,
                lang=lang,
                config=config,
                output_type=pytesseract.Output.DICT
            )

            text_parts = []
            confidences = []

            for i, word in enumerate(data['text']):
                if word.strip():
                    text_parts.append(word.strip())
                    conf = data['conf'][i]
                    if conf >= 0:
                        confidences.append(conf / 100.0)

            text = ' '.join(text_parts)
            confidence = sum(confidences) / len(confidences) if confidences else 0.0

            return OCRResult(text=text, confidence=confidence)

        except Exception as e:
            logger.error(f"Tesseract failed: {e}")
            return OCRResult(text="", confidence=0.0)

    def extract_title_text(
        self,
        image: np.ndarray,
        lang: str = 'eng'
    ) -> OCRResult:
        """
        OCR optimized for card title region.

        Tries raw grayscale first (works for most cards), then falls back
        through various preprocessing attempts for difficult backgrounds.

        Preprocessing attempts:
        1. Raw grayscale + PSM 7
        2. Otsu binarization + PSM 7
        3. Inverted binary + PSM 7
        4. CLAHE enhanced + PSM 6
        5. CLAHE + binarization + PSM 6

        Args:
            image: Title region crop (RGB, BGR, or grayscale)
            lang: Tesseract language code

        Returns:
            OCRResult with extracted text and confidence
        """
        if image is None or image.size == 0:
            logger.warning("Empty image provided to title OCR")
            return OCRResult(text="", confidence=0.0)

        gray = self._to_grayscale(image)
        config_psm7 = '--psm 7 --oem 3'
        config_psm6 = '--psm 6 --oem 3'

        # Try 1: Raw grayscale (works for most cards)
        result = self._run_tesseract(gray, config_psm7, lang)
        if result.text:
            logger.debug(f"Title OCR (raw): '{result.text}'")
            return result

        # Try 2: Otsu binarization
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        result = self._run_tesseract(binary, config_psm7, lang)
        if result.text:
            logger.debug(f"Title OCR (binary): '{result.text}'")
            return result

        # Try 3: Inverted binary
        inverted = cv2.bitwise_not(binary)
        result = self._run_tesseract(inverted, config_psm7, lang)
        if result.text:
            logger.debug(f"Title OCR (inverted): '{result.text}'")
            return result

        # Try 4: CLAHE enhanced + PSM 6
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(gray)
        result = self._run_tesseract(enhanced, config_psm6, lang)
        if result.text:
            logger.debug(f"Title OCR (CLAHE+PSM6): '{result.text}'")
            return result

        # Try 5: CLAHE + binarization
        _, clahe_binary = cv2.threshold(enhanced, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        result = self._run_tesseract(clahe_binary, config_psm6, lang)
        if result.text:
            logger.debug(f"Title OCR (CLAHE+binary): '{result.text}'")
            return result

        logger.debug("Title OCR: all attempts returned empty")
        return OCRResult(text="", confidence=0.0)

    def extract_collector_text(
        self,
        image: np.ndarray,
        lang: str = 'eng'
    ) -> OCRResult:
        """
        OCR optimized for collector region.

        Uses upscaling + CLAHE + binarization + PSM 6 pipeline.
        Small text is upscaled 2-3x for better recognition.

        Args:
            image: Collector region crop (RGB, BGR, or grayscale)
            lang: Tesseract language code

        Returns:
            OCRResult with extracted text and confidence
        """
        if image is None or image.size == 0:
            logger.warning("Empty image provided to collector OCR")
            return OCRResult(text="", confidence=0.0)

        # Upscale small crops for better OCR (collector text is typically tiny)
        h, w = image.shape[:2]
        if h < 50 or w < 200:
            scale = 3
            image = cv2.resize(image, (w * scale, h * scale), interpolation=cv2.INTER_CUBIC)

        gray = self._to_grayscale(image)

        # CLAHE enhancement
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(gray)

        # Binarize
        _, binary = cv2.threshold(enhanced, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        # PSM 6 = uniform block of text (handles multiple lines)
        config = '--psm 6 --oem 3'
        result = self._run_tesseract(binary, config, lang)

        if result.text:
            logger.debug(f"Collector OCR: '{result.text}' (conf: {result.confidence:.2f})")
            return result

        # Fallback: try inverted (for white-on-black text)
        inverted = cv2.bitwise_not(binary)
        result = self._run_tesseract(inverted, config, lang)

        logger.debug(f"Collector OCR (inverted): '{result.text}' (conf: {result.confidence:.2f})")
        return result

    def extract_text(
        self,
        image: np.ndarray,
        psm: int = OCR_PSM_MODE,
        lang: str = 'eng'
    ) -> OCRResult:
        """
        Generic text extraction (for backward compatibility).

        For best results on MTG cards, use region-specific methods:
        - extract_title_text() for card titles
        - extract_collector_text() for collector info

        Args:
            image: Input image as numpy array (RGB, BGR, or grayscale)
            psm: Page segmentation mode (default: 7 for single line)
            lang: Tesseract language code (default: 'eng')

        Returns:
            OCRResult with extracted text and confidence
        """
        if psm not in self.VALID_PSM_MODES:
            raise ValueError(f"Invalid PSM mode: {psm}. Must be 0-13.")

        if image is None or image.size == 0:
            logger.warning("Empty image provided to OCR")
            return OCRResult(text="", confidence=0.0)

        gray = self._to_grayscale(image)
        config = f'--psm {psm} --oem 3'
        return self._run_tesseract(gray, config, lang)

    def extract_text_with_boxes(
        self,
        image: np.ndarray,
        psm: int = OCR_PSM_MODE,
        lang: str = 'eng'
    ) -> dict:
        """
        Extract text with bounding box information.

        Useful for debugging and visualization.

        Args:
            image: Input image
            psm: Page segmentation mode
            lang: Language code

        Returns:
            Dictionary with 'text', 'boxes', and 'confidences'
        """
        if image is None or image.size == 0:
            return {'text': '', 'boxes': [], 'confidences': []}

        try:
            pytesseract = _get_pytesseract()
            gray = self._to_grayscale(image)
            config = f'--psm {psm} --oem 3'

            data = pytesseract.image_to_data(
                gray,
                lang=lang,
                config=config,
                output_type=pytesseract.Output.DICT
            )

            boxes = []
            confidences = []
            text_parts = []

            for i in range(len(data['text'])):
                if data['text'][i].strip():
                    text_parts.append(data['text'][i])
                    boxes.append({
                        'x': data['left'][i],
                        'y': data['top'][i],
                        'w': data['width'][i],
                        'h': data['height'][i]
                    })
                    conf = data['conf'][i]
                    confidences.append(conf / 100.0 if conf >= 0 else 0.0)

            return {
                'text': ' '.join(text_parts),
                'boxes': boxes,
                'confidences': confidences
            }

        except Exception as e:
            logger.error(f"OCR with boxes failed: {e}")
            return {'text': '', 'boxes': [], 'confidences': []}
