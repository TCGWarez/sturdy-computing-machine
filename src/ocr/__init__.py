"""
OCR package for MTG card text extraction and disambiguation.

This package provides:
- BaseOCRService: Abstract interface for OCR engines
- TesseractOCRService: Tesseract-based OCR implementation
- OCRDisambiguator: Disambiguation logic using OCR results
- Region cropping utilities for title and collector areas
- Set code fuzzy matching for OCR error correction
"""

from src.ocr.base_ocr import BaseOCRService, OCRResult
from src.ocr.tesseract_service import TesseractOCRService
from src.ocr.disambiguator import OCRDisambiguator, OCRDisambiguationResult, CollectorInfo
from src.ocr.region_crops import crop_title_for_ocr, crop_collector_for_ocr
from src.ocr.set_code_validator import (
    fuzzy_match_set_code,
    extract_set_code_from_text,
    load_valid_set_codes,
    get_valid_set_codes,
)

__all__ = [
    'BaseOCRService',
    'OCRResult',
    'TesseractOCRService',
    'OCRDisambiguator',
    'OCRDisambiguationResult',
    'CollectorInfo',
    'crop_title_for_ocr',
    'crop_collector_for_ocr',
    'fuzzy_match_set_code',
    'extract_set_code_from_text',
    'load_valid_set_codes',
    'get_valid_set_codes',
]
