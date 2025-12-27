#!/usr/bin/env python3
"""
OCR testing script for card recognition.

Usage:
    # Test with a single image
    uv run python scripts/tune_ocr_regions.py path/to/card.jpg

    # Save debug images
    uv run python scripts/tune_ocr_regions.py path/to/card.jpg --save-crops

    # Batch test on a directory
    uv run python scripts/tune_ocr_regions.py uploads/batch_id/ --batch
"""

import sys
import argparse
from pathlib import Path
import cv2

# Add project root
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.detection.card_detector import detect_and_warp
from src.ocr.tesseract_service import TesseractOCRService
from src.ocr.region_crops import crop_title_for_ocr, crop_collector_for_ocr
from src.ocr.set_code_validator import extract_set_code_from_text
from src.ocr.disambiguator import OCRDisambiguator


def test_single_image(image_path: Path, save_crops: bool = False):
    """Test OCR on a single image using production pipeline."""
    print(f"\n{'='*60}")
    print(f"Testing: {image_path.name}")
    print(f"{'='*60}")

    # Load and warp
    img = cv2.imread(str(image_path))
    if img is None:
        print(f"ERROR: Could not load {image_path}")
        return None

    try:
        warped, _, _ = detect_and_warp(img, debug=False)
        warped_rgb = cv2.cvtColor(warped, cv2.COLOR_BGR2RGB)
    except Exception as e:
        print(f"ERROR: Warp failed - {e}")
        return None

    h, w = warped_rgb.shape[:2]
    print(f"Warped size: {w}x{h}")

    # Initialize OCR
    ocr = TesseractOCRService()

    # Title OCR
    title_crop = crop_title_for_ocr(warped_rgb)
    title_result = ocr.extract_title_text(title_crop)
    print(f"\nTITLE:")
    print(f"  Text: '{title_result.text}'")
    print(f"  Confidence: {title_result.confidence:.2%}")

    if save_crops:
        crop_path = image_path.parent / f"{image_path.stem}_title_crop.jpg"
        cv2.imwrite(str(crop_path), cv2.cvtColor(title_crop, cv2.COLOR_RGB2BGR))
        print(f"  Saved: {crop_path}")

    # Collector OCR
    collector_crop = crop_collector_for_ocr(warped_rgb)
    collector_result = ocr.extract_collector_text(collector_crop)
    print(f"\nCOLLECTOR:")
    print(f"  Text: '{collector_result.text}'")
    print(f"  Confidence: {collector_result.confidence:.2%}")

    if save_crops:
        crop_path = image_path.parent / f"{image_path.stem}_collector_crop.jpg"
        cv2.imwrite(str(crop_path), cv2.cvtColor(collector_crop, cv2.COLOR_RGB2BGR))
        print(f"  Saved: {crop_path}")

    # Parse collector text
    collector_info = OCRDisambiguator._parse_collector_text(collector_result.text)
    print(f"\nPARSED:")
    print(f"  Number: {collector_info.collector_number}")
    print(f"  Set: {collector_info.set_code}")
    print(f"  Pattern: {collector_info.pattern_matched}")

    # Fuzzy set code matching
    fuzzy_set = extract_set_code_from_text(collector_result.text)
    print(f"\nFUZZY SET CODE:")
    print(f"  Matched: '{fuzzy_set}'")

    return {
        'title': title_result.text,
        'title_conf': title_result.confidence,
        'collector': collector_result.text,
        'collector_conf': collector_result.confidence,
        'set_code': fuzzy_set,
    }


def main():
    parser = argparse.ArgumentParser(description="Test OCR on card images")
    parser.add_argument('path', type=Path, help="Image file or directory")
    parser.add_argument('--batch', action='store_true', help="Process all images in directory")
    parser.add_argument('--save-crops', action='store_true', help="Save crop images for debugging")

    args = parser.parse_args()

    if args.batch and args.path.is_dir():
        images = list(args.path.glob("*.jpg")) + list(args.path.glob("*.png"))
        print(f"Found {len(images)} images")

        success = 0
        for img_path in images[:10]:  # Limit to 10 for quick testing
            result = test_single_image(img_path, save_crops=args.save_crops)
            if result and result.get('set_code'):
                success += 1

        print(f"\n{'='*60}")
        print(f"Results: {success}/{min(len(images), 10)} with set code extracted")
    else:
        test_single_image(args.path, save_crops=args.save_crops)


if __name__ == "__main__":
    main()
