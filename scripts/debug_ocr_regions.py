#!/usr/bin/env python3
"""
Debug script to visualize where text is detected on a card.
Runs OCR on the whole warped card and draws bounding boxes around detected text.
"""

import sys
import argparse
from pathlib import Path
import cv2
import numpy as np
import pytesseract

# Add project root
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.detection.card_detector import detect_and_warp
from src.config import (
    OCR_TITLE_X_START, OCR_TITLE_X_END, OCR_TITLE_Y_START, OCR_TITLE_Y_END,
    OCR_COLLECTOR_X_START, OCR_COLLECTOR_X_END, OCR_COLLECTOR_Y_START, OCR_COLLECTOR_Y_END,
)


def debug_ocr_regions(image_path: Path, output_path: Path = None):
    """
    Run OCR on whole card and draw bounding boxes around detected text.
    Also draws the configured crop regions for comparison.
    """
    print(f"\n{'='*60}")
    print(f"Debug OCR: {image_path.name}")
    print(f"{'='*60}")

    # Load and warp
    img = cv2.imread(str(image_path))
    if img is None:
        print(f"ERROR: Could not load {image_path}")
        return None

    try:
        warped, _, _ = detect_and_warp(img, debug=False)
    except Exception as e:
        print(f"ERROR: Warp failed - {e}")
        return None

    h, w = warped.shape[:2]
    print(f"Warped size: {w}x{h}")

    # Convert to RGB for OCR
    warped_rgb = cv2.cvtColor(warped, cv2.COLOR_BGR2RGB)

    # Run OCR on whole image with bounding box data
    print("\nRunning full-image OCR...")
    ocr_data = pytesseract.image_to_data(warped_rgb, output_type=pytesseract.Output.DICT)

    # Create output image (BGR for drawing)
    output = warped.copy()

    # Draw bounding boxes for each detected word
    n_boxes = len(ocr_data['text'])
    detected_texts = []

    for i in range(n_boxes):
        text = ocr_data['text'][i].strip()
        conf = int(ocr_data['conf'][i])

        if text and conf > 0:  # Has text and valid confidence
            x = ocr_data['left'][i]
            y = ocr_data['top'][i]
            w_box = ocr_data['width'][i]
            h_box = ocr_data['height'][i]

            # Color based on confidence (green=high, red=low)
            if conf > 70:
                color = (0, 255, 0)  # Green
            elif conf > 40:
                color = (0, 255, 255)  # Yellow
            else:
                color = (0, 0, 255)  # Red

            # Draw bounding box
            cv2.rectangle(output, (x, y), (x + w_box, y + h_box), color, 2)

            # Draw text label
            label = f"{text} ({conf}%)"
            cv2.putText(output, label, (x, y - 5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)

            detected_texts.append({
                'text': text,
                'conf': conf,
                'x': x, 'y': y,
                'w': w_box, 'h': h_box,
                'y_pct': y / h * 100,
                'x_pct': x / w * 100,
            })

    # Draw configured crop regions (blue dashed rectangles)
    h, w = output.shape[:2]

    # Title region (current config)
    title_x1 = int(w * OCR_TITLE_X_START)
    title_x2 = int(w * OCR_TITLE_X_END)
    title_y1 = int(h * OCR_TITLE_Y_START)
    title_y2 = int(h * OCR_TITLE_Y_END)
    cv2.rectangle(output, (title_x1, title_y1), (title_x2, title_y2), (255, 0, 0), 2)
    cv2.putText(output, "TITLE CROP", (title_x1, title_y1 - 5),
               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

    # Collector region (current config)
    coll_x1 = int(w * OCR_COLLECTOR_X_START)
    coll_x2 = int(w * OCR_COLLECTOR_X_END)
    coll_y1 = int(h * OCR_COLLECTOR_Y_START)
    coll_y2 = int(h * OCR_COLLECTOR_Y_END)
    cv2.rectangle(output, (coll_x1, coll_y1), (coll_x2, coll_y2), (255, 0, 0), 2)
    cv2.putText(output, "COLLECTOR CROP", (coll_x1, coll_y1 - 5),
               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

    # Print detected texts sorted by Y position
    print("\nDetected text regions (sorted by Y position):")
    print("-" * 60)
    detected_texts.sort(key=lambda x: x['y'])

    for item in detected_texts:
        print(f"  Y={item['y_pct']:5.1f}%  X={item['x_pct']:5.1f}%  "
              f"conf={item['conf']:3d}%  text='{item['text']}'")

    # Identify likely title region (first high-confidence text near top)
    title_candidates = [t for t in detected_texts if t['y_pct'] < 20 and t['conf'] > 50]
    if title_candidates:
        print(f"\nLikely title region: Y={title_candidates[0]['y_pct']:.1f}%")

    # Identify likely collector region (text near bottom)
    collector_candidates = [t for t in detected_texts if t['y_pct'] > 80 and t['conf'] > 30]
    if collector_candidates:
        print(f"Likely collector region: Y={collector_candidates[0]['y_pct']:.1f}%")

    # Save output
    if output_path is None:
        output_path = image_path.parent / f"{image_path.stem}_ocr_debug.jpg"

    cv2.imwrite(str(output_path), output)
    print(f"\nSaved debug image: {output_path}")

    # Also save the raw warped image for reference
    warped_path = image_path.parent / f"{image_path.stem}_warped.jpg"
    cv2.imwrite(str(warped_path), warped)
    print(f"Saved warped image: {warped_path}")

    return detected_texts


def main():
    parser = argparse.ArgumentParser(description="Debug OCR text detection on cards")
    parser.add_argument('path', type=Path, help="Image file path")
    parser.add_argument('--output', '-o', type=Path, help="Output path for debug image")

    args = parser.parse_args()
    debug_ocr_regions(args.path, args.output)


if __name__ == "__main__":
    main()
