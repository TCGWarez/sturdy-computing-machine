#!/usr/bin/env python3
"""
Debug script to visualize contour detection process.
Shows all detected contours and their aspect ratios.
"""

import sys
import argparse
from pathlib import Path
import cv2
import numpy as np

# Add project root
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.detection.card_detector import EXPECTED_ASPECT_RATIO, ASPECT_TOLERANCE


def debug_contours(image_path: Path):
    """Debug the contour detection process."""
    print(f"\n{'='*60}")
    print(f"Debug Contours: {image_path.name}")
    print(f"{'='*60}")

    # Load image
    img = cv2.imread(str(image_path))
    if img is None:
        print(f"ERROR: Could not load {image_path}")
        return

    h, w = img.shape[:2]
    img_area = h * w
    print(f"Input size: {w}x{h}, area: {img_area}")
    print(f"Expected aspect ratio: {EXPECTED_ASPECT_RATIO:.3f} ± {ASPECT_TOLERANCE*100:.0f}%")

    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Apply bilateral filter
    filtered = cv2.bilateralFilter(gray, 9, 75, 75)

    # Canny edge detection
    edges = cv2.Canny(filtered, 50, 150)

    # Save edges image
    edges_path = image_path.parent / f"{image_path.stem}_edges.jpg"
    cv2.imwrite(str(edges_path), edges)
    print(f"\nSaved edges: {edges_path}")

    # Morphological operations
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    edges_processed = cv2.dilate(edges, kernel, iterations=1)
    edges_processed = cv2.morphologyEx(edges_processed, cv2.MORPH_CLOSE, kernel, iterations=2)
    edges_processed = cv2.erode(edges_processed, kernel, iterations=1)

    # Save processed edges
    processed_path = image_path.parent / f"{image_path.stem}_edges_processed.jpg"
    cv2.imwrite(str(processed_path), edges_processed)
    print(f"Saved processed edges: {processed_path}")

    # Find contours
    contours, _ = cv2.findContours(edges_processed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    print(f"\nFound {len(contours)} contours")

    # Sort by area
    contours = sorted(contours, key=cv2.contourArea, reverse=True)

    # Create debug image
    debug_img = img.copy()

    # Analyze top contours
    print("\nTop 10 contours:")
    print("-" * 80)

    for i, contour in enumerate(contours[:10]):
        area = cv2.contourArea(contour)
        area_pct = area / img_area * 100

        rect = cv2.minAreaRect(contour)
        center, (rect_w, rect_h), angle = rect

        # Normalize for aspect ratio calc
        if rect_w > rect_h:
            rect_w, rect_h = rect_h, rect_w

        if rect_h > 0:
            aspect_ratio = rect_w / rect_h
        else:
            aspect_ratio = 0

        # Check if passes filters
        passes_area = area >= img_area * 0.1
        aspect_deviation = abs(aspect_ratio - EXPECTED_ASPECT_RATIO) / EXPECTED_ASPECT_RATIO if EXPECTED_ASPECT_RATIO > 0 else float('inf')
        passes_aspect = aspect_deviation < ASPECT_TOLERANCE

        status = ">> SELECTED" if (passes_area and passes_aspect) else ""
        if not passes_area:
            status = "X area<10%"
        elif not passes_aspect:
            status = f"X aspect {aspect_deviation:.1%} off"

        print(f"  [{i}] area={area_pct:5.1f}%  size={rect_w:.0f}x{rect_h:.0f}  "
              f"aspect={aspect_ratio:.3f}  angle={angle:.1f}°  {status}")

        # Draw contour
        color = (0, 255, 0) if (passes_area and passes_aspect) else (0, 0, 255)
        cv2.drawContours(debug_img, [contour], -1, color, 2)

        # Draw bounding box
        box = cv2.boxPoints(rect)
        box = box.astype(np.int32)
        cv2.drawContours(debug_img, [box], -1, (255, 255, 0), 2)

        # Label
        cx, cy = int(center[0]), int(center[1])
        cv2.putText(debug_img, f"[{i}]", (cx, cy),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    # Save debug image
    debug_path = image_path.parent / f"{image_path.stem}_contours_debug.jpg"
    cv2.imwrite(str(debug_path), debug_img)
    print(f"\nSaved contours debug: {debug_path}")


def main():
    parser = argparse.ArgumentParser(description="Debug contour detection")
    parser.add_argument('path', type=Path, help="Image file path")
    args = parser.parse_args()
    debug_contours(args.path)


if __name__ == "__main__":
    main()
