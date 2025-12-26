#!/usr/bin/env python3
"""
Debug script to visualize card detection and warp process.
Shows detected corners and their ordering.
"""

import sys
import argparse
from pathlib import Path
import cv2
import numpy as np

# Add project root
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.detection.card_detector import detect_and_warp, _detect_by_contours, _order_corners


def debug_warp(image_path: Path):
    """Debug the card detection and warp process."""
    print(f"\n{'='*60}")
    print(f"Debug Warp: {image_path.name}")
    print(f"{'='*60}")

    # Load image
    img = cv2.imread(str(image_path))
    if img is None:
        print(f"ERROR: Could not load {image_path}")
        return

    h, w = img.shape[:2]
    print(f"Input size: {w}x{h}")

    # Convert to grayscale for detection
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    filtered = cv2.bilateralFilter(gray, 9, 75, 75)

    # Try contour detection
    detection_result = _detect_by_contours(filtered)

    if detection_result is None:
        print("ERROR: Contour detection failed")
        return

    corners, rect = detection_result
    center, (rect_w, rect_h), angle = rect

    print(f"\nDetected rectangle:")
    print(f"  Center: ({center[0]:.1f}, {center[1]:.1f})")
    print(f"  Size: {rect_w:.1f} x {rect_h:.1f}")
    print(f"  Angle: {angle:.1f}°")
    print(f"  Aspect ratio: {min(rect_w, rect_h) / max(rect_w, rect_h):.3f}")

    print(f"\nOrdered corners:")
    for i, corner in enumerate(corners):
        labels = ["TL", "TR", "BR", "BL"]
        print(f"  {labels[i]}: ({corner[0]:.1f}, {corner[1]:.1f})")

    # Check landscape detection
    corner_width = np.linalg.norm(corners[1] - corners[0])  # TL to TR
    corner_height = np.linalg.norm(corners[3] - corners[0])  # TL to BL

    print(f"\nCorner-based dimensions:")
    print(f"  Width (TL->TR): {corner_width:.1f}")
    print(f"  Height (TL->BL): {corner_height:.1f}")
    print(f"  Is landscape: {corner_width > corner_height}")

    # Draw corners on original image
    debug_img = img.copy()
    colors = [(0, 0, 255), (0, 255, 0), (255, 0, 0), (255, 255, 0)]  # Red, Green, Blue, Yellow
    labels = ["0:TL", "1:TR", "2:BR", "3:BL"]

    for i, corner in enumerate(corners):
        pt = tuple(corner.astype(int))
        cv2.circle(debug_img, pt, 10, colors[i], -1)
        cv2.putText(debug_img, labels[i], (pt[0] + 15, pt[1]),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, colors[i], 2)

    # Draw lines between corners
    cv2.polylines(debug_img, [corners.astype(np.int32)], True, (0, 255, 255), 3)

    # Save debug image with corners
    corners_path = image_path.parent / f"{image_path.stem}_corners.jpg"
    cv2.imwrite(str(corners_path), debug_img)
    print(f"\nSaved corners debug: {corners_path}")

    # Now do the actual warp and save
    warped, _, _ = detect_and_warp(img, debug=False)
    warped_path = image_path.parent / f"{image_path.stem}_warped_debug.jpg"
    cv2.imwrite(str(warped_path), warped)
    print(f"Saved warped debug: {warped_path}")

    # Add orientation indicators to warped image
    warped_annotated = warped.copy()
    wh, ww = warped.shape[:2]
    cv2.putText(warped_annotated, "TOP", (ww//2 - 30, 30),
               cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.putText(warped_annotated, "BOTTOM", (ww//2 - 50, wh - 10),
               cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.putText(warped_annotated, "L", (10, wh//2),
               cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.putText(warped_annotated, "R", (ww - 30, wh//2),
               cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    annotated_path = image_path.parent / f"{image_path.stem}_warped_annotated.jpg"
    cv2.imwrite(str(annotated_path), warped_annotated)
    print(f"Saved annotated warped: {annotated_path}")


def main():
    parser = argparse.ArgumentParser(description="Debug card detection and warp")
    parser.add_argument('path', type=Path, help="Image file path")
    args = parser.parse_args()
    debug_warp(args.path)


if __name__ == "__main__":
    main()
