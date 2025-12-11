"""
src/detection/card_detector.py: Card edge detection and perspective warping
Following PRD.md Task 2 specifications
Detects card boundaries and warps to canonical 512x512 size
"""

import cv2
import numpy as np
from typing import Tuple, Optional, List
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


# Magic cards are 63mm x 88mm (2.5" x 3.5") = aspect ratio ~0.716
# Use 363x504 to maintain proper aspect ratio while keeping height at ~512
CANONICAL_WIDTH = 363
CANONICAL_HEIGHT = 504
CANONICAL_SIZE = 512  # Kept for backwards compatibility, but deprecated

# Expected aspect ratio for Magic cards (2.5:3.5 = 0.714)
EXPECTED_ASPECT_RATIO = 0.714
ASPECT_TOLERANCE = 0.15  # Allow 15% deviation


def _normalize_rect_dimensions(rect: tuple) -> tuple:
    """
    Normalize minAreaRect so width < height (portrait orientation).

    Args:
        rect: Result from cv2.minAreaRect() - ((cx, cy), (w, h), angle)

    Returns:
        Normalized rect with adjusted angle so width < height
    """
    center, (width, height), angle = rect

    if width > height:
        # Swap dimensions and adjust angle by 90 degrees
        width, height = height, width
        angle = angle + 90

    return (center, (width, height), angle)


def _rotate_image_by_angle(image: np.ndarray, angle: float, center: tuple = None) -> Tuple[np.ndarray, np.ndarray]:
    """
    Rotate image by specified angle around center point.

    Args:
        image: Input image
        angle: Rotation angle in degrees (positive = counter-clockwise)
        center: Center of rotation (default: image center)

    Returns:
        Tuple of (rotated_image, rotation_matrix)
    """
    h, w = image.shape[:2]

    if center is None:
        center = (w / 2, h / 2)

    # Get rotation matrix
    M = cv2.getRotationMatrix2D(center, angle, 1.0)

    # Calculate new image bounds to avoid cropping
    cos = np.abs(M[0, 0])
    sin = np.abs(M[0, 1])
    new_w = int(h * sin + w * cos)
    new_h = int(h * cos + w * sin)

    # Adjust translation to keep image centered
    M[0, 2] += (new_w - w) / 2
    M[1, 2] += (new_h - h) / 2

    # Apply rotation
    rotated = cv2.warpAffine(image, M, (new_w, new_h),
                             borderMode=cv2.BORDER_REPLICATE)

    return rotated, M


def _transform_points(points: np.ndarray, M: np.ndarray) -> np.ndarray:
    """
    Transform points using a 2x3 affine transformation matrix.

    Args:
        points: Nx2 array of points
        M: 2x3 transformation matrix

    Returns:
        Transformed Nx2 array of points
    """
    ones = np.ones((len(points), 1))
    points_homogeneous = np.hstack([points, ones])
    transformed = M.dot(points_homogeneous.T).T
    return transformed.astype(np.float32)


def detect_and_warp(
    image: np.ndarray,
    canonical_size: int = None,  # Deprecated - use canonical_width/height
    canonical_width: int = CANONICAL_WIDTH,
    canonical_height: int = CANONICAL_HEIGHT,
    debug: bool = False
) -> Tuple[np.ndarray, Optional[np.ndarray], Optional[np.ndarray]]:
    """
    Detect card boundaries and warp to canonical size

    Args:
        image: Input image (BGR or RGB)
        canonical_size: (Deprecated) Use canonical_width/height instead
        canonical_width: Target width for warped image (default 363 to match card aspect ratio)
        canonical_height: Target height for warped image (default 504 to match card aspect ratio)
        debug: If True, return debug visualization

    Returns:
        Tuple of:
        - warped_image: Perspective-warped card image (canonical_width x canonical_height)
        - mask: Binary mask of detected card region
        - boundary_points: 4 corner points of detected card
    """
    # Handle deprecated canonical_size parameter
    if canonical_size is not None and canonical_size != CANONICAL_SIZE:
        canonical_width = canonical_height = canonical_size
    if image is None or image.size == 0:
        raise ValueError("Empty input image")

    # Convert to grayscale for edge detection
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image.copy()

    # Apply bilateral filter to reduce noise while preserving edges
    filtered = cv2.bilateralFilter(gray, 9, 75, 75)

    # Keep track of original corners for return value
    original_corners = None
    rect = None

    # Try contour-based detection first (returns corners and rotation info)
    detection_result = _detect_by_contours(filtered)

    if detection_result is not None:
        corners, rect = detection_result
        original_corners = corners.copy()

        # Check if rotation correction is needed
        # minAreaRect angle is in range [-90, 0) for OpenCV 4.5+
        center, (rect_w, rect_h), angle = rect

        # Normalize dimensions and angle for portrait orientation
        if rect_w > rect_h:
            rect_w, rect_h = rect_h, rect_w
            angle = angle + 90

        # Determine if significant rotation correction is needed
        # Angles close to 0, -90, or 90 mean the card is roughly axis-aligned
        # We want to correct angles that are significantly off from these
        angle_from_horizontal = angle % 90
        if angle_from_horizontal > 45:
            angle_from_horizontal = 90 - angle_from_horizontal

        needs_rotation = angle_from_horizontal > 5  # More than 5 degrees off

        if needs_rotation:
            logger.debug(f"Applying rotation correction: {angle:.1f}° (deviation: {angle_from_horizontal:.1f}°)")

            # Rotate the entire image to make the card axis-aligned
            image, rotation_matrix = _rotate_image_by_angle(image, angle, center)

            # Transform corner points to new coordinate system
            corners = _transform_points(corners, rotation_matrix)

            # Update grayscale image for mask creation
            if len(image.shape) == 3:
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            else:
                gray = image.copy()

    # Fallback to Hough lines if contour detection fails
    if detection_result is None:
        logger.debug("Contour detection failed, trying Hough lines")
        corners = _detect_by_hough_lines(filtered)
        original_corners = corners

    # If still no detection, use the full image
    if detection_result is None and corners is None:
        logger.warning("Card detection failed, using full image")
        h, w = image.shape[:2]
        corners = np.array([
            [0, 0],
            [w-1, 0],
            [w-1, h-1],
            [0, h-1]
        ], dtype=np.float32)
        original_corners = corners

    # Use original corners for return if we haven't set them yet
    if original_corners is None:
        original_corners = corners

    # Check if the detected card region is in landscape orientation
    # (i.e., the card was scanned 90 degrees rotated)
    # We detect this by checking if the width of the detected region > height
    corner_width = np.linalg.norm(corners[1] - corners[0])  # top edge
    corner_height = np.linalg.norm(corners[3] - corners[0])  # left edge

    if corner_width > corner_height:
        # Card is in landscape orientation - rotate corners 90° clockwise
        # This remaps the corners so the perspective warp produces portrait output
        logger.debug(f"Detected landscape orientation ({corner_width:.0f}x{corner_height:.0f}), rotating corners for portrait warp")
        # Rotate corners: [TL, TR, BR, BL] -> [BL, TL, TR, BR]
        corners = np.roll(corners, 1, axis=0)

    # Warp the image to canonical size
    warped = _perspective_warp(image, corners, canonical_width, canonical_height)

    # Create mask (filled polygon of detected card)
    mask = np.zeros(gray.shape, dtype=np.uint8)
    cv2.fillPoly(mask, [corners.astype(np.int32)], 255)

    # Debug visualization if requested (show original detected corners on original image)
    if debug:
        debug_img = _create_debug_visualization(image.copy(), corners)
        cv2.imwrite("debug_card_detection.jpg", debug_img)
        logger.info("Saved debug visualization to debug_card_detection.jpg")

    # Return original_corners (pre-rotation) for API response boundary display
    return warped, mask, original_corners


def _detect_by_contours(gray: np.ndarray) -> Optional[Tuple[np.ndarray, tuple]]:
    """
    Detect card using contour detection with rotation-aware minAreaRect.

    Uses cv2.minAreaRect to detect cards at any rotation angle, not just
    axis-aligned rectangles. This allows proper detection of rotated scans.

    Args:
        gray: Grayscale filtered image

    Returns:
        Tuple of (corner_points, minAreaRect_result) or None if detection fails
        - corner_points: 4x2 array of box corners from minAreaRect
        - minAreaRect_result: ((cx, cy), (w, h), angle) for rotation info
    """
    # Canny edge detection with optimized thresholds
    edges = cv2.Canny(gray, 50, 150)

    # Morphological operations to clean up edges and connect fragments
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))

    # Dilation to connect nearby edges
    edges = cv2.dilate(edges, kernel, iterations=1)

    # Closing to fill small gaps
    edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel, iterations=2)

    # Erosion to thin edges back to proper width
    edges = cv2.erode(edges, kernel, iterations=1)

    # Find contours
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if not contours:
        return None

    # Sort contours by area (largest first)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)

    img_area = gray.shape[0] * gray.shape[1]

    # Try to find a rectangular contour with proper aspect ratio
    for contour in contours[:10]:  # Check top 10 largest contours
        # Check if area is reasonable (at least 10% of image)
        area = cv2.contourArea(contour)
        if area < img_area * 0.1:
            continue

        # Use minAreaRect for rotation-aware bounding box
        rect = cv2.minAreaRect(contour)
        center, (rect_w, rect_h), angle = rect

        # Skip if dimensions are invalid
        if rect_w <= 0 or rect_h <= 0:
            continue

        # Normalize so shorter dimension is width (portrait orientation)
        if rect_w > rect_h:
            rect_w, rect_h = rect_h, rect_w

        # Check aspect ratio (works regardless of rotation)
        aspect_ratio = rect_w / rect_h

        if abs(aspect_ratio - EXPECTED_ASPECT_RATIO) / EXPECTED_ASPECT_RATIO < ASPECT_TOLERANCE:
            # Found a valid card-shaped contour
            box = cv2.boxPoints(rect)
            corners = box.astype(np.float32)
            corners = _order_corners(corners)

            logger.debug(f"Found card with aspect ratio {aspect_ratio:.3f} "
                        f"(expected {EXPECTED_ASPECT_RATIO:.3f}), angle={angle:.1f}°")
            return corners, rect
        else:
            logger.debug(f"Rejected contour with aspect ratio {aspect_ratio:.3f} "
                        f"(expected {EXPECTED_ASPECT_RATIO:.3f})")

    return None


def _detect_by_hough_lines(gray: np.ndarray) -> Optional[np.ndarray]:
    """
    Detect card using Hough line detection as fallback

    Args:
        gray: Grayscale filtered image

    Returns:
        4 corner points or None if detection fails
    """
    # Edge detection
    edges = cv2.Canny(gray, 50, 150)

    # Hough line detection
    lines = cv2.HoughLinesP(
        edges,
        rho=1,
        theta=np.pi/180,
        threshold=100,
        minLineLength=100,
        maxLineGap=10
    )

    if lines is None or len(lines) < 4:
        return None

    # Group lines into horizontal and vertical
    horizontal_lines = []
    vertical_lines = []

    for line in lines:
        x1, y1, x2, y2 = line[0]
        angle = np.abs(np.arctan2(y2-y1, x2-x1) * 180 / np.pi)

        if angle < 30 or angle > 150:  # Nearly horizontal
            horizontal_lines.append(line[0])
        elif 60 < angle < 120:  # Nearly vertical
            vertical_lines.append(line[0])

    if len(horizontal_lines) < 2 or len(vertical_lines) < 2:
        return None

    # Find the outermost lines
    h_lines_sorted = sorted(horizontal_lines, key=lambda l: min(l[1], l[3]))
    v_lines_sorted = sorted(vertical_lines, key=lambda l: min(l[0], l[2]))

    top_line = h_lines_sorted[0]
    bottom_line = h_lines_sorted[-1]
    left_line = v_lines_sorted[0]
    right_line = v_lines_sorted[-1]

    # Find intersection points
    corners = []
    for h_line in [top_line, bottom_line]:
        for v_line in [left_line, right_line]:
            point = _line_intersection(h_line, v_line)
            if point is not None:
                corners.append(point)

    if len(corners) != 4:
        return None

    corners = np.array(corners, dtype=np.float32)
    corners = _order_corners(corners)
    return corners


def _line_intersection(line1: np.ndarray, line2: np.ndarray) -> Optional[np.ndarray]:
    """
    Find intersection point of two lines

    Args:
        line1, line2: Lines as [x1, y1, x2, y2]

    Returns:
        Intersection point [x, y] or None
    """
    x1, y1, x2, y2 = line1
    x3, y3, x4, y4 = line2

    denom = (x1-x2)*(y3-y4) - (y1-y2)*(x3-x4)
    if abs(denom) < 1e-10:
        return None

    t = ((x1-x3)*(y3-y4) - (y1-y3)*(x3-x4)) / denom

    x = x1 + t * (x2 - x1)
    y = y1 + t * (y2 - y1)

    return np.array([x, y])


def _order_corners(corners: np.ndarray) -> np.ndarray:
    """
    Order corners as: top-left, top-right, bottom-right, bottom-left

    Args:
        corners: 4x2 array of corner points

    Returns:
        Ordered 4x2 array
    """
    # Calculate center
    center = corners.mean(axis=0)

    # Sort by angle from center
    angles = np.arctan2(corners[:, 1] - center[1], corners[:, 0] - center[0])
    order = np.argsort(angles)

    # Reorder starting from top-left
    corners = corners[order]

    # Find top-left (minimum sum of coordinates)
    sums = corners[:, 0] + corners[:, 1]
    tl_idx = np.argmin(sums)

    # Rotate array to start with top-left
    corners = np.roll(corners, -tl_idx, axis=0)

    return corners


def _perspective_warp(image: np.ndarray, corners: np.ndarray, width: int, height: int, margin_reduction: float = -0.05) -> np.ndarray:
    """
    Apply perspective transformation to warp card to proper aspect ratio

    IMPROVED: Fixed to maintain proper card aspect ratio and optimal margin for collector numbers

    Args:
        image: Input image
        corners: 4 corner points (ordered: top-left, top-right, bottom-right, bottom-left)
        width: Target width for output image (363 for standard cards)
        height: Target height for output image (504 for standard cards)
        margin_reduction: Ratio to adjust boundaries (default -0.05 = 5% expansion outward)
                         Negative values expand outward to fully capture collector numbers
                         Positive values contract inward to crop tighter

    Returns:
        Warped image of shape (height, width, channels) maintaining proper card aspect ratio
    """
    # TUNED: Expand boundaries 5% outward to fully capture collector numbers and card edges
    # Negative margin_reduction expands outward from detected boundary
    # Calculate center point
    center = corners.mean(axis=0)

    # Adjust each corner relative to center
    adjusted_corners = corners.copy()
    for i in range(4):
        direction = corners[i] - center
        # Negative margin_reduction means expand outward, positive means contract inward
        adjusted_corners[i] = corners[i] - direction * margin_reduction

    # Ensure corners are within image bounds
    h, w = image.shape[:2]
    adjusted_corners[:, 0] = np.clip(adjusted_corners[:, 0], 0, w - 1)
    adjusted_corners[:, 1] = np.clip(adjusted_corners[:, 1], 0, h - 1)

    # Define destination points with PROPER ASPECT RATIO (not square!)
    # Magic cards are 63mm x 88mm (2.5" x 3.5") = 0.714 aspect ratio
    dst = np.array([
        [0, 0],
        [width-1, 0],
        [width-1, height-1],
        [0, height-1]
    ], dtype=np.float32)

    # Calculate perspective transform matrix
    M = cv2.getPerspectiveTransform(adjusted_corners, dst)

    # Apply transformation with proper dimensions (width x height, NOT square)
    warped = cv2.warpPerspective(image, M, (width, height))

    return warped


def _create_debug_visualization(image: np.ndarray, corners: np.ndarray) -> np.ndarray:
    """
    Create debug visualization with detected boundary

    Args:
        image: Input image
        corners: Detected corner points

    Returns:
        Image with boundary box drawn
    """
    # Draw the boundary polygon
    cv2.polylines(image, [corners.astype(np.int32)], True, (0, 255, 0), 3)

    # Draw corner points
    for i, corner in enumerate(corners):
        cv2.circle(image, tuple(corner.astype(int)), 8, (0, 0, 255), -1)
        # Label corners
        cv2.putText(image, str(i), tuple(corner.astype(int) + 10),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

    return image


# Utility functions for testing and CLI
def detect_card_from_file(
    image_path: Path,
    output_path: Optional[Path] = None,
    debug: bool = False
) -> np.ndarray:
    """
    Convenience function to detect and warp card from file

    Args:
        image_path: Path to input image
        output_path: Optional path to save warped image
        debug: Enable debug visualization

    Returns:
        Warped card image
    """
    # Load image
    image = cv2.imread(str(image_path))
    if image is None:
        raise ValueError(f"Could not load image: {image_path}")

    # Detect and warp
    warped, mask, corners = detect_and_warp(image, debug=debug)

    # Save if requested
    if output_path:
        cv2.imwrite(str(output_path), warped)
        logger.info(f"Saved warped image to {output_path}")

    return warped


if __name__ == "__main__":
    # Simple test
    import sys
    if len(sys.argv) > 1:
        input_path = Path(sys.argv[1])
        output_path = Path("warped_card.jpg")
        warped = detect_card_from_file(input_path, output_path, debug=True)
        print(f"Warped card saved to {output_path}")
        print(f"Shape: {warped.shape}")
    else:
        print("Usage: python card_detector.py <image_path>")