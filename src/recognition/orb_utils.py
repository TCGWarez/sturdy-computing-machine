"""
ORB keypoint matching utilities for card verification
"""

import cv2
import numpy as np
from pathlib import Path


def compute_orb_similarity(query_image: np.ndarray, reference_image_path: str, max_features: int = 500) -> float:
    """
    Compute ORB keypoint matching similarity between images

    Used for reranking candidates to verify visual similarity.

    Args:
        query_image: Query image as numpy array (RGB)
        reference_image_path: Path to reference card image
        max_features: Maximum ORB features to detect

    Returns:
        Similarity score [0, 1]
    """
    if not Path(reference_image_path).exists():
        return 0.0

    ref_img = cv2.imread(reference_image_path)
    if ref_img is None:
        return 0.0

    # Convert query to BGR
    if len(query_image.shape) == 2:
        query_bgr = cv2.cvtColor(query_image, cv2.COLOR_GRAY2BGR)
    elif query_image.shape[2] == 3:
        query_bgr = cv2.cvtColor(query_image, cv2.COLOR_RGB2BGR)
    else:
        query_bgr = cv2.cvtColor(query_image, cv2.COLOR_RGBA2BGR)

    # Resize to standard size
    target_size = (363, 504)
    query_resized = cv2.resize(query_bgr, target_size)
    ref_resized = cv2.resize(ref_img, target_size)

    # Convert to grayscale
    query_gray = cv2.cvtColor(query_resized, cv2.COLOR_BGR2GRAY)
    ref_gray = cv2.cvtColor(ref_resized, cv2.COLOR_BGR2GRAY)

    # ORB detection
    orb = cv2.ORB_create(nfeatures=max_features)

    try:
        kp1, des1 = orb.detectAndCompute(query_gray, None)
        kp2, des2 = orb.detectAndCompute(ref_gray, None)

        if des1 is None or des2 is None or len(des1) < 10 or len(des2) < 10:
            return 0.0

        # BFMatcher with Hamming distance
        bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
        matches = bf.knnMatch(des1, des2, k=2)

        # Lowe's ratio test
        good_matches = []
        for match_pair in matches:
            if len(match_pair) == 2:
                m, n = match_pair
                if m.distance < 0.75 * n.distance:
                    good_matches.append(m)

        if len(kp1) > 0:
            match_ratio = len(good_matches) / len(kp1)
            return min(1.0, match_ratio * 2.0)

        return 0.0

    except Exception:
        return 0.0
