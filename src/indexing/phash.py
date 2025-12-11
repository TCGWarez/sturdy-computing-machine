"""
src/indexing/phash.py: Perceptual hashing utilities
Computes pHash/dHash for fast candidate retrieval
Enhanced with 3-variant support per PRD.md Task 3
"""

import imagehash
import numpy as np
from PIL import Image
from typing import Union, List, Tuple, Dict


def compute_phash(image: Union[Image.Image, str], hash_size: int = 8) -> str:
    """
    Compute perceptual hash (pHash) for an image
    
    Args:
        image: PIL Image or path to image file
        hash_size: Size of hash (8 = 64-bit hash)
        
    Returns:
        Hash as hex string
    """
    if isinstance(image, str):
        img = Image.open(image)
    else:
        img = image
    
    # Convert to RGB if needed
    if img.mode != 'RGB':
        img = img.convert('RGB')
    
    # Compute pHash
    phash = imagehash.phash(img, hash_size=hash_size)
    
    return str(phash)


def compute_dhash(image: Union[Image.Image, str], hash_size: int = 8) -> str:
    """
    Compute difference hash (dHash) for an image
    
    Args:
        image: PIL Image or path to image file
        hash_size: Size of hash (8 = 64-bit hash)
        
    Returns:
        Hash as hex string
    """
    if isinstance(image, str):
        img = Image.open(image)
    else:
        img = image
    
    # Convert to RGB if needed
    if img.mode != 'RGB':
        img = img.convert('RGB')
    
    # Compute dHash
    dhash = imagehash.dhash(img, hash_size=hash_size)
    
    return str(dhash)


def hamming_distance(hash1: str, hash2: str) -> int:
    """
    Compute Hamming distance between two hashes
    
    Args:
        hash1: First hash as hex string
        hash2: Second hash as hex string
        
    Returns:
        Hamming distance (number of differing bits)
    """
    h1 = imagehash.hex_to_hash(hash1)
    h2 = imagehash.hex_to_hash(hash2)
    return h1 - h2


def phash_to_int(phash: str) -> int:
    """
    Convert pHash hex string to integer for faster comparison
    
    Args:
        phash: Hash as hex string
        
    Returns:
        Hash as integer
    """
    return int(phash, 16)


def int_to_phash(phash_int: int, hash_size: int = 8) -> str:
    """
    Convert integer hash back to hex string
    
    Args:
        phash_int: Hash as integer
        hash_size: Size of hash (for formatting)
        
    Returns:
        Hash as hex string
    """
    # Format as hex with appropriate length
    hex_str = format(phash_int, 'x')
    # Pad to expected length (hash_size^2 * 2 hex chars for 4 bits per char)
    expected_len = hash_size * hash_size // 4
    return hex_str.zfill(expected_len)


def batch_hamming_distance(query_hash: str, candidate_hashes: List[str]) -> np.ndarray:
    """
    Compute Hamming distances between query hash and multiple candidate hashes
    Uses vectorized operations for efficiency
    
    Args:
        query_hash: Query hash as hex string
        candidate_hashes: List of candidate hashes as hex strings
        
    Returns:
        Numpy array of Hamming distances
    """
    if not candidate_hashes:
        return np.array([])
    
    # Convert query hash to integer
    query_int = phash_to_int(query_hash)
    
    # Convert all candidate hashes to integers
    candidate_ints = np.array([phash_to_int(h) for h in candidate_hashes], dtype=np.uint64)
    
    # Compute Hamming distance using XOR and bit count
    # XOR gives bits that differ, then count set bits
    xor_result = query_int ^ candidate_ints
    
    # Count set bits (Hamming weight)
    # Use int.bit_count() if available (Python 3.10+), otherwise fall back to bin().count()
    try:
        # Python 3.10+ has int.bit_count() which is faster
        distances = np.array([x.bit_count() for x in xor_result], dtype=np.int32)
    except AttributeError:
        # Fallback for older Python versions
        distances = np.array([bin(x).count('1') for x in xor_result], dtype=np.int32)
    
    return distances


def filter_by_hamming_distance(
    query_hash: str,
    candidate_hashes: List[str],
    max_hamming: int,
    top_n: int = None
) -> List[Tuple[int, int]]:
    """
    Filter candidates by Hamming distance and return top N
    
    Args:
        query_hash: Query hash as hex string
        candidate_hashes: List of candidate hashes
        max_hamming: Maximum Hamming distance threshold
        top_n: Maximum number of results to return (None = all)
        
    Returns:
        List of tuples (index, distance) sorted by distance ascending
    """
    if not candidate_hashes:
        return []
    
    # Compute distances
    distances = batch_hamming_distance(query_hash, candidate_hashes)
    
    # Filter by threshold
    valid_indices = np.where(distances <= max_hamming)[0]
    
    if len(valid_indices) == 0:
        return []
    
    # Get distances for valid candidates
    valid_distances = distances[valid_indices]
    
    # Sort by distance
    sort_indices = np.argsort(valid_distances)
    sorted_indices = valid_indices[sort_indices]
    sorted_distances = valid_distances[sort_indices]
    
    # Limit to top N if specified
    if top_n is not None:
        sorted_indices = sorted_indices[:top_n]
        sorted_distances = sorted_distances[:top_n]
    
    # Return as list of tuples
    return [(int(idx), int(dist)) for idx, dist in zip(sorted_indices, sorted_distances)]


def compute_phash_variants(image: Union[Image.Image, str]) -> Dict[str, str]:
    """
    Compute 3 pHash variants for a card image per PRD.md

    Args:
        image: PIL Image or path to image file

    Returns:
        Dictionary with 'full', 'name', 'collector' pHash values
    """
    if isinstance(image, str):
        img = Image.open(image)
    else:
        img = image

    # Convert to RGB if needed
    if img.mode != 'RGB':
        img = img.convert('RGB')

    # Full image pHash
    full_hash = compute_phash(img)

    # Name region (top 15% of image)
    width, height = img.size
    name_height = int(height * 0.15)
    name_region = img.crop((0, 0, width, name_height))
    name_hash = compute_phash(name_region)

    # Collector region (bottom-left 12% height, 25% width)
    collector_height = int(height * 0.12)
    collector_width = int(width * 0.25)
    collector_region = img.crop((0, height - collector_height, collector_width, height))
    collector_hash = compute_phash(collector_region)

    return {
        'full': full_hash,
        'name': name_hash,
        'collector': collector_hash
    }


def combine_phash_scores(
    full_dist: int,
    name_dist: int,
    collector_dist: int,
    weights: Tuple[float, float, float] = (0.6, 0.3, 0.1)
) -> float:
    """
    Combine multiple pHash distances into a single score
    Following PRD.md scoring: 0.6*full + 0.3*name + 0.1*collector

    Args:
        full_dist: Hamming distance for full image
        name_dist: Hamming distance for name region
        collector_dist: Hamming distance for collector region
        weights: Weight tuple (full_weight, name_weight, collector_weight)

    Returns:
        Combined score (0-1, higher is better)
    """
    # Normalize distances to 0-1 (assuming 64-bit hash, max distance is 64)
    max_dist = 64
    full_score = 1.0 - min(full_dist / max_dist, 1.0)
    name_score = 1.0 - min(name_dist / max_dist, 1.0)
    collector_score = 1.0 - min(collector_dist / max_dist, 1.0)

    # Weight and combine
    combined = (
        weights[0] * full_score +
        weights[1] * name_score +
        weights[2] * collector_score
    )

    return combined

