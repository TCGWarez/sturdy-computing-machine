"""
Abstract base class for all embedding models

Provides a unified interface for CLIP and future custom vision models.
Supports composite embedding extraction with weighted region combinations.
"""

from abc import ABC, abstractmethod
from typing import Union, List, Dict, Optional
from pathlib import Path
import numpy as np
from PIL import Image

from src.utils.device import resolve_device


class BaseEmbedder(ABC):
    """
    Abstract base class for all embedding models

    Supports:
    - Single image embeddings
    - Batch image embeddings
    - Composite embeddings with weighted region combination
    """

    def __init__(self, device: Optional[str] = None):
        """
        Initialize embedder

        Args:
            device: Device to run model on ('cpu' or 'cuda'). Auto-detects if None.
        """
        self.device = resolve_device(device)
        self.embedding_dim = None  # Set by subclass (e.g., 512 for CLIP)

    @abstractmethod
    def get_image_embedding(self, image: Union[Image.Image, np.ndarray, str, Path]) -> np.ndarray:
        """
        Extract single embedding from image

        Args:
            image: PIL Image, numpy array, or path to image file

        Returns:
            Normalized embedding of shape (embedding_dim,)
        """
        pass

    @abstractmethod
    def get_batch_embeddings(self, images: List, batch_size: int = 32) -> np.ndarray:
        """
        Extract embeddings for multiple images

        Args:
            images: List of PIL Images, numpy arrays, or paths
            batch_size: Number of images to process at once

        Returns:
            Array of shape (n_images, embedding_dim)
        """
        pass

    def get_composite_embedding(
        self,
        full_image: Image.Image,
        regions: Optional[Dict[str, Image.Image]] = None,
        weights: Optional[Dict[str, float]] = None
    ) -> Dict[str, np.ndarray]:
        """
        Extract composite embedding from full image and regions

        This is the KEY method that fixes the retrieval-scoring mismatch.
        It pre-computes a weighted composite embedding that combines:
        - Full card appearance (45%)
        - Collector region with set code/number (30%)
        - Name region for variant discrimination (25%)

        The composite embedding is used for FAISS retrieval, ensuring that
        retrieval and final scoring are perfectly aligned.

        Args:
            full_image: Full card image (PIL Image)
            regions: Dict with 'collector' and 'name' cropped regions (PIL Images)
            weights: Dict with 'full', 'collector', 'name' weights
                    Defaults to 0.45, 0.30, 0.25

        Returns:
            Dict with keys:
            - 'composite': Pre-weighted composite embedding (512-dim)
            - 'full': Full card embedding (512-dim)
            - 'collector': Collector region embedding (512-dim) if regions provided
            - 'name': Name region embedding (512-dim) if regions provided
        """
        # Default weights: optimized for MTG card recognition
        if weights is None:
            weights = {
                'full': 0.45,       # Global appearance, art, borders
                'collector': 0.30,  # Most discriminative (set + collector #)
                'name': 0.25        # Variant discrimination
            }

        # Extract and normalize full image embedding
        full_emb = self.get_image_embedding(full_image)
        full_emb = full_emb / np.linalg.norm(full_emb)

        result = {
            'full': full_emb,
            'collector': None,
            'name': None,
            'composite': None
        }

        # Extract region embeddings if regions provided
        if regions:
            if 'collector' in regions and regions['collector'] is not None:
                coll_emb = self.get_image_embedding(regions['collector'])
                coll_emb = coll_emb / np.linalg.norm(coll_emb)
                result['collector'] = coll_emb

            if 'name' in regions and regions['name'] is not None:
                name_emb = self.get_image_embedding(regions['name'])
                name_emb = name_emb / np.linalg.norm(name_emb)
                result['name'] = name_emb

        # Compute weighted composite
        if result['collector'] is not None and result['name'] is not None:
            # Full composite with all three regions
            composite = (
                weights['full'] * result['full'] +
                weights['collector'] * result['collector'] +
                weights['name'] * result['name']
            )
            # Re-normalize composite
            composite = composite / np.linalg.norm(composite)
            result['composite'] = composite
        else:
            # Fallback: just use full embedding if regions not provided
            result['composite'] = result['full']

        return result

    def get_composite_embedding_batched(
        self,
        full_image: Image.Image,
        regions: Optional[Dict[str, Image.Image]] = None,
        weights: Optional[Dict[str, float]] = None
    ) -> Dict[str, np.ndarray]:
        """
        Extract composite embedding with batched CLIP inference.

        OPTIMIZED VERSION: Batches all 3 regions into a single CLIP forward pass,
        reducing GPU/CPU transfer overhead and kernel launch overhead.

        This provides ~1.3-1.5x speedup over the non-batched version.

        Args:
            full_image: Full card image (PIL Image)
            regions: Dict with 'collector' and 'name' cropped regions (PIL Images)
            weights: Dict with 'full', 'collector', 'name' weights

        Returns:
            Same as get_composite_embedding()
        """
        # Default weights
        if weights is None:
            weights = {
                'full': 0.45,
                'collector': 0.30,
                'name': 0.25
            }

        # Build list of images to batch
        images_to_embed = [full_image]
        has_collector = regions and 'collector' in regions and regions['collector'] is not None
        has_name = regions and 'name' in regions and regions['name'] is not None

        if has_collector:
            images_to_embed.append(regions['collector'])
        if has_name:
            images_to_embed.append(regions['name'])

        # Single batched call for all images
        all_embeddings = self.get_batch_embeddings(images_to_embed, batch_size=len(images_to_embed))

        # Extract individual embeddings (already normalized by get_batch_embeddings)
        full_emb = all_embeddings[0]
        full_emb = full_emb / np.linalg.norm(full_emb)  # Re-normalize just in case

        result = {
            'full': full_emb,
            'collector': None,
            'name': None,
            'composite': None
        }

        idx = 1
        if has_collector:
            coll_emb = all_embeddings[idx]
            result['collector'] = coll_emb / np.linalg.norm(coll_emb)
            idx += 1

        if has_name:
            name_emb = all_embeddings[idx]
            result['name'] = name_emb / np.linalg.norm(name_emb)

        # Compute weighted composite
        if result['collector'] is not None and result['name'] is not None:
            composite = (
                weights['full'] * result['full'] +
                weights['collector'] * result['collector'] +
                weights['name'] * result['name']
            )
            composite = composite / np.linalg.norm(composite)
            result['composite'] = composite
        else:
            result['composite'] = result['full']

        return result
