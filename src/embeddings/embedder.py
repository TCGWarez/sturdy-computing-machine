"""
src/embeddings/embedder.py: CLIP embedding baseline
Uses CLIP ViT-B/32 for 512-dim image embeddings with composite embedding support.
"""

import torch
import numpy as np
import open_clip
from PIL import Image
from pathlib import Path
from typing import Union, List, Optional
from concurrent.futures import ThreadPoolExecutor
import logging

from src.embeddings.base_embedder import BaseEmbedder

logger = logging.getLogger(__name__)

DEFAULT_CLIP_MODEL = 'ViT-B-32'
DEFAULT_PRETRAINED = 'openai'
DEFAULT_EMBEDDING_DIM = 512


class CLIPEmbedder(BaseEmbedder):
    """
    CLIP-based image embedder for card recognition
    Provides abstract interface for easy model swapping
    """

    def __init__(
        self,
        model_name: str = DEFAULT_CLIP_MODEL,
        pretrained: str = DEFAULT_PRETRAINED,
        device: Optional[str] = None,
        checkpoint_path: Optional[Path] = None
    ):
        """
        Initialize CLIP embedder

        Args:
            model_name: CLIP model name (default ViT-B-32)
            pretrained: Pretrained weights source (default 'openai')
            device: Device to run on ('cpu', 'cuda', or None for auto)
            checkpoint_path: Optional path to fine-tuned checkpoint
        """
        # Initialize base class (handles device auto-detection)
        super().__init__(device=device)

        logger.info(f"Initializing CLIP embedder on {self.device}")

        # Load model and preprocessing
        if checkpoint_path and checkpoint_path.exists():
            logger.info(f"Loading fine-tuned checkpoint from {checkpoint_path}")
            # Load fine-tuned model
            checkpoint = torch.load(checkpoint_path, map_location=device)
            self.model, _, self.preprocess = open_clip.create_model_and_transforms(
                model_name,
                pretrained=None
            )
            self.model.load_state_dict(checkpoint['model_state_dict'])
        else:
            # Load pretrained model
            logger.info(f"Loading pretrained CLIP model: {model_name}/{pretrained}")
            self.model, _, self.preprocess = open_clip.create_model_and_transforms(
                model_name,
                pretrained=pretrained
            )

        # Move to device and set to eval mode
        self.model = self.model.to(self.device)
        self.model.eval()

        # Store model info
        self.model_name = model_name
        self.embedding_dim = DEFAULT_EMBEDDING_DIM

        logger.info(f"CLIP embedder ready: {model_name} ({self.embedding_dim}-dim)")

    def get_image_embedding(self, image: Union[Image.Image, np.ndarray, str, Path]) -> np.ndarray:
        """
        Extract embedding for a single image

        Args:
            image: PIL Image, numpy array, or path to image file

        Returns:
            Embedding as numpy array of shape (embedding_dim,)
        """
        # Load image if path provided
        if isinstance(image, (str, Path)):
            image = Image.open(image)
        elif isinstance(image, np.ndarray):
            image = Image.fromarray(image)

        # Ensure RGB
        if image.mode != 'RGB':
            image = image.convert('RGB')

        # Preprocess
        image_tensor = self.preprocess(image).unsqueeze(0).to(self.device)

        # Extract features
        with torch.no_grad():
            image_features = self.model.encode_image(image_tensor)
            # Normalize features (CLIP standard)
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)

        # Convert to numpy and squeeze batch dimension
        embedding = image_features.cpu().numpy().squeeze(0)

        return embedding.astype(np.float32)

    def get_batch_embeddings(
        self,
        images: List[Union[Image.Image, np.ndarray, str, Path]],
        batch_size: int = 32
    ) -> np.ndarray:
        """
        Extract embeddings for multiple images

        Args:
            images: List of images (PIL, numpy, or paths)
            batch_size: Batch size for processing

        Returns:
            Array of shape (n_images, embedding_dim)
        """
        embeddings = []

        for i in range(0, len(images), batch_size):
            batch = images[i:i+batch_size]
            batch_tensors = []

            for img in batch:
                # Load image if needed
                if isinstance(img, (str, Path)):
                    img = Image.open(img)
                elif isinstance(img, np.ndarray):
                    img = Image.fromarray(img)

                # Ensure RGB
                if img.mode != 'RGB':
                    img = img.convert('RGB')

                # Preprocess
                img_tensor = self.preprocess(img)
                batch_tensors.append(img_tensor)

            # Stack into batch
            batch_tensor = torch.stack(batch_tensors).to(self.device)

            # Extract features
            with torch.no_grad():
                batch_features = self.model.encode_image(batch_tensor)
                # Normalize
                batch_features = batch_features / batch_features.norm(dim=-1, keepdim=True)

            # Convert to numpy
            batch_embeddings = batch_features.cpu().numpy()
            embeddings.append(batch_embeddings)

        # Concatenate all batches
        all_embeddings = np.vstack(embeddings)
        return all_embeddings.astype(np.float32)

    def get_batch_embeddings_parallel(
        self,
        images: List[Union[Image.Image, np.ndarray, str, Path]],
        batch_size: int = 32,
        max_workers: int = 4
    ) -> np.ndarray:
        """
        Extract embeddings for multiple images with parallel I/O loading.

        Uses ThreadPoolExecutor to parallelize image loading and preprocessing,
        which is GIL-friendly since it's I/O bound. Provides 2-3x speedup for
        batch processing from disk.

        Args:
            images: List of images (PIL, numpy, or paths)
            batch_size: Batch size for GPU/CPU inference
            max_workers: Number of parallel workers for image loading

        Returns:
            Array of shape (n_images, embedding_dim)
        """
        def load_and_preprocess(img):
            """Load image and apply CLIP preprocessing."""
            # Load image if path
            if isinstance(img, (str, Path)):
                img = Image.open(img)
            elif isinstance(img, np.ndarray):
                img = Image.fromarray(img)

            # Ensure RGB
            if img.mode != 'RGB':
                img = img.convert('RGB')

            # Apply CLIP preprocessing
            return self.preprocess(img)

        # Parallel image loading and preprocessing
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            preprocessed_tensors = list(executor.map(load_and_preprocess, images))

        # Batch inference
        embeddings = []
        for i in range(0, len(preprocessed_tensors), batch_size):
            batch_tensors = preprocessed_tensors[i:i+batch_size]

            # Stack into batch and move to device
            batch_tensor = torch.stack(batch_tensors).to(self.device)

            # Extract features
            with torch.no_grad():
                batch_features = self.model.encode_image(batch_tensor)
                batch_features = batch_features / batch_features.norm(dim=-1, keepdim=True)

            embeddings.append(batch_features.cpu().numpy())

        all_embeddings = np.vstack(embeddings)
        return all_embeddings.astype(np.float32)

    def save_checkpoint(self, checkpoint_path: Path):
        """
        Save model checkpoint (for fine-tuned models)

        Args:
            checkpoint_path: Path to save checkpoint
        """
        checkpoint = {
            'model_state_dict': self.model.state_dict(),
            'model_name': self.model_name,
            'embedding_dim': self.embedding_dim
        }
        torch.save(checkpoint, checkpoint_path)
        logger.info(f"Saved checkpoint to {checkpoint_path}")

    def load_checkpoint(self, checkpoint_path: Path):
        """
        Load model checkpoint

        Args:
            checkpoint_path: Path to checkpoint file
        """
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.eval()
        logger.info(f"Loaded checkpoint from {checkpoint_path}")


# Convenience functions for backward compatibility and easy swapping
_global_embedder = None


def get_embedder(
    device: Optional[str] = None,
    checkpoint_path: Optional[Path] = None,
    force_reload: bool = False
) -> CLIPEmbedder:
    """
    Get or create global embedder instance

    Args:
        device: Device to use
        checkpoint_path: Optional fine-tuned checkpoint
        force_reload: Force creation of new instance

    Returns:
        CLIPEmbedder instance
    """
    global _global_embedder

    if _global_embedder is None or force_reload:
        _global_embedder = CLIPEmbedder(
            device=device,
            checkpoint_path=checkpoint_path
        )

    return _global_embedder


def get_image_embedding(
    image: Union[Image.Image, np.ndarray, str, Path],
    device: Optional[str] = None
) -> np.ndarray:
    """
    Convenience function to extract embedding for a single image

    Args:
        image: Input image
        device: Device to use

    Returns:
        Embedding vector
    """
    embedder = get_embedder(device=device)
    return embedder.get_image_embedding(image)


def get_batch_embeddings(
    images: List[Union[Image.Image, np.ndarray, str, Path]],
    batch_size: int = 32,
    device: Optional[str] = None,
    parallel: bool = False,
    max_workers: int = 4
) -> np.ndarray:
    """
    Convenience function to extract embeddings for multiple images

    Args:
        images: List of images
        batch_size: Batch size
        device: Device to use
        parallel: If True, use parallel image loading (faster for disk I/O)
        max_workers: Number of parallel workers when parallel=True

    Returns:
        Array of embeddings
    """
    embedder = get_embedder(device=device)
    if parallel:
        return embedder.get_batch_embeddings_parallel(images, batch_size, max_workers)
    return embedder.get_batch_embeddings(images, batch_size)


if __name__ == "__main__":
    # Simple test
    import sys
    if len(sys.argv) > 1:
        test_image = sys.argv[1]
        embedding = get_image_embedding(test_image)
        print(f"Extracted embedding shape: {embedding.shape}")
        print(f"Embedding norm: {np.linalg.norm(embedding):.4f}")
        print(f"First 10 values: {embedding[:10]}")
    else:
        print("Usage: python embedder.py <image_path>")