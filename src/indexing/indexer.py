"""
src/indexing/indexer.py: Main indexing pipeline

- Uses CLIP embedder for 512-dim embeddings
- Skips detection for Scryfall images (already cropped)
- Computes 3 pHash variants (full, name, collector)
- Builds FAISS index for vector search
"""

import numpy as np
import logging
from pathlib import Path
from typing import List, Optional, Dict, Any
from tqdm import tqdm
from PIL import Image
from sqlalchemy.exc import IntegrityError
import struct

logger = logging.getLogger(__name__)

from src.config import (
    SCRYFALL_IMAGES_DIR, INDEXES_DIR,
    CANONICAL_WIDTH, CANONICAL_HEIGHT, INDEXING_BATCH_SIZE,
    COMPOSITE_WEIGHT_FULL, COMPOSITE_WEIGHT_COLLECTOR, COMPOSITE_WEIGHT_NAME
)
from src.database.schema import SessionLocal, init_db, Card, PhashVariant, CompositeEmbedding
from src.database import db as db_ops
from src.database.db import transaction
from src.embeddings.embedder import CLIPEmbedder
from src.embeddings.region_extractor import RegionExtractor
from src.indexing.phash import compute_phash_variants
from src.utils.scryfall import extract_metadata_from_path, detect_variant_type


def serialize_embedding(embedding: np.ndarray) -> bytes:
    """
    Serialize embedding vector to bytes for BLOB storage

    Args:
        embedding: numpy array of float32

    Returns:
        Bytes representation
    """
    return embedding.astype(np.float32).tobytes()


def deserialize_embedding(blob: bytes) -> np.ndarray:
    """
    Deserialize embedding from BLOB storage

    Args:
        blob: Bytes representation

    Returns:
        numpy array of float32
    """
    return np.frombuffer(blob, dtype=np.float32)


class Indexer:
    """Main indexing pipeline for building searchable index of card images."""

    def __init__(
        self,
        set_code: str,
        finish: str = 'nonfoil',
        device: Optional[str] = None,
        checkpoint_path: Optional[Path] = None
    ):
        """
        Initialize indexer

        Args:
            set_code: Set code to index (e.g., 'M21')
            finish: Finish type ('foil', 'nonfoil', 'etched')
            device: Device to run embedder on ('cpu', 'cuda', or None for auto)
            checkpoint_path: Path to fine-tuned CLIP checkpoint (optional)
        """
        self.set_code = set_code.upper()
        self.finish = finish.lower()
        self.device = device  # Store device for collector embedding extraction

        # Initialize database
        init_db()
        self.db = SessionLocal()

        # Initialize CLIP embedder (512-dim)
        logger.info("Initializing CLIP embedder...")
        self.embedder = CLIPEmbedder(device=device, checkpoint_path=checkpoint_path)
        self.embedding_dim = self.embedder.embedding_dim

        # Get set directory
        self.set_dir = SCRYFALL_IMAGES_DIR / self.set_code

        logger.info(f"Indexer initialized for {self.set_code}/{self.finish}")
        logger.debug(f"Embedding dimension: {self.embedding_dim}")

    def find_images(self) -> List[Path]:
        """
        Find all images for this set+finish combination

        Returns:
            List of image paths
        """
        if not self.set_dir.exists():
            raise ValueError(f"Set directory not found: {self.set_dir}")

        # Find all image files
        image_extensions = {'.jpg', '.jpeg', '.png', '.webp'}
        images = []

        # Check for finish-specific subdirectory
        finish_dir = self.set_dir / self.finish
        if finish_dir.exists():
            for ext in image_extensions:
                images.extend(finish_dir.glob(f'*{ext}'))
                images.extend(finish_dir.glob(f'*{ext.upper()}'))
        else:
            # Check main set directory, filter by finish in filename
            for ext in image_extensions:
                all_images = list(self.set_dir.glob(f'*{ext}')) + list(self.set_dir.glob(f'*{ext.upper()}'))
                # Filter by finish in path
                for img in all_images:
                    path_str = str(img).lower()
                    if self.finish == 'foil' and 'foil' in path_str:
                        images.append(img)
                    elif self.finish == 'nonfoil' and 'foil' not in path_str:
                        images.append(img)
                    elif self.finish == 'etched' and 'etched' in path_str:
                        images.append(img)

        logger.info(f"Found {len(images)} images for {self.set_code}/{self.finish}")
        return sorted(images)  # Sort for deterministic ordering

    def normalize_scryfall_image(self, image_path: Path) -> Image.Image:
        """
        Normalize Scryfall image to canonical 363x504 (no padding)

        CRITICAL: This must match the scanned image preprocessing in matcher.py
        Both pipelines must produce identical 363x504 images for CLIP to work.

        Scryfall images are already cropped, so skip detection.

        Args:
            image_path: Path to Scryfall image

        Returns:
            Normalized PIL Image at 363x504
        """
        img = Image.open(image_path)

        # Convert to RGB if needed
        if img.mode != 'RGB':
            img = img.convert('RGB')

        # Resize directly to canonical dimensions (363x504)
        # Scryfall images are already clean - just resize to match scanned preprocessing
        # Use LANCZOS for high-quality downsampling
        return img.resize((CANONICAL_WIDTH, CANONICAL_HEIGHT), Image.Resampling.LANCZOS)

    def _detect_variant_type(self, metadata: Dict[str, Any]) -> str:
        """
        Detect card variant type from metadata

        Args:
            metadata: Card metadata dict

        Returns:
            Variant type string ('normal', 'extended_art', 'showcase', etc.)
        """
        return detect_variant_type(metadata)

    def index_image(self, image_path: Path, force: bool = False) -> Optional[Dict[str, Any]]:
        """
        Index a single image

        Args:
            image_path: Path to image file
            force: If True, re-index even if already indexed

        Returns:
            Dict with card_id, embedding, phash_variants if successful, None otherwise
        """
        try:
            # Extract metadata
            json_path = image_path.with_suffix('.json')
            metadata = extract_metadata_from_path(image_path, json_path if json_path.exists() else None)

            # Check if card already exists
            existing = db_ops.get_card_by_image_path(self.db, str(image_path))

            if existing and not force:
                logger.debug(f"Skipping already indexed: {image_path.name}")
                return None

            # Normalize image (Scryfall images already cropped)
            normalized_img = self.normalize_scryfall_image(image_path)

            # Compute 3 pHash variants (full, name, collector)
            phash_variants = compute_phash_variants(normalized_img)

            # Extract regions for composite embedding
            regions = RegionExtractor.extract_all_regions(normalized_img)

            # Generate composite embedding with pre-weighted regions
            # This is the KEY change that fixes the FAISS search failure!
            embedding_result = self.embedder.get_composite_embedding(
                full_image=regions['full'],
                regions={'collector': regions['collector'], 'name': regions['name']},
                weights={
                    'full': COMPOSITE_WEIGHT_FULL,
                    'collector': COMPOSITE_WEIGHT_COLLECTOR,
                    'name': COMPOSITE_WEIGHT_NAME
                }
            )

            # Serialize embeddings for BLOB storage
            composite_blob = serialize_embedding(embedding_result['composite'])
            full_blob = serialize_embedding(embedding_result['full'])
            collector_blob = serialize_embedding(embedding_result['collector'])
            name_blob = serialize_embedding(embedding_result['name'])

            if existing and force:
                # Update existing card
                logger.debug(f"Updating existing card (force=True): {image_path.name}")
                existing.name = metadata.get('name', image_path.stem)
                existing.set_code = metadata.get('set_code', self.set_code)
                existing.collector_number = metadata.get('collector_number')
                existing.finish = self.finish
                existing.variant_type = self._detect_variant_type(metadata)

                # Update composite embedding
                if existing.composite_embedding:
                    existing.composite_embedding.embedding = composite_blob
                    existing.composite_embedding.full_embedding = full_blob
                    existing.composite_embedding.collector_embedding = collector_blob
                    existing.composite_embedding.name_embedding = name_blob
                    existing.composite_embedding.weight_full = COMPOSITE_WEIGHT_FULL
                    existing.composite_embedding.weight_collector = COMPOSITE_WEIGHT_COLLECTOR
                    existing.composite_embedding.weight_name = COMPOSITE_WEIGHT_NAME
                else:
                    new_composite = CompositeEmbedding(
                        card_id=existing.id,
                        embedding=composite_blob,
                        full_embedding=full_blob,
                        collector_embedding=collector_blob,
                        name_embedding=name_blob,
                        weight_full=COMPOSITE_WEIGHT_FULL,
                        weight_collector=COMPOSITE_WEIGHT_COLLECTOR,
                        weight_name=COMPOSITE_WEIGHT_NAME
                    )
                    self.db.add(new_composite)

                # Update pHash variants - delete old and create new
                for pv in existing.phash_variants:
                    self.db.delete(pv)

                for variant_type, phash_hex in phash_variants.items():
                    pv = PhashVariant(
                        card_id=existing.id,
                        variant_type=variant_type,
                        phash=phash_hex  # Store as hex string for SQLite compatibility
                    )
                    self.db.add(pv)

                self.db.flush()

                return {
                    'card_id': existing.id,
                    'embedding': embedding_result['composite'],
                    'phash_variants': phash_variants
                }
            else:
                # Create new card record
                # Generate unique card ID from filename (includes scryfall_id) + finish
                # This ensures uniqueness even with duplicate files
                collector_num = metadata.get('collector_number', 'unknown')
                card_name = metadata.get('name', 'unknown')
                # Extract scryfall_id from filename (the hash part)
                scryfall_id = metadata.get('scryfall_id', image_path.stem.split('_')[-1] if '_' in image_path.stem else image_path.stem)

                # Format: {collector}_{name}_{scryfall_id}_{finish}
                card_id = f"{collector_num}_{card_name.replace(' ', '_')}_{scryfall_id}_{self.finish}"

                # Check if card with this ID already exists (safety check)
                existing_by_id = self.db.query(Card).filter(Card.id == card_id).first()
                if existing_by_id:
                    logger.debug(f"Card with ID {card_id} already exists, skipping: {image_path.name}")
                    return None

                card = Card(
                    id=card_id,
                    scryfall_id=metadata.get('scryfall_id'),
                    name=metadata.get('name', image_path.stem),
                    set_code=metadata.get('set_code', self.set_code),
                    collector_number=metadata.get('collector_number'),
                    finish=self.finish,
                    variant_type=self._detect_variant_type(metadata),
                    image_path=str(image_path)
                )
                self.db.add(card)
                self.db.flush()  # Flush to get card ID

                # Create composite embedding record (single record with all components)
                composite_embedding_record = CompositeEmbedding(
                    card_id=card.id,
                    embedding=composite_blob,
                    full_embedding=full_blob,
                    collector_embedding=collector_blob,
                    name_embedding=name_blob,
                    weight_full=COMPOSITE_WEIGHT_FULL,
                    weight_collector=COMPOSITE_WEIGHT_COLLECTOR,
                    weight_name=COMPOSITE_WEIGHT_NAME
                )
                self.db.add(composite_embedding_record)

                # Create pHash variant records (3 per card)
                for variant_type, phash_hex in phash_variants.items():
                    pv = PhashVariant(
                        card_id=card.id,
                        variant_type=variant_type,
                        phash=phash_hex  # Store as hex string for SQLite compatibility
                    )
                    self.db.add(pv)

                return {
                    'card_id': card.id,
                    'embedding': embedding_result['composite'],
                    'phash_variants': phash_variants
                }

        except Exception as e:
            logger.error(f"Error indexing {image_path}: {e}", exc_info=True)
            return None

    def build_index(
        self,
        batch_size: int = INDEXING_BATCH_SIZE,
        force: bool = False,
        resume_from: int = 0
    ):
        """
        Build complete index for set+finish combination

        Args:
            batch_size: Number of images to process per batch before committing
            force: If True, re-index already indexed cards
            resume_from: Resume indexing from this image index (for checkpoint/resume)
        """
        logger.info(f"Building index for {self.set_code}/{self.finish}")
        logger.info(f"Batch size: {batch_size}, Force: {force}, Resume from: {resume_from}")

        # Find all images
        images = self.find_images()
        if not images:
            logger.warning(f"No images found for {self.set_code}/{self.finish}")
            return

        # Skip already processed images if resuming
        if resume_from > 0:
            images = images[resume_from:]
            logger.info(f"Resuming from image {resume_from}, {len(images)} images remaining")

        # Process images in batches
        all_embeddings = []
        all_card_ids = []
        failed_images = []

        total_images = len(images)
        num_batches = (total_images + batch_size - 1) // batch_size

        logger.info(f"Processing {total_images} images in {num_batches} batches...")

        for batch_idx in range(num_batches):
            batch_start = batch_idx * batch_size
            batch_end = min(batch_start + batch_size, total_images)
            batch_images = images[batch_start:batch_end]

            logger.info(f"Batch {batch_idx + 1}/{num_batches} (images {batch_start}-{batch_end-1})")

            # Process batch with transaction
            batch_embeddings = []
            batch_card_ids = []
            seen_card_ids_in_batch = set()  # Track duplicates within batch

            try:
                with transaction(self.db):
                    for image_path in tqdm(batch_images, desc=f"Batch {batch_idx + 1}"):
                        result = self.index_image(image_path, force=force)
                        if result:
                            # Skip if we've already processed this card_id in this batch
                            if result['card_id'] in seen_card_ids_in_batch:
                                logger.debug(f"Skipping duplicate in batch: {result['card_id']} ({image_path.name})")
                                continue
                            seen_card_ids_in_batch.add(result['card_id'])
                            batch_card_ids.append(result['card_id'])
                            batch_embeddings.append(result['embedding'])

                # If database transaction succeeded, accumulate embeddings
                if batch_embeddings:
                    all_embeddings.extend(batch_embeddings)
                    all_card_ids.extend(batch_card_ids)

                    logger.info(f"Batch {batch_idx + 1} completed: {len(batch_card_ids)} cards indexed")
                else:
                    logger.info(f"Batch {batch_idx + 1} had no new cards")

            except Exception as e:
                # Transaction automatically rolled back
                logger.error(f"Batch {batch_idx + 1} failed: {e}")
                logger.warning("Transaction rolled back, batch not indexed")
                # Ensure session is rolled back and ready for next transaction
                try:
                    self.db.rollback()
                except:
                    pass
                # Mark all images in this batch as failed
                failed_images.extend([str(img) for img in batch_images])
                # Continue with next batch

        if not all_embeddings:
            logger.warning("No images successfully indexed")
            if failed_images:
                logger.warning(f"Failed images: {len(failed_images)}")
            return

        logger.info(f"Index built successfully: {len(all_embeddings)} cards indexed")
        if failed_images:
            logger.warning(f"{len(failed_images)} images failed to index")

        # Return embeddings and card IDs for ANN index building
        return {
            'embeddings': np.array(all_embeddings, dtype=np.float32),
            'card_ids': all_card_ids,
            'set_code': self.set_code,
            'finish': self.finish
        }

    def close(self):
        """Close database connection"""
        self.db.close()


def main():
    """
    CLI entry point for indexing
    """
    import argparse

    parser = argparse.ArgumentParser(description='Index Scryfall card images')
    parser.add_argument('set_code', help='Set code to index (e.g., M21)')
    parser.add_argument('--finish', default='nonfoil', choices=['foil', 'nonfoil', 'etched'],
                        help='Finish type to index')
    parser.add_argument('--batch-size', type=int, default=INDEXING_BATCH_SIZE,
                        help='Batch size for processing')
    parser.add_argument('--force', action='store_true',
                        help='Re-index already indexed cards')
    parser.add_argument('--resume-from', type=int, default=0,
                        help='Resume from image index')
    parser.add_argument('--device', default=None,
                        help='Device to use (cpu/cuda)')
    parser.add_argument('--checkpoint', type=str, default=None,
                        help='Path to fine-tuned CLIP checkpoint')

    args = parser.parse_args()

    # Create indexer
    indexer = Indexer(
        set_code=args.set_code,
        finish=args.finish,
        device=args.device,
        checkpoint_path=Path(args.checkpoint) if args.checkpoint else None
    )

    try:
        # Build index
        result = indexer.build_index(
            batch_size=args.batch_size,
            force=args.force,
            resume_from=args.resume_from
        )

        if result:
            print(f"\nIndexing complete!")
            print(f"Total embeddings: {len(result['embeddings'])}")
            print(f"Embedding dimension: {result['embeddings'].shape[1]}")
            print(f"\nNext step: Build ANN index using:")
            print(f"  python -m src.ann.faiss_index --set {args.set_code} --finish {args.finish}")

    finally:
        indexer.close()


if __name__ == '__main__':
    main()
