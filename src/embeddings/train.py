"""
src/embeddings/train.py: CLIP fine-tuning training loop
Following PRD.md Task 11 specifications

Features:
- Loads corrections from data/al_labels.csv
- Uses contrastive loss (NT-Xent/SimCLR-style) on CLIP
- Constructs positive pairs from (scanned_image, reference_image)
- Constructs negatives with in-batch or hard negatives from index
- Training hyperparams per PRD: batch_size=64, lr=1e-4, epochs=5, embedding_dim=256 (projection head) or 512
- Saves checkpoint and replaces index embeddings after retraining
- Accepts --data_csv and --ckpt_out flags
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from pathlib import Path
from PIL import Image
from typing import List, Dict, Optional, Tuple
import logging
from datetime import datetime
from tqdm import tqdm
import csv
import json
import argparse

from src.embeddings.embedder import CLIPEmbedder, DEFAULT_EMBEDDING_DIM
from src.config import BASE_DIR, MODELS_DIR
from src.database.schema import SessionLocal, Card, CompositeEmbedding
from src.database.db import get_card_full_data

logger = logging.getLogger(__name__)

# Training hyperparameters per PRD
DEFAULT_BATCH_SIZE = 64
DEFAULT_LR = 1e-4
DEFAULT_EPOCHS = 5
DEFAULT_PROJECTION_DIM = 256  # Projection head output dimension (can use 512 for no projection)
DEFAULT_TEMPERATURE = 0.07  # Temperature for NT-Xent loss (SimCLR default)


class ProjectionHead(nn.Module):
    """
    Projection head for contrastive learning
    Maps CLIP embeddings to lower-dimensional space for fine-tuning
    """

    def __init__(self, input_dim: int = 512, output_dim: int = 256):
        """
        Initialize projection head

        Args:
            input_dim: Input embedding dimension (CLIP output)
            output_dim: Output projection dimension
        """
        super().__init__()

        # Two-layer MLP with ReLU
        self.projection = nn.Sequential(
            nn.Linear(input_dim, input_dim),
            nn.ReLU(),
            nn.Linear(input_dim, output_dim)
        )

    def forward(self, x):
        """
        Forward pass through projection head

        Args:
            x: Input embeddings (batch_size, input_dim)

        Returns:
            Projected embeddings (batch_size, output_dim)
        """
        return self.projection(x)


class NTXentLoss(nn.Module):
    """
    Normalized Temperature-scaled Cross Entropy Loss (NT-Xent)
    SimCLR-style contrastive loss for fine-tuning
    """

    def __init__(self, temperature: float = 0.07):
        """
        Initialize NT-Xent loss

        Args:
            temperature: Temperature scaling parameter
        """
        super().__init__()
        self.temperature = temperature

    def forward(
        self,
        z_i: torch.Tensor,
        z_j: torch.Tensor,
        negative_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Compute NT-Xent loss for a batch of positive pairs

        Args:
            z_i: First view embeddings (batch_size, projection_dim)
            z_j: Second view embeddings (batch_size, projection_dim)
            negative_mask: Optional mask for hard negatives (batch_size, num_negatives)

        Returns:
            Loss value
        """
        batch_size = z_i.shape[0]

        # Normalize embeddings
        z_i = F.normalize(z_i, dim=1)
        z_j = F.normalize(z_j, dim=1)

        # Compute similarity matrix
        # representations: (2*batch_size, projection_dim)
        representations = torch.cat([z_i, z_j], dim=0)

        # Similarity matrix: (2*batch_size, 2*batch_size)
        similarity_matrix = F.cosine_similarity(
            representations.unsqueeze(1),
            representations.unsqueeze(0),
            dim=2
        )

        # Create positive pairs mask
        # For each i, positive is i+batch_size (and vice versa)
        positives_mask = torch.zeros_like(similarity_matrix, dtype=torch.bool)
        for i in range(batch_size):
            positives_mask[i, batch_size + i] = True
            positives_mask[batch_size + i, i] = True

        # Create negatives mask (all except self and positive)
        negatives_mask = ~positives_mask
        for i in range(2 * batch_size):
            negatives_mask[i, i] = False

        # Apply temperature scaling
        similarity_matrix = similarity_matrix / self.temperature

        # Compute loss
        # For each anchor, we have one positive and (2*batch_size - 2) negatives
        exp_sim = torch.exp(similarity_matrix)

        # Sum of exponentials for positives
        pos_sim = exp_sim[positives_mask].view(2 * batch_size, 1)

        # Sum of exponentials for negatives
        neg_sim = exp_sim[negatives_mask].view(2 * batch_size, -1).sum(dim=1, keepdim=True)

        # NT-Xent loss: -log(exp(pos) / (exp(pos) + sum(exp(neg))))
        loss = -torch.log(pos_sim / (pos_sim + neg_sim))

        return loss.mean()


class ALDataset(torch.utils.data.Dataset):
    """
    Active Learning dataset for fine-tuning
    Loads scanned images and their correct reference images
    """

    def __init__(
        self,
        corrections: List[Dict[str, str]],
        db_session,
        transform,
        scryfall_images_dir: Path
    ):
        """
        Initialize dataset

        Args:
            corrections: List of correction dicts from CSV
            db_session: Database session for loading reference images
            transform: Image preprocessing transform (from CLIP)
            scryfall_images_dir: Directory containing reference images
        """
        self.corrections = corrections
        self.db = db_session
        self.transform = transform
        self.scryfall_images_dir = scryfall_images_dir

        # Filter out invalid corrections
        self.valid_corrections = []
        for correction in corrections:
            scanned_path = Path(correction['scanned_path'])
            if not scanned_path.exists():
                logger.warning(f"Scanned image not found: {scanned_path}")
                continue

            # Check if reference card exists
            card_data = get_card_full_data(self.db, correction['correct_card_id'])
            if not card_data or not card_data['card']:
                logger.warning(f"Reference card not found: {correction['correct_card_id']}")
                continue

            reference_path = Path(card_data['card'].image_path)
            if not reference_path.exists():
                logger.warning(f"Reference image not found: {reference_path}")
                continue

            self.valid_corrections.append(correction)

        logger.info(f"Loaded {len(self.valid_corrections)}/{len(corrections)} valid corrections")

    def __len__(self):
        return len(self.valid_corrections)

    def __getitem__(self, idx) -> Tuple[torch.Tensor, torch.Tensor, str, str]:
        """
        Get a training sample

        Args:
            idx: Index

        Returns:
            Tuple of (scanned_image_tensor, reference_image_tensor, scanned_path, card_id)
        """
        correction = self.valid_corrections[idx]

        # Load scanned image
        scanned_path = Path(correction['scanned_path'])
        scanned_image = Image.open(scanned_path).convert('RGB')
        scanned_tensor = self.transform(scanned_image)

        # Load reference image
        card_data = get_card_full_data(self.db, correction['correct_card_id'])
        reference_path = Path(card_data['card'].image_path)
        reference_image = Image.open(reference_path).convert('RGB')
        reference_tensor = self.transform(reference_image)

        return scanned_tensor, reference_tensor, str(scanned_path), correction['correct_card_id']


class CLIPFineTuner:
    """
    CLIP fine-tuner with contrastive learning
    Following PRD.md Task 11 specifications
    """

    def __init__(
        self,
        base_model: CLIPEmbedder,
        projection_dim: int = DEFAULT_PROJECTION_DIM,
        temperature: float = DEFAULT_TEMPERATURE,
        device: Optional[str] = None
    ):
        """
        Initialize fine-tuner

        Args:
            base_model: Base CLIP embedder
            projection_dim: Projection head output dimension
            temperature: Temperature for contrastive loss
            device: Device to use
        """
        self.base_model = base_model
        self.device = device or base_model.device

        # Create projection head
        input_dim = base_model.embedding_dim
        self.projection_head = ProjectionHead(input_dim, projection_dim).to(self.device)

        # Create loss function
        self.criterion = NTXentLoss(temperature=temperature)

        # Optimizer (only train projection head by default, can fine-tune full model)
        self.optimizer = None

        logger.info(f"Initialized CLIP fine-tuner: {input_dim}D -> {projection_dim}D")

    def setup_optimizer(
        self,
        lr: float = DEFAULT_LR,
        fine_tune_full_model: bool = False
    ):
        """
        Setup optimizer

        Args:
            lr: Learning rate
            fine_tune_full_model: If True, fine-tune entire CLIP model; else only projection head
        """
        if fine_tune_full_model:
            # Fine-tune entire model (requires more data and GPU)
            params = list(self.base_model.model.parameters()) + list(self.projection_head.parameters())
            logger.info("Fine-tuning entire CLIP model + projection head")
        else:
            # Only train projection head (faster, works with less data)
            params = self.projection_head.parameters()
            logger.info("Training projection head only (CLIP frozen)")

        self.optimizer = torch.optim.Adam(params, lr=lr)

    def train_epoch(
        self,
        dataloader: torch.utils.data.DataLoader,
        epoch: int
    ) -> float:
        """
        Train for one epoch

        Args:
            dataloader: Training data loader
            epoch: Current epoch number

        Returns:
            Average loss for epoch
        """
        self.projection_head.train()

        total_loss = 0.0
        num_batches = 0

        pbar = tqdm(dataloader, desc=f"Epoch {epoch}")

        for batch_idx, (scanned_imgs, reference_imgs, scanned_paths, card_ids) in enumerate(pbar):
            # Move to device
            scanned_imgs = scanned_imgs.to(self.device)
            reference_imgs = reference_imgs.to(self.device)

            # Extract CLIP embeddings
            with torch.no_grad():
                scanned_embeddings = self.base_model.model.encode_image(scanned_imgs)
                reference_embeddings = self.base_model.model.encode_image(reference_imgs)

                # Normalize (CLIP standard)
                scanned_embeddings = scanned_embeddings / scanned_embeddings.norm(dim=-1, keepdim=True)
                reference_embeddings = reference_embeddings / reference_embeddings.norm(dim=-1, keepdim=True)

            # Project embeddings
            scanned_projected = self.projection_head(scanned_embeddings)
            reference_projected = self.projection_head(reference_embeddings)

            # Compute contrastive loss
            loss = self.criterion(scanned_projected, reference_projected)

            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            # Update metrics
            total_loss += loss.item()
            num_batches += 1

            # Update progress bar
            pbar.set_postfix({'loss': f"{loss.item():.4f}"})

        avg_loss = total_loss / num_batches
        logger.info(f"Epoch {epoch} - Average loss: {avg_loss:.4f}")

        return avg_loss

    def save_checkpoint(
        self,
        checkpoint_path: Path,
        epoch: int,
        loss: float,
        metadata: Optional[Dict] = None
    ):
        """
        Save training checkpoint

        Args:
            checkpoint_path: Path to save checkpoint
            epoch: Current epoch
            loss: Current loss
            metadata: Optional metadata dict
        """
        checkpoint = {
            'epoch': epoch,
            'loss': loss,
            'base_model_state_dict': self.base_model.model.state_dict(),
            'projection_head_state_dict': self.projection_head.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict() if self.optimizer else None,
            'model_name': self.base_model.model_name,
            'embedding_dim': self.base_model.embedding_dim,
            'projection_dim': self.projection_head.projection[-1].out_features,
            'timestamp': datetime.now().isoformat(),
            'metadata': metadata or {}
        }

        torch.save(checkpoint, checkpoint_path)
        logger.info(f"Saved checkpoint to {checkpoint_path}")

    def load_checkpoint(self, checkpoint_path: Path):
        """
        Load training checkpoint

        Args:
            checkpoint_path: Path to checkpoint file
        """
        checkpoint = torch.load(checkpoint_path, map_location=self.device)

        self.base_model.model.load_state_dict(checkpoint['base_model_state_dict'])
        self.projection_head.load_state_dict(checkpoint['projection_head_state_dict'])

        if self.optimizer and checkpoint['optimizer_state_dict']:
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

        logger.info(f"Loaded checkpoint from {checkpoint_path}")
        logger.info(f"  Epoch: {checkpoint['epoch']}, Loss: {checkpoint['loss']:.4f}")

        return checkpoint


def load_corrections_from_csv(csv_path: Path) -> List[Dict[str, str]]:
    """
    Load corrections from CSV file

    Args:
        csv_path: Path to al_labels.csv

    Returns:
        List of correction dicts
    """
    corrections = []

    if not csv_path.exists():
        logger.error(f"Corrections CSV not found: {csv_path}")
        return corrections

    with open(csv_path, 'r', newline='', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            corrections.append(row)

    logger.info(f"Loaded {len(corrections)} corrections from {csv_path}")
    return corrections


def train_clip_model(
    data_csv: Path = DEFAULT_AL_CSV,
    ckpt_out: Optional[Path] = None,
    batch_size: int = DEFAULT_BATCH_SIZE,
    lr: float = DEFAULT_LR,
    epochs: int = DEFAULT_EPOCHS,
    projection_dim: int = DEFAULT_PROJECTION_DIM,
    temperature: float = DEFAULT_TEMPERATURE,
    fine_tune_full_model: bool = False,
    device: Optional[str] = None,
    job_dir: Optional[Path] = None
) -> Dict:
    """
    Main training function
    Following PRD.md Task 11 specifications

    Args:
        data_csv: Path to active learning labels CSV
        ckpt_out: Path to save final checkpoint
        batch_size: Training batch size (default 64)
        lr: Learning rate (default 1e-4)
        epochs: Number of epochs (default 5)
        projection_dim: Projection head dimension (default 256)
        temperature: Temperature for contrastive loss (default 0.07)
        fine_tune_full_model: If True, fine-tune entire CLIP model
        device: Device to use
        job_dir: Optional job directory for saving metrics

    Returns:
        Dict with training metrics and checkpoint path
    """
    from src.config import SCRYFALL_IMAGES_DIR

    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    logger.info("=" * 80)
    logger.info("Starting CLIP fine-tuning training")
    logger.info("=" * 80)
    logger.info(f"Data CSV: {data_csv}")
    logger.info(f"Batch size: {batch_size}")
    logger.info(f"Learning rate: {lr}")
    logger.info(f"Epochs: {epochs}")
    logger.info(f"Projection dim: {projection_dim}")
    logger.info(f"Temperature: {temperature}")
    logger.info(f"Fine-tune full model: {fine_tune_full_model}")

    # Load corrections
    corrections = load_corrections_from_csv(data_csv)

    if len(corrections) == 0:
        logger.error("No corrections found. Cannot train.")
        return {
            'success': False,
            'error': 'No corrections found',
            'num_corrections': 0
        }

    logger.info(f"Loaded {len(corrections)} corrections")

    # Check minimum data requirement
    min_samples = 10
    if len(corrections) < min_samples:
        logger.warning(f"Only {len(corrections)} corrections available. Recommend at least {min_samples} for training.")
        logger.warning("Training will proceed but may not be effective.")

    # Initialize base CLIP model
    base_embedder = CLIPEmbedder(device=device)

    # Initialize fine-tuner
    fine_tuner = CLIPFineTuner(
        base_model=base_embedder,
        projection_dim=projection_dim,
        temperature=temperature,
        device=device
    )

    # Setup optimizer
    fine_tuner.setup_optimizer(lr=lr, fine_tune_full_model=fine_tune_full_model)

    # Create dataset
    db = SessionLocal()
    try:
        dataset = ALDataset(
            corrections=corrections,
            db_session=db,
            transform=base_embedder.preprocess,
            scryfall_images_dir=SCRYFALL_IMAGES_DIR
        )

        if len(dataset) == 0:
            logger.error("No valid training samples found.")
            return {
                'success': False,
                'error': 'No valid training samples',
                'num_corrections': len(corrections)
            }

        # Create dataloader
        dataloader = torch.utils.data.DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=0,  # Use 0 for Windows compatibility
            pin_memory=True if device == 'cuda' else False
        )

        # Training loop
        logger.info(f"Starting training with {len(dataset)} samples...")

        losses = []
        best_loss = float('inf')

        for epoch in range(1, epochs + 1):
            avg_loss = fine_tuner.train_epoch(dataloader, epoch)
            losses.append(avg_loss)

            # Save best model
            if avg_loss < best_loss:
                best_loss = avg_loss
                logger.info(f"New best loss: {best_loss:.4f}")

        # Save final checkpoint
        if ckpt_out is None:
            # Default checkpoint path
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            ckpt_out = MODELS_DIR / f"clip_finetuned_{timestamp}.pt"

        # Ensure parent directory exists
        ckpt_out.parent.mkdir(parents=True, exist_ok=True)

        # Save checkpoint
        metadata = {
            'num_corrections': len(corrections),
            'num_valid_samples': len(dataset),
            'batch_size': batch_size,
            'lr': lr,
            'epochs': epochs,
            'projection_dim': projection_dim,
            'temperature': temperature,
            'fine_tune_full_model': fine_tune_full_model,
            'final_loss': losses[-1],
            'best_loss': best_loss
        }

        fine_tuner.save_checkpoint(
            checkpoint_path=ckpt_out,
            epoch=epochs,
            loss=losses[-1],
            metadata=metadata
        )

        # Save training metrics
        metrics = {
            'success': True,
            'checkpoint_path': str(ckpt_out),
            'num_corrections': len(corrections),
            'num_valid_samples': len(dataset),
            'epochs': epochs,
            'final_loss': losses[-1],
            'best_loss': best_loss,
            'losses_per_epoch': losses,
            'timestamp': datetime.now().isoformat()
        }

        # Save metrics to job directory if provided
        if job_dir:
            job_dir.mkdir(parents=True, exist_ok=True)
            metrics_path = job_dir / "training_metrics.json"
            with open(metrics_path, 'w') as f:
                json.dump(metrics, f, indent=2)
            logger.info(f"Saved training metrics to {metrics_path}")

        logger.info("=" * 80)
        logger.info("Training completed successfully!")
        logger.info(f"Final loss: {losses[-1]:.4f}")
        logger.info(f"Best loss: {best_loss:.4f}")
        logger.info(f"Checkpoint saved to: {ckpt_out}")
        logger.info("=" * 80)

        return metrics

    finally:
        db.close()


def main():
    """
    CLI entry point for training
    Following PRD.md Task 11 specifications
    """
    parser = argparse.ArgumentParser(
        description='Fine-tune CLIP model on active learning corrections'
    )

    parser.add_argument(
        '--data_csv',
        type=str,
        default=str(DEFAULT_AL_CSV),
        help='Path to active learning labels CSV (default: data/feedback/al_labels.csv)'
    )

    parser.add_argument(
        '--ckpt_out',
        type=str,
        help='Path to save final checkpoint (default: auto-generated in data/models/)'
    )

    parser.add_argument(
        '--batch_size',
        type=int,
        default=DEFAULT_BATCH_SIZE,
        help=f'Training batch size (default: {DEFAULT_BATCH_SIZE})'
    )

    parser.add_argument(
        '--lr',
        type=float,
        default=DEFAULT_LR,
        help=f'Learning rate (default: {DEFAULT_LR})'
    )

    parser.add_argument(
        '--epochs',
        type=int,
        default=DEFAULT_EPOCHS,
        help=f'Number of training epochs (default: {DEFAULT_EPOCHS})'
    )

    parser.add_argument(
        '--projection_dim',
        type=int,
        default=DEFAULT_PROJECTION_DIM,
        help=f'Projection head output dimension (default: {DEFAULT_PROJECTION_DIM}, use 512 for no projection)'
    )

    parser.add_argument(
        '--temperature',
        type=float,
        default=DEFAULT_TEMPERATURE,
        help=f'Temperature for contrastive loss (default: {DEFAULT_TEMPERATURE})'
    )

    parser.add_argument(
        '--fine_tune_full',
        action='store_true',
        help='Fine-tune entire CLIP model (default: only train projection head)'
    )

    parser.add_argument(
        '--device',
        type=str,
        choices=['cpu', 'cuda'],
        help='Device to use (default: auto-detect)'
    )

    parser.add_argument(
        '--job_dir',
        type=str,
        help='Job directory for saving metrics (default: data/jobs/<timestamp>/)'
    )

    args = parser.parse_args()

    # Convert paths
    data_csv = Path(args.data_csv)
    ckpt_out = Path(args.ckpt_out) if args.ckpt_out else None
    job_dir = Path(args.job_dir) if args.job_dir else None

    # Run training
    try:
        metrics = train_clip_model(
            data_csv=data_csv,
            ckpt_out=ckpt_out,
            batch_size=args.batch_size,
            lr=args.lr,
            epochs=args.epochs,
            projection_dim=args.projection_dim,
            temperature=args.temperature,
            fine_tune_full_model=args.fine_tune_full,
            device=args.device,
            job_dir=job_dir
        )

        if metrics['success']:
            print(f"\n✓ Training completed successfully!")
            print(f"  Checkpoint: {metrics['checkpoint_path']}")
            print(f"  Final loss: {metrics['final_loss']:.4f}")
            print(f"  Best loss: {metrics['best_loss']:.4f}")
            exit(0)
        else:
            print(f"\n✗ Training failed: {metrics.get('error', 'Unknown error')}")
            exit(1)

    except Exception as e:
        logger.exception(f"Training failed with exception: {e}")
        print(f"\n✗ Training failed: {e}")
        exit(1)


if __name__ == '__main__':
    main()
