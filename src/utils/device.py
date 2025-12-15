"""Device selection utilities for automatic GPU/CPU detection."""

import logging
from typing import Optional

import torch

logger = logging.getLogger(__name__)


def resolve_device(device: Optional[str] = None) -> str:
    """Resolve device with auto-detection.

    Args:
        device: Explicit device ('cpu' or 'cuda'). If None, auto-detects.

    Returns:
        'cuda' if available and device is None, otherwise 'cpu' or explicit device.
    """
    if device is not None:
        logger.info(f"Using explicit device: {device}")
        return device

    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        logger.info(f"Auto-detected GPU: {gpu_name} (cuda)")
        return 'cuda'
    else:
        logger.info("No GPU detected, using CPU")
        return 'cpu'
