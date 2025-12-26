"""Device selection utilities for automatic GPU/CPU detection."""

import logging
import os
from typing import Optional

import torch

logger = logging.getLogger(__name__)


def configure_cpu_threads(num_threads: int = None) -> int:
    """Configure PyTorch and related libraries to use multiple CPU threads.

    This should be called at application startup when running on CPU to leverage
    multi-core parallelism for inference operations.

    Args:
        num_threads: Number of threads to use. If None, uses all available CPU cores.

    Returns:
        The number of threads configured.
    """
    if num_threads is None:
        num_threads = os.cpu_count() or 4

    # PyTorch intra-op parallelism (within individual ops like matrix multiply)
    torch.set_num_threads(num_threads)

    # PyTorch inter-op parallelism (between independent ops)
    # Use half the threads for inter-op to avoid over-subscription
    interop_threads = max(1, num_threads // 2)
    torch.set_num_interop_threads(interop_threads)

    # Also configure OpenMP/MKL for numpy/scipy operations
    # These must be set BEFORE numpy is imported in some cases,
    # but setting them here can still help for new thread pools
    os.environ['OMP_NUM_THREADS'] = str(num_threads)
    os.environ['MKL_NUM_THREADS'] = str(num_threads)
    os.environ['OPENBLAS_NUM_THREADS'] = str(num_threads)

    logger.info(
        f"Configured CPU threading: {num_threads} intra-op threads, "
        f"{interop_threads} inter-op threads"
    )

    return num_threads


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
