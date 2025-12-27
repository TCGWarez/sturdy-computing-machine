"""
Queue management for concurrent batch processing.

Provides semaphore-based concurrency control to limit simultaneous
batch processing and prevent resource exhaustion under load.
"""

import asyncio
import logging
from contextlib import asynccontextmanager
from typing import Set, Optional
from dataclasses import dataclass

logger = logging.getLogger(__name__)

# Tunable: max concurrent batch processing jobs
# For CPU-based inference, limit based on cores (e.g., 3-5 for 8-core CPU)
MAX_CONCURRENT_BATCHES = 5


class BatchQueueManager:
    """
    Thread-safe queue manager for batch processing.

    Uses asyncio primitives that are properly initialized within the event loop
    to avoid issues with module-level initialization.
    """

    def __init__(self):
        self._semaphore: Optional[asyncio.Semaphore] = None
        self._lock: Optional[asyncio.Lock] = None
        self._active_batches: Set[str] = set()
        self._waiting_batches: Set[str] = set()
        self._initialized = False

    def _ensure_initialized(self):
        """Lazily initialize asyncio primitives within event loop context."""
        if not self._initialized:
            self._semaphore = asyncio.Semaphore(MAX_CONCURRENT_BATCHES)
            self._lock = asyncio.Lock()
            self._initialized = True

    @asynccontextmanager
    async def acquire_slot(self, batch_id: str):
        """
        Acquire a processing slot, waiting if at capacity.

        Uses asyncio.Semaphore to limit concurrent batch processing.
        Tracks batch state for monitoring endpoints with proper locking.

        Args:
            batch_id: Unique batch identifier

        Yields:
            None when slot is acquired
        """
        self._ensure_initialized()

        # Add to waiting set with lock protection
        async with self._lock:
            self._waiting_batches.add(batch_id)
        logger.info(f"Batch {batch_id}: Waiting for processing slot ({len(self._active_batches)}/{MAX_CONCURRENT_BATCHES} active)")

        try:
            async with self._semaphore:
                # Move from waiting to active with lock protection
                async with self._lock:
                    self._waiting_batches.discard(batch_id)
                    self._active_batches.add(batch_id)
                logger.info(f"Batch {batch_id}: Acquired processing slot ({len(self._active_batches)}/{MAX_CONCURRENT_BATCHES} active)")

                try:
                    yield
                finally:
                    # Remove from active with lock protection
                    async with self._lock:
                        self._active_batches.discard(batch_id)
                    logger.info(f"Batch {batch_id}: Released processing slot ({len(self._active_batches)}/{MAX_CONCURRENT_BATCHES} active)")
        except Exception:
            # Clean up waiting state on error
            async with self._lock:
                self._waiting_batches.discard(batch_id)
            raise

    def get_status(self) -> 'QueueStatus':
        """
        Return current queue status for monitoring.

        Returns:
            QueueStatus with active/waiting counts and available slots
        """
        return QueueStatus(
            active_batches=len(self._active_batches),
            waiting_batches=len(self._waiting_batches),
            max_concurrent=MAX_CONCURRENT_BATCHES,
            available_slots=max(0, MAX_CONCURRENT_BATCHES - len(self._active_batches)),
            active_batch_ids=list(self._active_batches),
            waiting_batch_ids=list(self._waiting_batches)
        )


@dataclass
class QueueStatus:
    """Current queue status for monitoring."""
    active_batches: int
    waiting_batches: int
    max_concurrent: int
    available_slots: int
    active_batch_ids: list
    waiting_batch_ids: list


# Global queue manager instance
_queue_manager = BatchQueueManager()


@asynccontextmanager
async def acquire_batch_slot(batch_id: str):
    """
    Acquire a processing slot, waiting if at capacity.

    Convenience wrapper around BatchQueueManager.acquire_slot().
    """
    async with _queue_manager.acquire_slot(batch_id):
        yield


def get_queue_status() -> QueueStatus:
    """
    Return current queue status for monitoring.

    Returns:
        QueueStatus with active/waiting counts and available slots
    """
    return _queue_manager.get_status()


async def run_with_concurrency_control(batch_id: str, sync_func, *args, **kwargs):
    """
    Run a synchronous function with concurrency control.

    Acquires a slot from the semaphore, then runs the function
    in a thread pool executor to avoid blocking the event loop.

    Args:
        batch_id: Batch identifier for tracking
        sync_func: Synchronous function to run
        *args, **kwargs: Arguments to pass to sync_func

    Returns:
        Result from sync_func
    """
    async with acquire_batch_slot(batch_id):
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            None,  # Use default thread pool
            lambda: sync_func(*args, **kwargs)
        )
