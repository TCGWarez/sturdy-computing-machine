"""
Cleanup service for old batches and uploaded files.

Removes batch records and associated uploaded images after a configurable
retention period to prevent disk space from growing indefinitely.
"""

import logging
import shutil
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

from sqlalchemy.orm import Session

from api.database import Batch, BatchResult, SessionLocal

logger = logging.getLogger(__name__)

# Default retention: 24 hours
DEFAULT_RETENTION_HOURS = 24

# Upload directory
UPLOAD_DIR = Path("uploads")


def cleanup_old_batches(
    retention_hours: int = DEFAULT_RETENTION_HOURS,
    db: Optional[Session] = None
) -> dict:
    """
    Delete batches older than retention period.

    Removes:
    - Batch records from database
    - BatchResult records (cascaded via FK)
    - Uploaded image files from disk

    Args:
        retention_hours: Hours to keep batches (default 24)
        db: Optional database session (creates one if not provided)

    Returns:
        Dict with cleanup statistics
    """
    close_db = False
    if db is None:
        db = SessionLocal()
        close_db = True

    try:
        cutoff_time = datetime.utcnow() - timedelta(hours=retention_hours)

        # Find old batches
        old_batches = db.query(Batch).filter(
            Batch.created_at < cutoff_time
        ).all()

        if not old_batches:
            logger.info(f"Cleanup: No batches older than {retention_hours} hours")
            return {"batches_deleted": 0, "files_deleted": 0, "bytes_freed": 0}

        batches_deleted = 0
        files_deleted = 0
        bytes_freed = 0

        for batch in old_batches:
            batch_id = batch.id
            batch_dir = UPLOAD_DIR / batch_id

            # Delete uploaded files
            if batch_dir.exists():
                try:
                    # Calculate size before deletion
                    for file_path in batch_dir.rglob("*"):
                        if file_path.is_file():
                            bytes_freed += file_path.stat().st_size
                            files_deleted += 1

                    shutil.rmtree(batch_dir)
                    logger.debug(f"Deleted upload directory: {batch_dir}")
                except Exception as e:
                    logger.warning(f"Failed to delete {batch_dir}: {e}")

            # Delete batch record (cascades to batch_results)
            db.delete(batch)
            batches_deleted += 1

        db.commit()

        # Convert bytes to human readable
        mb_freed = bytes_freed / (1024 * 1024)

        logger.info(
            f"Cleanup complete: {batches_deleted} batches, "
            f"{files_deleted} files, {mb_freed:.2f} MB freed"
        )

        return {
            "batches_deleted": batches_deleted,
            "files_deleted": files_deleted,
            "bytes_freed": bytes_freed
        }

    finally:
        if close_db:
            db.close()


def cleanup_orphaned_uploads() -> dict:
    """
    Delete upload directories that have no matching batch in the database.

    This handles cases where batches were deleted but files remained,
    or uploads that failed before creating a batch record.

    Returns:
        Dict with cleanup statistics
    """
    if not UPLOAD_DIR.exists():
        return {"orphans_deleted": 0, "bytes_freed": 0}

    db = SessionLocal()
    try:
        # Get all batch IDs from database
        batch_ids = {b.id for b in db.query(Batch.id).all()}

        orphans_deleted = 0
        bytes_freed = 0

        # Check each directory in uploads
        for dir_path in UPLOAD_DIR.iterdir():
            if dir_path.is_dir() and dir_path.name not in batch_ids:
                try:
                    # Calculate size
                    for file_path in dir_path.rglob("*"):
                        if file_path.is_file():
                            bytes_freed += file_path.stat().st_size

                    shutil.rmtree(dir_path)
                    orphans_deleted += 1
                    logger.debug(f"Deleted orphaned upload: {dir_path}")
                except Exception as e:
                    logger.warning(f"Failed to delete orphan {dir_path}: {e}")

        if orphans_deleted > 0:
            mb_freed = bytes_freed / (1024 * 1024)
            logger.info(f"Orphan cleanup: {orphans_deleted} dirs, {mb_freed:.2f} MB freed")

        return {"orphans_deleted": orphans_deleted, "bytes_freed": bytes_freed}

    finally:
        db.close()


def run_full_cleanup(retention_hours: int = DEFAULT_RETENTION_HOURS) -> dict:
    """
    Run complete cleanup: old batches + orphaned uploads.

    Args:
        retention_hours: Hours to keep batches

    Returns:
        Combined cleanup statistics
    """
    logger.info(f"Starting cleanup (retention: {retention_hours} hours)")

    batch_stats = cleanup_old_batches(retention_hours)
    orphan_stats = cleanup_orphaned_uploads()

    return {
        "batches_deleted": batch_stats["batches_deleted"],
        "files_deleted": batch_stats["files_deleted"],
        "orphans_deleted": orphan_stats["orphans_deleted"],
        "total_bytes_freed": batch_stats["bytes_freed"] + orphan_stats["bytes_freed"]
    }
