"""
src/cli/main.py: CLI entry point using Click

Commands:
- index --set M21 - Build indices
- recognize --dir /path/to/scans --out matches.csv --debug
- export --job <job_id> --format csv
- retrain - Kickoff fine-tune training
"""

import click
from pathlib import Path
from datetime import datetime
import json
import logging

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


@click.group()
def cli():
    """MTG Card Recognition System CLI"""
    pass


@cli.command()
@click.option('--set', 'set_code', required=True, help='Set code (e.g., M21, NEO)')
@click.option('--finish', type=click.Choice(['foil', 'nonfoil']), default='nonfoil', help='Finish type (default: nonfoil)')
@click.option('--device', default=None, type=click.Choice(['cpu', 'cuda']), help='Device to use (auto-detects GPU if available)')
@click.option('--batch-size', type=int, default=32, help='Batch size for embedding extraction (default: 32)')
@click.option('--force', is_flag=True, help='Force rebuild even if index exists')
@click.option('--checkpoint', type=click.Path(exists=True), help='Path to fine-tuned CLIP checkpoint')
def index(set_code, finish, device, batch_size, force, checkpoint):
    """
    Build indices for a set

    Example: index --set M21

    This command:
    1. Downloads/loads Scryfall images for the set
    2. Computes CLIP embeddings and pHash variants
    3. Builds FAISS/HNSW index
    4. Saves to data/indexes/<set>/<finish>/
    """
    from src.config import BASE_DIR
    from src.indexing.indexer import Indexer

    # Create job directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    job_dir = BASE_DIR / "data" / "jobs" / f"index_{set_code}_{finish}_{timestamp}"
    job_dir.mkdir(parents=True, exist_ok=True)

    logger.info("=" * 80)
    logger.info(f"Building index for {set_code}/{finish}")
    logger.info(f"Job directory: {job_dir}")
    logger.info("=" * 80)

    try:
        # Initialize indexer
        checkpoint_path = Path(checkpoint) if checkpoint else None
        indexer = Indexer(
            set_code=set_code,
            finish=finish,
            checkpoint_path=checkpoint_path,
            device=device
        )

        # Build index
        start_time = datetime.now()
        result = indexer.build_index(batch_size=batch_size, force=force)
        end_time = datetime.now()

        # Calculate metrics
        duration = (end_time - start_time).total_seconds()

        # Handle None result (indexing failed or no images)
        if result is None:
            logger.error("Indexing failed or no images were indexed")
            metrics = {
                'job_id': job_dir.name,
                'set_code': set_code,
                'finish': finish,
                'num_cards': 0,
                'num_embeddings': 0,
                'duration_seconds': duration,
                'timestamp': timestamp,
                'status': 'failed'
            }
        else:
            # Extract counts from result
            num_embeddings = len(result.get('embeddings', []))
            num_cards = len(result.get('card_ids', []))

            metrics = {
                'job_id': job_dir.name,
                'set_code': set_code,
                'finish': finish,
                'num_cards': num_cards,
                'num_embeddings': num_embeddings,
                'duration_seconds': duration,
                'timestamp': timestamp,
                'status': 'completed'
            }

        # Save metrics
        metrics_path = job_dir / "metrics.json"
        with open(metrics_path, 'w') as f:
            json.dump(metrics, f, indent=2)

        if result:
            logger.info("=" * 80)
            logger.info("Index built successfully!")
            logger.info(f"Cards indexed: {metrics['num_cards']}")
            logger.info(f"Duration: {duration:.2f} seconds")
            logger.info(f"Metrics saved to: {metrics_path}")
            logger.info("=" * 80)
        else:
            logger.error("=" * 80)
            logger.error("Indexing failed!")
            logger.error(f"Duration: {duration:.2f} seconds")
            logger.error(f"Metrics saved to: {metrics_path}")
            logger.error("=" * 80)

        indexer.close()

    except Exception as e:
        logger.exception(f"Error building index: {e}")
        # Save error to job directory
        error_path = job_dir / "error.json"
        with open(error_path, 'w') as f:
            json.dump({
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            }, f, indent=2)
        raise


@cli.command()
@click.option('--dir', 'scan_dir', required=True, type=click.Path(exists=True), help='Directory containing scanned images')
@click.option('--out', 'output_csv', required=True, type=click.Path(), help='Output CSV path for matches')
@click.option('--set', 'set_code', required=True, help='Set code (e.g., TLA, DMR)')
@click.option('--finish', type=click.Choice(['foil', 'nonfoil']), default='nonfoil', help='Finish type')
@click.option('--debug', is_flag=True, help='Enable debug mode')
@click.option('--device', default=None, type=click.Choice(['cpu', 'cuda']), help='Device to use (auto-detects GPU if available)')
def recognize(scan_dir, output_csv, set_code, finish, debug, device):
    """Recognize scanned card images from a directory."""
    from src.config import BASE_DIR
    from src.recognition.matcher import CardMatcher
    from tqdm import tqdm
    import csv

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    job_dir = BASE_DIR / "data" / "jobs" / f"recognize_{timestamp}"
    job_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"Recognizing images from {scan_dir}")
    logger.info(f"Set: {set_code}, Finish: {finish}")

    scan_dir_path = Path(scan_dir)
    image_extensions = {'.jpg', '.jpeg', '.png', '.webp', '.bmp'}
    image_paths = []

    for ext in image_extensions:
        image_paths.extend(scan_dir_path.glob(f'*{ext}'))
        image_paths.extend(scan_dir_path.glob(f'*{ext.upper()}'))

    if not image_paths:
        logger.error(f"No images found in {scan_dir}")
        return

    logger.info(f"Found {len(image_paths)} images to process")

    matcher = CardMatcher(set_code=set_code, finish=finish, device=device)

    results = []
    failures = []
    start_time = datetime.now()

    try:
        for img_path in tqdm(image_paths, desc="Processing images"):
            try:
                match_result = matcher.match_scanned(img_path, debug=debug)

                if match_result.match_card_id and match_result.candidates:
                    top = match_result.candidates[0]
                    result = {
                        'scanned_path': str(img_path),
                        'matched_card_id': top.card_id,
                        'card_name': top.card_name,
                        'confidence': match_result.confidence,
                        'status': 'matched'
                    }
                else:
                    result = {
                        'scanned_path': str(img_path),
                        'matched_card_id': None,
                        'card_name': None,
                        'confidence': 0.0,
                        'status': 'no_match'
                    }

                results.append(result)

                if result.get('confidence', 0) < 0.85:
                    failures.append(result)

            except Exception as e:
                logger.error(f"Error processing {img_path}: {e}")
                results.append({
                    'scanned_path': str(img_path),
                    'matched_card_id': None,
                    'card_name': None,
                    'confidence': 0.0,
                    'status': 'error'
                })

        end_time = datetime.now()
        duration = (end_time - start_time).total_seconds()

        output_path = Path(output_csv)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=['scanned_path', 'matched_card_id', 'card_name', 'confidence', 'status'])
            writer.writeheader()
            writer.writerows(results)

        num_matched = sum(1 for r in results if r.get('confidence', 0) >= 0.85)
        precision_at_1 = num_matched / len(results) if results else 0.0

        metrics = {
            'job_id': job_dir.name,
            'total_images': len(image_paths),
            'processed_images': len(results),
            'num_matched': num_matched,
            'num_failures': len(failures),
            'precision_at_1': precision_at_1,
            'avg_time_per_image': duration / len(results) if results else 0,
            'total_duration_seconds': duration,
            'timestamp': timestamp,
            'output_csv': str(output_path)
        }

        metrics_path = job_dir / "metrics.json"
        with open(metrics_path, 'w') as f:
            json.dump(metrics, f, indent=2)

        if failures:
            failures_dir = job_dir / "failures"
            failures_dir.mkdir(exist_ok=True)
            with open(failures_dir / "failures.json", 'w') as f:
                json.dump(failures, f, indent=2)

        logger.info(f"Recognition completed: {num_matched}/{len(results)} matched ({precision_at_1:.1%})")
        logger.info(f"Results: {output_path}")

        matcher.close()

    except Exception as e:
        logger.exception(f"Error during recognition: {e}")
        raise


@cli.command()
@click.option('--job', 'job_id', required=True, help='Job ID (timestamp from data/jobs/)')
@click.option('--format', 'export_format', type=click.Choice(['csv', 'json']), default='csv', help='Export format (default: csv)')
@click.option('--out', 'output_path', type=click.Path(), help='Output path (optional)')
def export(job_id, export_format, output_path):
    """
    Export job results

    Example: export --job <job_id> --format csv

    This command exports the results from a recognition job.
    """
    from src.config import BASE_DIR

    # Find job directory
    jobs_dir = BASE_DIR / "data" / "jobs"
    job_dir = jobs_dir / job_id

    if not job_dir.exists():
        logger.error(f"Job not found: {job_id}")
        logger.error(f"Searched in: {jobs_dir}")
        return

    logger.info(f"Exporting job: {job_id}")

    # Load metrics
    metrics_path = job_dir / "metrics.json"
    if not metrics_path.exists():
        logger.error(f"Metrics not found for job: {job_id}")
        return

    with open(metrics_path, 'r') as f:
        metrics = json.load(f)

    # Determine output path
    if output_path:
        out_path = Path(output_path)
    else:
        out_path = job_dir / f"export.{export_format}"

    # Export based on format
    if export_format == 'csv':
        # Check if CSV already exists
        existing_csv = metrics.get('output_csv')
        if existing_csv and Path(existing_csv).exists():
            # Copy existing CSV
            import shutil
            shutil.copy(existing_csv, out_path)
            logger.info(f"Exported to: {out_path}")
        else:
            logger.error("No CSV output found for this job")

    elif export_format == 'json':
        # Export metrics as JSON
        with open(out_path, 'w') as f:
            json.dump(metrics, f, indent=2)
        logger.info(f"Exported to: {out_path}")

    logger.info(f"Export completed: {out_path}")


@cli.command()
@click.option('--data-csv', type=click.Path(exists=True), required=True, help='Path to active learning labels CSV')
@click.option('--checkpoint', type=click.Path(), help='Output checkpoint path (default: auto-generated)')
@click.option('--batch-size', type=int, default=64, help='Training batch size (default: 64)')
@click.option('--lr', type=float, default=1e-4, help='Learning rate (default: 1e-4)')
@click.option('--epochs', type=int, default=5, help='Number of epochs (default: 5)')
@click.option('--projection-dim', type=int, default=256, help='Projection head dimension (default: 256)')
@click.option('--device', default=None, type=click.Choice(['cpu', 'cuda']), help='Device to use (auto-detects GPU if available)')
def retrain(data_csv, checkpoint, batch_size, lr, epochs, projection_dim, device):
    """
    Kickoff fine-tune training

    Example: retrain --data-csv data/al_labels.csv

    This command:
    1. Loads corrections from provided CSV file
    2. Fine-tunes CLIP with contrastive loss
    3. Saves checkpoint to data/models/
    4. Logs metrics to data/jobs/<timestamp>/
    """
    from src.embeddings.train import train_clip_model
    from src.config import BASE_DIR

    # Create job directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    job_dir = BASE_DIR / "data" / "jobs" / f"retrain_{timestamp}"
    job_dir.mkdir(parents=True, exist_ok=True)

    logger.info("=" * 80)
    logger.info("Starting CLIP fine-tuning")
    logger.info(f"Job directory: {job_dir}")
    logger.info("=" * 80)

    data_csv_path = Path(data_csv)

    # Determine checkpoint output path
    if checkpoint:
        ckpt_out = Path(checkpoint)
    else:
        ckpt_out = None  # Will be auto-generated

    try:
        # Run training
        metrics = train_clip_model(
            data_csv=data_csv_path,
            ckpt_out=ckpt_out,
            batch_size=batch_size,
            lr=lr,
            epochs=epochs,
            projection_dim=projection_dim,
            device=device,
            job_dir=job_dir
        )

        if metrics['success']:
            logger.info("=" * 80)
            logger.info("Fine-tuning completed successfully!")
            logger.info(f"Checkpoint: {metrics['checkpoint_path']}")
            logger.info(f"Final loss: {metrics['final_loss']:.4f}")
            logger.info(f"Metrics saved to: {job_dir / 'training_metrics.json'}")
            logger.info("=" * 80)
            logger.info("\nNext steps:")
            logger.info("1. Rebuild indices with new checkpoint:")
            logger.info(f"   index --set <SET> --checkpoint {metrics['checkpoint_path']}")
            logger.info("2. Test on scanned images:")
            logger.info(f"   recognize --dir <SCANS_DIR> --out matches.csv --checkpoint {metrics['checkpoint_path']}")
        else:
            logger.error(f"Fine-tuning failed: {metrics.get('error', 'Unknown error')}")

    except Exception as e:
        logger.exception(f"Error during fine-tuning: {e}")
        raise


if __name__ == '__main__':
    cli()

