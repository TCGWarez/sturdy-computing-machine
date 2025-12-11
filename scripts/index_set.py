#!/usr/bin/env python
"""
Streamlined set indexing script

Complete one-command indexing for card recognition system.
Automatically computes embeddings, pHash, variant types, and builds FAISS index.

Usage:
  # Index a single set (both nonfoil and foil/etched)
  uv run python scripts/index_set.py DMR

  # Index all sets in image directory
  uv run python scripts/index_set.py --all

  # Use GPU for faster CLIP embeddings
  uv run python scripts/index_set.py DMR --device cuda

  # Index multiple sets
  uv run python scripts/index_set.py DMR SLD NEO

  # Force re-index existing cards
  uv run python scripts/index_set.py DMR --force

  # Dry run to see what would be indexed
  uv run python scripts/index_set.py --all --dry-run
"""

import argparse
import sys
from pathlib import Path
import time
from typing import List, Tuple

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.indexing.indexer import Indexer
from src.database.schema import SessionLocal, Card, init_db
from src.config import SCRYFALL_IMAGES_DIR, INDEXES_DIR
from src.ann.faiss_index import build_composite_index_from_db


def discover_sets() -> List[str]:
    """Discover all set codes from the image directory"""
    if not SCRYFALL_IMAGES_DIR.exists():
        return []

    sets = []
    for item in SCRYFALL_IMAGES_DIR.iterdir():
        if item.is_dir() and not item.name.startswith('.'):
            sets.append(item.name.upper())

    return sorted(sets)


def get_available_finishes(set_code: str) -> List[str]:
    """Get list of available finish directories for a set"""
    set_dir = SCRYFALL_IMAGES_DIR / set_code.upper()
    if not set_dir.exists():
        return []

    finishes = []
    for finish in ['nonfoil', 'foil', 'etched']:
        finish_dir = set_dir / finish
        if finish_dir.exists() and any(finish_dir.glob('*.jpg')) or any(finish_dir.glob('*.png')):
            finishes.append(finish)

    return finishes


def check_set_availability(set_code: str, finish: str) -> dict:
    """Check if set is available and how many cards are already indexed"""
    set_dir = SCRYFALL_IMAGES_DIR / set_code.upper()

    if not set_dir.exists():
        return {
            'exists': False,
            'path': set_dir,
            'image_count': 0,
            'indexed_count': 0
        }

    # Count images
    image_count = 0
    finish_dir = set_dir / finish
    if finish_dir.exists():
        image_count = len(list(finish_dir.glob('*.jpg'))) + len(list(finish_dir.glob('*.png')))

    # Count already indexed
    db = SessionLocal()
    try:
        indexed_count = db.query(Card).filter(
            Card.set_code == set_code.upper(),
            Card.finish == finish
        ).count()
    finally:
        db.close()

    return {
        'exists': True,
        'path': set_dir,
        'image_count': image_count,
        'indexed_count': indexed_count
    }


def index_finish(set_code: str, finish: str, force: bool = False, device: str = None,
                 batch_size: int = 100) -> Tuple[bool, int]:
    """
    Index a single set+finish combination

    Returns:
        Tuple of (success, cards_indexed)
    """
    info = check_set_availability(set_code, finish)

    if info['image_count'] == 0:
        return True, 0  

    print(f"\n  {finish}: {info['image_count']} images, {info['indexed_count']} already indexed")

    if info['indexed_count'] > 0 and not force:
        to_index = info['image_count'] - info['indexed_count']
        if to_index == 0:
            print(f"    All cards already indexed")
            return True, 0

    indexer = Indexer(
        set_code=set_code,
        finish=finish,
        device=device
    )

    try:
        result = indexer.build_index(
            batch_size=batch_size,
            force=force
        )

        if result and 'embeddings' in result and len(result['embeddings']) > 0:
            return True, len(result['embeddings'])
        else:
            return True, 0

    except Exception as e:
        print(f"    [!] Error: {e}")
        return False, 0

    finally:
        indexer.close()


def build_faiss_index(set_code: str, finishes: List[str]) -> bool:
    """
    Build FAISS index for a set, combining finishes as needed

    - nonfoil -> {SET}_nonfoil_composite.faiss
    - foil + etched -> {SET}_foil_composite.faiss
    """
    INDEXES_DIR.mkdir(parents=True, exist_ok=True)

    success = True

    
    if 'nonfoil' in finishes:
        try:
            output_path = INDEXES_DIR / f"{set_code.upper()}_nonfoil_composite.faiss"
            build_composite_index_from_db(
                set_code=set_code.upper(),
                finish='nonfoil',
                index_type='flat',
                output_path=str(output_path)
            )
            print(f"    [+] Built: {output_path.name}")
        except Exception as e:
            print(f"    [!] Failed nonfoil index: {e}")
            success = False

    # Build foil index (combining foil + etched)
    foil_finishes = [f for f in finishes if f in ['foil', 'etched']]
    if foil_finishes:
        try:
            output_path = INDEXES_DIR / f"{set_code.upper()}_foil_composite.faiss"
            build_composite_index_from_db_multi_finish(
                set_code=set_code.upper(),
                finishes=foil_finishes,
                index_type='flat',
                output_path=str(output_path)
            )
            print(f"    [+] Built: {output_path.name} ({'+'.join(foil_finishes)})")
        except Exception as e:
            print(f"    [!] Failed foil index: {e}")
            success = False

    return success


def build_composite_index_from_db_multi_finish(
    set_code: str,
    finishes: List[str],
    index_type: str = 'flat',
    output_path: str = None
):
    """
    Build FAISS index from composite embeddings for MULTIPLE finishes

    Used to combine foil + etched into a single index.
    """
    from src.ann.faiss_index import FAISSIndex
    from src.database.schema import CompositeEmbedding
    from src.indexing.indexer import deserialize_embedding
    import numpy as np

    print(f"  Building composite FAISS index for {set_code}/{'+'.join(finishes)}")

    db = SessionLocal()
    try:
        
        cards_with_embeddings = (
            db.query(Card, CompositeEmbedding)
            .join(CompositeEmbedding, Card.id == CompositeEmbedding.card_id)
            .filter(Card.set_code == set_code.upper())
            .filter(Card.finish.in_([f.lower() for f in finishes]))
            .all()
        )

        if not cards_with_embeddings:
            raise ValueError(f"No cards with composite embeddings found for {set_code}/{finishes}")

        print(f"    Found {len(cards_with_embeddings)} cards")

        # Extract composite embeddings and card IDs
        embeddings_list = []
        card_ids = []

        for card, embedding_record in cards_with_embeddings:
            embedding = deserialize_embedding(embedding_record.embedding)
            embeddings_list.append(embedding)
            card_ids.append(card.id)

   
        embeddings = np.vstack(embeddings_list)

        
        index = FAISSIndex(
            embedding_dim=embeddings.shape[1],
            use_cosine=True,
            index_type=index_type
        )

        index.build_index(embeddings, card_ids)

        if output_path:
            index.save(
                output_path,
                metadata={
                    'set_code': set_code,
                    'finishes': finishes,
                    'index_type': index_type,
                    'embedding_type': 'composite'
                }
            )

        return index

    finally:
        db.close()


def index_set(set_code: str, force: bool = False, device: str = None, batch_size: int = 100) -> bool:
    """Index a complete set (all available finishes)"""

    print("\n" + "=" * 80)
    print(f"Indexing {set_code.upper()}")
    print("=" * 80)

    set_dir = SCRYFALL_IMAGES_DIR / set_code.upper()
    if not set_dir.exists():
        print(f"\n[!] Error: Set directory not found: {set_dir}")
        print(f"    Make sure images are downloaded to: {SCRYFALL_IMAGES_DIR}")
        return False

    
    available_finishes = get_available_finishes(set_code)
    if not available_finishes:
        print(f"\n[!] No finish directories found for {set_code}")
        return False

    print(f"\n  Directory: {set_dir}")
    print(f"  Available finishes: {', '.join(available_finishes)}")

    
    start_time = time.time()
    total_indexed = 0
    indexed_finishes = []

    print("\n  Indexing cards...")

    for finish in available_finishes:
        success, count = index_finish(
            set_code=set_code,
            finish=finish,
            force=force,
            device=device,
            batch_size=batch_size
        )

        if success and count > 0:
            total_indexed += count
            indexed_finishes.append(finish)
        elif not success:
            print(f"    [!] Failed to index {finish}")

    elapsed = time.time() - start_time

    if total_indexed > 0:
        print(f"\n  [+] Indexed {total_indexed} cards in {elapsed:.1f}s")

    # Build FAISS indexes
    print("\n  Building FAISS indexes...")

    # Use all available finishes (some may have been previously indexed)
    faiss_success = build_faiss_index(set_code, available_finishes)

    if faiss_success:
        print(f"\n[+] Success! Set {set_code.upper()} ready for recognition")
    else:
        print(f"\n[!] Warning: FAISS index build had issues (embeddings safe in database)")

    return True


def main():
    parser = argparse.ArgumentParser(
        description='Index card sets for recognition',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Index a single set (all finishes: nonfoil + foil/etched)
  %(prog)s DMR

  # Index all sets in image directory
  %(prog)s --all

  # Use GPU for faster CLIP embeddings
  %(prog)s DMR --device cuda

  # Index multiple specific sets
  %(prog)s DMR SLD NEO

  # Force re-index (recompute embeddings)
  %(prog)s DMR --force

  # Dry run to preview
  %(prog)s --all --dry-run

What it does:
  1. Computes composite embeddings (70% full + 20% collector + 10% name)
  2. Computes 3 pHash variants (full/name/collector)
  3. Auto-detects variant types (normal/extended_art/showcase)
  4. Builds FAISS indexes:
     - {SET}_nonfoil_composite.faiss (nonfoil cards)
     - {SET}_foil_composite.faiss (foil + etched cards combined)

Pipeline:
  1. Download images:  uv run python scripts/download_default_cards.py --sets DMR
  2. Index set:        uv run python scripts/index_set.py DMR
  3. Test recognition: uv run python scripts/test_recognition.py image.jpg --set DMR
        """
    )

    parser.add_argument(
        'sets',
        nargs='*',
        help='Set code(s) to index (e.g., DMR SLD NEO)'
    )
    parser.add_argument(
        '--all',
        action='store_true',
        help='Index all sets found in image directory'
    )
    parser.add_argument(
        '--force',
        action='store_true',
        help='Force re-index already indexed cards'
    )
    parser.add_argument(
        '--device',
        default=None,
        help='Device for embeddings: cpu, cuda, mps (default: auto)'
    )
    parser.add_argument(
        '--batch-size',
        type=int,
        default=100,
        help='Batch size for processing (default: 100)'
    )
    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='Show what would be indexed without doing it'
    )

    args = parser.parse_args()

    # Determine sets to index
    if args.all:
        sets_to_index = discover_sets()
        if not sets_to_index:
            print(f"[!] No sets found in {SCRYFALL_IMAGES_DIR}")
            return 1
    elif args.sets:
        sets_to_index = [s.upper() for s in args.sets]
    else:
        parser.print_help()
        print("\n[!] Error: Specify set codes or use --all")
        return 1

    # Initialize database
    init_db()

    print("\n" + "=" * 80)
    print("MTG Card Set Indexer")
    print("=" * 80)
    print(f"\nImage directory: {SCRYFALL_IMAGES_DIR}")
    print(f"Sets to index: {len(sets_to_index)} sets")

    if args.dry_run:
        print("\n[DRY RUN MODE - No changes will be made]")

    # Check all sets first
    print("\n" + "-" * 80)
    print("Checking set availability...")
    print("-" * 80)

    total_images = 0
    total_indexed = 0
    sets_info = []

    for set_code in sets_to_index:
        finishes = get_available_finishes(set_code)
        set_images = 0
        set_indexed = 0

        for finish in finishes:
            info = check_set_availability(set_code, finish)
            set_images += info['image_count']
            set_indexed += info['indexed_count']

        if set_images > 0:
            print(f"\n{set_code}:")
            print(f"  Finishes: {', '.join(finishes) if finishes else 'none'}")
            print(f"  Images: {set_images}, Indexed: {set_indexed}")

            total_images += set_images
            total_indexed += set_indexed
            sets_info.append((set_code, set_images, set_indexed, finishes))
        else:
            print(f"\n{set_code}: [!] No images found")

    # Summary
    total_to_index = total_images - (total_indexed if not args.force else 0)

    print("\n" + "-" * 80)
    print(f"Total: {total_images} images, {total_indexed} indexed, {total_to_index} to process")
    print(f"Sets with images: {len(sets_info)}")
    print("-" * 80)

    if args.dry_run:
        print("\n[Dry run complete - no changes made]")
        return 0

    if total_to_index == 0 and not args.force:
        print("\n[!] All cards already indexed. Use --force to re-index.")
        return 0

    # Confirm
    print()
    if args.force:
        response = input(f"Re-index all {total_images} cards across {len(sets_info)} sets? [y/N]: ")
    else:
        response = input(f"Index {total_to_index} cards across {len(sets_info)} sets? [y/N]: ")

    if response.lower() != 'y':
        print("Cancelled")
        return 0

    # Index each set
    overall_start = time.time()
    success_count = 0
    fail_count = 0

    for set_code, _, _, _ in sets_info:
        success = index_set(
            set_code=set_code,
            force=args.force,
            device=args.device,
            batch_size=args.batch_size
        )

        if success:
            success_count += 1
        else:
            fail_count += 1

    # Final summary
    overall_elapsed = time.time() - overall_start

    print("\n" + "=" * 80)
    print("INDEXING COMPLETE")
    print("=" * 80)
    print(f"\n  Success: {success_count} sets")
    if fail_count > 0:
        print(f"  Failed:  {fail_count} sets")
    print(f"  Total time: {overall_elapsed:.1f}s")

    print(f"\n  Test recognition with:")
    print(f"    uv run python scripts/test_recognition.py image.jpg --set SET_CODE")

    return 0 if fail_count == 0 else 1


if __name__ == '__main__':
    sys.exit(main())
