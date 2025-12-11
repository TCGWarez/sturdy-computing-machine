"""
Build a unified "ALL" FAISS index by merging existing per-set indexes.

This script combines all existing per-set FAISS indexes into a single
unified index for fast all-sets search without loading individual indexes.

Usage:
    python scripts/build_unified_index.py --finish nonfoil
    python scripts/build_unified_index.py --finish foil
    python scripts/build_unified_index.py --finish nonfoil --sets ONE,DSK,DMR  # Specific sets only

The unified index is saved as:
    data/indexes/ALL_{finish}_composite.faiss
    data/indexes/ALL_{finish}_composite.meta
"""

import argparse
import sys
from pathlib import Path
import numpy as np
import faiss
import pickle
from typing import List, Tuple, Optional

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.config import INDEXES_DIR


def discover_indexes(finish: str = 'nonfoil') -> List[Tuple[str, Path]]:
    """
    Discover all available per-set FAISS indexes for a given finish.

    Returns:
        List of (set_code, index_path) tuples
    """
    if not INDEXES_DIR.exists():
        return []

    pattern = f"*_{finish}_composite.faiss"
    index_files = list(INDEXES_DIR.glob(pattern))

    results = []
    for index_file in index_files:
        # Extract set code from filename like "DMR_nonfoil_composite.faiss"
        parts = index_file.stem.split('_')
        if len(parts) >= 3:
            set_code = parts[0]
            # Skip the ALL index if it already exists
            if set_code.upper() != 'ALL':
                results.append((set_code, index_file))

    return sorted(results, key=lambda x: x[0])


def load_index_data(index_path: Path) -> Tuple[np.ndarray, List[str], dict]:
    """
    Load vectors, card IDs, and metadata from a FAISS index.

    Args:
        index_path: Path to .faiss file

    Returns:
        Tuple of (embeddings array, card_ids list, metadata dict)
    """
    # Load FAISS index
    index = faiss.read_index(str(index_path))

    # Load metadata
    meta_path = index_path.with_suffix('.meta')
    if not meta_path.exists():
        raise FileNotFoundError(f"Metadata file not found: {meta_path}")

    with open(meta_path, 'rb') as f:
        meta = pickle.load(f)

    card_ids = meta['card_ids']
    embedding_dim = meta['embedding_dim']

    # Reconstruct vectors from index
    # This only works for Flat indexes (IndexFlatIP, IndexFlatL2)
    n_vectors = index.ntotal

    if n_vectors == 0:
        return np.array([]).reshape(0, embedding_dim), [], meta

    # Reconstruct all vectors
    embeddings = np.zeros((n_vectors, embedding_dim), dtype=np.float32)
    for i in range(n_vectors):
        embeddings[i] = index.reconstruct(i)

    return embeddings, card_ids, meta


def build_unified_index(
    finish: str = 'nonfoil',
    set_codes: Optional[List[str]] = None,
    output_path: Optional[Path] = None,
    index_type: str = 'flat'
) -> Tuple[int, int, Path]:
    """
    Build a unified FAISS index from existing per-set indexes.

    Args:
        finish: Finish type ('nonfoil' or 'foil')
        set_codes: Optional list of specific set codes to include.
                   If None, includes all available sets.
        output_path: Optional output path. Defaults to ALL_{finish}_composite.faiss
        index_type: FAISS index type ('flat', 'ivf', 'hnsw')

    Returns:
        Tuple of (num_sets_merged, num_vectors, output_path)
    """
    # Discover available indexes
    available = discover_indexes(finish)

    if not available:
        raise ValueError(f"No indexes found for finish '{finish}'")

    print(f"Found {len(available)} indexes for finish '{finish}'")

    # Filter to specific sets if requested
    if set_codes:
        set_codes_upper = [s.upper() for s in set_codes]
        available = [(code, path) for code, path in available if code.upper() in set_codes_upper]

        if not available:
            raise ValueError(f"None of the specified sets have indexes: {set_codes}")

    print(f"Will merge {len(available)} indexes: {[code for code, _ in available]}")

    # Collect all embeddings and card IDs
    all_embeddings = []
    all_card_ids = []
    merged_set_codes = []
    embedding_dim = None
    sample_meta = None

    for set_code, index_path in available:
        print(f"  Loading {set_code}...", end=" ")

        try:
            embeddings, card_ids, meta = load_index_data(index_path)

            if embeddings.shape[0] == 0:
                print(f"empty, skipping")
                continue

            # Verify dimension consistency
            if embedding_dim is None:
                embedding_dim = embeddings.shape[1]
                sample_meta = meta
            elif embeddings.shape[1] != embedding_dim:
                print(f"dimension mismatch ({embeddings.shape[1]} vs {embedding_dim}), skipping")
                continue

            all_embeddings.append(embeddings)
            all_card_ids.extend(card_ids)
            merged_set_codes.append(set_code)

            print(f"{len(card_ids)} cards")

        except Exception as e:
            print(f"error: {e}")
            continue

    if not all_embeddings:
        raise ValueError("No valid indexes could be loaded")

    # Stack all embeddings
    combined_embeddings = np.vstack(all_embeddings)
    print(f"\nCombined: {combined_embeddings.shape[0]} vectors, {embedding_dim} dimensions")

    # Verify card_ids match
    if len(all_card_ids) != combined_embeddings.shape[0]:
        raise ValueError(f"Card ID count ({len(all_card_ids)}) doesn't match vector count ({combined_embeddings.shape[0]})")

    # Create unified index
    print(f"\nBuilding unified {index_type} index...")

    # For large indexes (>100k vectors), consider IVF
    if index_type == 'flat':
        # Flat index - exact search
        unified_index = faiss.IndexFlatIP(embedding_dim)
    elif index_type == 'ivf':
        # IVF index - faster approximate search
        nlist = min(int(np.sqrt(combined_embeddings.shape[0])), 1000)
        quantizer = faiss.IndexFlatIP(embedding_dim)
        unified_index = faiss.IndexIVFFlat(quantizer, embedding_dim, nlist, faiss.METRIC_INNER_PRODUCT)
        print(f"  Training IVF with {nlist} clusters...")
        unified_index.train(combined_embeddings)
    else:
        raise ValueError(f"Unsupported index type: {index_type}")

    # Add vectors (already normalized from original indexes)
    unified_index.add(combined_embeddings)

    print(f"  Index built: {unified_index.ntotal} vectors")

    # Determine output path
    if output_path is None:
        output_path = INDEXES_DIR / f"ALL_{finish}_composite.faiss"

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Save FAISS index
    faiss.write_index(unified_index, str(output_path))
    print(f"\nSaved unified index to: {output_path}")

    # Save metadata
    meta_path = output_path.with_suffix('.meta')
    meta_dict = {
        'card_ids': all_card_ids,
        'embedding_dim': embedding_dim,
        'use_cosine': True,
        'index_type': index_type,
        'num_vectors': unified_index.ntotal,
        'metadata': {
            'set_codes': merged_set_codes,
            'finish': finish,
            'embedding_type': 'composite',
            'is_unified': True,
            'weights': sample_meta.get('metadata', {}).get('weights', {
                'full': 0.70,
                'collector': 0.20,
                'name': 0.10
            })
        }
    }

    with open(meta_path, 'wb') as f:
        pickle.dump(meta_dict, f)

    print(f"Saved metadata to: {meta_path}")
    print(f"\nMerged {len(merged_set_codes)} sets: {', '.join(merged_set_codes)}")

    return len(merged_set_codes), unified_index.ntotal, output_path


def main():
    parser = argparse.ArgumentParser(
        description='Build unified ALL index from existing per-set indexes'
    )
    parser.add_argument(
        '--finish',
        default='nonfoil',
        choices=['nonfoil', 'foil'],
        help='Finish type (default: nonfoil)'
    )
    parser.add_argument(
        '--sets',
        type=str,
        default=None,
        help='Comma-separated list of set codes to include (default: all)'
    )
    parser.add_argument(
        '--output',
        type=str,
        default=None,
        help='Output path (default: data/indexes/ALL_{finish}_composite.faiss)'
    )
    parser.add_argument(
        '--index-type',
        default='flat',
        choices=['flat', 'ivf'],
        help='FAISS index type. Use ivf for very large indexes (default: flat)'
    )
    parser.add_argument(
        '--list-only',
        action='store_true',
        help='Only list available indexes without building'
    )

    args = parser.parse_args()

    # Parse set codes if provided
    set_codes = None
    if args.sets:
        set_codes = [s.strip() for s in args.sets.split(',') if s.strip()]

    # List mode
    if args.list_only:
        available = discover_indexes(args.finish)
        print(f"Available {args.finish} indexes ({len(available)}):")
        for code, path in available:
            print(f"  {code}: {path}")
        return 0

    # Build unified index
    try:
        output_path = Path(args.output) if args.output else None

        num_sets, num_vectors, saved_path = build_unified_index(
            finish=args.finish,
            set_codes=set_codes,
            output_path=output_path,
            index_type=args.index_type
        )

        print(f"\n{'='*60}")
        print(f"SUCCESS: Unified index built!")
        print(f"  Sets merged: {num_sets}")
        print(f"  Total vectors: {num_vectors:,}")
        print(f"  Index file: {saved_path}")
        print(f"  Index type: {args.index_type}")
        print(f"{'='*60}")

        return 0

    except Exception as e:
        print(f"\nERROR: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == '__main__':
    sys.exit(main())
