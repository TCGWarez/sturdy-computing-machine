"""
src/ann/hnsw_index.py: HNSWlib-based ANN index wrapper
Alternative to FAISS following PRD.md Task 6 specifications
- Simpler and faster than FAISS for some use cases
- Build persistent index from embeddings
- Support build_index(embeddings) and query(embedding, top_k)
- Serialize index files with metadata
"""

import hnswlib
import numpy as np
import pickle
from pathlib import Path
from typing import List, Tuple, Optional, Dict, Any
from sqlalchemy.orm import Session

from src.config import INDEXES_DIR
from src.database.schema import SessionLocal, Card, CompositeEmbedding
from src.indexing.indexer import deserialize_embedding


class HNSWIndex:
    """
    HNSWlib-based ANN index for fast similarity search
    Following PRD.md specifications
    """

    def __init__(
        self,
        embedding_dim: int = 512,
        space: str = 'cosine',
        max_elements: int = 100000
    ):
        """
        Initialize HNSW index

        Args:
            embedding_dim: Dimension of embedding vectors (default 512 for CLIP)
            space: Distance space ('cosine', 'l2', 'ip')
            max_elements: Maximum number of elements (can be increased later)
        """
        self.embedding_dim = embedding_dim
        self.space = space
        self.max_elements = max_elements
        self.index = None
        self.card_ids = []  # Ordered list of card IDs corresponding to index positions
        self.metadata = {}  # Additional metadata (set_code, finish, etc.)

    def build_index(
        self,
        embeddings: np.ndarray,
        card_ids: List[str],
        ef_construction: int = 200,
        m: int = 16
    ):
        """
        Build index from embeddings

        Args:
            embeddings: Array of shape (n, embedding_dim)
            card_ids: List of card IDs corresponding to embeddings
            ef_construction: Controls index construction speed/quality tradeoff (higher = better quality, slower)
            m: Number of bi-directional links per element (higher = better recall, more memory)
        """
        if embeddings.shape[0] != len(card_ids):
            raise ValueError(f"Embeddings ({embeddings.shape[0]}) and card_ids ({len(card_ids)}) length mismatch")

        if embeddings.shape[1] != self.embedding_dim:
            raise ValueError(f"Embedding dim mismatch: expected {self.embedding_dim}, got {embeddings.shape[1]}")


        # Convert to float32 if needed
        embeddings = embeddings.astype(np.float32)

        # Create index
        self.index = hnswlib.Index(space=self.space, dim=self.embedding_dim)

        # Initialize index
        self.index.init_index(
            max_elements=max(embeddings.shape[0], self.max_elements),
            ef_construction=ef_construction,
            M=m
        )

        # Set ef for search (can be different from ef_construction)
        self.index.set_ef(50)  # Default search ef

        self.index.add_items(embeddings, np.arange(embeddings.shape[0]))

        self.card_ids = list(card_ids)

    def query(
        self,
        embedding: np.ndarray,
        top_k: int = 20,
        ef: Optional[int] = None
    ) -> List[Tuple[str, float]]:
        """
        Query index for similar vectors

        Args:
            embedding: Query embedding of shape (embedding_dim,) or (1, embedding_dim)
            top_k: Number of results to return
            ef: Controls search accuracy (higher = more accurate, slower). If None, uses default.

        Returns:
            List of tuples (card_id, similarity_score) sorted by similarity descending
        """
        if self.index is None or len(self.card_ids) == 0:
            return []

        # Ensure embedding is 1D for hnswlib and float32
        if embedding.ndim == 2:
            embedding = embedding.squeeze(0)
        embedding = embedding.astype(np.float32)

        # Set ef for this query if specified
        if ef is not None:
            self.index.set_ef(ef)

        # Search
        k = min(top_k, len(self.card_ids))
        indices, distances = self.index.knn_query(embedding, k=k)

        # Convert to results
        results = []
        for idx, dist in zip(indices[0], distances[0]):
            if idx >= len(self.card_ids):
                continue

            card_id = self.card_ids[idx]

            # Convert distance to similarity score
            if self.space == 'cosine':
                # Cosine distance in [0, 2], convert to similarity in [0, 1]
                score = 1.0 - (float(dist) / 2.0)
            elif self.space == 'ip':
                # Inner product (higher is better)
                score = float(dist)
            else:  # l2
                # L2 distance - convert to similarity
                score = 1.0 / (1.0 + float(dist))

            results.append((card_id, score))

        return results

    def save(
        self,
        index_path: Path,
        metadata: Optional[Dict[str, Any]] = None
    ):
        """
        Save index and metadata to disk

        Args:
            index_path: Path to save index file
            metadata: Additional metadata to save
        """
        if self.index is None:
            raise ValueError("No index to save")

        index_path = Path(index_path)
        index_path.parent.mkdir(parents=True, exist_ok=True)

        self.index.save_index(str(index_path))

        # Save metadata (card IDs and config)
        metadata_path = index_path.with_suffix('.meta')
        meta_dict = {
            'card_ids': self.card_ids,
            'embedding_dim': self.embedding_dim,
            'space': self.space,
            'max_elements': self.max_elements,
            'num_vectors': len(self.card_ids),
            'metadata': metadata or {}
        }

        # Merge additional metadata if provided
        if metadata:
            meta_dict['metadata'].update(metadata)

        with open(metadata_path, 'wb') as f:
            pickle.dump(meta_dict, f)

    def load(self, index_path: Path):
        """
        Load index and metadata from disk

        Args:
            index_path: Path to index file
        """
        index_path = Path(index_path)

        if not index_path.exists():
            raise FileNotFoundError(f"Index file not found: {index_path}")

        # Load metadata first to get config
        metadata_path = index_path.with_suffix('.meta')
        if not metadata_path.exists():
            raise FileNotFoundError(f"Metadata file not found: {metadata_path}")

        with open(metadata_path, 'rb') as f:
            meta_dict = pickle.load(f)

        self.card_ids = meta_dict['card_ids']
        self.embedding_dim = meta_dict['embedding_dim']
        self.space = meta_dict['space']
        self.max_elements = meta_dict['max_elements']
        self.metadata = meta_dict.get('metadata', {})

        # Create index and load
        self.index = hnswlib.Index(space=self.space, dim=self.embedding_dim)
        self.index.load_index(str(index_path), max_elements=self.max_elements)

    def get_num_vectors(self) -> int:
        """Get number of vectors in index"""
        return len(self.card_ids)


def build_index_from_db(
    set_code: str,
    finish: str = 'nonfoil',
    space: str = 'cosine',
    output_path: Optional[Path] = None,
    ef_construction: int = 200,
    m: int = 16
) -> HNSWIndex:
    """
    Build HNSW index from database embeddings

    Args:
        set_code: Set code to build index for
        finish: Finish type
        space: Distance space ('cosine', 'l2', 'ip')
        output_path: Optional path to save index
        ef_construction: Construction quality parameter
        m: Number of connections parameter

    Returns:
        Built HNSWIndex
    """

    # Query database for all cards with embeddings
    db = SessionLocal()
    try:
        # Query cards with composite embeddings for this set+finish
        cards_with_embeddings = (
            db.query(Card, CompositeEmbedding)
            .join(CompositeEmbedding, Card.id == CompositeEmbedding.card_id)
            .filter(Card.set_code == set_code.upper())
            .filter(Card.finish == finish.lower())
            .all()
        )

        if not cards_with_embeddings:
            raise ValueError(f"No cards with embeddings found for {set_code}/{finish}")

        # Extract embeddings and card IDs
        embeddings_list = []
        card_ids = []

        for card, embedding_record in cards_with_embeddings:
            # Deserialize embedding from BLOB
            embedding = deserialize_embedding(embedding_record.embedding)
            embeddings_list.append(embedding)
            card_ids.append(card.id)

        embeddings = np.vstack(embeddings_list)

        # Create and build index
        index = HNSWIndex(
            embedding_dim=embeddings.shape[1],
            space=space,
            max_elements=len(embeddings) * 2  # Allow for future growth
        )

        index.build_index(embeddings, card_ids, ef_construction=ef_construction, m=m)

        # Save if output path provided
        if output_path:
            index.save(
                output_path,
                metadata={
                    'set_code': set_code,
                    'finish': finish,
                    'space': space
                }
            )

        return index

    finally:
        db.close()


def main():
    """
    CLI entry point for building HNSW index
    """
    import argparse

    parser = argparse.ArgumentParser(description='Build HNSW index from database embeddings')
    parser.add_argument('--set', required=True, help='Set code (e.g., M21)')
    parser.add_argument('--finish', default='nonfoil', choices=['foil', 'nonfoil', 'etched'],
                        help='Finish type')
    parser.add_argument('--space', default='cosine', choices=['cosine', 'l2', 'ip'],
                        help='Distance space')
    parser.add_argument('--ef-construction', type=int, default=200,
                        help='Construction quality parameter (higher = better, slower)')
    parser.add_argument('--m', type=int, default=16,
                        help='Number of connections (higher = better recall, more memory)')
    parser.add_argument('--output', type=str, default=None,
                        help='Output path (default: data/indexes/{set}_{finish}.hnsw)')

    args = parser.parse_args()

    # Determine output path
    if args.output:
        output_path = Path(args.output)
    else:
        output_path = INDEXES_DIR / f"{args.set.upper()}_{args.finish}.hnsw"

    # Build index
    try:
        index = build_index_from_db(
            set_code=args.set,
            finish=args.finish,
            space=args.space,
            output_path=output_path,
            ef_construction=args.ef_construction,
            m=args.m
        )

        print(f"\nIndex built successfully!")
        print(f"Vectors: {index.get_num_vectors()}")
        print(f"Index saved to: {output_path}")
        print(f"\nYou can now use this index for recognition queries.")

    except Exception as e:
        print(f"Error building index: {e}")
        import traceback
        traceback.print_exc()
        return 1

    return 0


if __name__ == '__main__':
    exit(main())
