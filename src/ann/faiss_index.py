"""
src/ann/faiss_index.py: FAISS-based ANN index wrapper
Following PRD.md Task 6 specifications
- Build persistent index from embeddings
- Support build_index(embeddings) and query(embedding, top_k)
- Serialize index files with metadata
"""

import faiss
import numpy as np
import pickle
from pathlib import Path
from typing import List, Tuple, Optional, Dict, Any
from sqlalchemy.orm import Session

from src.config import INDEXES_DIR
from src.database.schema import SessionLocal, Card, CompositeEmbedding
from src.indexing.indexer import deserialize_embedding


class FAISSIndex:
    """
    FAISS-based ANN index for fast similarity search
    Following PRD.md specifications
    """

    def __init__(
        self,
        embedding_dim: int = 512,
        use_cosine: bool = True,
        index_type: str = 'flat'
    ):
        """
        Initialize FAISS index

        Args:
            embedding_dim: Dimension of embedding vectors (default 512 for CLIP)
            use_cosine: If True, use cosine similarity. If False, use L2 distance
            index_type: Type of index ('flat', 'ivf', 'hnsw')
        """
        self.embedding_dim = embedding_dim
        self.use_cosine = use_cosine
        self.index_type = index_type
        self.index = None
        self.card_ids = []  # Ordered list of card IDs corresponding to index positions
        self.metadata = {}  # Additional metadata (set_code, finish, etc.)

    def build_index(
        self,
        embeddings: np.ndarray,
        card_ids: List[str],
        nlist: int = 100,
        m: int = 32
    ):
        """
        Build index from embeddings

        Args:
            embeddings: Array of shape (n, embedding_dim)
            card_ids: List of card IDs corresponding to embeddings
            nlist: Number of clusters for IVF index
            m: Number of connections for HNSW index
        """
        if embeddings.shape[0] != len(card_ids):
            raise ValueError(f"Embeddings ({embeddings.shape[0]}) and card_ids ({len(card_ids)}) length mismatch")

        if embeddings.shape[1] != self.embedding_dim:
            raise ValueError(f"Embedding dim mismatch: expected {self.embedding_dim}, got {embeddings.shape[1]}")

        embeddings = embeddings.astype(np.float32)

        if self.use_cosine:
            faiss.normalize_L2(embeddings)

        if self.index_type == 'flat':
            if self.use_cosine:
                self.index = faiss.IndexFlatIP(self.embedding_dim)
            else:
                self.index = faiss.IndexFlatL2(self.embedding_dim)

        elif self.index_type == 'ivf':
            quantizer = faiss.IndexFlatL2(self.embedding_dim)
            if self.use_cosine:
                self.index = faiss.IndexIVFFlat(quantizer, self.embedding_dim, nlist, faiss.METRIC_INNER_PRODUCT)
            else:
                self.index = faiss.IndexIVFFlat(quantizer, self.embedding_dim, nlist, faiss.METRIC_L2)
            self.index.train(embeddings)

        elif self.index_type == 'hnsw':
            if self.use_cosine:
                self.index = faiss.IndexHNSWFlat(self.embedding_dim, m, faiss.METRIC_INNER_PRODUCT)
            else:
                self.index = faiss.IndexHNSWFlat(self.embedding_dim, m, faiss.METRIC_L2)

        else:
            raise ValueError(f"Unknown index type: {self.index_type}")

        self.index.add(embeddings)
        self.card_ids = list(card_ids)

    def query(
        self,
        embedding: np.ndarray,
        top_k: int = 20
    ) -> List[Tuple[str, float]]:
        """
        Query index for similar vectors

        Args:
            embedding: Query embedding of shape (embedding_dim,) or (1, embedding_dim)
            top_k: Number of results to return

        Returns:
            List of tuples (card_id, similarity_score) sorted by similarity descending
        """
        if self.index is None or self.index.ntotal == 0:
            return []

        # Ensure embedding is 2D and float32
        if embedding.ndim == 1:
            embedding = embedding.reshape(1, -1)
        embedding = embedding.astype(np.float32)

        # Normalize if using cosine similarity
        if self.use_cosine:
            faiss.normalize_L2(embedding)

        # Search
        k = min(top_k, self.index.ntotal)
        distances, indices = self.index.search(embedding, k)

        # Convert to results
        results = []
        for idx, dist in zip(indices[0], distances[0]):
            if idx == -1:  # FAISS returns -1 for empty slots
                continue

            if idx >= len(self.card_ids):
                continue

            card_id = self.card_ids[idx]

            if self.use_cosine:
                score = float(dist)
            else:
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

        faiss.write_index(self.index, str(index_path))

        metadata_path = index_path.with_suffix('.meta')
        meta_dict = {
            'card_ids': self.card_ids,
            'embedding_dim': self.embedding_dim,
            'use_cosine': self.use_cosine,
            'index_type': self.index_type,
            'num_vectors': self.index.ntotal,
            'metadata': metadata or {}
        }

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

        self.index = faiss.read_index(str(index_path))

        metadata_path = index_path.with_suffix('.meta')
        if not metadata_path.exists():
            raise FileNotFoundError(f"Metadata file not found: {metadata_path}")

        with open(metadata_path, 'rb') as f:
            meta_dict = pickle.load(f)

        self.card_ids = meta_dict['card_ids']
        self.embedding_dim = meta_dict['embedding_dim']
        self.use_cosine = meta_dict['use_cosine']
        self.index_type = meta_dict['index_type']
        self.metadata = meta_dict.get('metadata', {})

    def get_num_vectors(self) -> int:
        """Get number of vectors in index"""
        return self.index.ntotal if self.index else 0


def build_composite_index_from_db(
    set_code: str,
    finish: str = 'nonfoil',
    index_type: str = 'flat',
    output_path: Optional[Path] = None
) -> FAISSIndex:
    """Build FAISS index from composite embeddings using pre-weighted vectors."""
    db = SessionLocal()
    try:
        cards_with_embeddings = (
            db.query(Card, CompositeEmbedding)
            .join(CompositeEmbedding, Card.id == CompositeEmbedding.card_id)
            .filter(Card.set_code == set_code.upper())
            .filter(Card.finish == finish.lower())
            .all()
        )

        if not cards_with_embeddings:
            raise ValueError(f"No cards with composite embeddings found for {set_code}/{finish}")

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
                    'finish': finish,
                    'index_type': index_type,
                    'embedding_type': 'composite',
                    'weights': {
                        'full': embedding_record.weight_full,
                        'collector': embedding_record.weight_collector,
                        'name': embedding_record.weight_name
                    }
                }
            )

        return index

    finally:
        db.close()


def main():
    """CLI entry point for building composite FAISS index."""
    import argparse

    parser = argparse.ArgumentParser(description='Build composite FAISS index')
    parser.add_argument('--set', required=True, help='Set code (e.g., ONE, DSK)')
    parser.add_argument('--finish', default='nonfoil', choices=['foil', 'nonfoil', 'etched'])
    parser.add_argument('--index-type', default='flat', choices=['flat', 'ivf', 'hnsw'])
    parser.add_argument('--output', type=str, default=None)

    args = parser.parse_args()

    if args.output:
        output_path = Path(args.output)
    else:
        output_path = INDEXES_DIR / f"{args.set.upper()}_{args.finish}_composite.faiss"

    try:
        build_composite_index_from_db(
            set_code=args.set,
            finish=args.finish,
            index_type=args.index_type,
            output_path=output_path
        )
        return 0
    except Exception:
        return 1


if __name__ == '__main__':
    exit(main())
