"""
src/database/db.py: SQLite database operations
Provides helper functions for common database operations
"""

from sqlalchemy.orm import Session
from sqlalchemy import and_, or_
from typing import List, Optional, Dict, Any
from datetime import datetime
from contextlib import contextmanager

from src.database.schema import (
    Card, PhashVariant, CompositeEmbedding,
    SessionLocal, init_db
)


def get_card_by_id(db: Session, card_id: int) -> Optional[Card]:
    """Get card by ID"""
    return db.query(Card).filter(Card.id == card_id).first()


def get_card_by_image_path(db: Session, image_path: str) -> Optional[Card]:
    """Get card by image path"""
    return db.query(Card).filter(Card.image_path == image_path).first()


def get_cards_by_set_finish(db: Session, set_code: str, finish: str) -> List[Card]:
    """Get all cards for a specific set and finish"""
    return db.query(Card).filter(
        and_(Card.set_code == set_code, Card.finish == finish)
    ).all()


def get_card_by_collector_number(db: Session, set_code: str, collector_number: str, finish: str) -> Optional[Card]:
    """
    Get card by set code, collector number, and finish
    This is a unique identifier for most cards

    Args:
        db: Database session
        set_code: Set code (e.g., 'SLD', 'M21')
        collector_number: Collector number as string
        finish: Finish type ('foil' or 'nonfoil')

    Returns:
        Card object if found, None otherwise
    """
    return db.query(Card).filter(
        and_(
            Card.set_code == set_code,
            Card.collector_number == collector_number,
            Card.finish == finish
        )
    ).first()


def create_card(
    db: Session,
    name: str,
    set_code: str,
    collector_number: Optional[str],
    finish: str,
    image_path: str,
    phash: str,
    vector_id: int,
    metadata_json: Optional[Dict[str, Any]] = None,
    commit: bool = True
) -> Card:
    """
    Create a new card record
    Returns the created Card object
    
    Args:
        commit: If False, don't commit immediately (for batch operations)
    """
    card = Card(
        name=name,
        set_code=set_code,
        collector_number=collector_number,
        finish=finish,
        image_path=image_path,
        phash=phash,
        vector_id=vector_id,
        metadata_json=metadata_json
    )
    db.add(card)
    if commit:
        db.commit()
        db.refresh(card)
    else:
        db.flush()
    return card


def search_cards_by_phash(
    db: Session,
    phash: str,
    set_code: str,
    finish: str,
    max_hamming: int = 10,
    top_n: int = 200
) -> List[Card]:
    """
    Search cards by perceptual hash with Hamming distance threshold
    Optimized to query only pHash column and filter by set+finish
    
    Args:
        db: Database session
        phash: Query perceptual hash
        set_code: Set code to search in
        finish: Finish type to search in
        max_hamming: Maximum Hamming distance threshold
        top_n: Maximum number of results to return
        
    Returns:
        List of Card objects sorted by Hamming distance
    """
    # Query only pHash column for this set+finish (more efficient)
    phash_results = db.query(Card.id, Card.phash).filter(
        and_(Card.set_code == set_code, Card.finish == finish)
    ).all()
    
    if not phash_results:
        return []
    
    # Extract hashes and IDs
    candidate_hashes = [result.phash for result in phash_results]
    card_ids = [result.id for result in phash_results]
    
    # Use batch Hamming distance computation
    from src.indexing.phash import filter_by_hamming_distance
    matches = filter_by_hamming_distance(phash, candidate_hashes, max_hamming, top_n)
    
    if not matches:
        return []
    
    # Get card IDs for matches
    match_indices = [idx for idx, _ in matches]
    match_card_ids = [card_ids[idx] for idx in match_indices]
    
    # Query full Card objects for matched IDs
    # Use IN clause for efficiency
    cards = db.query(Card).filter(Card.id.in_(match_card_ids)).all()
    
    # Sort by distance (maintain order from matches)
    card_dict = {card.id: card for card in cards}
    sorted_cards = [card_dict[card_id] for card_id in match_card_ids if card_id in card_dict]

    return sorted_cards


@contextmanager
def transaction(db: Session):
    """
    Context manager for database transactions
    Automatically commits on success, rolls back on exception
    
    Usage:
        with transaction(db) as txn:
            # Do database operations
            db.add(some_object)
            # Commit happens automatically on exit
    """
    try:
        yield db
        db.commit()
    except Exception:
        db.rollback()
        raise


def begin_transaction(db: Session):
    """
    Begin a new transaction (explicit control)
    Note: SQLAlchemy uses autocommit=False by default, so transactions are implicit
    This is mainly for clarity and explicit transaction boundaries
    """
    # SQLAlchemy sessions are already transactional
    # This function is for explicit transaction boundaries
    pass


def commit_transaction(db: Session):
    """Commit the current transaction"""
    db.commit()


def rollback_transaction(db: Session):
    """Rollback the current transaction"""
    db.rollback()


# New functions for PhashVariant and CompositeEmbedding tables (PRD.md schema)

def get_phash_variants_by_card(db: Session, card_id: str) -> List[PhashVariant]:
    """
    Get all pHash variants for a card

    Args:
        db: Database session
        card_id: Card ID

    Returns:
        List of PhashVariant objects
    """
    return db.query(PhashVariant).filter(PhashVariant.card_id == card_id).all()


def get_phash_variant_by_type(db: Session, card_id: str, variant_type: str) -> Optional[PhashVariant]:
    """
    Get specific pHash variant for a card

    Args:
        db: Database session
        card_id: Card ID
        variant_type: Variant type ('full', 'name', 'collector')

    Returns:
        PhashVariant object if found, None otherwise
    """
    return db.query(PhashVariant).filter(
        and_(
            PhashVariant.card_id == card_id,
            PhashVariant.variant_type == variant_type
        )
    ).first()


def search_cards_by_phash_variant(
    db: Session,
    phash: int,
    variant_type: str,
    set_code: Optional[str] = None,
    finish: Optional[str] = None,
    max_hamming: int = 10,
    top_n: int = 200
) -> List[tuple]:
    """
    Search cards by pHash variant with Hamming distance threshold
    Following PRD.md specifications

    Args:
        db: Database session
        phash: Query perceptual hash (as integer)
        variant_type: Variant type ('full', 'name', 'collector')
        set_code: Optional set code filter
        finish: Optional finish type filter
        max_hamming: Maximum Hamming distance threshold
        top_n: Maximum number of results to return

    Returns:
        List of tuples (Card, PhashVariant, hamming_distance) sorted by distance
    """
    # Build query
    query = db.query(Card, PhashVariant).join(
        PhashVariant, Card.id == PhashVariant.card_id
    ).filter(PhashVariant.variant_type == variant_type)

    # Apply optional filters
    if set_code:
        query = query.filter(Card.set_code == set_code.upper())
    if finish:
        query = query.filter(Card.finish == finish.lower())

    # Get all candidates
    candidates = query.all()

    if not candidates:
        return []

    # Compute Hamming distances
    from src.indexing.phash import batch_hamming_distance, int_to_phash
    candidate_hashes = [int_to_phash(pv.phash) for _, pv in candidates]
    query_hash = int_to_phash(phash)

    distances = batch_hamming_distance(query_hash, candidate_hashes)

    # Filter by threshold and sort
    results = []
    for (card, pv), dist in zip(candidates, distances):
        if dist <= max_hamming:
            results.append((card, pv, int(dist)))

    # Sort by distance
    results.sort(key=lambda x: x[2])

    # Limit to top N
    return results[:top_n]


def get_embedding_by_card(db: Session, card_id: str) -> Optional[CompositeEmbedding]:
    """
    Get composite embedding for a card

    Args:
        db: Database session
        card_id: Card ID

    Returns:
        CompositeEmbedding object if found, None otherwise
    """
    return db.query(CompositeEmbedding).filter(CompositeEmbedding.card_id == card_id).first()


def get_cards_with_embeddings(
    db: Session,
    set_code: Optional[str] = None,
    finish: Optional[str] = None,
    limit: Optional[int] = None
) -> List[tuple]:
    """
    Get cards with their composite embeddings

    Args:
        db: Database session
        set_code: Optional set code filter
        finish: Optional finish type filter
        limit: Optional limit on number of results

    Returns:
        List of tuples (Card, CompositeEmbedding)
    """
    query = db.query(Card, CompositeEmbedding).join(
        CompositeEmbedding, Card.id == CompositeEmbedding.card_id
    )

    # Apply optional filters
    if set_code:
        query = query.filter(Card.set_code == set_code.upper())
    if finish:
        query = query.filter(Card.finish == finish.lower())

    # Apply limit
    if limit:
        query = query.limit(limit)

    return query.all()


def create_card_with_variants_and_embedding(
    db: Session,
    card_id: str,
    scryfall_id: Optional[str],
    name: str,
    set_code: str,
    collector_number: Optional[str],
    finish: str,
    image_path: str,
    phash_variants: Dict[str, int],
    embedding_blob: bytes,
    commit: bool = True
) -> Card:
    """
    Create a card with all its pHash variants and embedding in one transaction
    Following PRD.md schema

    Args:
        db: Database session
        card_id: Unique card ID
        scryfall_id: Scryfall ID
        name: Card name
        set_code: Set code
        collector_number: Collector number
        finish: Finish type
        image_path: Path to image
        phash_variants: Dict of {'full': phash_int, 'name': phash_int, 'collector': phash_int}
        embedding_blob: Serialized embedding as bytes
        commit: If False, don't commit (for batch operations)

    Returns:
        Created Card object
    """
    # Create card
    card = Card(
        id=card_id,
        scryfall_id=scryfall_id,
        name=name,
        set_code=set_code.upper(),
        collector_number=collector_number,
        finish=finish.lower(),
        image_path=image_path
    )
    db.add(card)
    db.flush()  # Flush to get card ID

    # Create pHash variants
    for variant_type, phash_int in phash_variants.items():
        pv = PhashVariant(
            card_id=card.id,
            variant_type=variant_type,
            phash=phash_int
        )
        db.add(pv)

    # Create composite embedding
    emb = CompositeEmbedding(
        card_id=card.id,
        embedding=embedding_blob
    )
    db.add(emb)

    if commit:
        db.commit()
        db.refresh(card)
    else:
        db.flush()

    return card


def get_card_full_data(db: Session, card_id: str) -> Optional[Dict[str, Any]]:
    """
    Get card with all its pHash variants and embedding

    Args:
        db: Database session
        card_id: Card ID

    Returns:
        Dict with card, phash_variants, and embedding, or None if not found
    """
    card = db.query(Card).filter(Card.id == card_id).first()
    if not card:
        return None

    # Get pHash variants
    phash_variants = db.query(PhashVariant).filter(PhashVariant.card_id == card_id).all()

    # Get composite embedding
    embedding = db.query(CompositeEmbedding).filter(CompositeEmbedding.card_id == card_id).first()

    return {
        'card': card,
        'phash_variants': {pv.variant_type: pv for pv in phash_variants},
        'embedding': embedding
    }

