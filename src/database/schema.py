"""
src/database/schema.py: Database schema definitions using SQLAlchemy
"""

from sqlalchemy import create_engine, Column, Integer, String, Float, DateTime, ForeignKey, JSON, BLOB
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, relationship
from datetime import datetime
import json

from src.config import DATABASE_PATH

Base = declarative_base()


class Card(Base):
    """Reference card from Scryfall dataset."""
    __tablename__ = "cards"

    id = Column(String(100), primary_key=True)  # Unique card ID (scryfall_id or composite)
    scryfall_id = Column(String(100), nullable=True, index=True)
    name = Column(String(255), nullable=False, index=True)
    set_code = Column(String(10), nullable=False, index=True)
    collector_number = Column(String(20), nullable=False, index=True)
    finish = Column(String(20), nullable=False, index=True)  # 'foil', 'nonfoil', 'etched'
    variant_type = Column(String(50), nullable=True, index=True)  # 'normal', 'extended_art', 'showcase', 'borderless', etc.
    frame_effects = Column(JSON, nullable=True)  # Additional frame treatments (textured, etched, gilded, etc.)
    image_path = Column(String(500), nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)

    # Relationships
    phash_variants = relationship("PhashVariant", back_populates="card", cascade="all, delete-orphan")
    composite_embedding = relationship("CompositeEmbedding", back_populates="card", uselist=False, cascade="all, delete-orphan")

    def __repr__(self):
        return f"<Card(id={self.id}, name='{self.name}', set='{self.set_code}', collector='{self.collector_number}')>"


class PhashVariant(Base):
    """pHash variants table (3 per card: full, name, collector)."""
    __tablename__ = "phash_variants"

    id = Column(Integer, primary_key=True, autoincrement=True)
    card_id = Column(String(100), ForeignKey("cards.id", ondelete="CASCADE"), nullable=False, index=True)
    variant_type = Column(String(20), nullable=False, index=True)  # 'full', 'name', 'collector'
    phash = Column(String(64), nullable=False, index=True)  # pHash as hex string (safer for SQLite)

    # Relationship
    card = relationship("Card", back_populates="phash_variants")

    def __repr__(self):
        return f"<PhashVariant(card_id={self.card_id}, type='{self.variant_type}', phash={self.phash})>"


class CompositeEmbedding(Base):
    """
    Composite embedding table - single pre-weighted embedding per card

    Combines three regions with optimal weights:
    - Full card (45%): Overall appearance, art, borders
    - Collector region (30%): Set code + collector number (most discriminative!)
    - Name region (25%): Variant discrimination

    This design fixes the systemic FAISS search failure by aligning
    retrieval with scoring. The composite embedding is used for FAISS
    search, ensuring correct cards are in the candidate pool.
    """
    __tablename__ = "composite_embeddings"

    card_id = Column(String(100), ForeignKey("cards.id", ondelete="CASCADE"), primary_key=True)

    # Pre-weighted composite embedding (512-dim)
    # This is the MAIN embedding used for FAISS retrieval
    embedding = Column(BLOB, nullable=False)

    # Individual component embeddings (stored for debugging/analysis)
    full_embedding = Column(BLOB, nullable=True)
    collector_embedding = Column(BLOB, nullable=True)
    name_embedding = Column(BLOB, nullable=True)

    # Weights used to create composite (for reproducibility)
    weight_full = Column(Float, default=0.45)
    weight_collector = Column(Float, default=0.30)
    weight_name = Column(Float, default=0.25)

    # Relationship
    card = relationship("Card", back_populates="composite_embedding")

    def __repr__(self):
        return f"<CompositeEmbedding(card_id={self.card_id}, weights={self.weight_full}/{self.weight_collector}/{self.weight_name})>"


# Database engine and session factory
engine = create_engine(f"sqlite:///{DATABASE_PATH}", echo=False)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)


def init_db():
    """Initialize database - create all tables."""
    Base.metadata.create_all(bind=engine)


def get_db():
    """
    Dependency function for FastAPI to get database session
    Yields a database session and ensures it's closed after use
    """
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

