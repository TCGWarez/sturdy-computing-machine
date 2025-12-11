"""
Database schema and operations for batch recognition
"""

from sqlalchemy import create_engine, Column, String, Integer, Float, Boolean, DateTime, Text, ForeignKey
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, relationship
from datetime import datetime
from pathlib import Path
import sys

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.config import DATABASE_PATH

engine = create_engine(f"sqlite:///{DATABASE_PATH}", echo=False)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

Base = declarative_base()

class Batch(Base):
    """Batch upload job"""
    __tablename__ = "batches"

    id = Column(String, primary_key=True)
    user_id = Column(String, nullable=True)
    status = Column(String, nullable=False, default="uploading")
    total_cards = Column(Integer, nullable=False, default=0)
    processed_cards = Column(Integer, nullable=False, default=0)
    set_code = Column(String, nullable=True)
    created_at = Column(DateTime, nullable=False, default=datetime.utcnow)
    completed_at = Column(DateTime, nullable=True)

    results = relationship("BatchResult", back_populates="batch", cascade="all, delete-orphan")

class BatchResult(Base):
    """Single card recognition result within a batch"""
    __tablename__ = "batch_results"

    id = Column(Integer, primary_key=True, autoincrement=True)
    batch_id = Column(String, ForeignKey("batches.id"), nullable=False)

    image_id = Column(String, nullable=False, unique=True)
    image_filename = Column(String, nullable=False)
    image_path = Column(String, nullable=False)

    matched_card_id = Column(String, nullable=False)
    confidence = Column(Float, nullable=False)

    clarity_score = Column(Float, nullable=False, default=1.0)
    is_ambiguous = Column(Boolean, nullable=False, default=False)

    candidates_json = Column(Text, nullable=True)

    is_corrected = Column(Boolean, nullable=False, default=False)
    corrected_card_id = Column(String, nullable=True)
    correction_reason = Column(Text, nullable=True)
    corrected_at = Column(DateTime, nullable=True)

    batch = relationship("Batch", back_populates="results")

def init_batch_tables():
    """Create batch tables if they don't exist"""
    Base.metadata.create_all(bind=engine)

def get_db():
    """Get database session (dependency injection for FastAPI)"""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

if __name__ == "__main__":
    init_batch_tables()
