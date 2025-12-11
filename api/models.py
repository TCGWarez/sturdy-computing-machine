"""
Pydantic models for API request/response schemas
"""

from pydantic import BaseModel, Field
from typing import Optional, List
from datetime import datetime
from enum import Enum

class BatchStatus(str, Enum):
    """Batch processing status"""
    UPLOADING = "uploading"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"

class BatchCreate(BaseModel):
    """Request to create a new batch"""
    set_code: Optional[str] = Field(None, description="Optional set code filter (e.g., 'TLA', 'DMR')")

class BatchInfo(BaseModel):
    """Batch metadata"""
    batch_id: str
    status: BatchStatus
    total_cards: int
    processed_cards: int
    created_at: datetime
    set_code: Optional[str] = None

class CardMatch(BaseModel):
    """Single card recognition result"""
    image_id: str = Field(..., description="Unique ID for this image in batch")
    image_filename: str
    image_url: str = Field(..., description="URL to view uploaded image")

    # Matched card info
    card_id: str
    card_name: str
    set_code: str
    collector_number: str
    finish: str
    confidence: float = Field(..., ge=0.0, le=1.0)

    # Reference image URL (Scryfall image for comparison)
    reference_image_url: Optional[str] = None

    # Clarity scoring (search-based matching)
    clarity_score: float = Field(1.0, ge=0.0, le=1.0, description="Gap between top 2 candidates")
    is_ambiguous: bool = Field(False, description="True if multiple high-confidence candidates")

    # Correction info
    is_corrected: bool = False
    corrected_card_id: Optional[str] = None
    correction_reason: Optional[str] = None

    # Variant info (if multiple exist)
    has_variants: bool = False
    variant_count: int = 0

    # Boundary detection (4 corners in original image for visualization)
    # Format: [[x1,y1], [x2,y2], [x3,y3], [x4,y4]]
    boundary_corners: Optional[List[List[float]]] = None

    # Top candidates for review mode (excluding the matched card)
    candidates: Optional[List["CandidateMatch"]] = None

class BatchResults(BaseModel):
    """Complete batch results"""
    batch_info: BatchInfo
    results: List[CardMatch]

class CorrectionRequest(BaseModel):
    """Request to correct a card match"""
    image_id: str
    correct_card_id: str
    reason: Optional[str] = Field(None, description="Why this correction was made")

class CorrectionResponse(BaseModel):
    """Response after correction"""
    success: bool
    image_id: str
    updated_match: CardMatch

class ExportFormat(str, Enum):
    """Export file format"""
    CSV = "csv"
    JSON = "json"
    MANAPOOL_CSV = "manapool_csv"

class CardSearchResult(BaseModel):
    """Card search result (for correction modal)"""
    card_id: str
    card_name: str
    set_code: str
    collector_number: str
    finish: str
    scryfall_image_url: Optional[str] = None


class CandidateMatch(BaseModel):
    """A candidate match for review mode"""
    card_id: str
    card_name: str
    set_code: str
    collector_number: str
    finish: str
    confidence: float
    reference_image_url: str  # URL to serve reference image


# Rebuild models to resolve forward references
CardMatch.model_rebuild()
