"""
Batch processing routes
Handles upload, recognition, corrections, and export
"""

from fastapi import APIRouter, UploadFile, File, Depends, HTTPException, BackgroundTasks
from fastapi.responses import FileResponse, StreamingResponse
from sqlalchemy.orm import Session
from typing import List, Optional, Tuple
import uuid
import shutil
import zipfile
from pathlib import Path
from datetime import datetime
import csv
import json
import io

from api.database import get_db, Batch, BatchResult
from api.models import (
    BatchCreate, BatchInfo, BatchResults, CardMatch,
    CorrectionRequest, CorrectionResponse, ExportFormat,
    CardSearchResult, BatchStatus, CandidateMatch
)
from api.services.recognition import recognize_card
from api.services.manapool_export import build_manapool_csv
from api.services.mtgsold_config import is_mtgsold_enabled
from src.database.schema import Card as DBCard

router = APIRouter()

UPLOAD_DIR = Path("uploads")
UPLOAD_DIR.mkdir(exist_ok=True)

MAX_BATCH_SIZE_NO_SET = 50

def _resolve_display_card(db: Session, batch_result: BatchResult):
    """Resolve the card to display (respecting corrections)."""
    card_id = batch_result.corrected_card_id if batch_result.is_corrected else batch_result.matched_card_id
    return db.query(DBCard).filter(DBCard.id == card_id).first()

def process_batch_background(batch_id: str, image_paths: List[Path], set_code: Optional[str], finish: Optional[str], prefer_foil: bool = False):
    """Background task to process batch recognition"""
    from api.database import SessionLocal
    db = SessionLocal()

    try:
        # Update batch status
        batch = db.query(Batch).filter(Batch.id == batch_id).first()
        batch.status = BatchStatus.PROCESSING
        db.commit()

        # Process each image
        for idx, image_path in enumerate(image_paths):
            try:
                # Run recognition with foil preference (device auto-detected)
                result = recognize_card(image_path, set_code=set_code, finish=finish, prefer_foil=prefer_foil)

                if 'error' not in result:
                    # Serialize candidates to JSON for storage
                    candidates_json = None
                    if 'candidates' in result and result['candidates']:
                        candidates_json = json.dumps(result['candidates'])

                    # Create batch result
                    image_id = str(uuid.uuid4())
                    batch_result = BatchResult(
                        batch_id=batch_id,
                        image_id=image_id,
                        image_filename=image_path.name,
                        image_path=str(image_path),
                        matched_card_id=result['card_id'],
                        confidence=result['combined_score'],
                        clarity_score=result.get('clarity_score', 1.0),
                        is_ambiguous=result.get('is_ambiguous', False),
                        candidates_json=candidates_json
                    )
                    db.add(batch_result)

                # Update progress
                batch.processed_cards = idx + 1
                db.commit()

            except Exception:
                continue

        # Mark batch as completed
        batch.status = BatchStatus.COMPLETED
        batch.completed_at = datetime.utcnow()
        db.commit()

    except Exception:
        batch = db.query(Batch).filter(Batch.id == batch_id).first()
        batch.status = BatchStatus.FAILED
        db.commit()

    finally:
        db.close()

@router.post("/upload", response_model=BatchInfo)
async def upload_batch(
    background_tasks: BackgroundTasks,
    files: List[UploadFile] = File(...),
    set_code: Optional[str] = None,
    finish: Optional[str] = None,
    prefer_foil: bool = False,
    db: Session = Depends(get_db)
):
    """
    Upload batch of card images for recognition
    Accepts multiple image files or a zip file

    Args:
        files: Image files or zip archive
        set_code: Set code filter (e.g., 'TLA', 'DMR'). If not provided, searches all indexed sets.
        finish: 'nonfoil' or 'foil' to filter search space
        prefer_foil: If true, prefer foil matches when ambiguous

    Note:
        When no set_code is provided, the system searches ALL indexed sets which is slower.
        Batch size is limited to 50 images in this case.
    """
    # Create batch record
    batch_id = str(uuid.uuid4())
    batch_dir = UPLOAD_DIR / batch_id
    batch_dir.mkdir(parents=True, exist_ok=True)

    # Handle files
    image_paths = []

    for file in files:
        filename = file.filename
        file_path = batch_dir / filename

        # Save file
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        # Check if zip
        if filename.endswith('.zip'):
            # Extract zip
            with zipfile.ZipFile(file_path, 'r') as zip_ref:
                zip_ref.extractall(batch_dir)

            # Find all images in extracted files
            for ext in ['*.jpg', '*.jpeg', '*.png']:
                image_paths.extend(batch_dir.rglob(ext))

            # Remove zip file
            file_path.unlink()
        else:
            # Regular image file
            image_paths.append(file_path)

    if not image_paths:
        raise HTTPException(status_code=400, detail="No valid images found")

    # Enforce batch size limit when no set_code is specified
    if not set_code and len(image_paths) > MAX_BATCH_SIZE_NO_SET:
        # Clean up uploaded files
        shutil.rmtree(batch_dir)
        raise HTTPException(
            status_code=400,
            detail=f"When no set_code is specified, batch size is limited to {MAX_BATCH_SIZE_NO_SET} images. "
                   f"You uploaded {len(image_paths)} images. Please specify a set_code or reduce batch size."
        )

    # Create batch in database
    batch = Batch(
        id=batch_id,
        status=BatchStatus.UPLOADING,
        total_cards=len(image_paths),
        processed_cards=0,
        set_code=set_code,
        created_at=datetime.utcnow()
    )
    db.add(batch)
    db.commit()

    # Start background processing
    background_tasks.add_task(process_batch_background, batch_id, image_paths, set_code, finish, prefer_foil)

    return BatchInfo(
        batch_id=batch_id,
        status=batch.status,
        total_cards=batch.total_cards,
        processed_cards=batch.processed_cards,
        created_at=batch.created_at,
        set_code=batch.set_code
    )

@router.get("/{batch_id}/status", response_model=BatchInfo)
async def get_batch_status(batch_id: str, db: Session = Depends(get_db)):
    """Get batch processing status"""
    batch = db.query(Batch).filter(Batch.id == batch_id).first()

    if not batch:
        raise HTTPException(status_code=404, detail="Batch not found")

    return BatchInfo(
        batch_id=batch.id,
        status=batch.status,
        total_cards=batch.total_cards,
        processed_cards=batch.processed_cards,
        created_at=batch.created_at,
        set_code=batch.set_code
    )

@router.get("/{batch_id}/results", response_model=BatchResults)
async def get_batch_results(batch_id: str, db: Session = Depends(get_db)):
    """Get complete batch results"""
    batch = db.query(Batch).filter(Batch.id == batch_id).first()

    if not batch:
        raise HTTPException(status_code=404, detail="Batch not found")

    # Get all results
    batch_results = db.query(BatchResult).filter(BatchResult.batch_id == batch_id).all()

    # Get card info for each result
    card_matches = []
    for br in batch_results:
        # Get matched card from database
        card = db.query(DBCard).filter(DBCard.id == br.matched_card_id).first()

        if not card:
            continue

        # Check if corrected
        display_card_id = br.corrected_card_id if br.is_corrected else br.matched_card_id
        display_card = db.query(DBCard).filter(DBCard.id == display_card_id).first()

        if not display_card:
            continue

        # Check for variants
        variant_count = db.query(DBCard).filter(
            DBCard.name == display_card.name,
            DBCard.set_code == display_card.set_code
        ).count()

        # Build reference image URL
        reference_image_url = f"/api/batch/reference/{display_card.id}"

        # Parse candidates from JSON if available
        candidates_list = None
        if br.candidates_json:
            try:
                raw_candidates = json.loads(br.candidates_json)
                candidates_list = [
                    CandidateMatch(
                        card_id=c['card_id'],
                        card_name=c['card_name'],
                        set_code=c['set_code'],
                        collector_number=c['collector_number'],
                        finish=c['finish'],
                        confidence=min(1.0, c['combined_score']),
                        reference_image_url=f"/api/batch/reference/{c['card_id']}"
                    )
                    for c in raw_candidates
                ]
            except (json.JSONDecodeError, KeyError):
                candidates_list = None

        card_matches.append(CardMatch(
            image_id=br.image_id,
            image_filename=br.image_filename,
            image_url=f"/api/batch/{batch_id}/image/{br.image_id}",
            card_id=display_card.id,
            card_name=display_card.name,
            set_code=display_card.set_code,
            collector_number=display_card.collector_number,
            finish=display_card.finish,
            confidence=min(1.0, br.confidence),  # Clamp to 1.0 max
            reference_image_url=reference_image_url,
            clarity_score=getattr(br, 'clarity_score', 1.0) or 1.0,
            is_ambiguous=getattr(br, 'is_ambiguous', False) or False,
            is_corrected=br.is_corrected,
            corrected_card_id=br.corrected_card_id,
            correction_reason=br.correction_reason,
            has_variants=(variant_count > 1),
            variant_count=variant_count,
            candidates=candidates_list
        ))

    return BatchResults(
        batch_info=BatchInfo(
            batch_id=batch.id,
            status=batch.status,
            total_cards=batch.total_cards,
            processed_cards=batch.processed_cards,
            created_at=batch.created_at,
            set_code=batch.set_code
        ),
        results=card_matches
    )

@router.post("/{batch_id}/correct", response_model=CorrectionResponse)
async def correct_match(
    batch_id: str,
    correction: CorrectionRequest,
    db: Session = Depends(get_db)
):
    """Correct a card match"""
    # Find batch result
    batch_result = db.query(BatchResult).filter(
        BatchResult.batch_id == batch_id,
        BatchResult.image_id == correction.image_id
    ).first()

    if not batch_result:
        raise HTTPException(status_code=404, detail="Image not found in batch")

    # Update correction
    batch_result.is_corrected = True
    batch_result.corrected_card_id = correction.correct_card_id
    batch_result.correction_reason = correction.reason
    batch_result.corrected_at = datetime.utcnow()
    db.commit()

    # Get corrected card info
    card = db.query(DBCard).filter(DBCard.id == correction.correct_card_id).first()

    if not card:
        raise HTTPException(status_code=404, detail="Card not found")

    return CorrectionResponse(
        success=True,
        image_id=correction.image_id,
        updated_match=CardMatch(
            image_id=batch_result.image_id,
            image_filename=batch_result.image_filename,
            image_url=f"/api/batch/{batch_id}/image/{batch_result.image_id}",
            card_id=card.id,
            card_name=card.name,
            set_code=card.set_code,
            collector_number=card.collector_number,
            finish=card.finish,
            confidence=min(1.0, batch_result.confidence),  # Clamp to 1.0 max
            is_corrected=True,
            corrected_card_id=card.id,
            correction_reason=correction.reason,
            has_variants=False,
            variant_count=0
        )
    )

@router.get("/{batch_id}/export")
async def export_batch(
    batch_id: str,
    format: ExportFormat = ExportFormat.CSV,
    db: Session = Depends(get_db)
):
    """Export batch results as CSV or JSON"""
    batch = db.query(Batch).filter(Batch.id == batch_id).first()

    if not batch:
        raise HTTPException(status_code=404, detail="Batch not found")

    # Get all results
    batch_results = db.query(BatchResult).filter(BatchResult.batch_id == batch_id).all()

    # Resolve cards once to avoid repeated DB queries
    resolved_rows: List[Tuple[BatchResult, DBCard]] = []
    for br in batch_results:
        card = _resolve_display_card(db, br)
        if card:
            resolved_rows.append((br, card))

    if format == ExportFormat.CSV:
        # Generate CSV
        output = io.StringIO()
        writer = csv.writer(output)

        # Header
        writer.writerow([
            'filename', 'card_name', 'set_code', 'collector_number',
            'finish', 'scryfall_id', 'confidence', 'corrected', 'correction_reason'
        ])

        # Rows
        for br, card in resolved_rows:
            writer.writerow([
                br.image_filename,
                card.name,
                card.set_code,
                card.collector_number,
                card.finish,
                card.scryfall_id or '',
                f"{br.confidence:.4f}",
                'Yes' if br.is_corrected else 'No',
                br.correction_reason or ''
            ])

        output.seek(0)
        return StreamingResponse(
            iter([output.getvalue()]),
            media_type="text/csv",
            headers={"Content-Disposition": f"attachment; filename=batch_{batch_id}.csv"}
        )

    if format == ExportFormat.JSON:
        # Generate JSON
        results = []
        for br, card in resolved_rows:
            results.append({
                'filename': br.image_filename,
                'card_name': card.name,
                'set_code': card.set_code,
                'collector_number': card.collector_number,
                'finish': card.finish,
                'scryfall_id': card.scryfall_id,
                'confidence': br.confidence,
                'corrected': br.is_corrected,
                'correction_reason': br.correction_reason
            })

        return StreamingResponse(
            iter([json.dumps(results, indent=2)]),
            media_type="application/json",
            headers={"Content-Disposition": f"attachment; filename=batch_{batch_id}.json"}
        )

    if format == ExportFormat.MANAPOOL_CSV:
        csv_string = build_manapool_csv(resolved_rows)
        return StreamingResponse(
            iter([csv_string]),
            media_type="text/csv",
            headers={"Content-Disposition": f"attachment; filename=batch_{batch_id}_manapool.csv"}
        )

    # Fallback when format is not recognized
    raise HTTPException(status_code=400, detail="Unsupported export format")

@router.get("/{batch_id}/image/{image_id}")
async def get_image(batch_id: str, image_id: str, db: Session = Depends(get_db)):
    """Serve uploaded image"""
    batch_result = db.query(BatchResult).filter(
        BatchResult.batch_id == batch_id,
        BatchResult.image_id == image_id
    ).first()

    if not batch_result:
        raise HTTPException(status_code=404, detail="Image not found")

    image_path = Path(batch_result.image_path)

    if not image_path.exists():
        raise HTTPException(status_code=404, detail="Image file not found")

    return FileResponse(image_path)


@router.get("/reference/{card_id}")
async def get_reference_image(card_id: str, db: Session = Depends(get_db)):
    """Serve reference image for a card from Scryfall images"""
    card = db.query(DBCard).filter(DBCard.id == card_id).first()

    if not card:
        raise HTTPException(status_code=404, detail="Card not found")

    if not card.image_path:
        raise HTTPException(status_code=404, detail="Card has no image path")

    image_path = Path(card.image_path)

    if not image_path.exists():
        raise HTTPException(status_code=404, detail="Reference image file not found")

    return FileResponse(image_path)

@router.get("/search", response_model=List[CardSearchResult])
async def search_cards(
    query: str,
    set_code: Optional[str] = None,
    limit: int = 50,
    offset: int = 0,
    db: Session = Depends(get_db)
):
    """Search for cards by name (for correction modal)"""
    filters = [DBCard.name.ilike(f"%{query}%")]

    if set_code:
        filters.append(DBCard.set_code == set_code.upper())

    cards = db.query(DBCard).filter(*filters).offset(offset).limit(limit).all()

    return [
        CardSearchResult(
            card_id=card.id,
            card_name=card.name,
            set_code=card.set_code,
            collector_number=card.collector_number,
            finish=card.finish,
            scryfall_image_url=None
        )
        for card in cards
    ]


@router.get("/sets", response_model=List[str])
async def get_available_sets(db: Session = Depends(get_db)):
    """Get list of all available set codes in the database"""
    sets = db.query(DBCard.set_code).distinct().order_by(DBCard.set_code).all()
    return [s[0] for s in sets]


@router.get("/config")
async def get_config():
    """
    Lightweight config endpoint for frontend toggles.
    Exposes mtgsold_enabled based on MTGSOLD_API_TOKEN presence.
    """
    enabled = is_mtgsold_enabled()
    return {"mtgsold_enabled": enabled}


@router.get("/{batch_id}/image/{image_id}/debug")
async def get_debug_image(batch_id: str, image_id: str, db: Session = Depends(get_db)):
    """
    Get annotated debug image showing card detection boundary

    Returns a JPEG image with the detected card boundary drawn as a polygon.
    Useful for verifying detection quality and understanding recognition results.
    """
    import cv2
    import numpy as np
    from io import BytesIO

    batch_result = db.query(BatchResult).filter(
        BatchResult.batch_id == batch_id,
        BatchResult.image_id == image_id
    ).first()

    if not batch_result:
        raise HTTPException(status_code=404, detail="Image not found")

    image_path = Path(batch_result.image_path)

    if not image_path.exists():
        raise HTTPException(status_code=404, detail="Image file not found")

    # Import detection module
    from src.detection.card_detector import detect_and_warp

    # Load image and run detection
    image = cv2.imread(str(image_path))
    if image is None:
        raise HTTPException(status_code=500, detail="Could not load image")

    try:
        warped, mask, corners = detect_and_warp(image, debug=False)

        # Draw boundary polygon on original image
        debug_img = image.copy()
        if corners is not None:
            pts = corners.astype(np.int32).reshape((-1, 1, 2))
            cv2.polylines(debug_img, [pts], isClosed=True, color=(0, 255, 0), thickness=3)

            # Draw corner points
            for i, corner in enumerate(corners):
                x, y = int(corner[0]), int(corner[1])
                cv2.circle(debug_img, (x, y), 8, (0, 0, 255), -1)
                cv2.putText(debug_img, str(i+1), (x+10, y-10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        # Encode as JPEG
        _, buffer = cv2.imencode('.jpg', debug_img, [cv2.IMWRITE_JPEG_QUALITY, 90])

        return StreamingResponse(
            BytesIO(buffer.tobytes()),
            media_type="image/jpeg",
            headers={"Content-Disposition": f"inline; filename=debug_{image_id}.jpg"}
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Detection failed: {str(e)}")
