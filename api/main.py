"""
FastAPI Backend for MTG Card Recognition Service
Handles batch uploads, recognition, and user corrections
"""

from contextlib import asynccontextmanager
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, FileResponse
from fastapi.templating import Jinja2Templates
from slowapi import _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded
from pathlib import Path
import sys
import logging

# Configure logging to show INFO level messages
logging.basicConfig(
    level=logging.INFO,
    format='%(levelname)s:     %(name)s - %(message)s'
)

logger = logging.getLogger(__name__)

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from api.routes import batch
from api.models import BatchStatus
from api.services.rate_limiter import limiter
from api.services.cleanup import run_full_cleanup
from src.utils.device import configure_cpu_threads, resolve_device

# Cleanup configuration
BATCH_RETENTION_HOURS = 24  # Delete batches older than 24 hours


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan: configure resources on startup."""
    # Configure CPU threading on startup
    device = resolve_device()
    if device == 'cpu':
        num_threads = configure_cpu_threads()
        logger.info(f"Running on CPU with {num_threads} threads")
    else:
        logger.info(f"Running on {device}")

    # Cleanup old batches on startup
    try:
        stats = run_full_cleanup(retention_hours=BATCH_RETENTION_HOURS)
        if stats["batches_deleted"] > 0 or stats["orphans_deleted"] > 0:
            mb_freed = stats["total_bytes_freed"] / (1024 * 1024)
            logger.info(
                f"Startup cleanup: {stats['batches_deleted']} batches, "
                f"{stats['orphans_deleted']} orphans, {mb_freed:.1f} MB freed"
            )
    except Exception as e:
        logger.warning(f"Startup cleanup failed: {e}")

    yield  # App runs here

    # Cleanup on shutdown (if needed)
    logger.info("Shutting down MTG Recognition API")


app = FastAPI(
    title="MTG Card Recognition API",
    description="Batch card recognition with visual review and corrections",
    version="1.0.0",
    lifespan=lifespan
)

# Attach rate limiter to app
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount("/static", StaticFiles(directory="web/static"), name="static")
templates = Jinja2Templates(directory="web/templates")
app.include_router(batch.router, prefix="/api/batch", tags=["batch"])

@app.get("/", response_class=HTMLResponse)
async def root(request: Request):
    """Serve upload page with MTGSold toggle"""
    from api.services.mtgsold_config import is_mtgsold_enabled
    return templates.TemplateResponse(
        "upload.html",
        {"request": request, "mtgsold_enabled": is_mtgsold_enabled()},
    )

@app.get("/batch/{batch_id}", response_class=HTMLResponse)
async def results_page(request: Request, batch_id: str):
    """Serve results page for a batch with MTGSold toggle"""
    from api.services.mtgsold_config import is_mtgsold_enabled
    return templates.TemplateResponse(
        "results.html",
        {"request": request, "batch_id": batch_id, "mtgsold_enabled": is_mtgsold_enabled()},
    )

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "service": "mtg-recognition"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api.main:app", host="0.0.0.0", port=8000, reload=True)
