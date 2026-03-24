"""
src/api/api_runner.py
──────────────────────
Step 5 — Starts the FastAPI server with uvicorn.

After running, open:
  http://localhost:8000/docs   ← Swagger UI (interactive)
  http://localhost:8000/redoc  ← ReDoc UI
  http://localhost:8000/health ← Health check
"""

import sys
from src.utils import get_logger

logger = get_logger("api.runner")


def run_step5() -> None:
    logger.info("╔══════════════════════════════════════════════════╗")
    logger.info("║   STAYWISE AI — STEP 5: FastAPI SERVER           ║")
    logger.info("╚══════════════════════════════════════════════════╝")

    try:
        import uvicorn
    except ImportError:
        logger.error("uvicorn not installed. Run: pip install -r requirements.txt")
        sys.exit(1)

    logger.info("\n  Starting API server...")
    logger.info("  ┌────────────────────────────────────────────┐")
    logger.info("  │  Swagger UI  →  http://localhost:8000/docs  │")
    logger.info("  │  Health      →  http://localhost:8000/health │")
    logger.info("  │  Press Ctrl+C to stop                       │")
    logger.info("  └────────────────────────────────────────────┘\n")

    uvicorn.run(
        "src.api.app:app",
        host       = "0.0.0.0",
        port       = 8000,
        reload     = False,
        log_level  = "info",
    )
