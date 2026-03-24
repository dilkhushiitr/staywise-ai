"""
src/api/app.py
───────────────
STEP 5D — Main FastAPI Application

Registers all routers and sets up:
  - CORS (allow all origins for development)
  - /health endpoint
  - /docs (Swagger UI — auto-generated)
  - /redoc (ReDoc UI — auto-generated)
"""

from fastapi              import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses    import JSONResponse
from src.utils            import load_config
from .schemas             import HealthResponse
from .search              import router as search_router
from .intelligence        import router as intelligence_router

# ── App setup ────────────────────────────────────────────────
app = FastAPI(
    title       = "StayWise AI API",
    description = """
## 🏨 StayWise AI — Hotel Decision Intelligence

AI-powered hotel recommendation system for India.

### Features
- **Search** — top hotels by city, budget, star rating
- **Dynamic Ranking** — query-aware ranking (best food / clean rooms / etc.)
- **Personalize** — travel-type based recommendations (family / couple / business)
- **Compare** — side-by-side hotel comparison
- **Ask** — RAG-grounded Q&A from real hotel data
- **Summarize** — pros/cons summary for any hotel
- **Recommend** — explainable AI recommendations with reasons
- **Issues** — hidden problem detection in hotels

### Data Sources
- MakeMyTrip · Goibibo · Google Reviews
- 5,579 hotels across 560 Indian cities
""",
    version     = "1.0.0",
    docs_url    = "/docs",
    redoc_url   = "/redoc",
)

# ── CORS — allow all origins (open for development) ──────────
app.add_middleware(
    CORSMiddleware,
    allow_origins     = ["*"],
    allow_credentials = True,
    allow_methods     = ["*"],
    allow_headers     = ["*"],
)

# ── Routers ───────────────────────────────────────────────────
app.include_router(search_router,       prefix="/api/v1")
app.include_router(intelligence_router, prefix="/api/v1")


# ─────────────────────────────────────────────────────────────
# HEALTH CHECK
# ─────────────────────────────────────────────────────────────

@app.get("/health", response_model=HealthResponse, tags=["System"])
def health_check():
    """Check if the API and data are loaded correctly."""
    try:
        import pandas as pd
        from pathlib import Path

        cfg      = load_config()
        csv_path = cfg["paths"]["processed_data"] / "unified_hotels.csv"
        n_hotels = 0
        if csv_path.exists():
            df       = pd.read_csv(csv_path, usecols=["hotel_id"])
            n_hotels = len(df)

        # Check ChromaDB
        from src.rag.vectorstore import get_client, COLLECTION_NAME
        client     = get_client()
        collection = client.get_collection(COLLECTION_NAME)
        n_chunks   = collection.count()

        return HealthResponse(
            status        = "ok",
            version       = "1.0.0",
            total_hotels  = n_hotels,
            vector_chunks = n_chunks,
        )
    except Exception as e:
        return JSONResponse(
            status_code = 503,
            content     = {"status": "error", "detail": str(e)},
        )


# ─────────────────────────────────────────────────────────────
# ROOT
# ─────────────────────────────────────────────────────────────

@app.get("/", tags=["System"])
def root():
    return {
        "name":    "StayWise AI API",
        "version": "1.0.0",
        "docs":    "/docs",
        "health":  "/health",
        "endpoints": {
            "search":      "POST /api/v1/search",
            "compare":     "POST /api/v1/compare",
            "personalize": "POST /api/v1/personalize",
            "issues":      "POST /api/v1/issues",
            "ask":         "POST /api/v1/ask",
            "summarize":   "POST /api/v1/summarize",
            "recommend":   "POST /api/v1/recommend",
        }
    }
