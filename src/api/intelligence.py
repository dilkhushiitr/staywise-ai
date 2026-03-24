"""
src/api/intelligence.py
────────────────────────
STEP 5C — LLM Intelligence Endpoints

POST /ask        → RAG-grounded Q&A
POST /summarize  → hotel pros/cons summary
POST /recommend  → explainable personalized recommendations
"""

from fastapi   import APIRouter, HTTPException
from src.llm   import HotelQA, HotelSummarizer, HotelExplainer
from .schemas  import (
    AskRequest,        AskResponse,
    SummarizeRequest,  SummarizeResponse,
    RecommendRequest,  RecommendResponse,
)

router = APIRouter()

# ── Shared LLM instances (loaded once at startup) ─────────────
_qa         = None
_summarizer = None
_explainer  = None

def get_qa() -> HotelQA:
    global _qa
    if _qa is None:
        _qa = HotelQA()
    return _qa

def get_summarizer() -> HotelSummarizer:
    global _summarizer
    if _summarizer is None:
        _summarizer = HotelSummarizer()
    return _summarizer

def get_explainer() -> HotelExplainer:
    global _explainer
    if _explainer is None:
        _explainer = HotelExplainer()
    return _explainer


# ─────────────────────────────────────────────────────────────
# /ask
# ─────────────────────────────────────────────────────────────

@router.post("/ask", response_model=AskResponse, tags=["Intelligence"])
def ask_question(req: AskRequest):
    """
    Ask any hotel-related question. Answers are grounded in real review data.

    Examples:
    - "Which hotels in Goa have a swimming pool?"
    - "Is Hotel Taj good for families?"
    - "Best budget hotels in Delhi under ₹2000"
    """
    try:
        result = get_qa().ask(
            question   = req.question,
            city       = req.city,
            max_price  = req.max_price,
            min_rating = req.min_rating,
        )
        return AskResponse(**result)
    except EnvironmentError as e:
        raise HTTPException(status_code=503, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ─────────────────────────────────────────────────────────────
# /summarize
# ─────────────────────────────────────────────────────────────

@router.post("/summarize", response_model=SummarizeResponse, tags=["Intelligence"])
def summarize_hotel(req: SummarizeRequest):
    """
    Get a structured pros/cons summary for any hotel.

    Provide either:
    - **hotel_name** + optional **city**
    - **hotel_id** (exact ID from /search results)
    """
    if not req.hotel_name and not req.hotel_id:
        raise HTTPException(
            status_code=422,
            detail="Provide either hotel_name or hotel_id"
        )
    try:
        s = get_summarizer()
        if req.hotel_id:
            result = s.summarize_by_id(req.hotel_id)
        else:
            result = s.summarize_by_name(req.hotel_name, city=req.city)

        return SummarizeResponse(
            hotel_name = result["hotel_name"],
            city       = result["city"],
            summary    = result["summary"],
        )
    except EnvironmentError as e:
        raise HTTPException(status_code=503, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ─────────────────────────────────────────────────────────────
# /recommend
# ─────────────────────────────────────────────────────────────

@router.post("/recommend", response_model=RecommendResponse, tags=["Intelligence"])
def recommend_hotels(req: RecommendRequest):
    """
    Get explainable, personalized hotel recommendations.

    The AI explains exactly WHY each hotel is recommended for your situation.

    - **travel_type**: family / couple / solo / business / luxury / budget
    - **priorities**: list of things you care about e.g. ["pool", "good food"]
    - **budget**: max price per night in INR
    """
    try:
        result = get_explainer().recommend(
            city        = req.city,
            travel_type = req.travel_type,
            budget      = req.budget,
            priorities  = req.priorities,
            top_k       = req.top_k,
        )
        return RecommendResponse(**result)
    except EnvironmentError as e:
        raise HTTPException(status_code=503, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
