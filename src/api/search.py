"""
src/api/search.py
──────────────────
STEP 5B — Search & Compare Endpoints

GET  /search     → top hotels by city + filters + dynamic query
POST /compare    → side-by-side comparison of multiple hotels
GET  /personalize → personalized hotels by travel type
GET  /issues     → hidden issue detection
"""

import pandas as pd
from fastapi       import APIRouter, HTTPException
from src.ranking   import HotelRanker, Personalizer
from .schemas      import (
    SearchRequest, SearchResponse,
    CompareRequest, CompareResponse, CompareRow,
    PersonalizeRequest, PersonalizeResponse,
    IssuesRequest, IssuesResponse,
    HotelCard, HiddenIssue,
)

router = APIRouter()


def _row_to_card(row: pd.Series) -> HotelCard:
    """Convert a DataFrame row to a HotelCard."""
    def safe(val):
        return None if pd.isna(val) else val

    return HotelCard(
        hotel_id           = str(row.get("hotel_id", "")),
        hotel_name         = str(row.get("hotel_name", "")),
        city               = str(row.get("city", "")),
        area               = safe(row.get("area")),
        star_rating        = safe(row.get("star_rating")),
        price_inr          = safe(row.get("price_inr")),
        total_price_inr    = safe(row.get("total_price_inr")),
        rating_overall     = safe(row.get("rating_overall")),
        trust_score        = safe(row.get("trust_score")),
        composite_score    = safe(row.get("composite_score")),
        dynamic_score      = safe(row.get("dynamic_score")),
        personalized_score = safe(row.get("personalized_score")),
        review_count       = int(row["review_count"]) if pd.notna(row.get("review_count")) else None,
        amenities          = safe(row.get("amenities")),
        source             = safe(row.get("source")),
        rank               = int(row["rank"]) if pd.notna(row.get("rank")) else None,
    )


# ── Shared ranker/personalizer instances (loaded once) ───────
_ranker      = None
_personalizer= None

def get_ranker() -> HotelRanker:
    global _ranker
    if _ranker is None:
        _ranker = HotelRanker()
    return _ranker

def get_personalizer() -> Personalizer:
    global _personalizer
    if _personalizer is None:
        _personalizer = Personalizer()
    return _personalizer


# ─────────────────────────────────────────────────────────────
# /search
# ─────────────────────────────────────────────────────────────

@router.post("/search", response_model=SearchResponse, tags=["Search"])
def search_hotels(req: SearchRequest):
    """
    Search hotels by city with optional filters and dynamic query ranking.

    - **city**: required city name
    - **query**: optional natural language (boosts relevant aspects)
    - **min_price / max_price**: price filter in INR
    - **min_stars**: minimum star rating
    - **top_n**: number of results (default 20)
    """
    ranker  = get_ranker()
    results = ranker.search(
        city       = req.city,
        query      = req.query or "",
        min_price  = req.min_price,
        max_price  = req.max_price,
        min_stars  = req.min_stars,
        min_rating = req.min_rating,
        top_n      = req.top_n,
    )

    if results.empty:
        return SearchResponse(
            city=req.city, query=req.query,
            total_found=0, hotels=[]
        )

    hotels = [_row_to_card(row) for _, row in results.iterrows()]
    return SearchResponse(
        city        = req.city,
        query       = req.query,
        total_found = len(hotels),
        hotels      = hotels,
    )


# ─────────────────────────────────────────────────────────────
# /compare
# ─────────────────────────────────────────────────────────────

@router.post("/compare", response_model=CompareResponse, tags=["Search"])
def compare_hotels(req: CompareRequest):
    """
    Compare 2–5 hotels side by side.
    Pass a list of hotel_ids from /search results.
    """
    ranker = get_ranker()
    rows   = []

    for hid in req.hotel_ids:
        hotel = ranker.get_hotel_by_id(hid)
        if hotel is None:
            raise HTTPException(
                status_code=404,
                detail=f"Hotel ID '{hid}' not found"
            )
        rows.append(hotel)

    def safe(val):
        return None if pd.isna(val) else val

    compare_rows = [
        CompareRow(
            hotel_id           = str(r.get("hotel_id", "")),
            hotel_name         = str(r.get("hotel_name", "")),
            city               = str(r.get("city", "")),
            star_rating        = safe(r.get("star_rating")),
            price_inr          = safe(r.get("price_inr")),
            rating_overall     = safe(r.get("rating_overall")),
            rating_cleanliness = safe(r.get("rating_cleanliness")),
            rating_food        = safe(r.get("rating_food")),
            rating_location    = safe(r.get("rating_location")),
            rating_service     = safe(r.get("rating_service")),
            rating_value       = safe(r.get("rating_value")),
            trust_score        = safe(r.get("trust_score")),
            composite_score    = safe(r.get("composite_score")),
            amenities          = safe(r.get("amenities")),
            source             = safe(r.get("source")),
        )
        for r in rows
    ]

    # Best value: highest rating_value score
    best_value = max(
        (r for r in compare_rows if r.rating_value is not None),
        key=lambda r: r.rating_value,
        default=None,
    )
    # Best rated: highest trust_score
    best_rated = max(
        (r for r in compare_rows if r.trust_score is not None),
        key=lambda r: r.trust_score,
        default=None,
    )

    return CompareResponse(
        hotels     = compare_rows,
        best_value = best_value.hotel_name if best_value else None,
        best_rated = best_rated.hotel_name if best_rated else None,
    )


# ─────────────────────────────────────────────────────────────
# /personalize
# ─────────────────────────────────────────────────────────────

@router.post("/personalize", response_model=PersonalizeResponse, tags=["Search"])
def personalize_hotels(req: PersonalizeRequest):
    """
    Get personalized hotel recommendations based on travel type.

    - **travel_type**: family / couple / solo / business / luxury / budget
    - **city**: required city name
    - **max_price**: optional budget cap
    """
    p       = get_personalizer()
    results = p.personalize(
        city        = req.city,
        travel_type = req.travel_type,
        max_price   = req.max_price,
        min_stars   = req.min_stars,
        top_n       = req.top_n,
    )

    if results.empty:
        return PersonalizeResponse(
            city=req.city, travel_type=req.travel_type,
            total_found=0, hotels=[]
        )

    hotels = [_row_to_card(row) for _, row in results.iterrows()]
    return PersonalizeResponse(
        city        = req.city,
        travel_type = req.travel_type,
        total_found = len(hotels),
        hotels      = hotels,
    )


# ─────────────────────────────────────────────────────────────
# /issues
# ─────────────────────────────────────────────────────────────

@router.post("/issues", response_model=IssuesResponse, tags=["Search"])
def detect_issues(req: IssuesRequest):
    """
    Detect hidden issues in hotels.
    Pass city to scan all hotels in that city,
    or hotel_id for a specific hotel.
    """
    p = get_personalizer()

    if req.hotel_id:
        result = p.detect_hidden_issues(req.hotel_id)
        if "error" in result:
            raise HTTPException(status_code=404, detail=result["error"])
        hotels = [HiddenIssue(**result)]
    else:
        flagged = p.scan_all_issues(city=req.city)
        hotels  = [
            HiddenIssue(
                hotel_id    = str(row["hotel_id"]),
                hotel_name  = str(row["hotel_name"]),
                city        = str(row["city"]),
                issues      = row["hidden_issues"],
                issue_count = int(row["issue_count"]),
                verdict     = "minor_concerns" if row["issue_count"] <= 2 else "major_concerns",
            )
            for _, row in flagged.iterrows()
        ]

    return IssuesResponse(total_flagged=len(hotels), hotels=hotels)
