"""
src/api/schemas.py
───────────────────
All Pydantic request & response models for the API.
Every endpoint input and output is strictly typed here.
"""

from typing import Optional
from pydantic import BaseModel, Field


# ─────────────────────────────────────────────────────────────
# SHARED MODELS
# ─────────────────────────────────────────────────────────────

class HotelCard(BaseModel):
    """Compact hotel representation used in search results."""
    hotel_id:          str
    hotel_name:        str
    city:              str
    area:              Optional[str]   = None
    star_rating:       Optional[float] = None
    price_inr:         Optional[float] = None
    total_price_inr:   Optional[float] = None
    rating_overall:    Optional[float] = None
    trust_score:       Optional[float] = None
    composite_score:   Optional[float] = None
    dynamic_score:     Optional[float] = None
    personalized_score:Optional[float] = None
    review_count:      Optional[int]   = None
    amenities:         Optional[str]   = None
    source:            Optional[str]   = None
    rank:              Optional[int]   = None


class HiddenIssue(BaseModel):
    """A single hidden issue flag for a hotel."""
    hotel_id:    str
    hotel_name:  str
    city:        str
    issues:      list[str]
    issue_count: int
    verdict:     str    # "clean" | "minor_concerns" | "major_concerns"


# ─────────────────────────────────────────────────────────────
# /search
# ─────────────────────────────────────────────────────────────

class SearchRequest(BaseModel):
    city:       str              = Field(..., example="goa",
                                        description="City name to search hotels in")
    query:      Optional[str]    = Field(None, example="best food hotel",
                                        description="Natural language query for dynamic ranking")
    min_price:  Optional[float]  = Field(None, example=1000,  description="Min price INR")
    max_price:  Optional[float]  = Field(None, example=5000,  description="Max price INR")
    min_stars:  Optional[int]    = Field(None, example=3,     description="Min star rating")
    min_rating: Optional[float]  = Field(None, example=3.5,   description="Min overall rating")
    top_n:      int              = Field(20,   example=20,    description="Number of results")


class SearchResponse(BaseModel):
    city:         str
    query:        Optional[str]
    total_found:  int
    hotels:       list[HotelCard]


# ─────────────────────────────────────────────────────────────
# /compare
# ─────────────────────────────────────────────────────────────

class CompareRequest(BaseModel):
    hotel_ids: list[str] = Field(
        ...,
        example=["GBB_000001", "MMT_000005"],
        description="List of hotel_ids to compare (2–5 hotels)",
        min_length=2,
        max_length=5,
    )


class CompareRow(BaseModel):
    hotel_id:          str
    hotel_name:        str
    city:              str
    star_rating:       Optional[float]
    price_inr:         Optional[float]
    rating_overall:    Optional[float]
    rating_cleanliness:Optional[float]
    rating_food:       Optional[float]
    rating_location:   Optional[float]
    rating_service:    Optional[float]
    rating_value:      Optional[float]
    trust_score:       Optional[float]
    composite_score:   Optional[float]
    amenities:         Optional[str]
    source:            Optional[str]


class CompareResponse(BaseModel):
    hotels:     list[CompareRow]
    best_value: Optional[str]     = None   # hotel_name with best value score
    best_rated: Optional[str]     = None   # hotel_name with highest trust score


# ─────────────────────────────────────────────────────────────
# /ask
# ─────────────────────────────────────────────────────────────

class AskRequest(BaseModel):
    question:   str             = Field(...,  example="Which hotels in Goa have a pool?")
    city:       Optional[str]   = Field(None, example="goa")
    max_price:  Optional[float] = Field(None, example=5000)
    min_rating: Optional[float] = Field(None, example=3.5)


class AskResponse(BaseModel):
    question:   str
    answer:     str
    sources:    list[str]
    num_chunks: int


# ─────────────────────────────────────────────────────────────
# /summarize
# ─────────────────────────────────────────────────────────────

class SummarizeRequest(BaseModel):
    hotel_name: Optional[str] = Field(None, example="Taj Lands End")
    hotel_id:   Optional[str] = Field(None, example="GBB_000042")
    city:       Optional[str] = Field(None, example="mumbai")


class SummarizeResponse(BaseModel):
    hotel_name: str
    city:       str
    summary:    str


# ─────────────────────────────────────────────────────────────
# /recommend
# ─────────────────────────────────────────────────────────────

class RecommendRequest(BaseModel):
    city:        str            = Field(...,  example="goa")
    travel_type: str            = Field("family", example="family",
                                        description="family/couple/solo/business/luxury/budget")
    budget:      Optional[float]= Field(None, example=5000)
    priorities:  list[str]      = Field([], example=["pool", "good food", "beach access"])
    top_k:       int            = Field(5,    example=5)


class RecommendResponse(BaseModel):
    city:            str
    travel_type:     str
    budget:          Optional[float]
    priorities:      list[str]
    recommendations: str
    hotels_found:    int


# ─────────────────────────────────────────────────────────────
# /personalize
# ─────────────────────────────────────────────────────────────

class PersonalizeRequest(BaseModel):
    city:        str            = Field(...,  example="mumbai")
    travel_type: str            = Field("family", example="business")
    max_price:   Optional[float]= Field(None, example=6000)
    min_stars:   Optional[int]  = Field(None, example=3)
    top_n:       int            = Field(10,   example=10)


class PersonalizeResponse(BaseModel):
    city:        str
    travel_type: str
    total_found: int
    hotels:      list[HotelCard]


# ─────────────────────────────────────────────────────────────
# /issues
# ─────────────────────────────────────────────────────────────

class IssuesRequest(BaseModel):
    city:     Optional[str] = Field(None, example="delhi")
    hotel_id: Optional[str] = Field(None, example="GBB_000042")


class IssuesResponse(BaseModel):
    total_flagged: int
    hotels:        list[HiddenIssue]


# ─────────────────────────────────────────────────────────────
# /health
# ─────────────────────────────────────────────────────────────

class HealthResponse(BaseModel):
    status:       str
    version:      str
    total_hotels: int
    vector_chunks:int
