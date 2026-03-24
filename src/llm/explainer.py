"""
src/llm/explainer.py
─────────────────────
STEP 3C — Explainable Recommendations

Given user preferences (budget, travel type, city, priorities),
finds the best matching hotels AND explains exactly WHY each one
is recommended — this is the core USP of StayWise AI.

Usage:
    from src.llm.explainer import HotelExplainer
    e = HotelExplainer()

    recs = e.recommend(
        city        = "goa",
        travel_type = "family",
        budget      = 5000,
        priorities  = ["pool", "clean rooms", "good food"],
        top_k       = 5,
    )
"""

from typing import Optional
from src.utils         import get_logger
from src.rag.retriever  import HotelRetriever
from .llm_client        import chat

logger = get_logger("llm.explainer")

EXPLAINER_SYSTEM_PROMPT = """You are StayWise AI, an expert hotel recommendation engine for India.

Given a list of hotels and a user's travel preferences, recommend the TOP hotels with clear explanations.

Output format for each hotel (repeat for each recommended hotel):

🏨 HOTEL NAME — City
⭐ Rating: X/5 | 💰 Price: ₹X/night | 🎯 Trust Score: X/10
✅ WHY WE RECOMMEND:
  • <specific reason 1 matching user's need>
  • <specific reason 2>
  • <specific reason 3>
⚠️  WATCH OUT FOR:
  • <one honest weakness or caveat>

---

Rules:
- Recommend ONLY hotels present in the provided data.
- WHY WE RECOMMEND must directly reference the user's stated preferences.
- Be specific — mention actual ratings, amenities, and features from the data.
- WATCH OUT FOR must be honest — mention any low scores or missing info.
- If price is unknown, say "Price not listed".
- Rank by best match to user preferences, not just trust score.
"""


def _build_preference_query(
    city:         str,
    travel_type:  str,
    budget:       Optional[float],
    priorities:   list[str],
) -> str:
    """Build a rich semantic query from user preferences."""
    parts = []

    type_map = {
        "family":   "family friendly hotel with kids activities",
        "couple":   "romantic hotel for couples",
        "solo":     "safe budget hotel for solo traveler",
        "business": "business hotel with wifi and meeting rooms",
        "luxury":   "luxury 5 star premium hotel",
        "budget":   "cheap budget hotel affordable price",
    }
    parts.append(type_map.get(travel_type.lower(), travel_type))

    if city:
        parts.append(f"in {city}")

    if budget:
        parts.append(f"under ₹{budget:,.0f} per night")

    if priorities:
        parts.extend(priorities)

    return " ".join(parts)


def _format_hotels_for_llm(results: list[dict]) -> str:
    """Format retrieved hotels into a clean block for the LLM."""
    lines = []
    seen  = set()

    for r in results:
        hid = r.get("hotel_id", "")
        if hid in seen:
            continue
        seen.add(hid)

        price = f"₹{r['price_inr']:,.0f}" if r.get("price_inr") else "Not listed"
        trust = r.get("trust_score", "N/A")
        rating= r.get("rating_overall", "N/A")
        stars = f"{int(r['star_rating'])}★" if r.get("star_rating") else "N/A"

        lines.append(f"Hotel: {r.get('hotel_name', '?')} | City: {r.get('city', '?')}")
        lines.append(f"Stars: {stars} | Rating: {rating}/5 | Trust: {trust}/10 | Price: {price}/night")
        lines.append(f"Data: {r.get('text', '')[:400]}")
        lines.append("")

    return "\n".join(lines)


class HotelExplainer:
    """
    Generates personalized, explainable hotel recommendations.

    Unlike a plain search, this tells the user exactly WHY each
    hotel was picked for their specific situation.
    """

    def __init__(self):
        self.retriever = HotelRetriever()
        logger.info("HotelExplainer initialized ✅")

    def recommend(
        self,
        city:         str,
        travel_type:  str   = "family",
        budget:       float = None,
        priorities:   list  = None,
        min_rating:   float = None,
        top_k:        int   = 5,
    ) -> dict:
        """
        Get explainable hotel recommendations for a user's needs.

        Args:
            city        : target city (e.g. "mumbai", "goa")
            travel_type : "family" / "couple" / "solo" / "business" / "luxury" / "budget"
            budget      : max price per night in INR
            priorities  : list of priorities e.g. ["pool", "good food", "clean rooms"]
            min_rating  : minimum overall rating (0–5)
            top_k       : number of hotels to recommend

        Returns:
            {
                "city":          str,
                "travel_type":   str,
                "budget":        float,
                "priorities":    list,
                "recommendations": str  (formatted explanation text),
                "hotels_found":  int,
            }
        """
        priorities = priorities or []
        logger.info(f"  Recommending for: {travel_type} in {city} | budget: {budget}")

        # ── Build rich search query ───────────────────────────
        query = _build_preference_query(city, travel_type, budget, priorities)

        # ── Retrieve candidates ───────────────────────────────
        results = self.retriever.search(
            query      = query,
            city       = city.lower().strip() if city else None,
            max_price  = budget,
            min_rating = min_rating,
            top_k      = top_k * 3,          # over-fetch, LLM picks best
        )

        if not results:
            return {
                "city":            city,
                "travel_type":     travel_type,
                "budget":          budget,
                "priorities":      priorities,
                "recommendations": f"No hotels found in {city} matching your criteria.",
                "hotels_found":    0,
            }

        # ── Format for LLM ────────────────────────────────────
        hotels_context = _format_hotels_for_llm(results)

        budget_str = f"₹{budget:,.0f}/night" if budget else "flexible"
        prio_str   = ", ".join(priorities) if priorities else "general comfort"

        user_prompt = f"""User preferences:
- City: {city}
- Travel type: {travel_type}
- Budget: {budget_str}
- Priorities: {prio_str}

Available hotels:
{hotels_context}

Recommend the top {top_k} hotels that best match these preferences with clear explanations."""

        recommendations = chat(
            system_prompt = EXPLAINER_SYSTEM_PROMPT,
            user_prompt   = user_prompt,
            max_tokens    = 800,
            temperature   = 0.3,
        )

        logger.info(f"  Recommendations generated from {len(results)} candidates")

        return {
            "city":            city,
            "travel_type":     travel_type,
            "budget":          budget,
            "priorities":      priorities,
            "recommendations": recommendations,
            "hotels_found":    len(results),
        }
