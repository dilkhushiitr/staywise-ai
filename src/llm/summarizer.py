"""
src/llm/summarizer.py
──────────────────────
STEP 3B — Review Summarization

For any hotel, generates a structured summary with:
  - One-line overview
  - Top 3 pros
  - Top 3 cons
  - Best suited for (family / couple / solo / business)
  - Verdict

All grounded in actual hotel data from ChromaDB — no hallucination.

Usage:
    from src.llm.summarizer import HotelSummarizer
    s = HotelSummarizer()
    summary = s.summarize_by_name("Taj Lands End", city="mumbai")
    summary = s.summarize_by_id("GBB_000042")
"""

from src.utils         import get_logger
from src.rag.retriever  import HotelRetriever
from .llm_client        import chat

logger = get_logger("llm.summarizer")

SUMMARIZER_SYSTEM_PROMPT = """You are StayWise AI, an expert hotel analyst for Indian hotels.

Given hotel data, generate a structured, honest summary.

Output format (use exactly these headings):
OVERVIEW: <one sentence describing the hotel>

PROS:
• <pro 1>
• <pro 2>
• <pro 3>

CONS:
• <con 1>
• <con 2>
• <con 3>

BEST FOR: <family / couple / solo traveler / business / budget traveler — pick most relevant>

VERDICT: <one sentence honest recommendation>

Rules:
- Base everything ONLY on the provided hotel data.
- If data is missing for pros/cons, say "Limited data available".
- Be honest — if ratings are low, reflect that in CONS and VERDICT.
- Keep each point brief (under 15 words).
"""


def _format_hotel_context(chunks: list[dict]) -> str:
    """Combine all chunks for a single hotel into one context block."""
    parts = []
    for c in chunks:
        parts.append(c.get("text", ""))
    return "\n\n".join(parts)


class HotelSummarizer:
    """Generates structured pros/cons summaries for any hotel."""

    def __init__(self):
        self.retriever = HotelRetriever()
        logger.info("HotelSummarizer initialized ✅")

    def summarize_by_name(self, hotel_name: str, city: str = None) -> dict:
        """
        Summarize a hotel by searching its name.

        Args:
            hotel_name : e.g. "Taj Mahal Palace"
            city       : optional, narrows search

        Returns:
            {
                "hotel_name": str,
                "city":       str,
                "summary":    str  (formatted pros/cons text),
                "raw_data":   str  (context used)
            }
        """
        query   = f"hotel {hotel_name} {city or ''}"
        results = self.retriever.search(query=query, city=city, top_k=5)

        if not results:
            return {
                "hotel_name": hotel_name,
                "city":       city or "",
                "summary":    "No data found for this hotel.",
                "raw_data":   "",
            }

        # Pick best match (top result)
        best        = results[0]
        matched_id  = best.get("hotel_id", "")
        matched_name= best.get("hotel_name", hotel_name)
        matched_city= best.get("city", city or "")

        # Get ALL chunks for this hotel
        all_chunks  = self.retriever.search_by_hotel_id(matched_id)
        context     = _format_hotel_context(all_chunks) if all_chunks else best["text"]

        return self._generate_summary(matched_name, matched_city, context)

    def summarize_by_id(self, hotel_id: str) -> dict:
        """
        Summarize a hotel by its exact hotel_id from unified_hotels.csv.

        Args:
            hotel_id : e.g. "GBB_000042"
        """
        chunks = self.retriever.search_by_hotel_id(hotel_id)
        if not chunks:
            return {
                "hotel_name": hotel_id,
                "city":       "",
                "summary":    "No data found for this hotel ID.",
                "raw_data":   "",
            }

        meta         = chunks[0]
        hotel_name   = meta.get("hotel_name", hotel_id)
        city         = meta.get("city", "")
        context      = _format_hotel_context(chunks)

        return self._generate_summary(hotel_name, city, context)

    def _generate_summary(
        self,
        hotel_name: str,
        city:       str,
        context:    str,
    ) -> dict:
        """Core: send context to LLM → get structured summary."""
        logger.info(f"  Summarizing: {hotel_name} ({city})")

        user_prompt = f"""Hotel: {hotel_name}
City: {city}

Hotel data:
{context}

Generate a structured summary for this hotel."""

        summary = chat(
            system_prompt = SUMMARIZER_SYSTEM_PROMPT,
            user_prompt   = user_prompt,
            max_tokens    = 400,
            temperature   = 0.3,
        )

        logger.info(f"  Summary generated for {hotel_name}")

        return {
            "hotel_name": hotel_name,
            "city":       city,
            "summary":    summary,
            "raw_data":   context[:500] + "..." if len(context) > 500 else context,
        }
