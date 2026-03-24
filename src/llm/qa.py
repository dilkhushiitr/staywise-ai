"""
src/llm/qa.py
──────────────
STEP 3A — RAG-Grounded Q&A

User asks a question about a hotel (or hotels in general).
System retrieves relevant chunks from ChromaDB → feeds to LLM → returns answer.

The LLM is STRICTLY instructed to answer ONLY from retrieved context.
No hallucination — if the data doesn't say it, the LLM says so.

Usage:
    from src.llm.qa import HotelQA
    qa = HotelQA()
    answer = qa.ask("Is Taj Mumbai good for families?")
    answer = qa.ask("Which Goa hotels have a pool?", city="goa")
"""

from src.utils        import get_logger
from src.rag.retriever import HotelRetriever
from .llm_client       import chat

logger = get_logger("llm.qa")

# ── System prompt ────────────────────────────────────────────
QA_SYSTEM_PROMPT = """You are StayWise AI, an expert hotel recommendation assistant for India.

Your job is to answer user questions about hotels using ONLY the hotel data provided below.

Rules:
- Base your answer STRICTLY on the provided hotel context. Do not make up information.
- If the context doesn't contain enough information, say: "I don't have enough data to answer that."
- Be concise, helpful and specific. Use hotel names when relevant.
- When mentioning ratings, cite the actual numbers from the context.
- Format your answer clearly. Use bullet points for lists of hotels.
"""


def _build_context(results: list[dict], max_chunks: int = 8) -> str:
    """Format retrieved chunks into a clean context block for the LLM."""
    lines = []
    seen_hotels = set()

    for r in results[:max_chunks]:
        hotel_id = r.get("hotel_id", "")
        if hotel_id in seen_hotels:
            continue
        seen_hotels.add(hotel_id)

        lines.append(f"--- Hotel: {r.get('hotel_name', 'Unknown')} ({r.get('city', '')}) ---")
        lines.append(r.get("text", ""))
        lines.append("")

    return "\n".join(lines)


class HotelQA:
    """
    RAG-powered Q&A over hotel data.

    Workflow:
        1. Embed user question
        2. Retrieve top-K relevant hotel chunks from ChromaDB
        3. Build a context string from those chunks
        4. Send context + question to LLM
        5. Return grounded answer
    """

    def __init__(self, top_k: int = 8):
        self.retriever = HotelRetriever()
        self.top_k     = top_k
        logger.info("HotelQA initialized ✅")

    def ask(
        self,
        question:    str,
        city:        str   = None,
        max_price:   float = None,
        min_rating:  float = None,
        min_stars:   int   = None,
    ) -> dict:
        """
        Answer a natural language hotel question.

        Args:
            question   : e.g. "Is this hotel good for families?"
            city       : optional city filter
            max_price  : optional max price filter (INR)
            min_rating : optional min rating filter (0–5)
            min_stars  : optional star rating filter

        Returns:
            {
                "question":  str,
                "answer":    str,
                "sources":   list of hotel names used,
                "num_chunks": int
            }
        """
        logger.info(f"  Q&A: \"{question}\"")

        # ── Retrieve relevant chunks ──────────────────────────
        results = self.retriever.search(
            query      = question,
            city       = city,
            max_price  = max_price,
            min_rating = min_rating,
            min_stars  = min_stars,
            top_k      = self.top_k,
        )

        if not results:
            return {
                "question":   question,
                "answer":     "I couldn't find any hotels matching your criteria.",
                "sources":    [],
                "num_chunks": 0,
            }

        # ── Build context ────────────────────────────────────
        context = _build_context(results)

        # ── Call LLM ─────────────────────────────────────────
        user_prompt = f"""Here is the hotel data:

{context}

User question: {question}

Answer the question based only on the hotel data above."""

        answer = chat(
            system_prompt = QA_SYSTEM_PROMPT,
            user_prompt   = user_prompt,
            max_tokens    = 500,
        )

        sources = list(dict.fromkeys(
            r.get("hotel_name", "") for r in results
            if r.get("hotel_name")
        ))

        logger.info(f"  Answer generated from {len(results)} chunks")

        return {
            "question":   question,
            "answer":     answer,
            "sources":    sources[:5],
            "num_chunks": len(results),
        }
