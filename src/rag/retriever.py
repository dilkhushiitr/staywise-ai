"""
src/rag/retriever.py
─────────────────────
STEP 2D — RAG Retriever

Exposes a clean query interface over ChromaDB.
Given a natural language question + optional filters,
returns the top-N most relevant hotel chunks.

Usage:
    from src.rag.retriever import HotelRetriever
    retriever = HotelRetriever()
    results   = retriever.search("best hotel for family in Goa", city="goa", top_k=5)
"""

from typing import Optional
from src.utils import load_config, get_logger
from .embedder    import load_model, EMBEDDING_MODEL
from .vectorstore import load_collection

logger = get_logger("rag.retriever")


class HotelRetriever:
    """
    Semantic search over indexed hotel chunks.

    Attributes:
        model      : SentenceTransformer for query embedding
        collection : ChromaDB collection
    """

    def __init__(self):
        logger.info("  Initializing HotelRetriever...")
        self.model      = load_model()
        self.collection = load_collection()
        count           = self.collection.count()
        logger.info(f"  Connected to ChromaDB — {count:,} chunks indexed")

    def search(
        self,
        query: str,
        city: Optional[str]  = None,
        min_price: Optional[float] = None,
        max_price: Optional[float] = None,
        min_rating: Optional[float] = None,
        min_stars: Optional[int]   = None,
        top_k: int = 10,
    ) -> list[dict]:
        """
        Semantic search for hotels matching the query + optional filters.

        Args:
            query      : natural language question e.g. "best hotel for family"
            city       : filter by city name (case insensitive)
            min_price  : minimum price in INR
            max_price  : maximum price in INR
            min_rating : minimum overall rating (0–5)
            min_stars  : minimum star rating (1–5)
            top_k      : number of results to return

        Returns:
            List of result dicts sorted by relevance:
            [
                {
                    "chunk_id":    str,
                    "text":        str,
                    "distance":    float,   # lower = more similar
                    "hotel_id":    str,
                    "hotel_name":  str,
                    "city":        str,
                    "price_inr":   float,
                    "trust_score": float,
                    ...all metadata fields
                },
                ...
            ]
        """
        # ── Embed the query ───────────────────────────────────
        query_embedding = self.model.encode(
            query,
            normalize_embeddings=True,
        ).tolist()

        # ── Build ChromaDB where-filter ───────────────────────
        where_clauses = []

        if city:
            where_clauses.append({"city": {"$eq": city.lower().strip()}})

        if min_price is not None:
            where_clauses.append({"price_inr": {"$gte": float(min_price)}})

        if max_price is not None:
            where_clauses.append({"price_inr": {"$lte": float(max_price)}})

        if min_rating is not None:
            where_clauses.append({"rating_overall": {"$gte": float(min_rating)}})

        if min_stars is not None:
            where_clauses.append({"star_rating": {"$gte": int(min_stars)}})

        where = None
        if len(where_clauses) == 1:
            where = where_clauses[0]
        elif len(where_clauses) > 1:
            where = {"$and": where_clauses}

        # ── Query ChromaDB ────────────────────────────────────
        query_params = {
            "query_embeddings": [query_embedding],
            "n_results":        min(top_k, self.collection.count()),
            "include":          ["documents", "metadatas", "distances"],
        }
        if where:
            query_params["where"] = where

        results = self.collection.query(**query_params)

        # ── Format results ────────────────────────────────────
        formatted = []
        for i in range(len(results["ids"][0])):
            meta = results["metadatas"][0][i]
            formatted.append({
                "chunk_id":    results["ids"][0][i],
                "text":        results["documents"][0][i],
                "distance":    round(results["distances"][0][i], 4),
                "relevance":   round(1 - results["distances"][0][i], 4),
                **meta,
            })

        return formatted

    def search_by_hotel_id(self, hotel_id: str) -> list[dict]:
        """Fetch all chunks for a specific hotel_id."""
        results = self.collection.get(
            where={"hotel_id": {"$eq": hotel_id}},
            include=["documents", "metadatas"],
        )
        return [
            {"chunk_id": cid, "text": doc, **meta}
            for cid, doc, meta in zip(
                results["ids"],
                results["documents"],
                results["metadatas"],
            )
        ]

    def get_stats(self) -> dict:
        """Return basic stats about the indexed collection."""
        return {
            "total_chunks":  self.collection.count(),
            "model":         EMBEDDING_MODEL,
            "collection":    self.collection.name,
        }
