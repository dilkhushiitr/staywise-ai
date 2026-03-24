"""
src/ingestion/google_places.py
───────────────────────────────
Google Places API — Live Hotel Data Fetcher

Fetches real hotel data from Google Places API for any city.
This replaces the static CSV files as the primary data source.

API used: Google Places API (New) — Text Search + Place Details
Docs: https://developers.google.com/maps/documentation/places/web-service

Get your free API key:
  1. Go to https://console.cloud.google.com
  2. Enable "Places API (New)"
  3. Create credentials → API Key
  4. Add to env.example: GOOGLE_PLACES_API_KEY=your-key-here

Free tier: $200/month credit (~2,000 searches free per month)
"""

import os
import time
import requests
import pandas as pd
from pathlib import Path
from src.utils import load_config, get_logger

logger = get_logger("ingestion.google_places")

# ── API config ────────────────────────────────────────────────
PLACES_TEXT_SEARCH_URL = "https://places.googleapis.com/v1/places:searchText"
PLACES_DETAILS_URL     = "https://places.googleapis.com/v1/places/{place_id}"

# Fields to fetch in Text Search (controls billing)
TEXT_SEARCH_FIELDS = ",".join([
    "places.id",
    "places.displayName",
    "places.formattedAddress",
    "places.location",
    "places.rating",
    "places.userRatingCount",
    "places.priceLevel",
    "places.types",
    "places.businessStatus",
])

# Fields to fetch in Place Details (controls billing)
DETAIL_FIELDS = ",".join([
    "id",
    "displayName",
    "formattedAddress",
    "location",
    "rating",
    "userRatingCount",
    "priceLevel",
    "types",
    "websiteUri",
    "internationalPhoneNumber",
    "regularOpeningHours",
    "photos",
    "editorialSummary",
    "amenities",
    "goodForGroups",
    "goodForWatchingSports",
    "liveMusic",
    "outdoorSeating",
    "servesBeer",
    "servesBreakfast",
    "servesLunch",
    "servesDinner",
    "servesVegetarianFood",
])

# Price level map
PRICE_LEVEL_MAP = {
    "PRICE_LEVEL_FREE":           0,
    "PRICE_LEVEL_INEXPENSIVE":    1,
    "PRICE_LEVEL_MODERATE":       2,
    "PRICE_LEVEL_EXPENSIVE":      3,
    "PRICE_LEVEL_VERY_EXPENSIVE": 4,
}


def _get_api_key() -> str:
    """Get Google Places API key from env.example."""
    key = os.getenv("GOOGLE_PLACES_API_KEY", "")
    if not key or key == "your-google-places-api-key-here":
        raise EnvironmentError(
            "\n\n❌  Google Places API key not set!\n"
            "  1. Go to https://console.cloud.google.com\n"
            "  2. Enable 'Places API (New)'\n"
            "  3. Create API Key under Credentials\n"
            "  4. Open env.example\n"
            "  5. Add: GOOGLE_PLACES_API_KEY=your-key-here\n"
            "  6. Save and run again\n"
        )
    return key


def _text_search(query: str, api_key: str, max_results: int = 20) -> list[dict]:
    """
    Search for hotels using Google Places Text Search API.
    Returns list of raw place dicts.
    """
    headers = {
        "Content-Type":     "application/json",
        "X-Goog-Api-Key":   api_key,
        "X-Goog-FieldMask": TEXT_SEARCH_FIELDS,
    }

    body = {
        "textQuery":          query,
        "includedType":       "lodging",
        "languageCode":       "en",
        "maxResultCount":     min(max_results, 20),   # API max is 20 per page
        "locationBias": {
            "circle": {
                "center": {"latitude": 20.5937, "longitude": 78.9629},
                "radius": 2000000.0,    # India bounding radius (meters)
            }
        }
    }

    response = requests.post(PLACES_TEXT_SEARCH_URL, json=body, headers=headers)
    response.raise_for_status()
    data = response.json()
    return data.get("places", [])


def _get_place_details(place_id: str, api_key: str) -> dict:
    """
    Fetch detailed info for a single place.
    """
    url     = PLACES_DETAILS_URL.format(place_id=place_id)
    headers = {
        "X-Goog-Api-Key":   api_key,
        "X-Goog-FieldMask": DETAIL_FIELDS,
    }
    response = requests.get(url, headers=headers)
    response.raise_for_status()
    return response.json()


def _parse_place(place: dict, city: str) -> dict:
    """
    Convert raw Google Places API response → clean flat dict
    matching our unified schema columns.
    """
    # Price level → approximate INR range
    price_map_inr = {0: None, 1: 1500, 2: 4000, 3: 8000, 4: 15000}
    price_level   = PRICE_LEVEL_MAP.get(place.get("priceLevel", ""), None)
    price_inr     = price_map_inr.get(price_level) if price_level is not None else None

    # Star rating approximation from price level
    star_map = {0: None, 1: 2, 2: 3, 3: 4, 4: 5}
    star_rating = star_map.get(price_level) if price_level is not None else None

    # Display name
    name = place.get("displayName", {})
    hotel_name = name.get("text", "") if isinstance(name, dict) else str(name)

    # Location
    loc       = place.get("location", {})
    latitude  = loc.get("latitude")
    longitude = loc.get("longitude")

    # Editorial summary → description
    summary     = place.get("editorialSummary", {})
    description = summary.get("text", "") if isinstance(summary, dict) else ""

    # Build amenities string from boolean fields
    amenity_fields = {
        "servesBreakfast":       "Breakfast",
        "servesLunch":           "Lunch",
        "servesDinner":          "Dinner",
        "servesVegetarianFood":  "Vegetarian Food",
        "outdoorSeating":        "Outdoor Seating",
        "liveMusic":             "Live Music",
        "goodForGroups":         "Good for Groups",
    }
    amenities = [label for field, label in amenity_fields.items()
                 if place.get(field) is True]

    return {
        "hotel_name":     hotel_name,
        "city":           city.title(),
        "area":           "",
        "state":          "",
        "country":        "India",
        "address":        place.get("formattedAddress", ""),
        "latitude":       latitude,
        "longitude":      longitude,
        "star_rating":    star_rating,
        "price_inr":      price_inr,
        "tax_inr":        None,
        "rating":         place.get("rating"),
        "review_count":   place.get("userRatingCount"),
        "hotel_type":     "Hotel",
        "description":    description,
        "amenities":      " | ".join(amenities),
        "facilities":     "",
        "room_facilities":"",
        "source":         "Google Places API",
        # Aspect ratings not available from Places API
        "rating_service":     None,
        "rating_amenities":   None,
        "rating_food":        None,
        "rating_value":       None,
        "rating_location":    None,
        "rating_cleanliness": None,
        # Keep raw place_id for future enrichment
        "_place_id": place.get("id", ""),
    }


def fetch_hotels_for_city(
    city:        str,
    api_key:     str,
    max_results: int = 20,
    fetch_details: bool = False,
    delay:       float = 0.2,
) -> list[dict]:
    """
    Fetch hotels for a single city from Google Places API.

    Args:
        city         : city name e.g. "Mumbai"
        api_key      : Google Places API key
        max_results  : max hotels to fetch per city (max 20 per API call)
        fetch_details: if True, makes extra API call per hotel for more data
        delay        : seconds between API calls (be nice to the API)

    Returns:
        List of clean hotel dicts
    """
    query  = f"hotels in {city} India"
    logger.info(f"  Fetching: {query} (max {max_results})")

    places = _text_search(query, api_key, max_results)
    hotels = []

    for place in places:
        if fetch_details and place.get("id"):
            try:
                time.sleep(delay)
                detailed = _get_place_details(place["id"], api_key)
                place.update(detailed)
            except Exception as e:
                logger.debug(f"  Detail fetch failed for {place.get('id')}: {e}")

        hotel = _parse_place(place, city)
        if hotel["hotel_name"]:
            hotels.append(hotel)
        time.sleep(delay)

    logger.info(f"  {city}: fetched {len(hotels)} hotels")
    return hotels


def fetch_all_cities(
    cities:       list[str],
    max_per_city: int   = 20,
    fetch_details:bool  = False,
    save_cache:   bool  = True,
) -> pd.DataFrame:
    """
    Fetch hotels for multiple cities and return as a unified DataFrame.

    Args:
        cities        : list of city names
        max_per_city  : max hotels per city
        fetch_details : fetch extra detail per hotel (uses more API quota)
        save_cache    : save raw results to data/processed/google_places_raw.csv

    Returns:
        DataFrame with all fetched hotels
    """
    api_key = _get_api_key()

    logger.info("=" * 55)
    logger.info("  GOOGLE PLACES API — FETCHING HOTEL DATA")
    logger.info("=" * 55)
    logger.info(f"  Cities       : {cities}")
    logger.info(f"  Max per city : {max_per_city}")
    logger.info(f"  Fetch details: {fetch_details}")

    all_hotels = []

    for i, city in enumerate(cities, 1):
        logger.info(f"\n  [{i}/{len(cities)}] {city}")
        try:
            hotels = fetch_hotels_for_city(
                city         = city,
                api_key      = api_key,
                max_results  = max_per_city,
                fetch_details= fetch_details,
            )
            all_hotels.extend(hotels)
        except Exception as e:
            logger.warning(f"  Failed for {city}: {e}")
        time.sleep(0.5)   # pause between cities

    df = pd.DataFrame(all_hotels)

    if df.empty:
        logger.warning("  No hotels fetched!")
        return df

    # Save cache
    if save_cache:
        cfg       = load_config()
        cache_path= cfg["paths"]["processed_data"] / "google_places_raw.csv"
        df.to_csv(cache_path, index=False)
        logger.info(f"\n  Cached → {cache_path.name}")

    logger.info(f"\n  Total fetched: {len(df):,} hotels from {len(cities)} cities")
    return df
