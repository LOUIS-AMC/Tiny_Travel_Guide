"""Lightweight data filtering helpers to prep NYC itinerary context."""
from __future__ import annotations

from pathlib import Path
from typing import Iterable, List, Optional

import pandas as pd

# Supported borough names and simple aliases for user input.
BOROUGH_ALIASES = {
    "manhattan": "Manhattan",
    "brooklyn": "Brooklyn",
    "queens": "Queens",
    "bronx": "Bronx",
    "staten island": "Staten Island",
    "staten_island": "Staten Island",
    "statenisland": "Staten Island",
}

# Map budget buckets to hotel star rating ranges.
BUDGET_TO_STARS = {
    "low": (1, 2),
    "medium": (2, 3),
    "high": (3, 4),
}


def normalize_boroughs(raw_boros: Iterable[str]) -> List[str]:
    """Normalize user-provided borough names to dataset labels."""
    names = []
    for boro in raw_boros:
        key = (boro or "").strip().lower()
        if not key:
            continue
        if key in BOROUGH_ALIASES:
            names.append(BOROUGH_ALIASES[key])
    # Deduplicate while preserving order.
    seen = set()
    ordered = []
    for name in names:
        if name not in seen:
            ordered.append(name)
            seen.add(name)
    return ordered


class TravelData:
    """Loads and slices NYC attraction, restaurant, and hotel data."""

    def __init__(self, data_dir: Optional[Path] = None) -> None:
        base_dir = Path(__file__).resolve().parents[1]
        self.data_dir = data_dir or base_dir / "cleaned_data"
        self.attractions = self._load_csv("nyc_attractions.csv")
        self.restaurants = self._load_csv("nyc_restaurants.csv")
        self.hotels = self._load_csv("nyc_hotel_encoded.csv")

    def _load_csv(self, name: str) -> pd.DataFrame:
        path = self.data_dir / name
        if not path.exists():
            raise FileNotFoundError(f"Expected data file missing: {path}")
        return pd.read_csv(path)

    def _filter_boros(self, df: pd.DataFrame, boros: List[str]) -> pd.DataFrame:
        if not boros:
            return df.copy()
        mask = df["BoroName"].str.lower().isin({b.lower() for b in boros})
        filtered = df.loc[mask].copy()
        if filtered.empty and boros:
            # Fall back to all data if filter wipes everything out.
            return df.copy()
        return filtered

    def hotels_for_budget(
        self, boros: List[str], budget: str, limit: int = 5
    ) -> pd.DataFrame:
        df = self._filter_boros(self.hotels, boros)
        rating_range = BUDGET_TO_STARS.get((budget or "").lower())
        if rating_range:
            low, high = rating_range
            df = df[df["star_rating"].between(low, high, inclusive="both")]
        df = df.sort_values(
            ["star_rating", "high_rate", "low_rate"],
            ascending=[False, True, True],
            na_position="last",
        )
        if len(df) > limit:
            df = df.head(limit)
        return df

    def pick_attractions(self, boros: List[str], limit: int = 12) -> pd.DataFrame:
        df = self._filter_boros(self.attractions, boros)
        if len(df) > limit:
            df = df.sample(n=limit, random_state=42)
        return df

    def pick_restaurants(self, boros: List[str], limit: int = 10) -> pd.DataFrame:
        df = self._filter_boros(self.restaurants, boros)
        if "Rating" in df.columns:
            df = df.sort_values("Rating", ascending=False, na_position="last")
        if len(df) > limit:
            df = df.head(limit)
        return df

    def format_hotels(self, df: pd.DataFrame) -> str:
        lines = []
        for _, row in df.iterrows():
            star = row.get("star_rating", "")
            price_span = f"${row.get('low_rate', '')}-{row.get('high_rate', '')}"
            address = row.get("address1", "")
            name = row.get("name", "")
            boro = row.get("BoroName", "")
            lines.append(
                f"- {name} ({boro}) | {star} stars | {price_span} | {address}"
            )
        return "\n".join(lines)

    def format_attractions(self, df: pd.DataFrame) -> str:
        lines = []
        for _, row in df.iterrows():
            spot = row.get("Tourist_Spot", "")
            addr = row.get("Address", "")
            region = row.get("Region", "")
            boro = row.get("BoroName", "")
            lines.append(f"- {spot} ({boro}, {region}) at {addr}")
        return "\n".join(lines)

    def format_restaurants(self, df: pd.DataFrame) -> str:
        lines = []
        for _, row in df.iterrows():
            name = row.get("Name", "")
            rating = row.get("Rating", "")
            price = row.get("Price Category", "")
            cuisine = row.get("Detailed Ratings", "")
            boro = row.get("BoroName", "")
            lines.append(
                f"- {name} ({boro}) | rating {rating} | {price} | {cuisine}"
            )
        return "\n".join(lines)


def build_context(
    boros: List[str], budget: str, days: int, data: TravelData
) -> str:
    """Create a concise context block for the LLM prompt."""
    hotels = data.hotels_for_budget(boros, budget, limit=6)
    attractions = data.pick_attractions(boros, limit=18)
    restaurants = data.pick_restaurants(boros, limit=18)

    hotel_text = data.format_hotels(hotels) or "- No matching hotels; suggest reasonable options."
    attraction_text = data.format_attractions(attractions) or "- No attractions found; suggest iconic sights."
    restaurant_text = data.format_restaurants(restaurants) or "- No restaurants found; suggest dependable picks."

    chosen_boros = ", ".join(boros) if boros else "All boroughs"
    budget_text = budget or "medium"

    return (
        f"Traveler request: {days} day(s) in NYC, boroughs: {chosen_boros}, budget: {budget_text}.\n"
        f"Hotels filtered by borough and budget:\n{hotel_text}\n\n"
        f"Attractions to pull from:\n{attraction_text}\n\n"
        f"Restaurants to mix in:\n{restaurant_text}\n"
    )
