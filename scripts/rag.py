"""Lightweight data filtering helpers to prep NYC itinerary context."""
from __future__ import annotations

from pathlib import Path
from typing import Iterable, List, Optional

import pandas as pd

from embedding_store import EmbeddingClient, top_k_by_embedding

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

# Map budget buckets to nightly high_rate ranges (USD).
BUDGET_TO_PRICE = {
    "low": (0, 100),
    "medium": (100, 300),
    "high": (300, float("inf")),
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

    def __init__(self, data_dir: Optional[Path] = None, use_embeddings: bool = True) -> None:
        base_dir = Path(__file__).resolve().parents[1]
        self.data_dir = data_dir or base_dir / "cleaned_data"
        self.attractions = self._load_csv("nyc_attractions.csv")
        self.restaurants = self._load_csv("nyc_restaurants.csv")
        self.hotels = self._load_csv("nyc_hotel_encoded.csv")
        self.use_embeddings = use_embeddings
        self._embedder: Optional[EmbeddingClient] = None

    @property
    def embedder(self) -> Optional[EmbeddingClient]:
        if not self.use_embeddings:
            return None
        if self._embedder is None:
            try:
                self._embedder = EmbeddingClient()
            except Exception:
                # If embedding setup fails, fall back to non-embedding flow.
                print("Failed")
                self._embedder = None
        return self._embedder

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
        budget_key = (budget or "").lower()

        # Ensure rates are numeric for filtering.
        for col in ("high_rate", "low_rate"):
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")

        if "budget_tier" in df.columns:
            df = df[df["budget_tier"] == budget_key]
        price_range = BUDGET_TO_PRICE.get(budget_key)
        if price_range and "high_rate" in df.columns:
            low, high = price_range
            df = df[(df["high_rate"] >= low) & (df["high_rate"] < high)]
        df = df.sort_values(
            ["star_rating", "high_rate", "low_rate"],
            ascending=[False, True, True],
            na_position="last",
        )
        if len(df) > limit:
            # Keep a top slice for quality, then sample for variety.
            top_slice = df.head(limit * 3)
            df = top_slice.sample(n=limit, random_state=None)

        cols = [
            c
            for c in [
                "name",
                "address1",
                "city",
                "state_province",
                "postal_code",
                "latitude",
                "longitude",
                "star_rating",
                "high_rate",
                "low_rate",
                "BoroName",
            ]
            if c in df.columns
        ]
        df = df[cols]

        embedder = self.embedder
        if embedder:
            try:
                texts = [
                    " | ".join(
                        str(row.get(col, ""))
                        for col in ["name", "address1", "city", "star_rating", "BoroName"]
                        if col in df.columns
                    )
                    for _, row in df.iterrows()
                ]
                idxs = top_k_by_embedding(
                    query=f"hotel options for {', '.join(boros) or 'all boroughs'} in NYC at {budget or 'medium'} budget",
                    items=texts,
                    embedder=embedder,
                    k=min(limit, len(texts)),
                )
                df = df.iloc[idxs]
            except Exception:
                pass
        return df

    def pick_attractions(self, boros: List[str], limit: int = 12) -> pd.DataFrame:
        df = self._filter_boros(self.attractions, boros)
        if "Tourist_Spot" in df.columns:
            df = df.drop_duplicates(subset=["Tourist_Spot"])
        if len(df) > limit:
            df = df.sample(n=limit, random_state=None)
        embedder = self.embedder
        if embedder:
            try:
                texts = [
                    " | ".join(
                        str(row.get(col, ""))
                        for col in ["Tourist_Spot", "Address", "Region", "BoroName"]
                        if col in df.columns
                    )
                    for _, row in df.iterrows()
                ]
                idxs = top_k_by_embedding(
                    query=f"NYC attractions clustered per borough: {', '.join(boros) or 'all boroughs'}",
                    items=texts,
                    embedder=embedder,
                    k=min(limit, len(texts)),
                )
                df = df.iloc[idxs]
            except Exception:
                pass
        return df

    def pick_restaurants(self, boros: List[str], limit: int = 10) -> pd.DataFrame:
        df = self._filter_boros(self.restaurants, boros)
        if "Name" in df.columns:
            df = df.drop_duplicates(subset=["Name"])
        if "Rating" in df.columns:
            df = df.sort_values("Rating", ascending=False, na_position="last")
        if len(df) > limit:
            df = df.head(limit)
        cols = [
            c
            for c in ["Name", "Rating", "Address", "latitude", "longitude", "ZipCode", "BoroName"]
            if c in df.columns
        ]
        df = df[cols]

        embedder = self.embedder
        if embedder:
            try:
                texts = [
                    " | ".join(str(row.get(col, "")) for col in df.columns)
                    for _, row in df.iterrows()
                ]
                idxs = top_k_by_embedding(
                    query=f"NYC trip food near attractions in {', '.join(boros) or 'all boroughs'}",
                    items=texts,
                    embedder=embedder,
                    k=min(limit, len(texts)),
                )
                df = df.iloc[idxs]
            except Exception:
                pass
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
            boro = row.get("BoroName", "")
            rating = row.get("Rating", "")
            addr = row.get("Address", "")
            addr_text = f" | {addr}" if addr else ""
            lines.append(f"- {name} ({boro}) | rating {rating}{addr_text}")
        return "\n".join(lines)


def build_context(
    boros: List[str], budget: str, days: int, data: TravelData
) -> str:
    """Create a concise context block for the LLM prompt."""
    hotels = data.hotels_for_budget(boros, budget, limit=6)
    attractions = data.pick_attractions(boros, limit=max(days * 3, 18))
    restaurants = data.pick_restaurants(boros, limit=max(days * 3, 18))

    hotel_text = data.format_hotels(hotels) or "- No matching hotels; suggest reasonable options."
    attraction_text = data.format_attractions(attractions) or "- No attractions found; suggest iconic sights."
    restaurant_text = data.format_restaurants(restaurants) or "- No restaurants found; suggest dependable picks."

    chosen_boros = ", ".join(boros) if boros else "All boroughs"
    budget_text = budget or "medium"

    return (
        f"Traveler request: {days} day(s) in NYC, boroughs: {chosen_boros}, budget: {budget_text}.\n"
        f"Hotels filtered by borough and budget (pick ONE as a home base for the whole trip):\n{hotel_text}\n\n"
        f"Attractions to pull from (with addresses; use each attraction at most once):\n{attraction_text}\n\n"
        f"Restaurants to mix in (name + rating + address; keep these near daily attractions):\n{restaurant_text}\n"
    )
