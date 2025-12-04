from __future__ import annotations

import os
import pandas as pd
import fiona
from shapely.geometry import shape, Point
from pyproj import Transformer
from dotenv import load_dotenv

load_dotenv()


# ----------------------------------------------------------------------
# Shared helpers
# ----------------------------------------------------------------------
def load_borough_shapes_and_transformer(shp_path: str):
    """
    Load nybb shapefile and return (borough_shapes, transformer).

    borough_shapes: list of (geom, props) where props["BoroName"] is the borough.
    transformer: transforms WGS84 (lon, lat) -> EPSG:2263 (nybb CRS).
    """
    borough_shapes: list[tuple] = []
    with fiona.open(shp_path) as src:
        # nybb is in EPSG:2263 (NAD_1983_StatePlane_New_York_Long_Island_FIPS_3104_Feet)
        for feat in src:
            geom = shape(feat["geometry"])
            props = feat["properties"]
            borough_shapes.append((geom, props))

    transformer = Transformer.from_crs("EPSG:4326", "EPSG:2263", always_xy=True)
    return borough_shapes, transformer


def boro_from_point(lat, lon, borough_geoms, transformer) -> str | None:
    """
    Given lat/lon in WGS84, return BoroName via point-in-polygon on nybb shapes.
    """
    if pd.isna(lat) or pd.isna(lon):
        return None
    x, y = transformer.transform(lon, lat)
    pt = Point(x, y)
    for geom, props in borough_geoms:
        if geom.contains(pt):
            return props["BoroName"]
    return None


# ----------------------------------------------------------------------
# 1. Hotels generator
# ----------------------------------------------------------------------
def generate_nyc_hotel_dataset(
    hotel_csv_path: str | None = None,
    borough_shp_path: str | None = None,
    output_path: str = "cleaned_data/nyc_hotel_encoded.csv",
) -> pd.DataFrame:
    """
    Load the raw hotel CSV, attach BoroName via point-in-polygon, filter to NYC,
    and write the cleaned CSV.
    """
    if hotel_csv_path is None:
        hotel_csv_path = os.getenv("NYC_HOTEL_PATH")
    if borough_shp_path is None:
        borough_shp_path = os.getenv("NYC_BOURUGH_COORDS")

    if not hotel_csv_path:
        raise ValueError("NYC_HOTEL_PATH is not set or hotel_csv_path not provided.")
    if not borough_shp_path:
        raise ValueError("NYC_BOURUGH_COORDS is not set or borough_shp_path not provided.")

    # 1) Load hotels
    hotels = pd.read_csv(hotel_csv_path, encoding="latin1")
    # Normalize rate columns to numeric
    hotels["high_rate"] = pd.to_numeric(hotels.get("high_rate"), errors="coerce")
    hotels["low_rate"] = pd.to_numeric(hotels.get("low_rate"), errors="coerce")

    # 2) Load borough shapes + transformer
    borough_shapes, transformer = load_borough_shapes_and_transformer(borough_shp_path)

    # 3) Assign borough to each hotel using point-in-polygon
    hotels["BoroName"] = hotels.apply(
        lambda row: boro_from_point(row["latitude"], row["longitude"], borough_shapes, transformer),
        axis=1,
    )

    # 4) Filter to only hotels inside NYC (one of the 5 boroughs)
    nyc_hotels = hotels[hotels["BoroName"].notna()].copy()

    # 5) Drop zero/invalid rates to avoid unusable entries
    nyc_hotels = nyc_hotels[
        (nyc_hotels["high_rate"] > 0) & (nyc_hotels["low_rate"] > 0)
    ].copy()

    # 6) Budget tier derived from high_rate
    def _budget_tier(rate: float | int | None) -> str | None:
        if pd.isna(rate):
            return None
        if rate < 100:
            return "low"
        if rate < 300:
            return "medium"
        return "high"

    nyc_hotels["budget_tier"] = nyc_hotels["high_rate"].apply(_budget_tier)

    # 7) Save
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    nyc_hotels.to_csv(output_path, index=False)

    print(f"[generate_nyc_hotel_dataset] Saved {len(nyc_hotels)} rows to {output_path}")
    return nyc_hotels


# ----------------------------------------------------------------------
# 2. Tourist locations generator
# ----------------------------------------------------------------------
def guess_boro_from_text(addr: str | None) -> str | None:
    """
    Heuristic borough guess from address string text.
    """
    if not isinstance(addr, str):
        return None
    a = addr.lower()
    if "staten island" in a:
        return "Staten Island"
    if "brooklyn" in a:
        return "Brooklyn"
    if "queens" in a:
        return "Queens"
    if "bronx" in a:
        return "Bronx"
    # A lot of places will say "Manhattan" or "New York, NY"
    if "manhattan" in a or "new york, ny" in a:
        return "Manhattan"
    return None


def generate_tourist_locations_dataset(
    borough_shp_path: str | None = None,
    output_path: str = "cleaned_data/nyc_attractions.csv",
) -> pd.DataFrame:
    """
    Load tourist locations, derive Region (last comma chunk of Address),
    infer BoroName via text (and optionally via geo if lat/lon exists),
    then write the enriched Excel file.
    """
    if borough_shp_path is None:
        borough_shp_path = os.getenv("NYC_BOURUGH_COORDS") or "nybb.shp"

    tourist_xlsx_path = os.getenv("NYC_ATTRACTIONS_PATH")
    # 1) Load tourist locations
    df = pd.read_excel(tourist_xlsx_path)

    # General region = last chunk after the final comma in the address
    df["Region"] = df["Address"].str.split(",").str[-1].str.strip()

    # 2) Text-based borough guess
    df["BoroName_text"] = df["Address"].apply(guess_boro_from_text)

    # 3) Final borough: prefer geo-based (if present), else text-based
    if "BoroName_geo" in df.columns:
        df["BoroName"] = df["BoroName_geo"].fillna(df["BoroName_text"])
    else:
        df["BoroName"] = df["BoroName_text"]

    # 5) Save subset
    out_cols = ["Tourist_Spot", "Address", "Zipcode", "Region", "BoroName"]
    df_out = df[out_cols].copy()
    df_out.to_csv(output_path, index=False)

    print(f"[generate_tourist_locations_dataset] Saved {len(df_out)} rows to {output_path}")
    return df_out


# ----------------------------------------------------------------------
# 3. Restaurants: keep only those within 5 boroughs (lon/lat given)
# ----------------------------------------------------------------------
def generate_restaurants_dataset(
    restaurants_csv_path: str | None = None,
    borough_shp_path: str | None = None,
    output_path: str = "cleaned_data/nyc_restaurants.csv",
) -> pd.DataFrame:
    """
    Load restaurant dataset with latitude/longitude, attach BoroName via
    point-in-polygon, filter to NYC, and write the cleaned CSV.
    """
    if restaurants_csv_path is None:
        restaurants_csv_path = os.getenv("NYC_RESTAURANTS_PATH") or "google_maps_restaurants(cleaned).csv"
    if borough_shp_path is None:
        borough_shp_path = os.getenv("NYC_BOURUGH_COORDS") or "nybb.shp"

    df = pd.read_csv(restaurants_csv_path)

    # Normalize Lat/Lon
    if "Lat" in df.columns:
        df.rename(columns={"Lat": "latitude"}, inplace=True)
    if "Lon" in df.columns:
        df.rename(columns={"Lon": "longitude"}, inplace=True)

    if "latitude" not in df.columns or "longitude" not in df.columns:
        raise ValueError("Dataset must contain Lat and Lon columns.")

    borough_shapes, transformer = load_borough_shapes_and_transformer(borough_shp_path)

    df["BoroName"] = df.apply(
        lambda r: boro_from_point(r["latitude"], r["longitude"], borough_shapes, transformer),
        axis=1,
    )

    df_nyc = df[df["BoroName"].notna()].copy()

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df_nyc.to_csv(output_path, index=False)

    print(f"[generate_restaurants_dataset] Saved {len(df_nyc)} rows to {output_path}")
    return df_nyc

if __name__ == "__main__":
    generate_nyc_hotel_dataset()
    generate_tourist_locations_dataset()
    generate_restaurants_dataset()
