# Tiny Travel Guide (NYC)
Local LLM-powered generator that builds morning/noon/evening itineraries for 1–7 day NYC trips using curated hotel, attraction, and restaurant datasets.

## Prerequisites
- Python 3.9+ and `pip`
- Ollama running locally
- NYC datasets downloaded to paths referenced in `.env` (see below)

## Quick Start
1) Install Ollama (https://ollama.com/download) so the `ollama` CLI is available.
2) Install Python deps and pull the model:
```bash
make setup          # installs requirements.txt and pulls the Ollama model
```
3) Add a `.env` file (see template below) pointing to your data files and preferred model.
4) Generate cleaned datasets (writes into `cleaned_data/`):
```bash
python data/generate_dataset.py
```
5) Optional: verify Ollama connectivity:
```bash
python scripts/test_ollama.py
```
6) Generate an itinerary:
```bash
python scripts/app.py
```

## .env template
Create a `.env` in the project root with the following keys:
```
NYC_HOTEL_PATH=/path/to/new_york_hotels.csv
NYC_BOURUGH_COORDS=/path/to/nybb.shp
NYC_ATTRACTIONS_PATH=/path/to/New_York_Tourist_Locations.xlsx
NYC_RESTAURANTS_PATH=/path/to/google_maps_restaurants(cleaned).csv
OLLAMA_MODEL=hf.co/bartowski/Qwen2.5-1.5B-Instruct-GGUF
```

## Data sources
- New York Hotels: https://www.kaggle.com/datasets/gdberrio/new-york-hotels
- NYC Boundaries: https://www.nyc.gov/content/planning/pages/resources/datasets/borough-boundaries
- New York City Restaurants: https://www.kaggle.com/datasets/beridzeg45/nyc-restaurants
- New York City Tourist Locations: https://www.kaggle.com/datasets/anirudhmunnangi/348-new-york-tourist-locations
