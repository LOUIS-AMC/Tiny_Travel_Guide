"""CLI entrypoint to craft a NYC itinerary with filtered data and an Ollama LLM."""
from __future__ import annotations

from typing import Dict, List

from llm_client import chat_with_model
from rag import BOROUGH_ALIASES, TravelData, build_context, normalize_boroughs


def _prompt_days() -> int:
    while True:
        raw = input("How many days do you want to stay in NYC? (1-7): ").strip()
        if not raw:
            continue
        if raw.isdigit():
            days = int(raw)
            if 1 <= days <= 7:
                return days
        print("Please enter a number between 1 and 7.")


def _prompt_boroughs() -> List[str]:
    valid_inputs = {
        *{k.lower() for k in BOROUGH_ALIASES.keys()},
        *{v.lower() for v in BOROUGH_ALIASES.values()},
    }
    prompt = (
        "Which boroughs would you like to explore (Manhattan, Brooklyn, Staten Island, Bronx, Queens)? "
        "(comma separated or 'all'): "
    )
    while True:
        raw = input(prompt).strip()
        if not raw or raw.lower() == "all":
            return []
        parts = [p.strip() for p in raw.split(",") if p.strip()]
        invalid = [p for p in parts if p.lower() not in valid_inputs]
        if invalid:
            print("Please enter only valid boroughs: Manhattan, Brooklyn, Queens, Bronx, Staten Island (or 'all').")
            continue
        boros = normalize_boroughs(parts)
        if boros:
            return boros
        print("No valid boroughs detected; please try again or enter 'all'.")


def _prompt_budget() -> str:
    allowed = {"low", "medium", "high"}
    prompt = "Trip budget? (low/medium/high): "
    while True:
        raw = input(prompt).strip().lower()
        if raw in allowed:
            return raw
        print("Please enter one of: low, medium, or high.")


def _prompt_season() -> str:
    month_aliases: Dict[str, str] = {
        "jan": "January",
        "january": "January",
        "feb": "February",
        "february": "February",
        "mar": "March",
        "march": "March",
        "apr": "April",
        "april": "April",
        "may": "May",
        "jun": "June",
        "june": "June",
        "jul": "July",
        "july": "July",
        "aug": "August",
        "august": "August",
        "sep": "September",
        "sept": "September",
        "september": "September",
        "oct": "October",
        "october": "October",
        "nov": "November",
        "november": "November",
        "dec": "December",
        "december": "December",
    }
    month_to_season: Dict[str, str] = {
        "january": "winter",
        "february": "winter",
        "march": "spring",
        "april": "spring",
        "may": "spring",
        "june": "summer",
        "july": "summer",
        "august": "summer",
        "september": "fall",
        "october": "fall",
        "november": "fall",
        "december": "winter",
    }
    season_aliases: Dict[str, str] = {
        "winter": "winter",
        "spring": "spring",
        "summer": "summer",
        "fall": "fall",
        "autumn": "fall",
    }
    prompt = "What month or season are you visiting NYC? (e.g., March, summer): "
    while True:
        raw = input(prompt).strip().lower()
        if not raw:
            print("Please enter a month (e.g., March) or a season (winter/spring/summer/fall).")
            continue
        key = raw.replace(".", "").replace(" ", "")
        if key in season_aliases:
            season = season_aliases[key]
            return season.title()
        if key in month_aliases:
            month = month_aliases[key]
            season = month_to_season[month.lower()]
            return f"{month} ({season})"
        print("Please enter a valid month (e.g., March) or season (winter, spring, summer, fall).")


def _prompt_pace() -> str:
    allowed = {"walk-heavy", "balanced", "ride-flexible"}
    prompt = "Preferred pace for getting around? (walk-heavy/balanced/ride-flexible): "
    while True:
        raw = input(prompt).strip().lower().replace(" ", "-")
        if raw in allowed:
            return raw
        print("Please enter one of: walk-heavy, balanced, or ride-flexible.")


def _build_prompt(
    context: str, days: int, boros: List[str], budget: str, season: str, pace: str
) -> str:
    borough_text = ", ".join(boros) if boros else "All boroughs"
    return f"""
    You are an expert NYC travel planner. Design a {days}-day itinerary with distinct Morning, Noon, and Evening plans for each day.
Rules:
- Use the provided borough list ({borough_text}); each day must stay within ONE borough from that list. If the user chose "all", still pick a single borough per day. Do not mix boroughs inside a day.
- Budget level: {budget}. Choose hotels, attractions, and restaurants that fit this budget.
- Travel pace: {pace}. Adjust activity density and walking vs transit suggestions accordingly.
- You must recommend one attraction and one restaurant for each time slot (Morning, Noon, Evening) per day.
- All attractions and restaurants for a given day must have the same BoroName as that day's borough. If none exist for a borough, pick another borough from the allowed list; do not cross-hop.
- Restaurants should be close to the day's attractions (same borough/region). Do not invent far-away venues; prefer those provided.
- Do NOT provide transit directions or step-by-step routing; keep focus on the places only.
- Season/month: {season}. Favor weather-appropriate picks (indoor vs outdoor) and include season-aware packing tips.
- Pick ONE primary hotel (from the list) for the whole trip; mention it once at the top as the "home base" and do not change hotels per day.
- Use the retrieved hotels, attractions, and restaurants as the primary pool; add famous staples only if the list lacks enough items in that borough.
- Balance variety across days (museums, parks, views, food) and keep travel reasonable by clustering nearby activities. Use attraction addresses provided to keep a day compact.
- Do NOT repeat the same attraction or restaurant across different days. If the pool is small, reuse only once and explain why, but prefer unique picks.
- When suggesting restaurants, prefer those listed; if adding new ones, keep cuisine/budget consistent and stay in the same borough for that day, choosing spots close to the day's attractions.

Deliverables:
1) Home Base and a day-by-day plan (Morning/Noon/Evening) with time windows (no transit directions).
2) Packing list tailored to the stated season/month in NYC (weather-aware items, footwear, MetroCard/tap guidance).

Context you can rely on:
{context}

Output format:
Home Base: <hotel name + borough + price note>
Day 1:
- Morning: ... (include attraction address if mentioned; note nearby restaurant)
- Noon: ... (include attraction address if mentioned; note nearby restaurant)
- Evening: ... (include attraction address if mentioned; note nearby restaurant)

Repeat the format for every day up to Day {days}. After the final day, include:
- Packing list: ...
    """


def main() -> None:
    print("NYC Itinerary Generator (local Ollama powered)")
    days = _prompt_days()
    boros = _prompt_boroughs()
    budget = _prompt_budget()
    season = _prompt_season()
    pace = _prompt_pace()

    data = TravelData()
    context = build_context(boros, budget, days, data, season=season, pace=pace)
    prompt = _build_prompt(context, days, boros, budget, season, pace)
    print("\nGenerating your itinerary...\n")
    itinerary = chat_with_model(prompt)
    print(itinerary or "No response from the model. Please verify Ollama is running.")


if __name__ == "__main__":
    main()
