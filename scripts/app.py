"""CLI entrypoint to craft a NYC itinerary with filtered data and an Ollama LLM."""
from __future__ import annotations

from typing import List

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


def _build_prompt(context: str, days: int, boros: List[str], budget: str) -> str:
    borough_text = ", ".join(boros) if boros else "All boroughs"
    return f"""
    You are an expert NYC travel planner. Design a {days}-day itinerary with distinct Morning, Noon, and Evening plans for each day.
Rules:
- Use the provided borough list ({borough_text}); each day should stay within ONE borough (or the provided boroughs if "All") and not hop to others.
- Pick ONE primary hotel (from the list) for the whole trip; mention it once at the top as the "home base" and do not change hotels per day.
- Use the retrieved hotels, attractions, and restaurants as the primary pool; add famous staples only if the list lacks enough items in that borough.
- Balance variety across days (museums, parks, views, food) and keep travel reasonable by clustering nearby activities. Use attraction addresses provided to keep a day compact; subway/bus/short ride is fine, you do NOT need to walk from the hotel.
- Do NOT repeat the same attraction or restaurant across different days. If the pool is small, reuse only once and explain why, but prefer unique picks.
- Use each attraction at most once. Rotate through restaurants as well to avoid repeats.
- When suggesting restaurants, prefer those listed; if adding new ones, keep cuisine/budget consistent and stay in the same borough for that day, choosing spots close to the day's attractions.

    Context you can rely on:
    {context}

    Output format:
    Home Base: <hotel name + borough + price note>
    Day 1:
    - Morning: ... (include attraction address if mentioned; note nearby restaurant)
    - Noon: ... (include attraction address if mentioned; note nearby restaurant)
    - Evening: ... (include attraction address if mentioned; note nearby restaurant)

    Repeat the format for every day up to Day {days}. Include brief tips for transit (subway/walk) and approximate time windows.
    """


def main() -> None:
    print("NYC Itinerary Generator (local Ollama powered)")
    days = _prompt_days()
    boros = _prompt_boroughs()
    budget = _prompt_budget()

    data = TravelData()
    context = build_context(boros, budget, days, data)
    prompt = _build_prompt(context, days, boros, budget)
    print("\nGenerating your itinerary...\n")
    itinerary = chat_with_model(prompt)
    print(itinerary or "No response from the model. Please verify Ollama is running.")


if __name__ == "__main__":
    main()
