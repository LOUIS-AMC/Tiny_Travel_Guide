"""CLI entrypoint to craft a NYC itinerary with filtered data and an Ollama LLM."""
from __future__ import annotations

from typing import List

from llm_client import chat_with_model
from rag import TravelData, build_context, normalize_boroughs


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
    raw = input(
        "Which boroughs would you like to explore (Manhanttan, Brooklyn, Staten Island, Bronx, Queens)? (comma separated or 'all'): "
    ).strip()
    if not raw or raw.lower() == "all":
        return []
    parts = [p.strip() for p in raw.split(",")]
    boros = normalize_boroughs(parts)
    if not boros:
        print("No valid boroughs detected; defaulting to all.")
    return boros


def _prompt_budget() -> str:
    raw = input("Trip budget? (low/medium/high) [medium]: ").strip().lower()
    if raw not in {"low", "medium", "high"}:
        return "medium"
    return raw


def _build_prompt(context: str, days: int, boros: List[str], budget: str) -> str:
    borough_text = ", ".join(boros) if boros else "All boroughs"
    return f"""
            You are an expert NYC travel planner. Design a {days}-day itinerary with distinct Morning, Noon, and Evening plans for each day.
            Keep recommendations aligned to the provided borough list ({borough_text}) and the budget tier '{budget}'.
            Use the retrieved hotels, attractions, and restaurants as the primary pool; only add famous staples if needed to complete days.
            Balance variety across days (museums, parks, views, food) and keep travel reasonable (cluster nearby activities).
            When suggesting restaurants, prefer those listed; if adding new ones, keep cuisine/budget consistent.

            Context you can rely on:
            {context}

            Output format:
            Day 1:
            - Morning: ...
            - Noon: ...
            - Evening: ...

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
