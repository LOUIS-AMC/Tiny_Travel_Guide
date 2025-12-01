"""Thin wrapper around Ollama for itinerary generation."""
from __future__ import annotations
import ollama
import os
from dotenv import load_dotenv
from typing import Dict, Optional

load_dotenv()

DEFAULT_MODEL = os.getenv("OLLAMA_MODEL", "hf.co/bartowski/Qwen2.5-1.5B-Instruct-GGUF")


def chat_with_model(
    prompt: str, *, model: Optional[str] = None, temperature: float = 0.6
) -> str:
    """Send a chat request to the local Ollama model."""

    chosen_model = model or DEFAULT_MODEL
    try:
        response: Dict[str, Dict[str, str]] = ollama.chat(
            model=chosen_model,
            messages=[{"role": "user", "content": prompt}],
            options={"temperature": temperature},
        )
    except Exception as exc:  # pragma: no cover - protection around client call
        raise RuntimeError(
            f"Ollama call failed. Ensure the model '{chosen_model}' is available and Ollama is running."
        ) from exc
    message = response['message']['content']
    return message.strip()
