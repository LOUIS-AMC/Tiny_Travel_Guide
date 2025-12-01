"""Quick health check to verify a local Ollama server is reachable."""
from __future__ import annotations

import json
import os
import sys
import urllib.error
import urllib.request

def check_ollama(host: str | None = None, timeout: int = 5) -> int:
    """
    Ping the Ollama `/api/tags` endpoint to confirm the server is up.
    Returns 0 on success, 1 on failure.
    """
    base = host or os.environ.get("OLLAMA_HOST", "http://127.0.0.1:11434")
    url = f"{base.rstrip('/')}/api/tags"
    try:
        with urllib.request.urlopen(url, timeout=timeout) as resp:
            payload = resp.read().decode("utf-8")
            data = json.loads(payload)
            available = [m.get("name", "") for m in data.get("models", [])]
            print(f"Ollama reachable at {base} (HTTP {resp.status}).")
            if available:
                print("Available models:")
                for name in available:
                    print(f"- {name}")
            else:
                print("No models returned; you may need to pull one.")
        return 0
    except urllib.error.URLError as exc:
        print(f"Failed to reach Ollama at {base}: {exc}")
        print("Start Ollama (e.g., `ollama serve` or open the app) and ensure the host/port are correct.")
        return 1
    except json.JSONDecodeError:
        print(f"Ollama at {base} responded with non-JSON content.")
        return 1


if __name__ == "__main__":
        check_ollama()