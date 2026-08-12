"""Verify local model setup (Ollama).

Run after following SETUP_MODELS.md. Reports whether the local Ollama server is
reachable and which models are installed, without failing hard.

    python verify_local_models.py
"""
from __future__ import annotations

import os


def check_ollama() -> None:
    print("\n[1] Ollama (generative SLMs) ...")
    base = os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434").rstrip("/")
    if base.endswith("/v1"):
        base = base[: -len("/v1")]
    try:
        import requests
    except Exception:
        print("   ✗ 'requests' not installed (it is a core dep — run `pip install .`)")
        return
    try:
        resp = requests.get(f"{base}/api/tags", timeout=3)
        resp.raise_for_status()
        models = [m.get("name") for m in resp.json().get("models", [])]
    except Exception as exc:
        print(f"   ✗ Ollama not reachable at {base}: {exc}")
        print("   → fix: `brew install ollama` then `ollama serve`")
        return
    if not models:
        print(f"   ✓ Ollama up at {base}, but no models pulled.")
        print("   → run: ollama pull qwen2.5:3b")
        return
    print(f"   ✓ Ollama up at {base}; models: {models}")
    try:
        gen = requests.post(
            f"{base}/api/generate",
            json={"model": models[0], "prompt": "Reply with one word: ok", "stream": False},
            timeout=60,
        )
        gen.raise_for_status()
        print(f"   ✓ generate({models[0]}) -> {gen.json().get('response', '').strip()[:40]!r}")
    except Exception as exc:
        print(f"   ! generation test skipped: {exc}")


def main() -> None:
    print("=== Local model verification ===")
    check_ollama()
    print("\nSee SETUP_MODELS.md for full instructions.")


if __name__ == "__main__":
    main()
