import os
import json

import requests


BASE_URL = os.getenv("BASE_URL", "http://127.0.0.1:8000")


def _print_response(title: str, response: requests.Response) -> None:
    print(f"\n{title} [{response.status_code}]")
    try:
        print(json.dumps(response.json(), indent=2, ensure_ascii=False))
    except Exception:
        print(response.text)


def main() -> None:
    text = "I am angry because the issue was ignored."

    pred = requests.post(f"{BASE_URL}/predict", json={"text": text, "top_k": 3}, timeout=30)
    _print_response("PREDICT", pred)

    rewrite = requests.post(
        f"{BASE_URL}/rewrite",
        json={
            "text": text,
            "target_tone": "professional",
            "user_instruction": "Be calm and keep it short",
        },
        timeout=30,
    )
    _print_response("REWRITE", rewrite)

    combined = requests.post(
        f"{BASE_URL}/analyze-rewrite",
        json={
            "text": text,
            "target_tone": "professional",
            "user_instruction": "Keep it short",
        },
        timeout=30,
    )
    _print_response("ANALYZE-REWRITE", combined)

    suggestions = requests.get(f"{BASE_URL}/tones/suggestions", params={"emotion": "anger"}, timeout=30)
    _print_response("TONE SUGGESTIONS", suggestions)


if __name__ == "__main__":
    main()

