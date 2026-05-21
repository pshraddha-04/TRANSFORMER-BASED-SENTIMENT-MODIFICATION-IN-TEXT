"""Simple latency benchmark for the /predict endpoint."""

import argparse
import statistics
import time

import requests


def run_benchmark(base_url: str, text: str, iterations: int, warmup: int, api_key: str | None) -> dict:
    endpoint = f"{base_url.rstrip('/')}/predict"
    payload = {"text": text, "top_k": 3}
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["X-API-Key"] = api_key

    for _ in range(warmup):
        requests.post(endpoint, json=payload, headers=headers, timeout=20)

    latencies = []
    errors = 0
    for _ in range(iterations):
        started = time.perf_counter()
        response = requests.post(endpoint, json=payload, headers=headers, timeout=20)
        elapsed_ms = (time.perf_counter() - started) * 1000
        if response.status_code == 200:
            latencies.append(elapsed_ms)
        else:
            errors += 1

    if not latencies:
        return {"iterations": iterations, "errors": errors, "success": 0}

    p50 = statistics.median(latencies)
    sorted_latencies = sorted(latencies)
    index_95 = max(0, int(round(0.95 * (len(sorted_latencies) - 1))))
    p95 = sorted_latencies[index_95]
    return {
        "iterations": iterations,
        "success": len(latencies),
        "errors": errors,
        "avg_ms": round(statistics.mean(latencies), 2),
        "p50_ms": round(p50, 2),
        "p95_ms": round(p95, 2),
        "min_ms": round(min(latencies), 2),
        "max_ms": round(max(latencies), 2),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark /predict latency")
    parser.add_argument("--base-url", default="http://127.0.0.1:8000", help="API base URL")
    parser.add_argument("--text", default="I am upset about this delay", help="Input text for inference")
    parser.add_argument("--iterations", type=int, default=50, help="Measured requests")
    parser.add_argument("--warmup", type=int, default=5, help="Warmup requests")
    parser.add_argument("--api-key", default=None, help="Optional X-API-Key value")
    args = parser.parse_args()

    result = run_benchmark(
        base_url=args.base_url,
        text=args.text,
        iterations=max(1, args.iterations),
        warmup=max(0, args.warmup),
        api_key=args.api_key,
    )

    print("Benchmark result")
    for key, value in result.items():
        print(f"- {key}: {value}")


if __name__ == "__main__":
    main()

