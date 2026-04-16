"""
KV Cache Workload Generator + Metrics Collector for vLLM + LMCache
Demonstrates cache hits via prefix sharing patterns.

Usage:
    python workload.py --host <cloudlab-ip> --port 8000
"""

import argparse
import json
import time
import threading
import requests
import csv
import sys
from datetime import datetime
from pathlib import Path

# ── Workload patterns designed to stress KV cache ──────────────────────────
# Pattern 1: Same system prompt, different continuations → prefix cache hits
# Pattern 2: Repeated identical prompts → full cache hits
# Pattern 3: Unique prompts → cache misses (baseline)

SYSTEM_PROMPT = (
    "You are an expert assistant in distributed systems and storage. "
    "Answer concisely and technically."
)

SHARED_PREFIX = (
    "Context: A distributed key-value store uses consistent hashing with "
    "virtual nodes to distribute data across a cluster of 16 nodes. Each "
    "node stores 3 replicas. The cluster uses Raft for consensus.\n\n"
)

WORKLOADS = {
    "prefix_reuse": [
        # Same prefix, different questions → should hit prefix cache
        SHARED_PREFIX + "Question: What happens during a node failure?",
        SHARED_PREFIX + "Question: How does rebalancing work when a new node joins?",
        SHARED_PREFIX + "Question: Explain the read path for a key lookup.",
        SHARED_PREFIX + "Question: What is the write amplification factor here?",
        SHARED_PREFIX + "Question: How does the cluster handle network partitions?",
        SHARED_PREFIX + "Question: Describe the compaction strategy.",
        SHARED_PREFIX + "Question: What are the latency implications of 3 replicas?",
        SHARED_PREFIX + "Question: How does leader election work in Raft here?",
    ],
    "exact_repeats": [
        # Exact same prompt repeated → full cache hit
        SHARED_PREFIX + "Question: Summarize the architecture in one paragraph.",
    ] * 6,
    "cold_misses": [
        # Unique prompts with no shared prefix → cache misses
        f"Explain concept {i}: {topic}"
        for i, topic in enumerate([
            "B-tree vs LSM-tree trade-offs",
            "MVCC in PostgreSQL",
            "Write-ahead logging in SQLite",
            "Copy-on-write in ZFS",
            "Extent-based allocation in ext4",
            "Log-structured file systems",
        ])
    ],
}

# ── Metrics collection ──────────────────────────────────────────────────────

class MetricsCollector(threading.Thread):
    """Polls vLLM Prometheus metrics endpoint and logs to CSV."""

    METRICS_OF_INTEREST = [
        "vllm:gpu_cache_usage_perc",
        "vllm:cpu_cache_usage_perc",
        "vllm:num_requests_running",
        "vllm:num_requests_waiting",
        "vllm:num_requests_swapped",
        "vllm:prompt_tokens_total",
        "vllm:generation_tokens_total",
        # LMCache metrics (if exposed)
        "lmcache_local_cache_size",
        "lmcache_remote_cache_size",
        "lmcache_hit_tokens",
        "lmcache_miss_tokens",
    ]

    def __init__(self, host, port, output_csv, interval=1.0):
        super().__init__(daemon=True)
        self.url = f"http://{host}:{port}/metrics"
        self.output_csv = output_csv
        self.interval = interval
        self.running = False
        self._data = []
        self._lock = threading.Lock()

    def run(self):
        self.running = True
        with open(self.output_csv, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["timestamp"] + self.METRICS_OF_INTEREST)
            while self.running:
                row = self._poll()
                if row:
                    writer.writerow(row)
                    f.flush()
                    with self._lock:
                        self._data.append(row)
                time.sleep(self.interval)

    def _poll(self):
        try:
            resp = requests.get(self.url, timeout=3)
            resp.raise_for_status()
            values = self._parse_prometheus(resp.text)
            ts = datetime.utcnow().isoformat()
            return [ts] + [values.get(m, 0.0) for m in self.METRICS_OF_INTEREST]
        except Exception as e:
            print(f"[metrics] poll error: {e}", file=sys.stderr)
            return None

    def _parse_prometheus(self, text):
        values = {}
        for line in text.splitlines():
            if line.startswith("#"):
                continue
            parts = line.rsplit(" ", 1)
            if len(parts) == 2:
                name_labels, val = parts
                name = name_labels.split("{")[0]
                try:
                    values[name] = float(val)
                except ValueError:
                    pass
        return values

    def stop(self):
        self.running = False

    @property
    def data(self):
        with self._lock:
            return list(self._data)


# ── Request sender ──────────────────────────────────────────────────────────

def send_request(host, port, model, messages, request_id, results):
    url = f"http://{host}:{port}/v1/chat/completions"
    payload = {
        "model": model,
        "messages": messages,
        "max_tokens": 150,
        "temperature": 0.0,  # deterministic for cache testing
    }
    start = time.perf_counter()
    try:
        resp = requests.post(url, json=payload, timeout=60)
        resp.raise_for_status()
        data = resp.json()
        elapsed = time.perf_counter() - start
        usage = data.get("usage", {})
        results.append({
            "id": request_id,
            "latency_s": round(elapsed, 3),
            "prompt_tokens": usage.get("prompt_tokens", 0),
            "completion_tokens": usage.get("completion_tokens", 0),
            "timestamp": datetime.utcnow().isoformat(),
        })
        print(
            f"  [{request_id:>3}] {elapsed:.2f}s | "
            f"prompt={usage.get('prompt_tokens',0)} "
            f"compl={usage.get('completion_tokens',0)}"
        )
    except Exception as e:
        print(f"  [{request_id:>3}] ERROR: {e}", file=sys.stderr)
        results.append({"id": request_id, "error": str(e)})


def run_workload(host, port, model, workload_name, prompts, results, delay=0.5):
    print(f"\n{'='*60}")
    print(f"Workload: {workload_name} ({len(prompts)} requests)")
    print(f"{'='*60}")
    for i, prompt in enumerate(prompts):
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ]
        send_request(host, port, model, messages, f"{workload_name}_{i}", results)
        time.sleep(delay)


# ── Main ────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="KV Cache workload generator")
    parser.add_argument("--host", default="localhost")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--model", default="Qwen/Qwen2.5-3B-Instruct")
    parser.add_argument("--delay", type=float, default=0.5,
                        help="Seconds between requests")
    parser.add_argument("--metrics-interval", type=float, default=1.0,
                        help="Seconds between metric polls")
    parser.add_argument("--output-dir", default="results")
    args = parser.parse_args()

    out = Path(args.output_dir)
    out.mkdir(exist_ok=True)
    metrics_csv = out / "metrics.csv"
    requests_json = out / "requests.json"

    # Start metrics collector
    collector = MetricsCollector(
        args.host, args.port, str(metrics_csv), args.metrics_interval
    )
    collector.start()
    print(f"[*] Collecting metrics → {metrics_csv}")
    time.sleep(2)  # let collector warm up

    all_results = []

    # Run workloads in order:
    # 1. Cold misses first (establish baseline)
    # 2. Prefix reuse (should see cache hits)
    # 3. Exact repeats (should see high hit rate)
    for name in ["cold_misses", "prefix_reuse", "exact_repeats"]:
        run_workload(
            args.host, args.port, args.model,
            name, WORKLOADS[name], all_results, args.delay
        )
        time.sleep(2)  # pause between workloads so metrics separate clearly

    time.sleep(3)
    collector.stop()

    with open(requests_json, "w") as f:
        json.dump(all_results, f, indent=2)

    print(f"\n[*] Done. Results → {requests_json}")
    print(f"[*] Metrics  → {metrics_csv}")
    print("\nRun:  python visualize.py  to see the plots.")


if __name__ == "__main__":
    main()
