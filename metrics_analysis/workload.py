"""
KV Cache Workload Runner
Reads workload definitions from input/*.json and saves each run to
output/<workload_name>_<YYYYMMDD_HHMMSS>/

Usage:
    # Run all workloads in input/
    python workload.py

    # Run a specific workload file
    python workload.py --workload input/heavy_cache_reuse.json

    # Run against a remote host
    python workload.py --host <ip> --port 8000
"""

import argparse
import csv
import json
import sys
import threading
import time
from datetime import datetime
from pathlib import Path

import requests


# ── Metrics collector ───────────────────────────────────────────────────────

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
                time.sleep(self.interval)

    def _poll(self):
        try:
            resp = requests.get(self.url, timeout=3)
            resp.raise_for_status()
            values = self._parse(resp.text)
            ts = datetime.utcnow().isoformat()
            return [ts] + [values.get(m, 0.0) for m in self.METRICS_OF_INTEREST]
        except Exception as e:
            print(f"[metrics] poll error: {e}", file=sys.stderr)
            return None

    def _parse(self, text):
        values = {}
        for line in text.splitlines():
            if line.startswith("#"):
                continue
            parts = line.rsplit(" ", 1)
            if len(parts) == 2:
                name = parts[0].split("{")[0]
                try:
                    values[name] = float(parts[1])
                except ValueError:
                    pass
        return values

    def stop(self):
        self.running = False


# ── Request sender ──────────────────────────────────────────────────────────

def send_request(host, port, model, system_prompt, user_prompt, request_id, results):
    url = f"http://{host}:{port}/v1/chat/completions"
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user",   "content": user_prompt},
    ]
    payload = {
        "model":       model,
        "messages":    messages,
        "max_tokens":  150,
        "temperature": 0.0,
    }

    start = time.perf_counter()
    try:
        resp = requests.post(url, json=payload, timeout=120)
        resp.raise_for_status()
        data    = resp.json()
        elapsed = time.perf_counter() - start
        usage   = data.get("usage", {})
        details = usage.get("prompt_tokens_details") or {}
        cached  = details.get("cached_tokens", -1)

        results.append({
            "id":                request_id,
            "latency_s":         round(elapsed, 3),
            "prompt_tokens":     usage.get("prompt_tokens", 0),
            "completion_tokens": usage.get("completion_tokens", 0),
            "cached_tokens":     cached,
            "system_prompt":     system_prompt,
            "user_prompt":       user_prompt,
            "timestamp":         datetime.utcnow().isoformat(),
        })
        cached_str = f"cached={cached}" if cached >= 0 else "cached=n/a"
        print(
            f"  [{request_id:>26}] {elapsed:.2f}s | "
            f"prompt={usage.get('prompt_tokens', 0):>5} "
            f"compl={usage.get('completion_tokens', 0):>4} "
            f"{cached_str}"
        )
    except Exception as e:
        print(f"  [{request_id:>26}] ERROR: {e}", file=sys.stderr)
        results.append({"id": request_id, "error": str(e)})


# ── Workload runner ─────────────────────────────────────────────────────────

def load_workload(path: Path) -> dict:
    with open(path) as f:
        return json.load(f)


def run_workload(host, port, model, workload_def: dict, out_dir: Path, delay: float, metrics_interval: float):
    name        = workload_def["name"]
    system_prompt = workload_def.get("system_prompt", "You are a helpful assistant.")
    prompts     = workload_def["prompts"]

    # Output directory: output/<name>_<timestamp>/
    ts      = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = out_dir / f"{name}_{ts}"
    run_dir.mkdir(parents=True, exist_ok=True)

    metrics_csv   = run_dir / "metrics.csv"
    requests_json = run_dir / "requests.json"
    meta_json     = run_dir / "meta.json"

    # Save workload metadata
    with open(meta_json, "w") as f:
        json.dump({
            "name":        name,
            "description": workload_def.get("description", ""),
            "model":       model,
            "host":        host,
            "port":        port,
            "num_prompts": len(prompts),
            "started_at":  ts,
        }, f, indent=2)

    # Start metrics collector
    collector = MetricsCollector(host, port, str(metrics_csv), metrics_interval)
    collector.start()

    print(f"\n{'='*64}")
    print(f"Workload : {name}")
    print(f"Prompts  : {len(prompts)}")
    print(f"Output   : {run_dir}")
    print(f"{'='*64}")
    time.sleep(2)  # let collector warm up

    results = []
    for i, user_prompt in enumerate(prompts):
        request_id = f"{name}_{i}"
        send_request(host, port, model, system_prompt, user_prompt, request_id, results)
        time.sleep(delay)

    time.sleep(3)
    collector.stop()

    with open(requests_json, "w") as f:
        json.dump(results, f, indent=2)

    # Per-run summary
    good = [r for r in results if "error" not in r]
    if good:
        latencies = [r["latency_s"] for r in good]
        cached    = [r["cached_tokens"] for r in good if r.get("cached_tokens", -1) >= 0]
        print(f"\n  Summary for '{name}':")
        print(f"    Requests      : {len(good)} ok / {len(results)} total")
        print(f"    Latency p50   : {sorted(latencies)[len(latencies)//2]:.2f}s")
        print(f"    Latency mean  : {sum(latencies)/len(latencies):.2f}s")
        if cached:
            print(f"    Cached tokens : min={min(cached)} max={max(cached)} mean={sum(cached)/len(cached):.0f}")

    print(f"\n  Results → {requests_json}")
    print(f"  Metrics → {metrics_csv}")
    return run_dir


# ── Entry point ─────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="KV-cache workload runner")
    parser.add_argument("--host",             default="localhost")
    parser.add_argument("--port",             type=int, default=8000)
    parser.add_argument("--model",            default="Qwen/Qwen2.5-3B-Instruct")
    parser.add_argument("--workload",         default=None,
                        help="Path to a single input/*.json file. "
                             "Omit to run ALL files in --input-dir.")
    parser.add_argument("--input-dir",        default="input",
                        help="Directory containing workload JSON files (default: input/)")
    parser.add_argument("--output-dir",       default="output",
                        help="Root output directory (default: output/)")
    parser.add_argument("--delay",            type=float, default=0.5,
                        help="Seconds between requests within a workload")
    parser.add_argument("--metrics-interval", type=float, default=1.0,
                        help="Seconds between Prometheus metric polls")
    args = parser.parse_args()

    out_dir   = Path(args.output_dir)
    input_dir = Path(args.input_dir)

    # Collect workload files to run
    if args.workload:
        wf_paths = [Path(args.workload)]
    else:
        wf_paths = sorted(input_dir.glob("*.json"))
        if not wf_paths:
            sys.exit(f"ERROR: no *.json files found in {input_dir}/")

    print(f"[*] Will run {len(wf_paths)} workload(s):")
    for p in wf_paths:
        print(f"    {p}")

    run_dirs = []
    for wf_path in wf_paths:
        wf_def  = load_workload(wf_path)
        run_dir = run_workload(
            host             = args.host,
            port             = args.port,
            model            = args.model,
            workload_def     = wf_def,
            out_dir          = out_dir,
            delay            = args.delay,
            metrics_interval = args.metrics_interval,
        )
        run_dirs.append(run_dir)
        time.sleep(3)  # pause between workloads so metrics separate cleanly

    print(f"\n{'='*64}")
    print("[*] All workloads complete. Output directories:")
    for d in run_dirs:
        print(f"    {d}")
    print("\nTo visualise a run:")
    print(f"    python visualize.py --run-dir output/<name>_<timestamp>")


if __name__ == "__main__":
    main()
