#!/usr/bin/env python3
"""
Drive concurrent vLLM traffic and sample cache-related Prometheus metrics.

What it does:
- Sends repeated-prefix chat completion requests to a vLLM OpenAI-compatible server
- Keeps the model busy with configurable concurrency
- Scrapes /metrics periodically
- Extracts KV/prefix/LMCache counters
- Writes:
    1) raw_metrics.csv      raw sampled values
    2) delta_metrics.csv    per-sample deltas for counters
    3) requests.csv         per-request latency / status

Example:
    python kv_probe.py \
      --base-url http://localhost:8000 \
      --model Qwen/Qwen2.5-3B-Instruct \
      --duration 120 \
      --concurrency 8 \
      --sample-interval 1.0 \
      --output-dir kv_probe_out
"""

from __future__ import annotations

import argparse
import asyncio
import csv
import json
import math
import random
import signal
import sys
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import aiohttp


METRIC_NAMES = [
    "vllm:kv_cache_usage_perc",
    "vllm:prefix_cache_queries_total",
    "vllm:prefix_cache_hits_total",
    "vllm:external_prefix_cache_queries_total",
    "vllm:external_prefix_cache_hits_total",
    "lmcache:num_vllm_hit_tokens_total",
    "lmcache:num_requested_tokens_total",
    "lmcache:num_hit_tokens_total",
    "lmcache:num_store_requests_total",
    "lmcache:num_retrieve_requests_total",
    "lmcache:local_cache_usage",
    "lmcache:remote_cache_usage",
]


PREFIXES = [
    (
        "p1",
        "KV cache test prefix KV cache test prefix KV cache test prefix KV cache test prefix "
        "KV cache test prefix. "
    ),
    (
        "p2",
        "Repeated prompt prefix experiment repeated prompt prefix experiment repeated prompt "
        "prefix experiment repeated prompt prefix experiment. "
    ),
    (
        "p3",
        "Caching benchmark shared context caching benchmark shared context caching benchmark "
        "shared context caching benchmark shared context. "
    ),
]


SUFFIXES = [
    "Tell me what caching means.",
    "Explain why repeated prefixes help.",
    "Give a short definition of prefix caching.",
    "Why can shared prompt prefixes reduce prefill work?",
    "Summarize how KV reuse affects inference latency.",
    "Explain the difference between prefill and decode.",
    "Describe how cached prompt states help repeated requests.",
    "Why might a repeated prefix not produce a 100 percent hit rate?",
]


@dataclass
class RequestRecord:
    timestamp: float
    worker_id: int
    request_id: int
    prefix_id: str
    prompt_chars: int
    latency_s: float
    http_status: int
    ok: int
    error: str


@dataclass
class MetricSample:
    timestamp: float
    sample_idx: int
    values: Dict[str, float]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--base-url", default="http://localhost:8000")
    p.add_argument("--model", required=True)
    p.add_argument("--duration", type=float, default=60.0)
    p.add_argument("--concurrency", type=int, default=4)
    p.add_argument("--sample-interval", type=float, default=1.0)
    p.add_argument("--request-timeout", type=float, default=120.0)
    p.add_argument("--max-tokens", type=int, default=64)
    p.add_argument("--temperature", type=float, default=0.0)
    p.add_argument("--output-dir", default="kv_probe_out")
    p.add_argument(
        "--mode",
        choices=["mixed", "single-prefix", "alternating"],
        default="mixed",
        help="Traffic pattern. mixed=random shared prefixes, single-prefix=maximizes hits, alternating=cycle deterministically.",
    )
    p.add_argument(
        "--long-prefix-repeat",
        type=int,
        default=1,
        help="Repeat each built-in prefix N times to make shared prefixes longer and hits more obvious.",
    )
    return p.parse_args()


def build_prefixes(repeat: int) -> List[Tuple[str, str]]:
    out = []
    for prefix_id, text in PREFIXES:
        out.append((prefix_id, text * max(1, repeat)))
    return out


def parse_prometheus_metrics(text: str, wanted_names: List[str]) -> Dict[str, float]:
    wanted = set(wanted_names)
    results: Dict[str, float] = {}
    for line in text.splitlines():
        if not line or line.startswith("#"):
            continue
        # format: metric_name{...} value
        # or     metric_name value
        parts = line.strip().split()
        if len(parts) < 2:
            continue
        left, right = parts[0], parts[-1]
        if "{" in left:
            metric_name = left.split("{", 1)[0]
        else:
            metric_name = left
        if metric_name not in wanted:
            continue
        try:
            results[metric_name] = float(right)
        except ValueError:
            continue
    for name in wanted_names:
        results.setdefault(name, math.nan)
    return results


async def fetch_metrics(session: aiohttp.ClientSession, base_url: str) -> Dict[str, float]:
    async with session.get(f"{base_url}/metrics") as resp:
        resp.raise_for_status()
        text = await resp.text()
    return parse_prometheus_metrics(text, METRIC_NAMES)


def choose_prompt(
    mode: str,
    req_idx: int,
    prefixes: List[Tuple[str, str]],
) -> Tuple[str, str]:
    if mode == "single-prefix":
        prefix_id, prefix = prefixes[0]
        suffix = random.choice(SUFFIXES)
        return prefix_id, prefix + suffix

    if mode == "alternating":
        prefix_id, prefix = prefixes[req_idx % len(prefixes)]
        suffix = SUFFIXES[req_idx % len(SUFFIXES)]
        return prefix_id, prefix + suffix

    prefix_id, prefix = random.choice(prefixes)
    suffix = random.choice(SUFFIXES)
    return prefix_id, prefix + suffix


async def one_request(
    session: aiohttp.ClientSession,
    base_url: str,
    model: str,
    prompt: str,
    temperature: float,
    max_tokens: int,
) -> Tuple[int, Optional[dict], str]:
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": temperature,
        "max_tokens": max_tokens,
    }
    try:
        async with session.post(
            f"{base_url}/v1/chat/completions",
            json=payload,
        ) as resp:
            status = resp.status
            text = await resp.text()
            if status != 200:
                return status, None, text[:500]
            try:
                return status, json.loads(text), ""
            except json.JSONDecodeError:
                return status, None, f"bad json: {text[:500]}"
    except Exception as e:
        return -1, None, repr(e)


async def worker_loop(
    worker_id: int,
    session: aiohttp.ClientSession,
    args: argparse.Namespace,
    stop_event: asyncio.Event,
    request_records: List[RequestRecord],
    prefixes: List[Tuple[str, str]],
    counter_ref: Dict[str, int],
) -> None:
    while not stop_event.is_set():
        req_id = counter_ref["value"]
        counter_ref["value"] += 1

        prefix_id, prompt = choose_prompt(args.mode, req_id, prefixes)
        t0 = time.time()
        status, _body, error = await one_request(
            session=session,
            base_url=args.base_url,
            model=args.model,
            prompt=prompt,
            temperature=args.temperature,
            max_tokens=args.max_tokens,
        )
        latency = time.time() - t0
        request_records.append(
            RequestRecord(
                timestamp=t0,
                worker_id=worker_id,
                request_id=req_id,
                prefix_id=prefix_id,
                prompt_chars=len(prompt),
                latency_s=latency,
                http_status=status,
                ok=int(status == 200),
                error=error,
            )
        )


async def metrics_sampler(
    session: aiohttp.ClientSession,
    args: argparse.Namespace,
    stop_event: asyncio.Event,
    samples: List[MetricSample],
) -> None:
    idx = 0
    while not stop_event.is_set():
        t = time.time()
        try:
            values = await fetch_metrics(session, args.base_url)
        except Exception:
            values = {name: math.nan for name in METRIC_NAMES}
        samples.append(MetricSample(timestamp=t, sample_idx=idx, values=values))
        idx += 1
        await asyncio.sleep(args.sample_interval)

    # one final scrape after workers stop
    try:
        t = time.time()
        values = await fetch_metrics(session, args.base_url)
        samples.append(MetricSample(timestamp=t, sample_idx=idx, values=values))
    except Exception:
        pass


def write_raw_metrics_csv(samples: List[MetricSample], out_path: Path) -> None:
    fieldnames = ["timestamp", "sample_idx", *METRIC_NAMES]
    with out_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for s in samples:
            row = {"timestamp": s.timestamp, "sample_idx": s.sample_idx}
            row.update(s.values)
            writer.writerow(row)


def write_delta_metrics_csv(samples: List[MetricSample], out_path: Path) -> None:
    counter_metrics = [
        "vllm:prefix_cache_queries_total",
        "vllm:prefix_cache_hits_total",
        "vllm:external_prefix_cache_queries_total",
        "vllm:external_prefix_cache_hits_total",
        "lmcache:num_vllm_hit_tokens_total",
        "lmcache:num_requested_tokens_total",
        "lmcache:num_hit_tokens_total",
        "lmcache:num_store_requests_total",
        "lmcache:num_retrieve_requests_total",
    ]
    gauge_metrics = [
        "vllm:kv_cache_usage_perc",
        "lmcache:local_cache_usage",
        "lmcache:remote_cache_usage",
    ]

    fieldnames = [
        "timestamp",
        "sample_idx",
        "dt_s",
        *[f"delta::{m}" for m in counter_metrics],
        *[f"gauge::{m}" for m in gauge_metrics],
        "derived::prefix_hit_rate_delta",
        "derived::external_prefix_hit_rate_delta",
    ]

    with out_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for prev, cur in zip(samples, samples[1:]):
            dt_s = cur.timestamp - prev.timestamp
            row = {
                "timestamp": cur.timestamp,
                "sample_idx": cur.sample_idx,
                "dt_s": dt_s,
            }

            for m in counter_metrics:
                pv = prev.values.get(m, math.nan)
                cv = cur.values.get(m, math.nan)
                row[f"delta::{m}"] = cv - pv if not (math.isnan(pv) or math.isnan(cv)) else math.nan

            for m in gauge_metrics:
                row[f"gauge::{m}"] = cur.values.get(m, math.nan)

            dq = row["delta::vllm:prefix_cache_queries_total"]
            dh = row["delta::vllm:prefix_cache_hits_total"]
            row["derived::prefix_hit_rate_delta"] = (dh / dq) if dq and dq > 0 else math.nan

            edq = row["delta::vllm:external_prefix_cache_queries_total"]
            edh = row["delta::vllm:external_prefix_cache_hits_total"]
            row["derived::external_prefix_hit_rate_delta"] = (edh / edq) if edq and edq > 0 else math.nan

            writer.writerow(row)


def write_requests_csv(records: List[RequestRecord], out_path: Path) -> None:
    fieldnames = list(asdict(records[0]).keys()) if records else [
        "timestamp", "worker_id", "request_id", "prefix_id",
        "prompt_chars", "latency_s", "http_status", "ok", "error"
    ]
    with out_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in records:
            writer.writerow(asdict(r))


def print_summary(samples: List[MetricSample], records: List[RequestRecord]) -> None:
    ok_records = [r for r in records if r.ok == 1]
    if records:
        print(f"requests_total={len(records)} ok={len(ok_records)} errors={len(records) - len(ok_records)}")
    if ok_records:
        latencies = sorted(r.latency_s for r in ok_records)
        p50 = latencies[len(latencies) // 2]
        p95 = latencies[min(len(latencies) - 1, int(len(latencies) * 0.95))]
        print(f"latency_p50_s={p50:.3f} latency_p95_s={p95:.3f}")

    if len(samples) >= 2:
        first, last = samples[0], samples[-1]
        def delta(name: str) -> float:
            a = first.values.get(name, math.nan)
            b = last.values.get(name, math.nan)
            return b - a if not (math.isnan(a) or math.isnan(b)) else math.nan

        dq = delta("vllm:prefix_cache_queries_total")
        dh = delta("vllm:prefix_cache_hits_total")
        print(f"delta_prefix_queries={dq}")
        print(f"delta_prefix_hits={dh}")
        if dq and dq > 0:
            print(f"overall_prefix_hit_rate={dh / dq:.4f}")

        print(f"final_kv_cache_usage_perc={last.values.get('vllm:kv_cache_usage_perc', math.nan)}")
        print(f"delta_lmcache_vllm_hit_tokens={delta('lmcache:num_vllm_hit_tokens_total')}")


async def main_async(args: argparse.Namespace) -> int:
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    timeout = aiohttp.ClientTimeout(total=args.request_timeout)
    connector = aiohttp.TCPConnector(limit=max(args.concurrency * 4, 32))
    stop_event = asyncio.Event()

    loop = asyncio.get_running_loop()
    for sig in (signal.SIGINT, signal.SIGTERM):
        try:
            loop.add_signal_handler(sig, stop_event.set)
        except NotImplementedError:
            pass

    request_records: List[RequestRecord] = []
    metric_samples: List[MetricSample] = []
    request_counter = {"value": 0}
    prefixes = build_prefixes(args.long_prefix_repeat)

    async with aiohttp.ClientSession(timeout=timeout, connector=connector) as session:
        sampler_task = asyncio.create_task(metrics_sampler(session, args, stop_event, metric_samples))
        worker_tasks = [
            asyncio.create_task(
                worker_loop(
                    worker_id=i,
                    session=session,
                    args=args,
                    stop_event=stop_event,
                    request_records=request_records,
                    prefixes=prefixes,
                    counter_ref=request_counter,
                )
            )
            for i in range(args.concurrency)
        ]

        try:
            await asyncio.sleep(args.duration)
        finally:
            stop_event.set()
            await asyncio.gather(*worker_tasks, return_exceptions=True)
            await asyncio.gather(sampler_task, return_exceptions=True)

    write_raw_metrics_csv(metric_samples, out_dir / "raw_metrics.csv")
    write_delta_metrics_csv(metric_samples, out_dir / "delta_metrics.csv")
    write_requests_csv(request_records, out_dir / "requests.csv")
    print_summary(metric_samples, request_records)

    print(f"wrote {out_dir / 'raw_metrics.csv'}")
    print(f"wrote {out_dir / 'delta_metrics.csv'}")
    print(f"wrote {out_dir / 'requests.csv'}")
    return 0


def main() -> int:
    args = parse_args()
    return asyncio.run(main_async(args))


if __name__ == "__main__":
    raise SystemExit(main())
