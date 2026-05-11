import argparse
from pathlib import Path

from transformers import AutoTokenizer

from src.inference.engine import VLLMEngine
from src.metrics.collector import KVEventCollector
from src.metrics.visualizer import build_token_lifecycle, render_html_report


def get_prompts(profile: str, num_prompts: int) -> list[str]:
    if profile == "default":
        return [
            "Explain vLLM in one sentence.",
            "What is the KV cache and why is it useful?",
            "Give a two-line explanation of cache eviction policies.",
        ]

    shared_prefix = (
        "You are analyzing production cache behavior for an LLM inference service. "
        "Use concise bullet points and focus on actionable diagnostics.\n\n"
    )
    long_context = (
        "System context: Requests share repeated policy/header prefixes, while request tails vary by user. "
        "During burst traffic, memory pressure increases and stale blocks can be evicted. "
        "Latency tends to spike when reusable prefixes are not found and prefill is recomputed. "
        "Telemetry records block-store and block-remove events with timestamps and block hashes. "
        "Operators suspect intermittent cache thrashing and want evidence-backed mitigation ideas. "
    ) * 10

    tasks = [
        "Summarize why prefix reuse lowers prefill cost in exactly 6 bullets.",
        "Create a triage checklist for diagnosing cache thrashing in a busy service.",
        "Write a brief incident note describing likely causes of repeated evictions.",
        "Propose 8 experiments to validate prefix-caching effectiveness.",
        "Draft a compact runbook section with headings and operator actions.",
        "List indicators that distinguish healthy churn from harmful eviction storms.",
    ]

    if profile == "eviction-stress":
        prompts = []
        for i in range(num_prompts):
            unique_prefix = (
                f"Request template id={i}. "
                "This prompt intentionally varies prefix tokens to minimize cache reuse. "
            )
            unique_body = (
                "Operational telemetry includes request ids, policy snapshots, and time-window aggregates. "
                "Memory pressure rises under concurrent long-context requests with distinct prefixes. "
                "When cache capacity is constrained, old prefix blocks should be evicted. "
            ) * 18
            unique_tail = (
                f"Task: For request {i}, produce 10 concise bullets that diagnose cache pressure and mitigation tradeoffs."
            )
            prompts.append(shared_prefix + unique_prefix + unique_body + unique_tail)
        return prompts

    return [
        f"{shared_prefix}{long_context}Task: {tasks[i % len(tasks)]}"
        for i in range(num_prompts)
    ]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run vLLM generation and visualize KV cache fill/evict events."
    )
    parser.add_argument(
        "--model",
        type=str,
        default="mistralai/Mistral-7B-v0.1",
        help="Model name/path for vLLM and tokenizer.",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=64,
        help="Max new tokens per prompt.",
    )
    parser.add_argument(
        "--events-out",
        type=Path,
        default=Path("artifacts/kv_events.jsonl"),
        help="Path to write raw KV events as JSONL.",
    )
    parser.add_argument(
        "--html-out",
        type=Path,
        default=Path("artifacts/kv_cache_report.html"),
        help="Path to write the interactive HTML report.",
    )
    parser.add_argument(
        "--topic",
        type=str,
        default="kv-events",
        help="ZMQ topic for KV events.",
    )
    parser.add_argument(
        "--endpoint-bind",
        type=str,
        default="tcp://*:5557",
        help="Publisher endpoint used by vLLM.",
    )
    parser.add_argument(
        "--endpoint-connect",
        type=str,
        default="tcp://127.0.0.1:5557",
        help="Subscriber endpoint used by the collector.",
    )
    parser.add_argument(
        "--no-token-decode",
        action="store_true",
        help="Disable token-id to token-text decoding for hover labels.",
    )
    parser.add_argument(
        "--gpu-memory-utilization",
        type=float,
        default=0.9,
        help=(
            "Fraction of total GPU memory vLLM should reserve. "
            "Lower this on busy/shared GPUs if startup fails (must be in (0, 1])."
        ),
    )
    parser.add_argument(
        "--prompt-profile",
        type=str,
        default="default",
        choices=["default", "cache-stress", "eviction-stress"],
        help="Prompt set to run. Use cache-stress to generate richer cache activity.",
    )
    parser.add_argument(
        "--num-prompts",
        type=int,
        default=8,
        help="Number of prompts to generate for cache-stress profile.",
    )
    parser.add_argument(
        "--num-gpu-blocks-override",
        type=int,
        default=None,
        help="Optional vLLM KV block capacity override for eviction experiments.",
    )
    parser.add_argument(
        "--max-model-len",
        type=int,
        default=None,
        help="Optional max model sequence length to fit in available KV cache.",
    )
    args = parser.parse_args()

    if not (0.0 < args.gpu_memory_utilization <= 1.0):
        raise ValueError("--gpu-memory-utilization must be in (0, 1].")

    if args.num_prompts < 1:
        raise ValueError("--num-prompts must be >= 1.")
    if args.num_gpu_blocks_override is not None and args.num_gpu_blocks_override < 1:
        raise ValueError("--num-gpu-blocks-override must be >= 1.")
    if args.max_model_len is not None and args.max_model_len < 1:
        raise ValueError("--max-model-len must be >= 1.")

    prompts = get_prompts(args.prompt_profile, args.num_prompts)

    collector = KVEventCollector(endpoint=args.endpoint_connect, topic=args.topic)
    collector.start()

    engine = VLLMEngine(
        model_name=args.model,
        kv_events_endpoint=args.endpoint_bind,
        kv_events_topic=args.topic,
        enable_prefix_caching=True,
        gpu_memory_utilization=args.gpu_memory_utilization,
        num_gpu_blocks_override=args.num_gpu_blocks_override,
        max_model_len=args.max_model_len,
    )

    outputs = engine.generate(prompts, max_tokens=args.max_tokens)

    for i, out in enumerate(outputs):
        print(f"\nPrompt {i}: {prompts[i]}")
        print(f"Output: {out}")

    collector.wait_until_events(min_events=1, timeout_s=5.0)
    collector.stop()
    collector.write_jsonl(args.events_out)

    lifecycle, summary = build_token_lifecycle(collector.events)

    if args.num_gpu_blocks_override is not None:
        summary["kv_capacity_blocks"] = args.num_gpu_blocks_override

    token_decode_lookup = None
    if not args.no_token_decode:
        tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
        all_token_ids = {
            token_id
            for event in lifecycle
            for token_id in event.get("token_ids", [])
        }
        token_decode_lookup = {
            token_id: tokenizer.decode([token_id], skip_special_tokens=False)
            for token_id in all_token_ids
        }

    render_html_report(
        lifecycle=lifecycle,
        summary=summary,
        output_html=args.html_out,
        token_decode_lookup=token_decode_lookup,
    )

    print(f"\nWrote raw events to: {args.events_out}")
    print(f"Wrote HTML report to: {args.html_out}")
    print(f"Summary: {summary}")


if __name__ == "__main__":
    main()
