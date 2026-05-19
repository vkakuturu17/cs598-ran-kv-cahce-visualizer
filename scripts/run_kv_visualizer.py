import argparse
import sys
from pathlib import Path
from typing import Any

from transformers import AutoTokenizer
from pypdf import PdfReader

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


_SUPPORTED_INPUT_TYPES = {"P", "U"}


def _parse_inputs(path: Path) -> list[tuple[str, str]]:
    """Return a list of (type, content) pairs from the input file.

    Each non-blank, non-comment line must start with a known type identifier
    followed by a space and the content:
        P Tell me about prefix caching.
    """
    inputs: list[tuple[str, str]] = []
    for lineno, raw in enumerate(path.read_text().splitlines(), 1):
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if len(line) < 3 or line[1] != " ":
            sys.exit(f"{path}:{lineno}: expected '<TYPE> <content>', got: {raw!r}")
        kind, content = line[0].upper(), line[2:].strip()
        if kind not in _SUPPORTED_INPUT_TYPES:
            sys.exit(
                f"{path}:{lineno}: unknown input type {kind!r}. "
                f"Supported: {', '.join(sorted(_SUPPORTED_INPUT_TYPES))}"
            )
        if not content:
            sys.exit(f"{path}:{lineno}: empty content after type identifier")
        inputs.append((kind, content))
    return inputs


def _extract_pdf_text(pdf_path: Path) -> str:
    reader = PdfReader(pdf_path)
    pages = [page.extract_text() or "" for page in reader.pages]
    return "\n\n".join(pages).strip()


def _build_pdf_message(pdf_path: Path) -> str:
    text = _extract_pdf_text(pdf_path)
    return f"[Uploaded document: {pdf_path.name}]\n\n{text}"


def load_inputs(path: Path) -> list[str]:
    if not path.is_file():
        sys.exit(f"Input file not found: {path}")

    turns = _parse_inputs(path)
    if not turns:
        sys.exit(f"No inputs found in {path}")

    prompts: list[str] = []
    for kind, content in turns:
        if kind == "U":
            pdf_path = Path(content)
            if not pdf_path.is_file():
                sys.exit(f"PDF not found: {content}")
            prompts.append(_build_pdf_message(pdf_path))
        else:
            prompts.append(content)
    return prompts


def _build_preview(text: str, limit: int = 140) -> str:
    collapsed = " ".join(text.split())
    if len(collapsed) <= limit:
        return collapsed
    return collapsed[: limit - 3] + "..."


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
        "--inputs",
        type=Path,
        default=None,
        help=(
            "Optional input file. Each non-blank line: '<TYPE> <content>'. "
            "Supported types: P (paragraph text), U (PDF file path). "
            "Lines starting with # are comments."
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

    if args.inputs is not None:
        prompts = load_inputs(args.inputs)
    else:
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

    tokenizer = None
    if not args.no_token_decode:
        tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)

    outputs: list[str] = []
    turn_windows: list[tuple[int, int]] = []
    turns: list[dict[str, Any]] = []

    for i, prompt in enumerate(prompts):
        start_idx = len(collector.events)
        out = engine.generate([prompt], max_tokens=args.max_tokens)[0]
        outputs.append(out)

        collector.wait_until_events(min_events=start_idx + 1, timeout_s=2.0)
        end_idx = len(collector.events)
        turn_windows.append((start_idx, end_idx))

        prompt_tokens = 0
        generated_tokens = 0
        if tokenizer is not None:
            prompt_tokens = len(tokenizer.encode(prompt, add_special_tokens=False))
            generated_tokens = len(tokenizer.encode(out, add_special_tokens=False))

        turns.append(
            {
                "turn": i + 1,
                "input_preview": _build_preview(prompt),
                "prompt_tokens": prompt_tokens,
                "cached_tokens": 0,
                "computed_tokens": 0,
                "generated_tokens": generated_tokens,
                "cache_hit_rate": 0.0,
                "kv_cache_utilization": 0.0,
                "ttft_s": 0.0,
                "e2e_latency_s": 0.0,
                "event_index_start": start_idx,
                "event_index_end": end_idx,
            }
        )

    for i, out in enumerate(outputs):
        print(f"\nPrompt {i}: {prompts[i]}")
        print(f"Output: {out}")

    collector.wait_until_events(min_events=1, timeout_s=5.0)
    collector.stop()
    collector.write_jsonl(args.events_out)

    lifecycle, summary = build_token_lifecycle(
        collector.events,
        turn_windows=turn_windows,
    )

    if args.num_gpu_blocks_override is not None:
        summary["kv_capacity_blocks"] = args.num_gpu_blocks_override

    token_decode_lookup = None
    if tokenizer is not None:
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
        turns=turns,
        model_name=args.model,
    )

    print(f"\nWrote raw events to: {args.events_out}")
    print(f"Wrote HTML report to: {args.html_out}")
    print(f"Summary: {summary}")


if __name__ == "__main__":
    main()
