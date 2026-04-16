import argparse
from pathlib import Path

from transformers import AutoTokenizer

from src.inference.engine import VLLMEngine
from src.metrics.collector import KVEventCollector
from src.metrics.visualizer import build_token_lifecycle, render_html_report


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
    args = parser.parse_args()

    prompts = [
        "Explain vLLM in one sentence.",
        "What is the KV cache and why is it useful?",
        "Give a two-line explanation of cache eviction policies.",
    ]

    collector = KVEventCollector(endpoint=args.endpoint_connect, topic=args.topic)
    collector.start()

    engine = VLLMEngine(
        model_name=args.model,
        kv_events_endpoint=args.endpoint_bind,
        kv_events_topic=args.topic,
        enable_prefix_caching=True,
    )

    outputs = engine.generate(prompts, max_tokens=args.max_tokens)

    for i, out in enumerate(outputs):
        print(f"\nPrompt {i}: {prompts[i]}")
        print(f"Output: {out}")

    collector.wait_until_events(min_events=1, timeout_s=5.0)
    collector.stop()
    collector.write_jsonl(args.events_out)

    lifecycle, summary = build_token_lifecycle(collector.events)

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
