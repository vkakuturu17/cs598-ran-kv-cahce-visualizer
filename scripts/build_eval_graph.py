import argparse
import json
import re
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def extract_kv_data(html_path: Path) -> dict:
    html = html_path.read_text(encoding="utf-8")

    match = re.search(
        r'<script\s+id="kv-data"\s+type="application/json">(.*?)</script>',
        html,
        flags=re.DOTALL,
    )

    if not match:
        raise ValueError(f"Could not find kv-data JSON in {html_path}")

    return json.loads(match.group(1))


def build_event_series(label: str, html_path: Path) -> tuple[pd.DataFrame, dict]:
    data = extract_kv_data(html_path)

    lifecycle = data.get("lifecycle", [])
    summary = data.get("summary", {})

    rows = []
    fills = 0
    evicts = 0
    live_blocks = 0
    live_set = set()

    capacity = summary.get("kv_capacity_blocks")

    for i, event in enumerate(lifecycle, start=1):
        action = event.get("action")
        block_hash = event.get("block_hash")

        if action == "fill":
            fills += 1
            if block_hash is not None:
                live_set.add(block_hash)
        elif action == "evict":
            evicts += 1
            if block_hash is not None:
                live_set.discard(block_hash)

        live_blocks = len(live_set)

        fill_event_pct = fills / i * 100
        live_block_pct = (live_blocks / capacity * 100) if capacity else None

        rows.append(
            {
                "scenario": label,
                "event_step": i,
                "action": action,
                "cumulative_fills": fills,
                "cumulative_evicts": evicts,
                "live_blocks": live_blocks,
                "fill_event_pct": fill_event_pct,
                "live_block_pct": live_block_pct,
            }
        )

    df = pd.DataFrame(rows)

    metrics = {
        "Scenario": label,
        "Total events": len(lifecycle),
        "Total fills": fills,
        "Total evicts": evicts,
        "Ending fill %": round((fills / len(lifecycle) * 100), 1) if lifecycle else 0,
    }

    return df, metrics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--same", required=True, help="Path to same_prompt HTML file")
    parser.add_argument("--different", required=True, help="Path to different_prompts HTML file")
    parser.add_argument("--outdir", default="output/evaluation", help="Output folder")
    args = parser.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    same_df, same_metrics = build_event_series("same_prompt", Path(args.same))
    diff_df, diff_metrics = build_event_series("different_prompts", Path(args.different))

    all_df = pd.concat([same_df, diff_df], ignore_index=True)
    metrics_df = pd.DataFrame([same_metrics, diff_metrics])

    all_df.to_csv(outdir / "fill_percent_over_time.csv", index=False)
    metrics_df.to_csv(outdir / "aggregate_metrics.csv", index=False)

    plt.figure(figsize=(9, 5))

    for scenario, group in all_df.groupby("scenario"):
        plt.plot(
            group["event_step"],
            group["fill_event_pct"],
            label=scenario,
            linewidth=2,
        )

    plt.xlabel("Event Step")
    plt.ylabel("Fill Event %")
    plt.title("Fill Event % Over Time: same_prompt vs different_prompts")
    plt.ylim(50, 100)
    plt.yticks(range(50, 101, 5))

    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()

    plt.savefig(outdir / "fill_percent_overlay.png", dpi=300)
    plt.close()

    with open(outdir / "aggregate_metrics.md", "w", encoding="utf-8") as f:
        f.write(metrics_df.to_markdown(index=False))

    print("\nAggregate metrics:")
    print(metrics_df.to_string(index=False))

    print(f"\nSaved outputs to: {outdir}")
    print(f"- {outdir / 'fill_percent_overlay.png'}")
    print(f"- {outdir / 'aggregate_metrics.csv'}")
    print(f"- {outdir / 'aggregate_metrics.md'}")
    print(f"- {outdir / 'fill_percent_over_time.csv'}")


if __name__ == "__main__":
    main()