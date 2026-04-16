import json
from collections import Counter
from pathlib import Path
from typing import Any

import plotly.graph_objects as go


def _safe_token_label(token_id: int, decoded_lookup: dict[int, str] | None) -> str:
    if decoded_lookup and token_id in decoded_lookup:
        token_text = decoded_lookup[token_id].replace("\n", "\\n")
        return f"{token_id}: {token_text}"
    return str(token_id)


def build_token_lifecycle(
    events: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    block_to_tokens: dict[str, list[int]] = {}
    lifecycle: list[dict[str, Any]] = []

    fills = 0
    evictions = 0

    for item in events:
        event = item["event"]
        event_type = event.get("type")

        if event_type == "AllBlocksCleared":
            block_to_tokens.clear()
            continue

        if event_type == "BlockStored":
            block_hashes = event["block_hashes"]
            token_ids = event["token_ids"]
            block_size = event["block_size"]

            for i, block_hash in enumerate(block_hashes):
                start = i * block_size
                end = start + block_size
                block_tokens = token_ids[start:end]
                block_to_tokens[block_hash] = block_tokens
                fills += 1

                lifecycle.append(
                    {
                        "event_index": item["event_index"],
                        "timestamp": item["timestamp"],
                        "action": "fill",
                        "block_hash": block_hash,
                        "token_ids": block_tokens,
                        "group_idx": event.get("group_idx"),
                        "medium": event.get("medium"),
                    }
                )

        if event_type == "BlockRemoved":
            for block_hash in event["block_hashes"]:
                evictions += 1
                evicted_tokens = block_to_tokens.pop(block_hash, [])
                lifecycle.append(
                    {
                        "event_index": item["event_index"],
                        "timestamp": item["timestamp"],
                        "action": "evict",
                        "block_hash": block_hash,
                        "token_ids": evicted_tokens,
                        "group_idx": event.get("group_idx"),
                        "medium": event.get("medium"),
                    }
                )

    return lifecycle, {
        "num_batches": len(events),
        "num_lifecycle_events": len(lifecycle),
        "num_fills": fills,
        "num_evictions": evictions,
        "num_live_blocks": len(block_to_tokens),
    }


def render_html_report(
    lifecycle: list[dict[str, Any]],
    summary: dict[str, int],
    output_html: str | Path,
    token_decode_lookup: dict[int, str] | None = None,
) -> None:
    output_html = Path(output_html)
    output_html.parent.mkdir(parents=True, exist_ok=True)

    x_fill: list[int] = []
    y_fill: list[str] = []
    hover_fill: list[str] = []

    x_evict: list[int] = []
    y_evict: list[str] = []
    hover_evict: list[str] = []

    token_fill_count: Counter[int] = Counter()
    token_evict_count: Counter[int] = Counter()

    for event in lifecycle:
        token_ids = event["token_ids"]
        label = ", ".join(_safe_token_label(t, token_decode_lookup) for t in token_ids)

        if event["action"] == "fill":
            x_fill.append(event["event_index"])
            y_fill.append(event["block_hash"])
            hover_fill.append(
                f"tokens=[{label}]<br>group={event['group_idx']}<br>medium={event['medium']}"
            )
            token_fill_count.update(token_ids)
        else:
            x_evict.append(event["event_index"])
            y_evict.append(event["block_hash"])
            hover_evict.append(
                f"tokens=[{label}]<br>group={event['group_idx']}<br>medium={event['medium']}"
            )
            token_evict_count.update(token_ids)

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=x_fill,
            y=y_fill,
            mode="markers",
            marker={"color": "#1f77b4", "size": 9, "symbol": "circle"},
            name="Block fill",
            hovertext=hover_fill,
            hoverinfo="text+x+y",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=x_evict,
            y=y_evict,
            mode="markers",
            marker={"color": "#d62728", "size": 10, "symbol": "x"},
            name="Block eviction",
            hovertext=hover_evict,
            hoverinfo="text+x+y",
        )
    )

    fig.update_layout(
        title="vLLM KV Cache Lifecycle",
        xaxis_title="Event index",
        yaxis_title="Block hash",
        template="plotly_white",
        height=640,
        legend={"orientation": "h", "y": 1.05, "x": 0.0},
    )

    top_filled = token_fill_count.most_common(10)
    top_evicted = token_evict_count.most_common(10)

    html_summary = {
        "summary": summary,
        "top_filled_tokens": top_filled,
        "top_evicted_tokens": top_evicted,
    }

    # Embed summary payload in HTML for quick inspection.
    fig_html = fig.to_html(full_html=True, include_plotlyjs="cdn")
    payload_tag = "<script id='kv-summary-json' type='application/json'>"
    summary_blob = json.dumps(html_summary, indent=2)

    final_html = fig_html.replace(
        "</body>",
        f"{payload_tag}{summary_blob}</script></body>",
    )

    output_html.write_text(final_html, encoding="utf-8")
