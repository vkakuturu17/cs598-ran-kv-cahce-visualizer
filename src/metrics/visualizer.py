import json
from collections import Counter
from pathlib import Path
from typing import Any


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
                        "sequence": item.get("sequence"),
                        "data_parallel_rank": item.get("data_parallel_rank"),
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
                        "sequence": item.get("sequence"),
                        "data_parallel_rank": item.get("data_parallel_rank"),
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

    token_fill_count: Counter[int] = Counter()
    token_evict_count: Counter[int] = Counter()

    for event in lifecycle:
        token_ids = event["token_ids"]
        if event["action"] == "fill":
            token_fill_count.update(token_ids)
        else:
            token_evict_count.update(token_ids)

    top_filled = token_fill_count.most_common(12)
    top_evicted = token_evict_count.most_common(12)

    payload = {
        "summary": summary,
        "top_filled_tokens": top_filled,
        "top_evicted_tokens": top_evicted,
        "lifecycle": lifecycle,
        "token_decode_lookup": token_decode_lookup or {},
    }

    payload_blob = json.dumps(payload, ensure_ascii=True).replace("</", "<\\/")

    html_template = """<!doctype html>
<html lang=\"en\">
    <head>
        <meta charset=\"utf-8\" />
        <meta name=\"viewport\" content=\"width=device-width, initial-scale=1\" />
        <title>KV Cache Report</title>
        <link rel=\"preconnect\" href=\"https://fonts.googleapis.com\" />
        <link rel=\"preconnect\" href=\"https://fonts.gstatic.com\" crossorigin />
        <link
            href=\"https://fonts.googleapis.com/css2?family=Barlow:wght@300;400;600;700&family=JetBrains+Mono:wght@400;600&display=swap\"
            rel=\"stylesheet\"
        />
        <script src=\"https://cdn.plot.ly/plotly-2.30.0.min.js\"></script>
        <style>
            :root {{
                color-scheme: dark;
                --bg: #0b1117;
                --bg-alt: #0f1a24;
                --card: #141e2a;
                --border: #223243;
                --text: #e7edf3;
                --muted: #9fb0c0;
                --accent: #2fb8ac;
                --accent-warm: #f59e0b;
            }}
            * {{ box-sizing: border-box; }}
            body {{
                margin: 0;
                font-family: "Barlow", system-ui, sans-serif;
                background: radial-gradient(circle at top left, #15202c, var(--bg));
                color: var(--text);
            }}
            .page {{
                max-width: 1280px;
                margin: 0 auto;
                padding: 32px 24px 56px;
            }}
            .header {{
                display: grid;
                grid-template-columns: 1.3fr 1fr;
                gap: 24px;
                align-items: center;
                margin-bottom: 28px;
            }}
            .eyebrow {{
                color: var(--accent);
                letter-spacing: 0.16em;
                text-transform: uppercase;
                font-size: 12px;
                font-weight: 600;
            }}
            h1 {{
                margin: 8px 0 6px;
                font-size: 34px;
                font-weight: 700;
            }}
            .sub {{
                color: var(--muted);
                max-width: 520px;
            }}
            .meta {{
                display: grid;
                grid-template-columns: repeat(3, 1fr);
                gap: 12px;
            }}
            .meta-item {{
                background: var(--bg-alt);
                border: 1px solid var(--border);
                border-radius: 12px;
                padding: 14px 16px;
                text-align: right;
                animation: fadeIn 0.6s ease;
            }}
            .meta-item span {{
                display: block;
                color: var(--muted);
                font-size: 12px;
                text-transform: uppercase;
                letter-spacing: 0.14em;
            }}
            .meta-item strong {{
                font-family: "JetBrains Mono", monospace;
                font-size: 20px;
            }}
            .cards {{
                display: grid;
                grid-template-columns: repeat(5, 1fr);
                gap: 12px;
                margin-bottom: 24px;
            }}
            .card {{
                background: var(--card);
                border: 1px solid var(--border);
                border-radius: 14px;
                padding: 16px;
                animation: fadeIn 0.7s ease;
            }}
            .card h3 {{
                margin: 0 0 8px;
                font-size: 13px;
                font-weight: 600;
                letter-spacing: 0.08em;
                text-transform: uppercase;
                color: var(--muted);
            }}
            .card strong {{
                font-family: "JetBrains Mono", monospace;
                font-size: 22px;
                color: var(--accent);
            }}
            .controls {{
                display: grid;
                grid-template-columns: auto 1fr auto;
                gap: 16px;
                align-items: center;
                background: var(--bg-alt);
                border: 1px solid var(--border);
                border-radius: 14px;
                padding: 14px 16px;
                margin-bottom: 24px;
            }}
            .controls button {{
                background: transparent;
                border: 1px solid var(--border);
                color: var(--text);
                border-radius: 10px;
                padding: 8px 12px;
                font-weight: 600;
                cursor: pointer;
            }}
            .controls button.active {{
                border-color: var(--accent);
                color: var(--accent);
            }}
            .slider-wrap {{
                display: grid;
                gap: 8px;
            }}
            input[type=range] {{
                width: 100%;
            }}
            .control-readout {{
                color: var(--muted);
                font-size: 13px;
            }}
            .grid {{
                display: grid;
                grid-template-columns: 1.6fr 1fr;
                gap: 16px;
                margin-bottom: 20px;
            }}
            .panel {{
                background: var(--card);
                border: 1px solid var(--border);
                border-radius: 16px;
                padding: 16px;
            }}
            .panel h3 {{
                margin: 0 0 12px;
                font-size: 15px;
            }}
            .kv-list {{
                display: grid;
                grid-template-columns: 1fr 1.4fr;
                gap: 8px 12px;
                font-size: 13px;
                color: var(--muted);
            }}
            .kv-list span {{
                color: var(--text);
                font-family: "JetBrains Mono", monospace;
            }}
            .token-list {{
                display: flex;
                flex-wrap: wrap;
                gap: 6px;
                font-size: 12px;
            }}
            .token-chip {{
                border: 1px solid var(--border);
                border-radius: 999px;
                padding: 4px 8px;
                background: #101926;
                font-family: "JetBrains Mono", monospace;
            }}
            .top-tokens {{
                display: grid;
                grid-template-columns: repeat(2, 1fr);
                gap: 10px;
                font-size: 13px;
            }}
            .top-tokens ul {{
                margin: 0;
                padding-left: 18px;
                color: var(--muted);
            }}
            .top-tokens li span {{
                color: var(--text);
                font-family: "JetBrains Mono", monospace;
            }}
            @keyframes fadeIn {{
                from {{ opacity: 0; transform: translateY(6px); }}
                to {{ opacity: 1; transform: translateY(0); }}
            }}
            @media (max-width: 980px) {{
                .header {{ grid-template-columns: 1fr; }}
                .meta {{ grid-template-columns: repeat(2, 1fr); }}
                .cards {{ grid-template-columns: repeat(2, 1fr); }}
                .grid {{ grid-template-columns: 1fr; }}
                .controls {{ grid-template-columns: 1fr; }}
            }}
        </style>
    </head>
    <body>
        <div class=\"page\">
            <header class=\"header\">
                <div>
                    <div class=\"eyebrow\">KV Cache Visualizer</div>
                    <h1>Post-run Analysis</h1>
                    <p class=\"sub\">Step through KV cache fills and evictions captured from vLLM.</p>
                </div>
                <div class=\"meta\">
                    <div class=\"meta-item\"><span>Batches</span><strong id=\"meta-batches\">0</strong></div>
                    <div class=\"meta-item\"><span>Lifecycle Events</span><strong id=\"meta-events\">0</strong></div>
                    <div class=\"meta-item\"><span>Live Blocks</span><strong id=\"meta-live\">0</strong></div>
                </div>
            </header>

            <section class=\"cards\">
                <div class=\"card\"><h3>Fills</h3><strong id=\"card-fills\">0</strong></div>
                <div class=\"card\"><h3>Evictions</h3><strong id=\"card-evicts\">0</strong></div>
                <div class=\"card\"><h3>Current Live</h3><strong id=\"card-live\">0</strong></div>
                <div class=\"card\"><h3>Current Fill %</h3><strong id=\"card-fill-rate\">0%</strong></div>
                <div class=\"card\"><h3>Event Index</h3><strong id=\"card-index\">0</strong></div>
            </section>

            <section class=\"controls\">
                <div>
                    <button id=\"step-back\">Step Back</button>
                    <button id=\"step-forward\">Step Forward</button>
                    <button id=\"toggle-play\">Play</button>
                </div>
                <div class=\"slider-wrap\">
                    <input id=\"event-slider\" type=\"range\" min=\"0\" max=\"0\" value=\"0\" step=\"1\" />
                    <div class=\"control-readout\">
                        Event <span id=\"current-index\">0</span> / <span id=\"max-index\">0</span> &middot;
                        <span id=\"current-time\">--</span>
                    </div>
                </div>
                <div class=\"control-readout\">
                    Tokens in event: <span id=\"current-token-count\">0</span>
                </div>
            </section>

            <section class=\"grid\">
                <div class=\"panel\">
                    <h3>Lifecycle Timeline</h3>
                    <div id=\"timeline\" style=\"height: 520px;\"></div>
                </div>
                <div class=\"panel\">
                    <h3>Current Event</h3>
                    <div class=\"kv-list\">
                        <div>Action</div><span id=\"detail-action\">--</span>
                        <div>Block Hash</div><span id=\"detail-block\">--</span>
                        <div>Group</div><span id=\"detail-group\">--</span>
                        <div>Medium</div><span id=\"detail-medium\">--</span>
                        <div>Sequence</div><span id=\"detail-sequence\">--</span>
                        <div>Data Parallel</div><span id=\"detail-rank\">--</span>
                        <div>Timestamp</div><span id=\"detail-time\">--</span>
                    </div>
                    <h3 style=\"margin-top: 20px;\">Cumulative Stats</h3>
                    <div class=\"kv-list\">
                        <div>Fills</div><span id=\"detail-fills\">0</span>
                        <div>Evictions</div><span id=\"detail-evicts\">0</span>
                        <div>Live Blocks</div><span id=\"detail-live\">0</span>
                    </div>
                </div>
            </section>

            <section class=\"grid\">
                <div class=\"panel\">
                    <h3>Token Preview</h3>
                    <div class=\"token-list\" id=\"token-list\"></div>
                </div>
                <div class=\"panel\">
                    <h3>Top Tokens</h3>
                    <div class=\"top-tokens\">
                        <div>
                            <strong style=\"color: var(--accent);\">Filled</strong>
                            <ul id=\"top-filled\"></ul>
                        </div>
                        <div>
                            <strong style=\"color: var(--accent-warm);\">Evicted</strong>
                            <ul id=\"top-evicted\"></ul>
                        </div>
                    </div>
                </div>
            </section>
        </div>

        <script id=\"kv-data\" type=\"application/json\">{payload_blob}</script>
        <script>
            const payload = JSON.parse(document.getElementById("kv-data").textContent);
            const lifecycle = payload.lifecycle || [];
            const tokenLookup = payload.token_decode_lookup || {};
            const summary = payload.summary || {};

            const formatValue = (value) => (value === null || value === undefined ? "n/a" : String(value));
            const formatTime = (ts) => (ts ? ts.toFixed(3) + " s" : "--");
            const tokenLabel = (tokenId) => {
                const decoded = tokenLookup[String(tokenId)] ?? tokenLookup[tokenId];
                if (decoded === undefined) return String(tokenId);
                return tokenId + ": " + String(decoded).replace(/\\n/g, "\\n");
            };
            const tokenPreview = (tokenIds, maxTokens) => {
                const slice = tokenIds.slice(0, maxTokens);
                let label = slice.map(tokenLabel).join(", ");
                if (tokenIds.length > maxTokens) {
                    label += ", ... +" + (tokenIds.length - maxTokens);
                }
                return label;
            };

            document.getElementById("meta-batches").textContent = formatValue(summary.num_batches);
            document.getElementById("meta-events").textContent = lifecycle.length;
            document.getElementById("meta-live").textContent = formatValue(summary.num_live_blocks);
            document.getElementById("card-fills").textContent = formatValue(summary.num_fills);
            document.getElementById("card-evicts").textContent = formatValue(summary.num_evictions);

            const slider = document.getElementById("event-slider");
            const maxIndex = Math.max(lifecycle.length - 1, 0);
            slider.max = String(maxIndex);
            document.getElementById("max-index").textContent = String(maxIndex);

            const cumFills = [];
            const cumEvicts = [];
            const liveBlocks = [];
            const liveSet = new Set();
            let fills = 0;
            let evicts = 0;

            lifecycle.forEach((event) => {
                if (event.action === "fill") {
                    fills += 1;
                    liveSet.add(event.block_hash);
                } else {
                    evicts += 1;
                    liveSet.delete(event.block_hash);
                }
                cumFills.push(fills);
                cumEvicts.push(evicts);
                liveBlocks.push(liveSet.size);
            });

            const buildTopTokens = (list, containerId) => {
                const container = document.getElementById(containerId);
                container.innerHTML = "";
                list.forEach(([tokenId, count]) => {
                    const li = document.createElement("li");
                    const span = document.createElement("span");
                    span.textContent = tokenLabel(tokenId);
                    li.appendChild(span);
                    li.appendChild(document.createTextNode(" x" + count));
                    container.appendChild(li);
                });
            };

            buildTopTokens(payload.top_filled_tokens || [], "top-filled");
            buildTopTokens(payload.top_evicted_tokens || [], "top-evicted");

            const fillX = [];
            const fillY = [];
            const fillHover = [];
            const evictX = [];
            const evictY = [];
            const evictHover = [];

            lifecycle.forEach((event) => {
                const hover =
                    "tokens=[" + tokenPreview(event.token_ids || [], 24) + "]<br>" +
                    "group=" + formatValue(event.group_idx) + "<br>" +
                    "medium=" + formatValue(event.medium);
                if (event.action === "fill") {
                    fillX.push(event.event_index);
                    fillY.push(event.block_hash);
                    fillHover.push(hover);
                } else {
                    evictX.push(event.event_index);
                    evictY.push(event.block_hash);
                    evictHover.push(hover);
                }
            });

            const plotData = [
                {
                    x: fillX,
                    y: fillY,
                    mode: "markers",
                    marker: { color: "#2fb8ac", size: 8, symbol: "circle" },
                    name: "Fill",
                    hovertext: fillHover,
                    hoverinfo: "text+x+y",
                },
                {
                    x: evictX,
                    y: evictY,
                    mode: "markers",
                    marker: { color: "#f97316", size: 9, symbol: "x" },
                    name: "Evict",
                    hovertext: evictHover,
                    hoverinfo: "text+x+y",
                },
                {
                    x: [],
                    y: [],
                    mode: "markers",
                    marker: { color: "#f59e0b", size: 14, symbol: "diamond" },
                    name: "Current",
                    hoverinfo: "skip",
                },
            ];

            const plotLayout = {
                paper_bgcolor: "rgba(0,0,0,0)",
                plot_bgcolor: "rgba(0,0,0,0)",
                xaxis: { title: "Event index", gridcolor: "#1f2a36" },
                yaxis: { title: "Block hash", gridcolor: "#1f2a36" },
                height: 500,
                margin: { l: 60, r: 20, t: 20, b: 50 },
                legend: { orientation: "h", y: 1.05, x: 0 },
            };

            Plotly.newPlot("timeline", plotData, plotLayout, { displayModeBar: false, responsive: true });

            const updateEvent = (index) => {
                const event = lifecycle[index];
                if (!event) return;
                slider.value = String(index);
                document.getElementById("current-index").textContent = String(index);
                document.getElementById("card-index").textContent = String(index);
                document.getElementById("current-time").textContent = formatTime(event.timestamp);
                document.getElementById("current-token-count").textContent = String((event.token_ids || []).length);
                document.getElementById("detail-action").textContent = formatValue(event.action);
                document.getElementById("detail-block").textContent = formatValue(event.block_hash);
                document.getElementById("detail-group").textContent = formatValue(event.group_idx);
                document.getElementById("detail-medium").textContent = formatValue(event.medium);
                document.getElementById("detail-sequence").textContent = formatValue(event.sequence);
                document.getElementById("detail-rank").textContent = formatValue(event.data_parallel_rank);
                document.getElementById("detail-time").textContent = formatTime(event.timestamp);
                document.getElementById("detail-fills").textContent = String(cumFills[index] || 0);
                document.getElementById("detail-evicts").textContent = String(cumEvicts[index] || 0);
                document.getElementById("detail-live").textContent = String(liveBlocks[index] || 0);
                document.getElementById("card-live").textContent = String(liveBlocks[index] || 0);
                const fillRate = event.action === "fill" ? ((cumFills[index] / Math.max(1, index + 1)) * 100) : ((cumFills[index] / Math.max(1, index + 1)) * 100);
                document.getElementById("card-fill-rate").textContent = fillRate.toFixed(1) + "%";

                const tokenList = document.getElementById("token-list");
                tokenList.innerHTML = "";
                (event.token_ids || []).slice(0, 120).forEach((tokenId) => {
                    const chip = document.createElement("span");
                    chip.className = "token-chip";
                    chip.textContent = tokenLabel(tokenId);
                    tokenList.appendChild(chip);
                });

                Plotly.restyle("timeline", { x: [[event.event_index]], y: [[event.block_hash]] }, [2]);
                Plotly.relayout("timeline", {
                    shapes: [
                        {
                            type: "line",
                            x0: event.event_index,
                            x1: event.event_index,
                            y0: 0,
                            y1: 1,
                            yref: "paper",
                            line: { color: "#f59e0b", width: 1, dash: "dot" },
                        },
                    ],
                });
            };

            const stepBack = document.getElementById("step-back");
            const stepForward = document.getElementById("step-forward");
            const togglePlay = document.getElementById("toggle-play");
            let playTimer = null;

            stepBack.addEventListener("click", () => {
                const next = Math.max(0, Number(slider.value) - 1);
                updateEvent(next);
            });
            stepForward.addEventListener("click", () => {
                const next = Math.min(maxIndex, Number(slider.value) + 1);
                updateEvent(next);
            });
            slider.addEventListener("input", (event) => {
                updateEvent(Number(event.target.value));
            });
            togglePlay.addEventListener("click", () => {
                if (playTimer) {
                    clearInterval(playTimer);
                    playTimer = null;
                    togglePlay.textContent = "Play";
                    togglePlay.classList.remove("active");
                    return;
                }
                togglePlay.textContent = "Pause";
                togglePlay.classList.add("active");
                playTimer = setInterval(() => {
                    const current = Number(slider.value);
                    const next = current + 1;
                    if (next > maxIndex) {
                        clearInterval(playTimer);
                        playTimer = null;
                        togglePlay.textContent = "Play";
                        togglePlay.classList.remove("active");
                        return;
                    }
                    updateEvent(next);
                }, 350);
            });

            updateEvent(0);
        </script>
    </body>
</html>
"""

    html_template = html_template.replace("{{", "{").replace("}}", "}")
    html = html_template.replace("{payload_blob}", payload_blob)

    output_html.write_text(html, encoding="utf-8")
