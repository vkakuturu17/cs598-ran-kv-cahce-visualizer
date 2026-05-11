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
                grid-template-columns: repeat(3, 1fr);
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
            .block-grid {{
                display: grid;
                grid-template-columns: repeat(var(--cols, 12), 1fr);
                gap: 6px;
            }}
            .block-cell {{
                aspect-ratio: 1 / 1;
                border-radius: 8px;
                border: 1px solid var(--border);
                background: #0f1823;
                transition: transform 0.2s ease, background 0.2s ease;
            }}
            .block-cell.filled {{
                background: var(--accent);
                border-color: rgba(47, 184, 172, 0.7);
                box-shadow: 0 0 12px rgba(47, 184, 172, 0.35);
                transform: translateY(-2px);
            }}
            .block-cell.selected {{
                outline: 2px solid var(--accent-warm);
                outline-offset: 2px;
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
            .token-compare {{
                display: grid;
                grid-template-columns: repeat(2, minmax(0, 1fr));
                gap: 12px;
            }}
            .token-heading {{
                display: flex;
                justify-content: space-between;
                align-items: baseline;
                color: var(--muted);
                font-size: 12px;
                margin-bottom: 6px;
            }}
            .token-heading strong {{
                color: var(--text);
                font-size: 13px;
            }}
            .token-hint {{
                color: var(--muted);
                font-size: 12px;
                margin-top: 8px;
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
                    <h3>KV Blocks <span style=\"color: var(--muted);\" id=\"block-capacity\"></span></h3>
                    <div id=\"block-grid\" class=\"block-grid\"></div>
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
                    <h3>Block Tokens</h3>
                    <div class=\"token-compare\">
                        <div>
                            <div class=\"token-heading\">
                                <strong>Current event</strong>
                                <span id=\"token-source-current\">--</span>
                            </div>
                            <div class=\"token-list\" id=\"token-list-current\"></div>
                        </div>
                        <div>
                            <div class=\"token-heading\">
                                <strong>Selected block</strong>
                                <span id=\"token-source-selected\">None</span>
                            </div>
                            <div class=\"token-list\" id=\"token-list-selected\"></div>
                        </div>
                    </div>
                    <div class=\"token-hint\">Format: token_id: decoded_token</div>
                </div>
                <div class=\"panel\">
                    <h3>All-time Stats</h3>
                    <div class=\"kv-list\" style=\"margin-bottom: 12px;\">
                        <div>Fills</div><span id=\"stat-fills\">0</span>
                        <div>Evictions</div><span id=\"stat-evicts\">0</span>
                        <div>Lifecycle Events</div><span id=\"stat-events\">0</span>
                    </div>
                    <div class=\"top-tokens\">
                        <div>
                            <strong style=\"color: var(--accent);\">Top Filled</strong>
                            <ul id=\"top-filled\"></ul>
                        </div>
                        <div>
                            <strong style=\"color: var(--accent-warm);\">Top Evicted</strong>
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
            document.getElementById("stat-fills").textContent = formatValue(summary.num_fills);
            document.getElementById("stat-evicts").textContent = formatValue(summary.num_evictions);
            document.getElementById("stat-events").textContent = String(lifecycle.length);

            const slider = document.getElementById("event-slider");
            const maxIndex = Math.max(lifecycle.length - 1, 0);
            slider.max = String(maxIndex);
            document.getElementById("max-index").textContent = String(maxIndex);

            const cumFills = [];
            const cumEvicts = [];
            const liveBlocks = [];
            const liveSet = new Set();
            const liveHashesByIndex = [];
            const liveSlotsByIndex = [];
            const slotMapByIndex = [];
            const blockOrder = [];
            const blockIndex = new Map();
            const blockTokensByHash = new Map();
            let fills = 0;
            let evicts = 0;
            const capacityBlocks = summary.kv_capacity_blocks || null;
            const slotForBlock = new Map();
            const slotOrder = [];
            const freeSlots = capacityBlocks ? Array.from({ length: capacityBlocks }, (_, i) => i) : [];

            lifecycle.forEach((event) => {
                if (event.action === "fill") {
                    fills += 1;
                    if (!blockIndex.has(event.block_hash)) {
                        blockIndex.set(event.block_hash, blockOrder.length);
                        blockOrder.push(event.block_hash);
                    }
                    if (!blockTokensByHash.has(event.block_hash)) {
                        blockTokensByHash.set(event.block_hash, event.token_ids || []);
                    }
                    liveSet.add(event.block_hash);

                    if (capacityBlocks) {
                        if (!slotForBlock.has(event.block_hash)) {
                            let slot = freeSlots.shift();
                            if (slot === undefined) {
                                const evictedHash = slotOrder.shift();
                                if (evictedHash !== undefined) {
                                    slot = slotForBlock.get(evictedHash);
                                    slotForBlock.delete(evictedHash);
                                }
                            }
                            if (slot !== undefined) {
                                slotForBlock.set(event.block_hash, slot);
                                slotOrder.push(event.block_hash);
                            }
                        }
                    }
                } else {
                    evicts += 1;
                    liveSet.delete(event.block_hash);

                    if (capacityBlocks) {
                        const slot = slotForBlock.get(event.block_hash);
                        if (slot !== undefined) {
                            slotForBlock.delete(event.block_hash);
                            freeSlots.push(slot);
                            const orderIndex = slotOrder.indexOf(event.block_hash);
                            if (orderIndex >= 0) {
                                slotOrder.splice(orderIndex, 1);
                            }
                        }
                    }
                }
                cumFills.push(fills);
                cumEvicts.push(evicts);
                liveBlocks.push(liveSet.size);
                liveHashesByIndex.push(new Set(liveSet));
                if (capacityBlocks) {
                    liveSlotsByIndex.push(new Set(slotForBlock.values()));
                    const slotMap = new Map();
                    slotForBlock.forEach((slot, blockHash) => {
                        slotMap.set(slot, blockHash);
                    });
                    slotMapByIndex.push(slotMap);
                }
            });

            const blockGrid = document.getElementById("block-grid");
            const maxLive = liveBlocks.length ? Math.max(...liveBlocks) : 0;
            const totalBlocks = Math.max(capacityBlocks || blockOrder.length, 1);
            const gridColumns = Math.min(16, Math.max(6, Math.ceil(Math.sqrt(totalBlocks))));
            blockGrid.style.setProperty("--cols", String(gridColumns));
            blockGrid.innerHTML = "";
            const gridCells = [];
            let selectedBlockIndex = null;
            const totalCells = totalBlocks;
            const capacityLabel = document.getElementById("block-capacity");
            capacityLabel.textContent = capacityBlocks ? "(capacity: " + String(capacityBlocks) + ")" : "";
            for (let i = 0; i < totalCells; i += 1) {
                const cell = document.createElement("div");
                cell.className = "block-cell";
                cell.title = "Block " + String(i + 1);
                cell.addEventListener("click", () => {
                    if (selectedBlockIndex === i) {
                        selectedBlockIndex = null;
                    } else {
                        selectedBlockIndex = i;
                    }
                    updateEvent(Number(slider.value));
                });
                blockGrid.appendChild(cell);
                gridCells.push(cell);
            }

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

            const tokenListCurrent = document.getElementById("token-list-current");
            const tokenListSelected = document.getElementById("token-list-selected");
            const tokenSourceCurrent = document.getElementById("token-source-current");
            const tokenSourceSelected = document.getElementById("token-source-selected");
            const renderTokenList = (target, tokenIds) => {
                target.innerHTML = "";
                (tokenIds || []).slice(0, 120).forEach((tokenId) => {
                    const chip = document.createElement("span");
                    chip.className = "token-chip";
                    chip.textContent = tokenLabel(tokenId);
                    target.appendChild(chip);
                });
            };

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

                tokenSourceCurrent.textContent = "Event " + String(index);
                renderTokenList(tokenListCurrent, event.token_ids || []);

                if (selectedBlockIndex !== null) {
                    let blockHash = null;
                    if (capacityBlocks) {
                        const slotMap = slotMapByIndex[index];
                        blockHash = slotMap ? slotMap.get(selectedBlockIndex) : null;
                    } else {
                        blockHash = blockOrder[selectedBlockIndex] || null;
                    }

                    if (blockHash) {
                        const tokenIds = blockTokensByHash.get(blockHash) || [];
                        tokenSourceSelected.textContent = "Block " + String(selectedBlockIndex + 1);
                        renderTokenList(tokenListSelected, tokenIds);
                    } else {
                        tokenSourceSelected.textContent = "Empty slot";
                        renderTokenList(tokenListSelected, []);
                    }
                } else {
                    tokenSourceSelected.textContent = "None";
                    renderTokenList(tokenListSelected, []);
                }

                const liveAtIndex = capacityBlocks
                    ? (liveSlotsByIndex[index] || new Set())
                    : (liveHashesByIndex[index] || new Set());
                for (let i = 0; i < gridCells.length; i += 1) {
                    const filled = capacityBlocks
                        ? liveAtIndex.has(i)
                        : (blockOrder[i] && liveAtIndex.has(blockOrder[i]));
                    gridCells[i].classList.toggle("filled", Boolean(filled));
                    gridCells[i].classList.toggle("selected", i === selectedBlockIndex);
                }
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
