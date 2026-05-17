"""
token_mapper.py — Maps vLLM KV cache blocks back to prompt tokens / words.

Given a prompt and the number of cached_tokens reported by vLLM, this module:
  1. Tokenizes the full chat-formatted prompt using Qwen's tokenizer
  2. Splits the token sequence into block_size chunks (matching vLLM's PagedAttention)
  3. Labels each block as HIT / PARTIAL / MISS based on the cached_tokens boundary
  4. Decodes each block back to human-readable text
  5. Plots a token-level heatmap across all requests (one row per request)

Exported API
------------
TokenBlockMapper          — main class
plot_token_heatmap(...)   — standalone plot function used by visualize.py
"""

import textwrap
from pathlib import Path
from typing import List, Dict, Any, Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.ticker import MaxNLocator

# ── Colour palette ─────────────────────────────────────────────────────────
COLORS = {
    "hit":     "#2ecc71",   # green
    "partial": "#f39c12",   # amber
    "miss":    "#e74c3c",   # red
    "unknown": "#bdc3c7",   # grey  (cached_tokens not reported)
}

PHASE_LABEL_COLORS = {
    "cold_misses":   "#e74c3c",
    "prefix_reuse":  "#3498db",
    "exact_repeats": "#2ecc71",
}


# ── Core mapper ────────────────────────────────────────────────────────────

class TokenBlockMapper:
    """
    Tokenises prompts and maps each KV-cache block to a text span.

    Parameters
    ----------
    model_name : str
        HuggingFace model ID — must match what vLLM is serving so the
        tokenizer and chat template are identical.
    block_size : int
        vLLM PagedAttention block size (default 16 tokens).
        Check with:  grep block_size /proc/$(pgrep -f vllm)/cmdline
        or leave at 16 (the vLLM default).
    """

    def __init__(self, model_name: str = "Qwen/Qwen2.5-3B-Instruct", block_size: int = 16):
        self.block_size = block_size
        self.tokenizer = None
        self._load_tokenizer(model_name)

    def _load_tokenizer(self, model_name: str):
        try:
            from transformers import AutoTokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(
                model_name, trust_remote_code=True
            )
            print(f"[token_mapper] Tokenizer loaded: {model_name}")
        except Exception as e:
            print(f"[token_mapper] WARNING: could not load tokenizer ({e}). "
                  "Block text will show token-index ranges instead of words.")

    # ── Public interface ────────────────────────────────────────────────────

    def map_request(
        self,
        system_prompt: str,
        user_prompt: str,
        cached_tokens: int,              # from usage.prompt_tokens_details.cached_tokens
        request_id: str = "",
    ) -> List[Dict[str, Any]]:
        """
        Returns a list of block dicts, one per KV-cache block in the prompt.

        Each dict has:
            index         — block index (0-based)
            token_start   — first token index in this block
            token_end     — last token index (inclusive)
            text          — decoded text of the block (or placeholder)
            status        — "hit" | "partial" | "miss" | "unknown"
            hit_fraction  — fraction of tokens in block that were cached (0–1)
            request_id    — passed through for reference
        """
        token_ids = self._tokenize(system_prompt, user_prompt)
        return self._build_blocks(token_ids, cached_tokens, request_id)

    def render_ascii(self, blocks: List[Dict], request_id: str = "", width: int = 72):
        """Print a colour-coded ASCII visualisation to stdout."""
        RESET  = "\033[0m"
        GREEN  = "\033[42m\033[30m"
        AMBER  = "\033[43m\033[30m"
        RED    = "\033[41m\033[37m"
        GREY   = "\033[100m\033[37m"
        MAP    = {"hit": GREEN, "partial": AMBER, "miss": RED, "unknown": GREY}

        print(f"\n  Request: {request_id}")
        print(f"  {'─'*width}")
        cell_w = max(12, width // max(len(blocks), 1))
        line   = ""
        for blk in blocks:
            label = blk["text"][:cell_w - 2].replace("\n", " ")
            label = f" {label:<{cell_w - 2}} "
            line += MAP.get(blk["status"], GREY) + label + RESET
        # wrap into rows of ~width chars
        cell_px = cell_w
        per_row = max(1, width // cell_px)
        for start in range(0, len(blocks), per_row):
            row_blocks = blocks[start:start + per_row]
            row = ""
            for blk in row_blocks:
                label = blk["text"][:cell_px - 2].replace("\n", " ")
                label = f" {label:<{cell_px - 2}} "
                row += MAP.get(blk["status"], GREY) + label + RESET
            print(f"  {row}")
        print(f"  {'─'*width}")
        hit_pct = 100 * sum(b["hit_fraction"] for b in blocks) / len(blocks)
        print(f"  Blocks: {len(blocks)} | Est. cache hit: {hit_pct:.0f}%\n")

    # ── Internals ───────────────────────────────────────────────────────────

    def _tokenize(self, system_prompt: str, user_prompt: str) -> List[int]:
        """Build the exact token sequence vLLM will process."""
        if self.tokenizer is None:
            return []
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user",   "content": user_prompt},
        ]
        try:
            text = self.tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            return self.tokenizer.encode(text)
        except Exception:
            # Fallback: plain concatenation
            raw = f"{system_prompt}\n{user_prompt}"
            return self.tokenizer.encode(raw)

    def _build_blocks(
        self,
        token_ids: List[int],
        cached_tokens: int,
        request_id: str,
    ) -> List[Dict[str, Any]]:
        blocks = []
        total  = len(token_ids)

        # If tokenizer unavailable, produce placeholder blocks from cached_tokens count
        if not token_ids:
            n_blocks = max(1, (cached_tokens or 0) + self.block_size) // self.block_size
            for i in range(n_blocks):
                t_start = i * self.block_size
                t_end   = t_start + self.block_size - 1
                status, hit_frac = self._status(t_start, t_end, cached_tokens)
                blocks.append({
                    "index": i, "token_start": t_start, "token_end": t_end,
                    "text": f"tokens {t_start}–{t_end}",
                    "status": status, "hit_fraction": hit_frac,
                    "request_id": request_id,
                })
            return blocks

        for i in range(0, total, self.block_size):
            chunk = token_ids[i: i + self.block_size]
            t_start = i
            t_end   = i + len(chunk) - 1
            status, hit_frac = self._status(t_start, t_end, cached_tokens)

            try:
                text = self.tokenizer.decode(chunk, skip_special_tokens=True).strip()
            except Exception:
                text = f"tokens {t_start}–{t_end}"

            blocks.append({
                "index":       i // self.block_size,
                "token_start": t_start,
                "token_end":   t_end,
                "text":        text or f"[block {i // self.block_size}]",
                "status":      status,
                "hit_fraction": hit_frac,
                "request_id":  request_id,
            })
        return blocks

    def _status(self, t_start: int, t_end: int, cached_tokens: int):
        """
        cached_tokens is a count: tokens with index < cached_tokens are cached.
        Returns (status_str, hit_fraction).
        """
        if cached_tokens < 0:
            return "unknown", 0.0
        block_len = t_end - t_start + 1
        cached_in_block = max(0, min(cached_tokens, t_end + 1) - t_start)
        hit_frac = cached_in_block / block_len

        if cached_tokens == 0:
            return "miss", 0.0
        elif t_end < cached_tokens:           # entire block cached
            return "hit", 1.0
        elif t_start >= cached_tokens:        # entire block uncached
            return "miss", 0.0
        else:                                  # block straddles boundary
            return "partial", hit_frac


# ── Standalone plot ────────────────────────────────────────────────────────

def plot_token_heatmap(
    all_blocks: List[List[Dict]],
    request_ids: List[str],
    out_path: Path,
    block_size: int = 16,
):
    """
    Render a token-level KV-cache hit/miss heatmap.

    Layout: one row per request, one cell per KV-cache block.
    Cell colour: green=hit, amber=partial, red=miss, grey=unknown.
    Cell text: first ~14 chars of the decoded block content.

    Parameters
    ----------
    all_blocks   : list of block-list (one per request, from map_request)
    request_ids  : matching request ID strings
    out_path     : Path to save the PNG
    block_size   : used for x-axis label
    """
    n_req    = len(all_blocks)
    max_blks = max((len(b) for b in all_blocks), default=1)

    cell_w   = 1.6          # inches per block column
    cell_h   = 0.55         # inches per request row
    fig_w    = min(28, max(10, max_blks * cell_w + 3))
    fig_h    = max(5,  n_req * cell_h + 2.5)

    fig, ax  = plt.subplots(figsize=(fig_w, fig_h))
    fig.suptitle(
        f"Token-Level KV-Cache Block Map  (block_size = {block_size} tokens)",
        fontsize=13, fontweight="bold", y=0.98,
    )

    TEXT_MAX = 13          # chars shown inside each cell

    for row_i, (blocks, req_id) in enumerate(zip(all_blocks, request_ids)):
        y_center = row_i

        # Phase label colour for row label
        phase = req_id.rsplit("_", 1)[0] if "_" in req_id else req_id
        lbl_color = PHASE_LABEL_COLORS.get(phase, "#555555")

        for blk in blocks:
            col_i  = blk["index"]
            color  = COLORS.get(blk["status"], COLORS["unknown"])
            rect   = mpatches.FancyBboxPatch(
                (col_i + 0.05, y_center - 0.38),
                0.90, 0.76,
                boxstyle="round,pad=0.02",
                linewidth=0.4,
                edgecolor="#555555",
                facecolor=color,
                alpha=0.85,
            )
            ax.add_patch(rect)

            # Block text (truncated + wrapped to 2 lines)
            label = blk["text"][:TEXT_MAX].replace("\n", " ").strip()
            if len(blk["text"]) > TEXT_MAX:
                label += "…"
            ax.text(
                col_i + 0.5, y_center,
                label,
                ha="center", va="center",
                fontsize=5.5, color="#1a1a1a",
                clip_on=True,
            )

            # Token-index annotation at top of cell
            ax.text(
                col_i + 0.5, y_center + 0.30,
                f"{blk['token_start']}",
                ha="center", va="center",
                fontsize=4, color="#333333", style="italic",
            )

        # Row label (request id, coloured by phase)
        ax.text(
            -0.3, y_center,
            req_id,
            ha="right", va="center",
            fontsize=7, color=lbl_color, fontweight="bold",
        )

        # Hit-rate badge at end of row
        hit_pct = 100 * sum(b["hit_fraction"] for b in blocks) / max(len(blocks), 1)
        ax.text(
            len(blocks) + 0.15, y_center,
            f"{hit_pct:.0f}% hit",
            ha="left", va="center",
            fontsize=7, color="#2c3e50",
        )

    # ── Axes cosmetics ──────────────────────────────────────────────────────
    ax.set_xlim(-2.5, max_blks + 1.5)
    ax.set_ylim(-0.8, n_req - 0.2)
    ax.set_xlabel(f"KV-Cache Block Index  (each block = {block_size} tokens)", fontsize=9)
    ax.set_yticks([])
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax.invert_yaxis()      # first request at top
    ax.grid(axis="x", linestyle="--", alpha=0.2)
    ax.spines[["top", "right", "left"]].set_visible(False)

    # ── Legend ──────────────────────────────────────────────────────────────
    legend_patches = [
        mpatches.Patch(facecolor=COLORS["hit"],     label="Cache HIT  (all tokens cached)"),
        mpatches.Patch(facecolor=COLORS["partial"],  label="Partial HIT (boundary block)"),
        mpatches.Patch(facecolor=COLORS["miss"],     label="Cache MISS (cold)"),
        mpatches.Patch(facecolor=COLORS["unknown"],  label="Unknown (not reported)"),
    ]
    phase_patches = [
        mpatches.Patch(facecolor=c, label=n.replace("_", " ").title())
        for n, c in PHASE_LABEL_COLORS.items()
    ]
    leg1 = ax.legend(handles=legend_patches, loc="lower right",
                     title="Block status", fontsize=8, framealpha=0.9)
    ax.add_artist(leg1)
    ax.legend(handles=phase_patches, loc="upper right",
              title="Workload phase (row colour)", fontsize=8, framealpha=0.9)

    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"[*] Token heatmap saved → {out_path}")
    plt.close(fig)
