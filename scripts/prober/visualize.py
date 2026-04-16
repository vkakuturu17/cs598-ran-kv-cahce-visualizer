"""
KV Cache Metrics Visualizer
Reads results/metrics.csv and results/requests.json produced by workload.py

Usage:
    python visualize.py [--output-dir results]
"""

import argparse
import json
import sys
from pathlib import Path

import pandas as pd
import matplotlib
matplotlib.use("Agg")           # headless-safe; change to "TkAgg" if you have a display
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec

# Workload phase colours
PHASE_COLORS = {
    "cold_misses":   "#e74c3c",
    "prefix_reuse":  "#3498db",
    "exact_repeats": "#2ecc71",
}


def load_metrics(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, parse_dates=["timestamp"])
    df = df.sort_values("timestamp").reset_index(drop=True)
    # Elapsed seconds from start
    df["elapsed"] = (df["timestamp"] - df["timestamp"].iloc[0]).dt.total_seconds()
    return df


def load_requests(path: Path):
    with open(path) as f:
        data = json.load(f)
    rows = []
    for r in data:
        if "error" in r:
            continue
        phase = r["id"].rsplit("_", 1)[0]
        rows.append({
            "id": r["id"],
            "phase": phase,
            "latency_s": r["latency_s"],
            "prompt_tokens": r["prompt_tokens"],
            "completion_tokens": r["completion_tokens"],
            "timestamp": pd.to_datetime(r["timestamp"]),
        })
    return pd.DataFrame(rows)


def add_phase_bands(ax, req_df, metrics_t0):
    """Shade background by workload phase."""
    for phase, color in PHASE_COLORS.items():
        ph = req_df[req_df["phase"] == phase]
        if ph.empty:
            continue
        t_start = (ph["timestamp"].min() - metrics_t0).total_seconds()
        t_end   = (ph["timestamp"].max() - metrics_t0).total_seconds()
        ax.axvspan(t_start, t_end, alpha=0.08, color=color, label=f"_{phase}")


def plot(metrics: pd.DataFrame, reqs: pd.DataFrame, out_dir: Path):
    metrics_t0 = metrics["timestamp"].iloc[0]

    fig = plt.figure(figsize=(16, 14))
    fig.suptitle("vLLM + LMCache — KV Cache Visualisation", fontsize=15, fontweight="bold")
    gs = GridSpec(4, 2, figure=fig, hspace=0.55, wspace=0.35)

    # ── 1. GPU cache utilisation ──────────────────────────────────────────
    ax1 = fig.add_subplot(gs[0, :])
    ax1.set_title("GPU KV-Cache Utilisation (%)")
    col = "vllm:gpu_cache_usage_perc"
    if col in metrics.columns:
        ax1.plot(metrics["elapsed"], metrics[col] * 100, color="#8e44ad", linewidth=1.5)
    add_phase_bands(ax1, reqs, metrics_t0)
    ax1.set_ylabel("% used")
    ax1.set_xlabel("Time (s)")
    ax1.set_ylim(0, 100)
    ax1.grid(True, alpha=0.3)

    # ── 2. LMCache hit vs miss tokens ────────────────────────────────────
    ax2 = fig.add_subplot(gs[1, 0])
    ax2.set_title("LMCache Hit / Miss Tokens")
    hit_col  = "lmcache_hit_tokens"
    miss_col = "lmcache_miss_tokens"
    if hit_col in metrics.columns and metrics[hit_col].max() > 0:
        ax2.plot(metrics["elapsed"], metrics[hit_col],  label="Hit tokens",  color="#27ae60")
        ax2.plot(metrics["elapsed"], metrics[miss_col], label="Miss tokens", color="#e74c3c")
        ax2.legend(fontsize=8)
    else:
        ax2.text(0.5, 0.5, "LMCache token metrics\nnot exposed",
                 ha="center", va="center", transform=ax2.transAxes, color="grey")
    add_phase_bands(ax2, reqs, metrics_t0)
    ax2.set_ylabel("Tokens")
    ax2.set_xlabel("Time (s)")
    ax2.grid(True, alpha=0.3)

    # ── 3. Requests running / waiting ────────────────────────────────────
    ax3 = fig.add_subplot(gs[1, 1])
    ax3.set_title("Requests Running / Waiting")
    run_col  = "vllm:num_requests_running"
    wait_col = "vllm:num_requests_waiting"
    if run_col in metrics.columns:
        ax3.fill_between(metrics["elapsed"], metrics[run_col],  alpha=0.5, label="Running",  color="#3498db")
        ax3.fill_between(metrics["elapsed"], metrics[wait_col], alpha=0.5, label="Waiting",  color="#e67e22")
        ax3.legend(fontsize=8)
    add_phase_bands(ax3, reqs, metrics_t0)
    ax3.set_ylabel("Count")
    ax3.set_xlabel("Time (s)")
    ax3.grid(True, alpha=0.3)

    # ── 4. Per-request latency by phase ──────────────────────────────────
    ax4 = fig.add_subplot(gs[2, 0])
    ax4.set_title("Per-Request Latency by Phase")
    if not reqs.empty:
        for i, row in reqs.iterrows():
            color = PHASE_COLORS.get(row["phase"], "grey")
            ax4.bar(i, row["latency_s"], color=color, width=0.8)
        ax4.set_xlabel("Request index")
        ax4.set_ylabel("Latency (s)")
    ax4.grid(True, alpha=0.3, axis="y")

    # ── 5. Latency box-plot per phase ─────────────────────────────────────
    ax5 = fig.add_subplot(gs[2, 1])
    ax5.set_title("Latency Distribution per Phase")
    if not reqs.empty:
        phases = list(PHASE_COLORS.keys())
        data = [reqs[reqs["phase"] == p]["latency_s"].values for p in phases]
        bp = ax5.boxplot(data, patch_artist=True, labels=phases)
        for patch, phase in zip(bp["boxes"], phases):
            patch.set_facecolor(PHASE_COLORS[phase])
            patch.set_alpha(0.7)
        ax5.set_ylabel("Latency (s)")
    ax5.grid(True, alpha=0.3, axis="y")

    # ── 6. Token throughput total ─────────────────────────────────────────
    ax6 = fig.add_subplot(gs[3, :])
    ax6.set_title("Cumulative Tokens Generated (prompt + completion)")
    p_col = "vllm:prompt_tokens_total"
    g_col = "vllm:generation_tokens_total"
    if p_col in metrics.columns:
        ax6.plot(metrics["elapsed"], metrics[p_col],  label="Prompt tokens total",     color="#2980b9")
        ax6.plot(metrics["elapsed"], metrics[g_col],  label="Generation tokens total", color="#16a085")
        ax6.legend(fontsize=8)
    add_phase_bands(ax6, reqs, metrics_t0)
    ax6.set_ylabel("Tokens")
    ax6.set_xlabel("Time (s)")
    ax6.grid(True, alpha=0.3)

    # Legend for phase bands
    patches = [mpatches.Patch(color=c, alpha=0.3, label=n.replace("_", " ").title())
               for n, c in PHASE_COLORS.items()]
    fig.legend(handles=patches, loc="upper right", title="Workload phase",
               fontsize=9, framealpha=0.8)

    out = out_dir / "kv_cache_dashboard.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"[*] Dashboard saved → {out}")

    # Also save individual summary stats
    if not reqs.empty:
        summary = reqs.groupby("phase")["latency_s"].agg(["mean", "median", "min", "max", "count"])
        print("\nLatency summary (seconds):")
        print(summary.to_string())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", default="results")
    args = parser.parse_args()

    out = Path(args.output_dir)
    metrics_csv = out / "metrics.csv"
    requests_json = out / "requests.json"

    if not metrics_csv.exists():
        sys.exit(f"ERROR: {metrics_csv} not found — run workload.py first")

    metrics = load_metrics(metrics_csv)
    reqs = load_requests(requests_json) if requests_json.exists() else pd.DataFrame()

    plot(metrics, reqs, out)


if __name__ == "__main__":
    main()
