# KV Cache Visualizer — vLLM + LMCache

Demonstrates and visualises vLLM's KV-cache prefix caching using four
workload patterns (cold misses, prefix reuse, exact repeats, heavy cache
reuse). Shows which input tokens were served from the cache and which were
recomputed.

### Project layout

```
kv-cache-viz/
├── input/                          # one JSON file per workload definition
│   ├── cold_misses.json            # 10 unique prompts — cache miss baseline
│   ├── prefix_reuse.json           # 8 prompts sharing a paragraph prefix
│   ├── exact_repeats.json          # same prompt ×6 — maximum hit rate
│   └── heavy_cache_reuse.json      # 20 prompts over a ~700-token document
├── output/                         # auto-created; one subfolder per run
│   └── <name>_<YYYYMMDD_HHMMSS>/
│       ├── requests.json           # per-request latency + cached_tokens
│       ├── metrics.csv             # Prometheus time-series
│       └── meta.json               # workload metadata
├── workload.py                     # runner — reads input/, writes output/
├── visualize.py                    # dashboard + token heatmap PNGs
└── token_mapper.py                 # maps cached_tokens → words/blocks
```

---

## Tested Environment

| Component | Version |
|-----------|---------|
| OS | Ubuntu 22.04 |
| Python | 3.10.x |
| GPU | NVIDIA A30 24 GB (sm_80) — any CUDA-capable GPU works |
| CUDA Driver | 13.x (CUDA 12.x driver also works) |
| vLLM | 0.21.0 |
| LMCache | 0.4.5 |
| PyTorch | 2.11.0+cu130 |
| transformers | 5.8.1 |

---

## Prerequisites

- NVIDIA GPU with ≥ 8 GB VRAM (24 GB recommended for full model)
- NVIDIA driver installed (`nvidia-smi` must work)
- Python 3.10+ (`python3 --version`)
- `pip3` and `python3-venv` available

Install venv support if missing:

```bash
sudo apt install python3-venv python3-pip -y
```

---

## Step 1 — Clone / enter the project directory

```bash
cd kv-cache-viz
```

---

## Step 2 — Clear pip cache (frees ~6 GB, prevents stale package conflicts)

```bash
pip3 cache purge
```

---

## Step 3 — Create a fresh virtual environment

```bash
python3 -m venv ~/vllm-env
```

---

## Step 4 — Install vLLM

This installs vLLM and all its dependencies (PyTorch, transformers, etc.).
Takes 5–10 minutes depending on network speed.

```bash
~/vllm-env/bin/pip install --upgrade pip setuptools wheel
~/vllm-env/bin/pip install vllm==0.21.0 --no-cache-dir
```

---

## Step 5 — Install LMCache

```bash
~/vllm-env/bin/pip install lmcache==0.4.5 --no-cache-dir
```

---

## Step 6 — Install workload dependencies

```bash
~/vllm-env/bin/pip install pandas matplotlib --no-cache-dir
```

---

## Step 7 — Activate the virtual environment

Do this in **every new terminal session** before running any command:

```bash
source ~/vllm-env/bin/activate
```

You should see `(vllm-env)` in your prompt.

---

## Step 8 — Start the vLLM server

Run this in **Terminal 1**. The server takes ~60 seconds to load the model
and compile CUDA kernels on first launch.

```bash
VLLM_USE_FLASHINFER_SAMPLER=0 vllm serve Qwen/Qwen2.5-3B-Instruct \
  --max-model-len 4096 \
  --enable-prefix-caching \
  --gpu-memory-utilization 0.85 \
  --enable-prompt-tokens-details \
  > /tmp/vllm_serve.log 2>&1 &
```

Wait for the server to be ready:

```bash
until curl -s http://localhost:8000/health > /dev/null; do sleep 5; done && echo "Server is ready"
```

> **Why `VLLM_USE_FLASHINFER_SAMPLER=0`?**
> FlashInfer's sampling kernels require JIT compilation with a CUDA 12+
> toolkit (`nvcc`). If your system only has an older CUDA toolkit (e.g.
> CUDA 11.x), this compilation fails. Setting this env var makes vLLM
> fall back to its built-in Triton sampler with no loss of functionality.
>
> **Why `--enable-prompt-tokens-details`?**
> Without this flag vLLM omits `cached_tokens` from API responses entirely,
> making every request show `cached=n/a` even when caching is working.

---

## Step 9 — Run the workloads

Open **Terminal 2**, activate the venv, then run:

```bash
source ~/vllm-env/bin/activate
cd kv-cache-viz
```

**Run all workloads** (cold_misses → prefix_reuse → exact_repeats → heavy_cache_reuse):
```bash
python workload.py
```

**Run a single workload** by pointing at its input file:
```bash
python workload.py --workload input/heavy_cache_reuse.json
```

**Run against a remote host:**
```bash
python workload.py --host <IP> --port 8000
```

Each run is saved to its own timestamped folder under `output/`:
```
output/heavy_cache_reuse_20260516_213045/
    requests.json   ← per-request latency + cached_tokens
    metrics.csv     ← Prometheus time-series
    meta.json       ← workload metadata
```

Expected output for `heavy_cache_reuse`:
```
Workload : heavy_cache_reuse
Prompts  : 20
Output   : output/heavy_cache_reuse_20260516_213045

  [     heavy_cache_reuse_0]  2.10s | prompt=  620 compl= 150  cached=n/a
  [     heavy_cache_reuse_1]  0.45s | prompt=  622 compl= 150  cached=608
  [     heavy_cache_reuse_2]  0.45s | prompt=  619 compl= 150  cached=608
  ...
  Summary for 'heavy_cache_reuse':
    Requests      : 20 ok / 20 total
    Latency p50   : 0.45s
    Cached tokens : min=0 max=608 mean=579
```

---

## Step 10 — Visualise results and see the token-level cache map

**Auto-pick the most recent run:**
```bash
python visualize.py
```

**Visualise a specific run:**
```bash
python visualize.py --run-dir output/heavy_cache_reuse_20260516_213045
```

This prints a colour-coded ASCII block map to the terminal showing exactly
which words/tokens were served from the KV cache (green) vs recomputed
(red), and saves two PNG files inside the run directory:

| File | Contents |
|------|----------|
| `output/<run>/token_heatmap.png` | One row per request; each cell = 16-token KV block; green = HIT, red = MISS |
| `output/<run>/kv_cache_dashboard.png` | Time-series: GPU cache utilisation, latency per phase, token throughput |

---

## Understanding the output

### `cached` column in workload output

| Value | Meaning |
|-------|---------|
| `n/a` | First-ever request — nothing in cache yet; vLLM returns `null` for `prompt_tokens_details` |
| `16` | System-prompt block (16 tokens) was cached |
| `64` | Shared context prefix (4 blocks × 16 tokens) was cached |
| `80` | Nearly the full prompt was cached (exact repeat hit) |

### Why latency drops

- **cold_misses** ~1.45 s — all tokens computed fresh every time
- **prefix_reuse** — first request cold; subsequent ones skip the shared
  context block (~64 tokens), saving prefill compute
- **exact_repeats** ~0.61 s — 80 of 82 prompt tokens hit the cache;
  only 2 new tokens prefilled
- **heavy_cache_reuse** ~0.45 s after the first request — the ~600-token
  Nexus spec document is cached after request 0; each subsequent question
  only prefills its short unique tail (~10–20 tokens)

### How the token mapper works

vLLM prefix-caches in 16-token blocks. When `cached_tokens=64`, the first
4 blocks (tokens 0–63) were served from cache. `token_mapper.py`
tokenises the prompt with the same chat template vLLM uses, splits into
blocks, then decodes each block back to human-readable text and labels it
HIT / PARTIAL / MISS.

---

## Useful commands

```bash
# Check server is running
curl http://localhost:8000/health

# Check which model is loaded
curl http://localhost:8000/v1/models

# Watch server logs live
tail -f /tmp/vllm_serve.log

# Check GPU memory usage
nvidia-smi

# Stop the server
pkill -f "vllm serve"

# Run all workloads
python workload.py

# Run only the heavy cache reuse workload
python workload.py --workload input/heavy_cache_reuse.json

# Run a specific workload against a remote host
python workload.py --workload input/prefix_reuse.json --host <IP> --port 8000

# Visualise the most recent run (auto-detected)
python visualize.py

# Visualise a specific run
python visualize.py --run-dir output/heavy_cache_reuse_20260516_213045

# List all completed runs
ls output/
```

---

## Troubleshooting

### `cached=n/a` for every request
Server was started without `--enable-prompt-tokens-details`. Stop the
server, add the flag, and restart (Step 8).

### `AttributeError: Qwen2Tokenizer has no attribute all_special_tokens_extended`
The `transformers` version is incompatible with your vLLM version.
Upgrade vLLM to 0.21.0+ which supports transformers 5.x (Step 4).

### FlashInfer ninja build failure / `cuda/functional: No such file`
The system CUDA toolkit is too old for FlashInfer's JIT compiler.
Add `VLLM_USE_FLASHINFER_SAMPLER=0` to the serve command (Step 8).

### `Free memory on device … is less than desired GPU memory utilization`
Another process is using the GPU. Check with `nvidia-smi`, kill leftover
vllm processes with `pkill -f "vllm serve"`, then restart.

### Server takes a long time on first launch
Normal — vLLM runs `torch.compile` and CUDA graph capture on first start.
Subsequent launches use a cache and are faster (~20 s vs ~60 s).
