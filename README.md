# cs598-ran-kv-cache-visualizer

Build on top of vLLM and visualize when KV cache blocks are filled and evicted.

## What this project does

- Enables vLLM KV cache events (`BlockStored`, `BlockRemoved`) via built-in `kv_events_config`.
- Subscribes to vLLM's ZMQ event stream.
- Reconstructs token-level lifecycle from block-level events.
- Writes:
	- raw event log: `artifacts/kv_events.jsonl`
	- interactive HTML timeline: `artifacts/kv_cache_report.html`

## Install

```bash
pip install -r requirements.txt
```

## Run the visualizer

From the project root:

```bash
PYTHONPATH=. python scripts/run_kv_visualizer.py \
	--model mistralai/Mistral-7B-v0.1 \
	--max-tokens 64
```

Optional flags:

- `--events-out artifacts/kv_events.jsonl`
- `--html-out artifacts/kv_cache_report.html`
- `--endpoint-bind tcp://*:5557`
- `--endpoint-connect tcp://127.0.0.1:5557`
- `--topic kv-events`
- `--no-token-decode` (skip tokenizer decoding for token labels)
- `--gpu-memory-utilization 0.9` (lower this on busy/shared GPUs if vLLM startup fails due limited free VRAM)

## Notes

- KV events are published by vLLM itself; this project consumes and visualizes them.
- Eviction events do not carry token IDs directly in vLLM events, so this project
	maps evicted block hashes back to tokens observed during prior fill events.
