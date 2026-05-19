"""
Sanity-check script: verifies vLLM + LMCache are running and reachable.
Run this before workload.py.

Usage:
    python check_setup.py --host <cloudlab-ip> --port 8000
"""

import argparse
import sys
import requests


def check(host, port):
    base = f"http://{host}:{port}"
    ok = True

    # 1. Health endpoint
    print(f"[1] Checking health at {base}/health ...")
    try:
        r = requests.get(f"{base}/health", timeout=5)
        if r.status_code == 200:
            print("    OK — server is up")
        else:
            print(f"    WARN — status {r.status_code}: {r.text[:200]}")
    except Exception as e:
        print(f"    FAIL — {e}")
        ok = False

    # 2. Models endpoint
    print(f"[2] Checking models at {base}/v1/models ...")
    try:
        r = requests.get(f"{base}/v1/models", timeout=5)
        r.raise_for_status()
        models = [m["id"] for m in r.json().get("data", [])]
        print(f"    OK — models served: {models}")
    except Exception as e:
        print(f"    FAIL — {e}")
        ok = False

    # 3. Metrics endpoint (Prometheus)
    print(f"[3] Checking metrics at {base}/metrics ...")
    try:
        r = requests.get(f"{base}/metrics", timeout=5)
        r.raise_for_status()
        lines = [l for l in r.text.splitlines() if not l.startswith("#") and l.strip()]
        kv_lines = [l for l in lines if "cache" in l.lower()]
        print(f"    OK — {len(lines)} metric lines found")
        print(f"    KV cache related metrics:")
        for l in kv_lines[:15]:
            print(f"      {l}")
        lmcache_lines = [l for l in lines if "lmcache" in l.lower()]
        if lmcache_lines:
            print(f"    LMCache metrics found:")
            for l in lmcache_lines[:10]:
                print(f"      {l}")
        else:
            print("    NOTE: No 'lmcache_*' metrics found — LMCache may not expose Prometheus metrics.")
            print("          KV cache hits will still be visible via vllm:gpu_cache_usage_perc.")
    except Exception as e:
        print(f"    FAIL — {e}")
        ok = False

    # 4. Quick inference test
    print(f"[4] Sending test inference request ...")
    try:
        r = requests.post(
            f"{base}/v1/chat/completions",
            json={
                "model": "Qwen/Qwen2.5-3B-Instruct",
                "messages": [{"role": "user", "content": "Say 'hello' in one word."}],
                "max_tokens": 5,
                "temperature": 0.0,
            },
            timeout=30,
        )
        r.raise_for_status()
        data = r.json()
        reply = data["choices"][0]["message"]["content"]
        usage = data.get("usage", {})
        print(f"    OK — reply: {reply!r}")
        print(f"    Usage: {usage}")
    except Exception as e:
        print(f"    FAIL — {e}")
        ok = False

    print()
    if ok:
        print("Setup looks good! Run:  python workload.py --host", host, "--port", port)
    else:
        print("Some checks failed. Fix the issues above before running the workload.")
    return ok


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="localhost")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()
    sys.exit(0 if check(args.host, args.port) else 1)
