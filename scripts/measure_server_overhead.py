"""SIMULATOR.md §8 `overhead`: what a real SGLang server adds to each decode step.

The cost grid times forwards on a bare ModelRunner. A server also runs its scheduler on the
CPU every step (batch bookkeeping, sampling output, logprobs, metrics); with the overlap
schedule that hides behind the GPU while the GPU step is long, and shows when it is short.
Run 20260912-072938's tail (batch < 100) decoded 2-8 ms per step slower than the grid
predicts, and that may not be fitted on a run trace (§8), so it is measured here.

Launches a server with the run's args, then for each batch size B sends B requests at once
(real-text prompts of --ctx tokens, `ignore_eos`), and reads the scheduler's own
"Decode batch, #running-req: B ... gen throughput" lines: step = B / throughput, taken only
while all B decode and nothing waits. Each B runs with and without `return_logprob`
(the run asked for logprobs on every token).

    python scripts/measure_server_overhead.py --gpus 0,1,2,3 \\
        --tokens analysis/costgrid/real-tokens-030010.jsonl --out analysis/costgrid/b200-tp4-server
"""

import argparse
import json
import os
import re
import subprocess
import sys
import threading
import time
from pathlib import Path

import numpy as np
import requests

ap = argparse.ArgumentParser()
ap.add_argument("--model", default="/data0/shared/Qwen3-30B-A3B-Instruct-2507")
ap.add_argument("--tp", type=int, default=4)
ap.add_argument("--gpus", default="0,1,2,3")
ap.add_argument("--port", type=int, default=31777)
ap.add_argument("--tokens", required=True, help="jsonl of real token-id lists")
ap.add_argument("--ctx", type=int, nargs="+", default=[2048, 16384])
ap.add_argument("--batches", type=int, nargs="+", default=[1, 2, 4, 8, 16, 32, 64, 128, 224])
ap.add_argument("--new-tokens", type=int, default=640)
ap.add_argument("--out", type=Path, required=True)
a = ap.parse_args()
a.out.mkdir(parents=True, exist_ok=True)
log_path = a.out / "server.log"

env = dict(os.environ, CUDA_VISIBLE_DEVICES=a.gpus)
for k in ("HTTP_PROXY", "HTTPS_PROXY", "http_proxy", "https_proxy", "ALL_PROXY", "all_proxy"):
    env.pop(k, None)   # the host proxy turns local HTTP into 404s
cmd = [sys.executable, "-m", "sglang.launch_server", "--model-path", a.model, "--tp", str(a.tp),
       "--port", str(a.port), "--attention-backend", "trtllm_mha", "--page-size", "64",
       "--moe-runner-backend", "triton", "--max-running-requests", "256",
       "--mem-fraction-static", "0.75", "--context-length", "65536", "--enable-metrics",
       "--log-level", "info", "--decode-log-interval", "40"]
server = subprocess.Popen(cmd, env=env, stdout=open(log_path, "w"), stderr=subprocess.STDOUT)
http = requests.Session()
http.trust_env = False
url = f"http://127.0.0.1:{a.port}"
try:
    for _ in range(600):
        try:
            if http.get(f"{url}/health", timeout=2).status_code == 200:
                break
        except requests.RequestException:
            pass
        if server.poll() is not None:
            raise SystemExit(f"server exited, see {log_path}")
        time.sleep(2)
    else:
        raise SystemExit("server did not come up")

    real = np.concatenate([np.asarray(json.loads(line), np.int64) for line in open(a.tokens)])
    rng = np.random.default_rng(0)
    line_re = re.compile(r"Decode batch, #running-req: (\d+), #token: (\d+).*?gen throughput "
                         r"\(token/s\): ([\d.]+), #queue-req: (\d+)")

    def run(b, ctx, logprob):
        http.post(f"{url}/flush_cache", timeout=60)
        start = log_path.stat().st_size
        prompts = [real[s:s + ctx].tolist() for s in rng.integers(0, len(real) - ctx, b)]

        def one(ids):
            http.post(f"{url}/generate", timeout=3600, json=dict(
                input_ids=ids, return_logprob=logprob,
                sampling_params=dict(max_new_tokens=a.new_tokens, ignore_eos=True,
                                     temperature=1.0)))
        th = [threading.Thread(target=one, args=(p,)) for p in prompts]
        for t in th:
            t.start()
        for t in th:
            t.join()
        time.sleep(1)
        with open(log_path) as f:
            f.seek(start)
            text = f.read()
        steps = []
        for m in line_re.finditer(text):
            running, _, thr, queued = int(m[1]), int(m[2]), float(m[3]), int(m[4])
            if running == b and queued == 0 and thr > 0:
                steps.append(b / thr)
        # the first line after the prefills can straddle one; drop it
        steps = steps[1:]
        return steps

    rows = []
    for ctx in a.ctx:
        for b in a.batches:
            if b * (ctx + a.new_tokens) > 4_000_000:
                continue
            for logprob in (True, False):
                st = run(b, ctx, logprob)
                if not st:
                    print(f"B={b} ctx={ctx} logprob={logprob}: no clean decode lines", flush=True)
                    continue
                row = dict(batch=b, ctx=ctx, logprob=logprob, n=len(st),
                           step_med=float(np.median(st)), step_p10=float(np.percentile(st, 10)),
                           step_p90=float(np.percentile(st, 90)))
                rows.append(row)
                print(json.dumps(row), flush=True)
    with open(a.out / "server_steps.jsonl", "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    meta = dict(cmd=cmd, new_tokens=a.new_tokens, started=time.strftime("%Y-%m-%dT%H:%M:%S"))
    (a.out / "meta.json").write_text(json.dumps(meta, indent=1))
finally:
    server.terminate()
    try:
        server.wait(60)
    except subprocess.TimeoutExpired:
        server.kill()
