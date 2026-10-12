"""SIMULATOR.md §15 5(c): a real workload's IR, as token ids, through stubbed SGLang schedulers.

Loads one training step of one of our runs, realises it, fills each request's cap with
verl's rule, and runs it through stubbed replicas configured like the run's servers. Every
forward costs a constant, so the timeline is not a prediction; the point is the token
stream. The run's `cached_tokens` per request is the check: SGLang's radix cache only
serves what the ids actually share, so if the ids are built right the stub should
reproduce what the real servers served.

    python scripts/simulator_dev/feed_workload.py \
        /data0/shared/kewen.liu/agentic-timelines/20260912-072938 --step 1
"""

import argparse
import collections
import dataclasses
import json
import logging
import time
from pathlib import Path

import numpy as np

from simulator.adapters.priors import default_priors
from simulator.dev.agentic_timeline import AgenticTimelineAdapter
from simulator.dev.ir_driver import run
from simulator.engine.sglang_stub import StubScheduler, stub_server_args
from simulator.ir import Prefix, validate
from simulator.realise import realise

ap = argparse.ArgumentParser()
ap.add_argument("path", type=Path)
ap.add_argument("--step", type=int, default=1)
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--model", default="/data0/shared/Qwen3-30B-A3B-Instruct-2507")
ap.add_argument("--replicas", type=int, default=2)
# Run 20260912-072938's servers (configs/grpo_swebench.yaml, resolved by SGLang on B200)
ap.add_argument("--max-total-tokens", type=int, default=4_560_512)
ap.add_argument("--chunked-prefill", type=int, default=16384)
ap.add_argument("--max-running", type=int, default=256)
# verl: max_new_tokens = min(response_length, prompt_length + response_length - prompt),
# clamped to max_model_len - prompt - 1 (async_sglang_server.generate)
ap.add_argument("--prompt-length", type=int, default=4096)
ap.add_argument("--response-length", type=int, default=61440)
ap.add_argument("--max-model-len", type=int, default=65536)
# 8 workers x AGENTIC_MAX_LIVE_CONTAINERS 33; 8 x AGENTIC_MAX_EVAL_CONTAINERS 4
ap.add_argument("--containers", type=int, default=264)
ap.add_argument("--evals", type=int, default=32)
ap.add_argument("--step-s", type=float, default=0.044)
# Tokens every group's first prompt shares (system prompt + tool schemas). The timeline has no
# text, so the adapter cannot see it; for run 0912 it is 1045, from the harness templates at
# b2f97c8 rendered with the Qwen3 tokenizer (common prefix of two different problems).
ap.add_argument("--shared-root", type=int, default=0)
ap.add_argument("--limit-sessions", type=int, default=0, help="first N sessions only (smoke)")
ap.add_argument("--out", type=Path, default=None)
a = ap.parse_args()
logging.disable(logging.INFO)   # SGLang logs every decode batch

t0 = time.time()
conv = AgenticTimelineAdapter(step=a.step).convert(a.path, default_priors())
w = realise(conv.workload, seed=a.seed)


def cap(prompt):
    return max(0, min(a.response_length, a.prompt_length + a.response_length - prompt,
                      a.max_model_len - prompt - 1))


sessions = w.sessions[:a.limit_sessions] if a.limit_sessions else w.sessions
sessions = tuple(dataclasses.replace(s, requests=tuple(
    dataclasses.replace(r, max_new_tokens=cap(r.prompt_tokens)) for r in s.requests))
    for s in sessions)
w = dataclasses.replace(w, sessions=sessions)
if a.shared_root:
    root = Prefix(id="shared-root", tokens=a.shared_root, prov={"tokens": "derived"})
    w = dataclasses.replace(w, prefixes=(root,) + tuple(
        dataclasses.replace(p, parent=root.id) if p.parent is None else p for p in w.prefixes))
    validate(w)
n_req = sum(len(s.requests) for s in w.sessions)
print(f"{w.source}: {len(w.sessions)} sessions, {n_req} requests, {len(w.prefixes)} prefixes, "
      f"loaded in {time.time() - t0:.0f}s")

args = stub_server_args(model_path=a.model, page_size=64, max_running_requests=a.max_running,
                        max_total_tokens=a.max_total_tokens, context_length=a.max_model_len,
                        chunked_prefill_size=a.chunked_prefill, schedule_policy="fcfs",
                        random_seed=0)
reps = [StubScheduler(args) for _ in range(a.replicas)]
pools = {"container": a.containers, "eval": a.evals}
res = run(w, reps, pools, step_s=a.step_s)
steps = sum(len(s) for s in res.steps)
print(f"{steps} forwards in {res.wall_s:.0f}s wall, virtual end {res.end:.0f}s, "
      f"retracted {res.retracted}, short parents {res.short}")

# --- the check: cached_tokens against the run's --------------------------------------
obs = {o.id: o for o in conv.observations.requests}
req = {r.id: r for s in w.sessions for r in s.requests}
first, later = [], []
for rid, rr in res.requests.items():
    o = obs.get(rid)
    if o is None:
        continue
    r = req[rid]
    (first if r.prefix_parent in {p.id for p in w.prefixes} or r.prefix_parent is None
     else later).append((rr.cached_tokens, o.cached_tokens, r.prompt_tokens, rid))


def describe(name, rows):
    if not rows:
        return {}
    sim = np.array([x[0] for x in rows])
    real = np.array([x[1] for x in rows])
    prompt = np.array([x[2] for x in rows])
    d = sim - real
    out = dict(n=len(rows), exact=float(np.mean(d == 0)),
               within_page=float(np.mean(np.abs(d) <= 64)),
               diff_p1=float(np.percentile(d, 1)), diff_p50=float(np.percentile(d, 50)),
               diff_p99=float(np.percentile(d, 99)),
               sim_total=int(sim.sum()), real_total=int(real.sum()),
               sim_hit=float(sim.sum() / prompt.sum()), real_hit=float(real.sum() / prompt.sum()))
    print(f"  {name:6s} n={out['n']:6d}  exact {out['exact']:.2%}  |diff|<=64 "
          f"{out['within_page']:.2%}  diff p1/p50/p99 {out['diff_p1']:.0f}/{out['diff_p50']:.0f}/"
          f"{out['diff_p99']:.0f}  hit sim {out['sim_hit']:.3f} vs run {out['real_hit']:.3f}")
    return out


print("cached_tokens, stub minus run:")
summary = dict(first=describe("first", first), later=describe("later", later),
               forwards=steps, wall_s=res.wall_s, retracted=res.retracted, short=res.short,
               per_replica=[dict(collections.Counter(s.mode for s in st)) for st in res.steps],
               requests_per_replica=collections.Counter(rr.replica for rr in res.requests.values()))
if a.out:
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(summary, indent=1, default=str))
    rows = a.out.with_suffix(".requests.jsonl")
    with rows.open("w") as f:
        for rid, rr in res.requests.items():
            o = obs.get(rid)
            f.write(json.dumps(dict(id=rid, **dataclasses.asdict(rr),
                                    run_cached=o.cached_tokens if o else None,
                                    parent=req[rid].prefix_parent,
                                    shared=req[rid].shared_prefix_tokens)) + "\n")
