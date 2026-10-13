"""SIMULATOR.md §15 5(d): one training step of one of our runs through the engine.

Loads the step, realises it, caps it and builds the system from a config bundle, and runs
the engine with the bundle's cost model (or a constant price). Then compares with the
run's own answers, in two groups:

- not set by the clock: ``cached_tokens`` on follow-up turns, which 5(c) matched 99.63%;
- set by the clock: cold first turns (which member of a group reaches a replica first),
  TTFT, decode seconds per token, session ends and the rollout span. These preview the
  second and third levels of §12. Until the run's ~1.5x decode slowdown is explained
  they are expected to come out fast, so they are a measurement, not a pass/fail.

    python scripts/simulator_dev/simulate_step.py \
        /data0/shared/kewen.liu/agentic-timelines/20260912-072938 \
        --bundle configs/simulator/run-20260912-072938.yaml --step 1 --shared-root 1045 \
        --out analysis/20260912-072938/engine/step1-model.json
"""

import argparse
import dataclasses
import json
import logging
import time
from collections import Counter
from pathlib import Path

import numpy as np

from simulator.adapters.priors import default_priors
from simulator.bundle import fill_caps, load_bundle
from simulator.dev.agentic_timeline import AgenticTimelineAdapter
from simulator.engine.engine import constant, run
from simulator.engine.system import build
from simulator.ir import Prefix, validate
from simulator.realise import realise

ap = argparse.ArgumentParser()
ap.add_argument("path", type=Path)
ap.add_argument("--bundle", type=Path, required=True)
ap.add_argument("--step", type=int, default=1)
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--price", choices=["model", "constant"], default="model")
ap.add_argument("--step-s", type=float, default=0.044, help="with --price constant")
# Tokens every group's first prompt shares (system prompt + tool schemas), which a
# timeline without text cannot show; 1045 for run 0912 (see feed_workload.py).
ap.add_argument("--shared-root", type=int, default=0)
ap.add_argument("--limit-sessions", type=int, default=0, help="first N sessions only (smoke)")
ap.add_argument("--out", type=Path, default=None)
a = ap.parse_args()
logging.disable(logging.INFO)   # SGLang logs every decode batch

t0 = time.time()
bundle = load_bundle(a.bundle)
conv = AgenticTimelineAdapter(step=a.step).convert(a.path, default_priors())
w = realise(conv.workload, seed=a.seed)
if a.limit_sessions:
    w = dataclasses.replace(w, sessions=w.sessions[:a.limit_sessions])
w = fill_caps(w, bundle)
if a.shared_root:
    root = Prefix(id="shared-root", tokens=a.shared_root, prov={"tokens": "derived"})
    w = dataclasses.replace(w, prefixes=(root,) + tuple(
        dataclasses.replace(p, parent=root.id) if p.parent is None else p for p in w.prefixes))
    validate(w)
system = build(bundle)
price = system.cost.price if a.price == "model" else constant(a.step_s)
print(f"{w.source}: {len(w.sessions)} sessions, "
      f"{sum(len(s.requests) for s in w.sessions)} requests, pools {system.pools}, "
      f"price {a.price}, loaded in {time.time() - t0:.0f}s")

res = run(w, system.replicas, system.pools, price)
modes = [Counter(f.step.mode for f in fs) for fs in res.forwards]
print(f"{sum(len(fs) for fs in res.forwards)} forwards in {res.wall_s:.0f}s wall {modes}, "
      f"retracted {res.retracted}, short parents {res.short}, "
      f"outside grid {system.cost.outside_share:.2%}")

# --- the run's answers ----------------------------------------------------------------
obs = {o.id: o for o in conv.observations.requests}
req = {r.id: r for s in w.sessions for r in s.requests}
prefix_ids = {p.id for p in w.prefixes}
gen = next((t, e) for n, t, e in conv.observations.phases if n == "gen")
summary = dict(source=w.source, bundle=str(a.bundle), price=a.price, seed=a.seed,
               forwards=[dict(m) for m in modes], wall_s=res.wall_s, retracted=res.retracted,
               outside_grid=system.cost.outside_share)


def pct(x, qs=(10, 50, 90)):
    return {f"p{q}": float(np.percentile(x, q)) for q in qs} if len(x) else {}


def compare(name, sim, real):
    sim, real = np.asarray(sim, float), np.asarray(real, float)
    out = dict(n=len(sim), sim=pct(sim), run=pct(real),
               ratio=pct(sim / np.where(real > 0, real, np.nan)) if len(sim) else {})
    s, r, q = out["sim"], out["run"], out["ratio"]
    print(f"  {name:22s} n={len(sim):6d}  sim p50 {s.get('p50', 0):9.3f}  run p50 "
          f"{r.get('p50', 0):9.3f}  sim/run p10/p50/p90 {q.get('p10', 0):.2f}/"
          f"{q.get('p50', 0):.2f}/{q.get('p90', 0):.2f}")
    return out


# not set by the clock
later = [(rr.cached_tokens, obs[rid].cached_tokens) for rid, rr in res.requests.items()
         if rid in obs and req[rid].prefix_parent not in prefix_ids]
d = np.array([s - r for s, r in later])
summary["later_cached"] = dict(n=len(d), exact=float(np.mean(d == 0)),
                               within_page=float(np.mean(np.abs(d) <= 64)))
print(f"cached_tokens on follow-up turns: n={len(d)} exact {np.mean(d == 0):.2%}, "
      f"within a page {np.mean(np.abs(d) <= 64):.2%}")

# set by the clock
page = int(bundle.server.args["page_size"])
floor = a.shared_root // page * page
first = [(res.requests[rid].cached_tokens, obs[rid].cached_tokens)
         for rid in res.requests if rid in obs and req[rid].prefix_parent in prefix_ids]
cold = dict(n=len(first), sim=sum(s <= floor for s, _ in first),
            run=sum(r <= floor for _, r in first))
summary["first_turn_cold"] = cold
print(f"first turns that missed their group's prefix (cached <= {floor}): "
      f"sim {cold['sim']} / run {cold['run']} of {cold['n']}")

both = [rid for rid in res.requests if rid in obs]
print("timing, sim against run:")
summary["ttft_s"] = compare("TTFT s", [res.requests[r].t_first_token - res.requests[r].t_submit
                                       for r in both], [obs[r].ttft_s for r in both])
multi = [r for r in both if res.requests[r].completion_tokens > 1]
summary["decode_s_per_token"] = compare(
    "decode s/token", [(res.requests[r].t_finish - res.requests[r].t_first_token)
                       / (res.requests[r].completion_tokens - 1) for r in multi],
    [obs[r].decode_s / (res.requests[r].completion_tokens - 1) for r in multi])
end = {}
for rid, rr in res.requests.items():
    end[req[rid].session] = max(end.get(req[rid].session, 0.0), rr.t_finish)
task_session = {t.id: s.id for s in w.sessions for t in s.tasks}
for tid, tr in res.tasks.items():
    sid = task_session[tid]
    end[sid] = max(end.get(sid, 0.0), tr.t_end)
osess = {o.id: o for o in conv.observations.sessions if o.id in end}
summary["session_end_s"] = compare("session end s", [end[s] for s in osess],
                                   [o.t_end - gen[0] for o in osess.values()])
summary["span_s"] = dict(sim=res.end, run=gen[1] - gen[0])
print(f"  rollout span           sim {res.end:.0f}s  run gen {gen[1] - gen[0]:.0f}s  "
      f"sim/run {res.end / (gen[1] - gen[0]):.2f}")

if a.out:
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(summary, indent=1, default=str))
