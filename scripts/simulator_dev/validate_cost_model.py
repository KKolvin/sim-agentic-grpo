"""SIMULATOR.md §12, first level: does the cost model account for a real server's time?

For one training step of one of our runs, take the work each replica actually did in each
interval between two metrics samples -- which requests were decoding and at what context,
how many tokens they produced, which EXTENDs ran -- and price it with the cost model. If
the model is right, the priced time equals the interval. Nothing here is a simulation: the
schedule is the real one, only the per-forward cost is predicted.

Runs since 2026-10-04 log which replica served each request, and then that is used. Older
runs (0912) did not; verl's router is deterministic (sticky by session, else fewest in
flight, first on a tie), so there it is replayed from the request start/finish order; the
per-replica running counts in srv-metrics check either.

    python scripts/simulator_dev/validate_cost_model.py \
        /data0/shared/kewen.liu/agentic-timelines/20260912-072938 --step 1 \
        --calibration analysis/costgrid/b200-tp4/calibration.json
"""

import argparse
import bisect
import collections
import json
from pathlib import Path

import numpy as np

from simulator.adapters.priors import default_priors
from simulator.dev.agentic_timeline import AgenticTimelineAdapter
from simulator.engine.cost_model import Calibration, CostModel, ModelShape

ap = argparse.ArgumentParser()
ap.add_argument("path", type=Path)
ap.add_argument("--step", type=int, default=1)
ap.add_argument("--calibration", required=True)
ap.add_argument("--model", default="/data0/shared/Qwen3-30B-A3B-Instruct-2507")
ap.add_argument("--chunk", type=int, default=16384, help="chunked_prefill_size")
ap.add_argument("--sub", type=int, default=40, help="slices per interval")
ap.add_argument("--same-forward-ms", type=float, default=20.0,
                help="prefill-finished stamps closer than this belong to one EXTEND")
ap.add_argument("--out", type=Path, default=None)
a = ap.parse_args()

conv = AgenticTimelineAdapter(step=a.step).convert(a.path, default_priors())
req = {r.id: r for s in conv.workload.sessions for r in s.requests}
obs = [o for o in conv.observations.requests if o.id in req]

# --- which replica served each request: logged, else replay verl's router ----------------
if all(o.replica is not None for o in obs):
    routing = "logged replica"
    replica = {o.id: int(o.replica) for o in obs}
else:
    routing = "router replay"
    ev = []
    for o in obs:
        ev.append((o.t_arrival, 0, o.id))
        ev.append((o.t_last_token, 1, o.id))
    ev.sort()
    inflight, sticky, replica = [0, 0], {}, {}
    for t, kind, rid in ev:
        if kind == 0:
            s = req[rid].session
            i = sticky.get(s)
            if i is None:
                i = inflight.index(min(inflight))
                sticky[s] = i
            replica[rid] = i
            inflight[i] += 1
        else:
            inflight[replica[rid]] -= 1
reps = sorted(set(replica.values()))

srv = collections.defaultdict(list)
for line in (a.path / next(p.name for p in a.path.glob("srv-metrics-*.jsonl"))).open():
    d = json.loads(line)
    srv[int(d["replica"])].append(d)
t0 = min(o.t_arrival for o in obs)
t1 = max(o.t_last_token for o in obs)

# per replica: decoding intervals and EXTEND batches
dec = {i: [] for i in reps}
live = {i: [] for i in reps}   # [arrival, last token): the replica has work for this request
firsts = {i: [] for i in reps}
for o in obs:
    r, i = req[o.id], replica[o.id]
    dec[i].append((o.t_first_token, o.t_last_token, r.prompt_tokens, r.completion_tokens))
    live[i].append((o.t_arrival, o.t_last_token))
    firsts[i].append((o.t_first_token, r.prompt_tokens - o.cached_tokens, o.cached_tokens))
# Requests prefilled in one EXTEND finish it together, but SGLang stamps prefill_finished per
# request inside its result loop (radix inserts in between), so one forward's stamps spread
# over up to ~10 ms. Two forwards are at least the ~31 ms EXTEND floor apart, so a chain of
# stamps each closer than --same-forward-ms is one EXTEND, keyed by its first stamp. On
# 030010 (logged replica) the gaps split cleanly: 0.1-12 ms, then nothing until 34 ms.
ext = {i: collections.defaultdict(list) for i in reps}
for i in reps:
    key, prev = None, -1e18
    for t, new, pre in sorted(firsts[i]):
        if t - prev >= a.same_forward_ms / 1e3:
            key = t
        ext[i][key].append((new, pre))
        prev = t

shape = ModelShape.from_config(a.model)
cal = Calibration.load(a.calibration)
models = {
    "full": CostModel(cal, shape),
}


def running(cands, t):
    return [(p + c * (t - ts) / max(te - ts, 1e-9), c / max(te - ts, 1e-9))
            for ts, te, p, c in cands if ts <= t < te]


arr = {i: np.array([(s0, s1) for s0, s1 in live[i]]) for i in reps}


def extend_cost(m, items):
    """One EXTEND batch, split into chunks of chunked_prefill_size new tokens."""
    total, out = 0.0, []
    for new, pre in items:
        while new > 0:
            take = min(new, a.chunk - sum(e for e, _ in out))
            out.append((take, pre))
            pre += take
            new -= take
            if sum(e for e, _ in out) >= a.chunk:
                total += m.extend([e for e, _ in out], [p for _, p in out])
                out = []
    if out:
        total += m.extend([e for e, _ in out], [p for _, p in out])
    return total


rows = []
check = []
for i in reps:
    samples = [d for d in srv[i] if t0 <= d["t"] <= t1]
    ext_t = sorted(ext[i])
    for d0, d1 in zip(samples, samples[1:]):
        ta, tb = d0["t"], d1["t"]
        dt = tb - ta
        if dt <= 0:
            continue
        # replay check at the sample instant
        m = (arr[i][:, 0] < tb) & (arr[i][:, 1] > ta)
        idx = np.nonzero(m)[0]
        cands = [dec[i][k] for k in idx]
        spans = [live[i][k] for k in idx]
        run_now = running(cands, ta)
        check.append((d0["num_running_reqs"], len(run_now), d0["decode_sum_seq_lens"],
                      sum(c for c, _ in run_now)))
        subs = [ta + dt * (k + 0.5) / a.sub for k in range(a.sub)]
        busy = sum(any(s0 <= t < s1 for s0, s1 in spans) for t in subs) * dt / a.sub
        if busy < 0.5 * dt:
            continue                      # mostly idle: too little time to compare against
        snaps = [s for s in (running(cands, t) for t in subs) if s]
        dec_time = dec_tokens = 0.0
        naive = 0.0
        for s in snaps:
            b = len(s)
            rate = sum(r for _, r in s)                # tokens/s the batch produced
            steps = rate / b * (dt / a.sub)            # decode forwards in this slice
            ctx = [int(c) for c, _ in s]
            dec_time += steps * models["full"].decode(ctx)
            dec_tokens += rate * dt / a.sub
            mean = int(np.mean(ctx))
            naive += steps * models["full"].decode([mean] * b)
        lo, hi = bisect.bisect_left(ext_t, ta), bisect.bisect_left(ext_t, tb)
        batches = [ext[i][ext_t[k]] for k in range(lo, hi)]
        ext_time = sum(extend_cost(models["full"], items) for items in batches)
        rows.append(dict(replica=i, t=ta - t0, dt=busy, window=dt, batch=np.mean([len(s) for s in snaps]),
                         ctx_mean=np.mean([np.mean([c for c, _ in s]) for s in snaps]),
                         ctx_max=np.mean([max(c for c, _ in s) for s in snaps]),
                         decode=dec_time, decode_uniform=naive, extend=ext_time,
                         n_extend=len(batches), tokens=dec_tokens,
                         srv_gen=d1["gen_throughput"], srv_running=d0["num_running_reqs"]))

chk = np.array(check, float)
busy = chk[:, 0] > 0
print(f"{conv.workload.source}: {len(obs)} requests, {len(rows)} intervals busy >= half the time "
      f"({len(reps)} replica(s), ~{np.median([r['dt'] for r in rows]):.2f} s each)")
rel = np.abs(chk[busy, 1] - chk[busy, 0]) / chk[busy, 0]
rel_s = np.abs(chk[busy, 3] - chk[busy, 2]) / np.maximum(chk[busy, 2], 1)
print(f"{routing} vs srv-metrics: running count off by p50 {np.median(rel):.1%} "
      f"p90 {np.percentile(rel, 90):.1%}; summed context p50 {np.median(rel_s):.1%} "
      f"p90 {np.percentile(rel_s, 90):.1%}")

R = {k: np.array([r[k] for r in rows]) for k in rows[0]}
for name, num in (("cost model (decode + extend)", R["decode"] + R["extend"]),
                  ("  decode only", R["decode"]),
                  ("  uniform batch at mean ctx + extend", R["decode_uniform"] + R["extend"]),
                  ("  uniform batch at mean ctx, no extend (§8 as first written)",
                   R["decode_uniform"])):
    ratio = num / R["dt"]
    print(f"{name:62s} priced / actual  p10 {np.percentile(ratio, 10):.2f}  "
          f"p50 {np.median(ratio):.2f}  p90 {np.percentile(ratio, 90):.2f}")
steady = R["batch"] >= 100
for name, m in (("steady (batch >= 100)", steady), ("tail (batch < 100)", ~steady)):
    if m.any():
        ratio = (R["decode"] + R["extend"])[m] / R["dt"][m]
        print(f"{name:22s} n={m.sum():5d} priced/actual p50 {np.median(ratio):.2f}; time shares "
              f"decode {np.sum(R['decode'][m]) / np.sum(R['dt'][m]):.2f}, extend "
              f"{np.sum(R['extend'][m]) / np.sum(R['dt'][m]):.2f}; batch {np.median(R['batch'][m]):.0f}, "
              f"ctx mean {np.median(R['ctx_mean'][m]):.0f} max {np.median(R['ctx_max'][m]):.0f}, "
              f"extends/s {np.median(R['n_extend'][m] / R['dt'][m]):.1f}")
if a.out:
    a.out.parent.mkdir(parents=True, exist_ok=True)
    with a.out.open("w") as f:
        for r in rows:
            f.write(json.dumps({k: float(v) for k, v in r.items()}) + "\n")
