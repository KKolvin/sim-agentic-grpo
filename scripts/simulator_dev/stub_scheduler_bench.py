"""SIMULATOR.md §15 5(b): wall time of one simulated rollout through stubbed SGLang schedulers.

The workload has the shape of run 20260912-072938 step 1 (2,048 chains of mean 23.5
requests, first prompt 2,507 tokens, 463 new tokens per follow-up, 162 output tokens)
split over its two TP=4 replicas. The step time is a constant: this measures the
simulator, not the system.

    python scripts/simulator_dev/stub_scheduler_bench.py --chains 2048 --replicas 2
"""

import argparse
import collections
import time

from simulator.dev.stub_driver import run, synthetic
from simulator.engine.sglang_stub import StubScheduler, stub_server_args

ap = argparse.ArgumentParser()
ap.add_argument("--model", default="/data0/shared/Qwen3-30B-A3B-Instruct-2507")
ap.add_argument("--chains", type=int, default=2048)
ap.add_argument("--replicas", type=int, default=2)
ap.add_argument("--no-overlap", action="store_true")
ap.add_argument("--max-wall", type=float, default=None)
a = ap.parse_args()

args = stub_server_args(model_path=a.model, page_size=64, max_running_requests=256,
                        max_total_tokens=4_560_512, context_length=65536,
                        chunked_prefill_size=8192, schedule_policy="fcfs",
                        disable_overlap_schedule=a.no_overlap, random_seed=0)
t0 = time.time()
reps = [StubScheduler(args) for _ in range(a.replicas)]
chains = synthetic(0, a.chains, mean_len=23.5, root=2507, new_mean=463, out_mean=162,
                   gap_mean=1.0)
# verl's router pins a chain to one replica; round-robin by chain stands in for it here
replica_of = {c.cid: reps[i % len(reps)] for i, c in enumerate(chains)}
print(f"{a.replicas} schedulers up in {time.time() - t0:.1f}s")

t0 = time.time()
traces, virt = run(chains, replica_of, max_wall_s=a.max_wall)
wall = time.time() - t0
steps = sum(len(t.steps) for t in traces.values())
print(f"{a.chains} chains, {sum(len(c.out) for c in chains)} requests, "
      f"{sum(sum(c.out) for c in chains)} output tokens")
print(f"{steps} forwards in {wall:.1f}s wall ({1e3 * wall / steps:.2f} ms per forward, all in); "
      f"virtual end {virt:.0f}s at a constant step time")
for i, r in enumerate(reps):
    t = traces[id(r)]
    modes = collections.Counter(st.mode for st in t.steps)
    reasons = collections.Counter(f.reason for f in t.finished)
    print(f"  replica {i}: {len(t.steps)} forwards {dict(modes)}, finished {dict(reasons)}, "
          f"retracted {r.s.num_retracted_reqs}")
