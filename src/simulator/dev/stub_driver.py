"""A synthetic closed-loop workload driven through stubbed schedulers (SIMULATOR.md §15 5(a)/(b)).

Not the engine: no router, no pools, a constant step time instead of a cost model. It
exists to answer two questions before the engine is built: can several stubbed
`Scheduler`s share a process without leaking into each other (5(a)), and how much wall
time does a simulated rollout cost (5(b)).
"""

from __future__ import annotations

import hashlib
import heapq
import itertools
import random
import time
from dataclasses import dataclass, field

from simulator.engine.sglang_stub import Finished, Step, StubScheduler


def _ids(key: str, n: int) -> list[int]:
    out, i = [], 0
    while len(out) < n:
        h = hashlib.blake2b(f"{key}:{i}".encode(), digest_size=32).digest()
        out.extend(int.from_bytes(h[j:j + 4], "little") % 150_000 for j in range(0, 32, 4))
        i += 1
    return out[:n]


@dataclass
class Chain:
    """Requests in order; request k's prompt is request k-1's prompt + output + `new[k]` fresh tokens."""

    cid: str
    root: list[int]                # shared with the rest of the group
    new: list[int]
    out: list[int]
    gap_s: list[float]             # wait after request k finishes before k+1 arrives
    group: str = ""


def synthetic(seed: int, n: int, *, mean_len: float, root: int, new_mean: float,
              out_mean: float, gap_mean: float, group_size: int = 8,
              max_len: int = 100) -> list[Chain]:
    rng = random.Random(seed)
    chains = []
    for i in range(n):
        g = f"{seed}-g{i // group_size}"
        k = min(max_len, max(1, int(rng.expovariate(1 / mean_len))))
        chains.append(Chain(
            cid=f"{seed}-c{i}", group=g, root=_ids(g, root),
            new=[0] + [max(1, int(rng.expovariate(1 / new_mean))) for _ in range(k - 1)],
            out=[max(1, int(rng.expovariate(1 / out_mean))) for _ in range(k)],
            gap_s=[rng.expovariate(1 / gap_mean) for _ in range(k)]))
    return chains


@dataclass
class Trace:
    steps: list[Step] = field(default_factory=list)
    finished: list[Finished] = field(default_factory=list)
    prompt_len: dict[str, int] = field(default_factory=dict)
    step_wall_s: list[float] = field(default_factory=list)


def run(chains: list[Chain], replica_of: dict[str, StubScheduler], *, step_s: float = 0.044,
        max_new_tokens: int = 4096, max_wall_s: float | None = None) -> tuple[dict, float]:
    """Closed loop on one virtual clock; every forward costs `step_s`.

    Arrivals sort before steps at the same instant, and an idle replica sleeps until an
    arrival wakes it, so a replica's event sequence does not depend on its neighbours.
    Returns per-replica traces and the virtual end time.
    """
    seq = itertools.count()
    ev: list = []
    traces = {id(r): Trace() for r in set(replica_of.values())}
    prompt = {c.cid: list(c.root) for c in chains}
    k_of = {c.cid: 0 for c in chains}
    chain_of = {c.cid: c for c in chains}
    awake = {id(r): False for r in replica_of.values()}
    for c in chains:
        heapq.heappush(ev, (0.0, 0, next(seq), c.cid))
    now, t0 = 0.0, time.time()
    while ev:
        now, kind, _, x = heapq.heappop(ev)
        if kind == 0:
            c, k = chain_of[x], k_of[x]
            rep = replica_of[x]
            prompt[x] = prompt[x] + _ids(f"{x}/{k}", c.new[k])
            rid = f"{x}/{k}"
            traces[id(rep)].prompt_len[rid] = len(prompt[x])
            rep.submit(rid, prompt[x], c.out[k], max_new_tokens)
            if not awake[id(rep)]:
                awake[id(rep)] = True
                heapq.heappush(ev, (now, 1, next(seq), rep))
            continue
        rep, tr = x, traces[id(x)]
        w0 = time.perf_counter()
        st = rep.step()
        tr.step_wall_s.append(time.perf_counter() - w0)
        end = now + (step_s if st else 0.0)
        if st:
            tr.steps.append(st)
        for f in rep.finished():
            tr.finished.append(f)
            cid, k = f.rid.rsplit("/", 1)
            if f.reason == "abort":
                continue
            prompt[cid] = prompt[cid] + list(f.output_ids)
            k_of[cid] = int(k) + 1
            c = chain_of[cid]
            if k_of[cid] < len(c.out):
                heapq.heappush(ev, (end + c.gap_s[int(k)], 0, next(seq), cid))
        if st or rep.busy:
            heapq.heappush(ev, (end, 1, next(seq), rep))
        else:
            awake[id(rep)] = False
        if max_wall_s and time.time() - t0 > max_wall_s:
            break
    return traces, now
