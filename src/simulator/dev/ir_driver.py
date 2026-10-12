"""A realised IR workload driven through stubbed schedulers (SIMULATOR.md §15 5(c)).

Not the engine yet: every forward costs the same `step_s`, and pools are flat counters
(no per-worker semaphores). What it does have is the part 5(c) is about: token ids built
from the IR's structure (:mod:`simulator.tokens`), each turn's prompt extending the ids its
parent actually produced, and verl's router, so the real scheduler decides what the radix
cache serves. Its `cached_tokens` per request is what gets compared with the run's.

Generic over the IR: requests, tasks, `after`, `offset_s`, leases and task pools. It knows
nothing of what a session is for.
"""

from __future__ import annotations

import heapq
import itertools
import time
from collections import defaultdict, deque
from dataclasses import dataclass, field

from simulator.engine.sglang_stub import Step, StubScheduler
from simulator.ir import Request, Task, Workload, require_runnable
from simulator.tokens import TokenIds


class Router:
    """verl's GlobalRequestLoadBalancer (verl/workers/rollout/llm_server.py): a key seen
    before goes back to its replica, otherwise to the replica with the fewest requests in
    flight, first in order on a tie. The LRU of 10,000 keys never evicts at our sizes."""

    def __init__(self, n: int):
        self.inflight = [0] * n
        self.sticky: dict[str, int] = {}

    def acquire(self, key: str) -> int:
        i = self.sticky.get(key)
        if i is None:
            i = self.inflight.index(min(self.inflight))
            self.sticky[key] = i
        self.inflight[i] += 1
        return i

    def release(self, i: int) -> None:
        self.inflight[i] -= 1


@dataclass
class RequestResult:
    replica: int
    t_submit: float
    t_finish: float = -1.0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    cached_tokens: int = 0
    reason: str = ""


@dataclass
class Result:
    requests: dict[str, RequestResult] = field(default_factory=dict)
    steps: list[list[Step]] = field(default_factory=list)
    end: float = 0.0
    wall_s: float = 0.0
    short: dict[str, int] = field(default_factory=dict)
    retracted: list[int] = field(default_factory=list)


_ARRIVE, _STEP = 0, 1   # arrivals before steps at the same instant, as in stub_driver


def run(w: Workload, replicas: list[StubScheduler], pools: dict[str, int], *,
        step_s: float = 0.044) -> Result:
    require_runnable(w)
    if any(s.spawned_by for s in w.sessions):
        raise NotImplementedError("subagent sessions are the engine's job (5(d))")
    tok = TokenIds(w)
    router = Router(len(replicas))
    res = Result(steps=[[] for _ in replicas])

    node = {n.id: n for s in w.sessions for n in s.nodes()}
    waiting_on = {nid: len(n.after) for nid, n in node.items()}
    dependents = defaultdict(list)
    for nid, n in node.items():
        for d in n.after:
            dependents[d].append(nid)
    lease_at = defaultdict(list)       # node -> pools to acquire before it starts
    lease_end = defaultdict(list)      # node -> pools to release after it completes
    for s in w.sessions:
        for ls in s.leases:
            lease_at[ls.acquire_before].append(ls.pool)
            lease_end[ls.release_after].append(ls.pool)
    for p in {p for s in w.sessions for ls in s.leases for p in [ls.pool]} | \
             {t.pool for s in w.sessions for t in s.tasks if t.pool}:
        if p not in pools:
            raise ValueError(f"workload uses pool {p!r} but the config gives it no size")
    free = dict(pools)
    queue: dict[str, deque] = defaultdict(deque)   # pool -> (node id, pools still to take)

    seq = itertools.count()
    ev: list = []
    awake = [False] * len(replicas)
    where: dict[str, int] = {}
    now = 0.0

    def push(t, kind, x):
        heapq.heappush(ev, (t, kind, next(seq), x))

    def ready(nid, t):
        n = node[nid]
        push(t + n.offset_s, _ARRIVE, ("acquire", nid, tuple(lease_at[nid]) +
                                       ((n.pool,) if isinstance(n, Task) and n.pool else ())))

    def acquire(nid, need, t):
        for i, p in enumerate(need):
            if free[p] > 0 and not queue[p]:
                free[p] -= 1
            else:
                queue[p].append((nid, need[i + 1:]))
                return
        start(nid, t)

    def release(p, t):
        if queue[p]:
            nid, rest = queue[p].popleft()
            acquire(nid, rest, t)
        else:
            free[p] += 1

    def start(nid, t):
        n = node[nid]
        if isinstance(n, Task):
            push(t + n.duration_s, _ARRIVE, ("done", nid, None))
            return
        r: Request = n
        i = router.acquire(r.session)
        where[nid] = i
        ids = tok.prompt(nid)
        res.requests[nid] = RequestResult(replica=i, t_submit=t, prompt_tokens=len(ids))
        replicas[i].submit(nid, ids, r.completion_tokens, r.max_new_tokens)
        if not awake[i]:
            awake[i] = True
            push(t, _STEP, i)

    def done(nid, t):
        n = node[nid]
        if isinstance(n, Task) and n.pool:
            release(n.pool, t)
        for p in lease_end[nid]:
            release(p, t)
        for d in dependents[nid]:
            waiting_on[d] -= 1
            if not waiting_on[d]:
                ready(d, t)

    for s in w.sessions:
        for n in s.nodes():
            if not n.after:
                ready(n.id, s.release_offset_s)

    t0 = time.time()
    while ev:
        now, kind, _, x = heapq.heappop(ev)
        if kind == _ARRIVE:
            what, nid, need = x
            if what == "acquire":
                acquire(nid, need, now)
            else:
                done(nid, now)
            continue
        i, rep = x, replicas[x]
        st = rep.step()
        end = now + (step_s if st else 0.0)
        if st:
            res.steps[i].append(st)
        for f in rep.finished():
            rr = res.requests[f.rid]
            rr.t_finish, rr.completion_tokens = end, f.completion_tokens
            rr.cached_tokens, rr.reason = f.cached_tokens, f.reason
            router.release(where.pop(f.rid))
            tok.finish(f.rid, f.output_ids)
            push(end, _ARRIVE, ("done", f.rid, None))
        if st or rep.busy:
            push(end, _STEP, i)
        else:
            awake[i] = False
    res.end, res.wall_s = now, time.time() - t0
    res.short = dict(tok.short)
    res.retracted = [r.s.num_retracted_reqs for r in replicas]
    unfinished = [nid for nid, k in waiting_on.items() if k] + \
                 [nid for nid, rr in res.requests.items() if rr.t_finish < 0]
    if unfinished:
        raise RuntimeError(f"{len(unfinished)} nodes never ran, e.g. {unfinished[:3]}")
    return res
