"""The engine's clock: a realised workload through stubbed SGLang replicas (SIMULATOR.md §2, §7).

Generic over the IR: requests, tasks, ``after``, ``offset_s``, leases and task pools. It
knows nothing of what a session is for. What it adds over :mod:`simulator.dev.ir_driver`
is the clock: every forward is priced by ``price`` (the cost model, or a constant), and
each replica's time follows SGLang's own event loop instead of a fixed step.

With zero CPU time per iteration, which is what a standalone server showed (§8: overlap
hid the scheduler's CPU), the two loops are:

- **normal** (``event_loop_normal``): an iteration at ``T`` forms batch *i*, runs it and
  processes its result. The batch ends at ``T + price(i)``; its finished requests leave
  then, and the next iteration starts then.
- **overlap** (``event_loop_overlap``): an iteration at ``T`` forms batch *i* and launches
  it, then processes batch *i-1*, which waits for that batch to end at ``E(i-1)``. Batch
  *i* starts on the GPU at ``max(T, E(i-1))``; batch *i-1*'s finished requests leave at
  that same moment, and the next iteration starts then. So each batch is formed while
  the one before it is still running, and a request that arrives during batch *i* is
  first seen by batch *i+2*.

A request's first token is when the forward that prefilled the last of its prompt has
its result processed: that is where SGLang stamps ``prefill_finished_ts``.

If the run's decode slowdown (§12) proves to be scheduler CPU that overlap does not
hide, a per-iteration CPU time goes into the two ``max``s above.
"""

from __future__ import annotations

import heapq
import itertools
import time
from collections import defaultdict, deque
from dataclasses import dataclass, field
from typing import Callable

from simulator.engine.router import Router
from simulator.engine.sglang_stub import Step, StubScheduler
from simulator.ir import Request, Task, Workload, require_runnable
from simulator.tokens import TokenIds

#: Seconds one forward takes, from what the scheduler put in it.
Price = Callable[[Step], float]


def constant(step_s: float) -> Price:
    """Every forward costs the same: for checking scheduling without a cost model."""
    return lambda st: step_s


@dataclass
class RequestResult:
    replica: int
    t_ready: float                 # its dependencies were done (plus offset)
    t_submit: float                # it reached the server, after any lease it waits for
    t_first_token: float = -1.0
    t_finish: float = -1.0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    cached_tokens: int = 0
    reason: str = ""


@dataclass
class TaskResult:
    t_ready: float
    t_start: float = -1.0          # after the pool slots it needs
    t_end: float = -1.0


@dataclass(frozen=True)
class Forward:
    t_start: float
    t_end: float
    step: Step


@dataclass
class Result:
    requests: dict[str, RequestResult] = field(default_factory=dict)
    tasks: dict[str, TaskResult] = field(default_factory=dict)
    forwards: list[list[Forward]] = field(default_factory=list)   # per replica, in order
    end: float = 0.0
    wall_s: float = 0.0
    short: dict[str, int] = field(default_factory=dict)
    retracted: list[int] = field(default_factory=list)


_ARRIVE, _STEP = 0, 1   # arrivals before steps at the same instant

#: Iterations in a row at one instant that launch nothing and process nothing while the
#: replica still has work: SGLang would spin forever, so it is a bug, not a wait.
_MAX_IDLE_SPINS = 3


def run(w: Workload, replicas: list[StubScheduler], pools: dict[str, int],
        price: Price) -> Result:
    require_runnable(w)
    if any(s.spawned_by for s in w.sessions):
        raise NotImplementedError("subagent sessions are not wired into the engine yet")
    tok = TokenIds(w)
    router = Router(len(replicas))
    res = Result(forwards=[[] for _ in replicas])

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
    for p in {ls.pool for s in w.sessions for ls in s.leases} | \
             {t.pool for s in w.sessions for t in s.tasks if t.pool}:
        if p not in pools:
            raise ValueError(f"workload uses pool {p!r} but the config gives it no size")
    free = dict(pools)
    queue: dict[str, deque] = defaultdict(deque)   # pool -> (node id, pools still to take)
    t_ready: dict[str, float] = {}

    seq = itertools.count()
    ev: list = []
    awake = [False] * len(replicas)
    gpu_free = [0.0] * len(replicas)               # when the last launched batch ends
    pending: list[Forward | None] = [None] * len(replicas)   # launched, result not processed
    spins = [0] * len(replicas)
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
            res.tasks[nid] = TaskResult(t_ready=t_ready[nid], t_start=t, t_end=t + n.duration_s)
            push(t + n.duration_s, _ARRIVE, ("done", nid, None))
            return
        r: Request = n
        i = router.acquire(r.session)
        where[nid] = i
        ids = tok.prompt(nid)
        res.requests[nid] = RequestResult(replica=i, t_ready=t_ready[nid], t_submit=t,
                                          prompt_tokens=len(ids))
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

    def first_tokens(f: Forward, t):
        if f.step.mode not in ("EXTEND", "MIXED"):
            return
        for rid, n in zip(f.step.rids, f.step.seq_lens):
            rr = res.requests[rid]
            if rr.t_first_token < 0 and n >= rr.prompt_tokens:
                rr.t_first_token = t

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
                t_ready[nid] = now
                acquire(nid, need, now)
            else:
                done(nid, now)
            continue
        i, rep = x, replicas[x]
        st = rep.step()
        if rep.overlap:
            t_out = max(now, gpu_free[i])
            processed, pending[i] = pending[i], None
            if st:
                f = Forward(t_out, t_out + price(st), st)
                gpu_free[i], pending[i] = f.t_end, f
            nxt = t_out
        else:
            processed = None
            if st:
                f = Forward(now, now + price(st), st)
                gpu_free[i], processed = f.t_end, f
            t_out = nxt = gpu_free[i] if st else now
        if st:
            res.forwards[i].append(f)
        if processed is not None:
            first_tokens(processed, t_out)
        finished = rep.finished()
        for fin in finished:
            rr = res.requests[fin.rid]
            rr.t_finish, rr.completion_tokens = t_out, fin.completion_tokens
            rr.cached_tokens, rr.reason = fin.cached_tokens, fin.reason
            router.release(where.pop(fin.rid))
            tok.finish(fin.rid, fin.output_ids)
            push(t_out, _ARRIVE, ("done", fin.rid, None))
        if st or rep.busy:
            spins[i] = spins[i] + 1 if not st and processed is None and not finished \
                and nxt == now else 0
            if spins[i] > _MAX_IDLE_SPINS:
                raise RuntimeError(f"replica {i} has work but schedules nothing at t={now:.3f}")
            push(nxt, _STEP, i)
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
