"""Token ids for a workload, built from its structure (SIMULATOR.md §7 "token id 从哪来").

SGLang's radix cache matches content, not lengths, so a real scheduler can only reuse a
prefix if the ids are actually the same. The IR says how long things are and what extends
what; this turns that into id sequences:

- a :class:`~simulator.ir.Prefix` is its parent's ids followed by its own segment;
- a request's prompt is the first ``shared_prefix_tokens`` of its parent's content,
  followed by a fresh segment up to ``prompt_tokens``;
- a request's content, as the next turn sees it, is its prompt plus the output ids the
  scheduler actually produced. That part is only known after the request finishes, so
  :meth:`TokenIds.finish` hands it over.

Fresh segments are drawn from a generator seeded by (kind, id), so they never collide with
each other or with the stub's own output ids except by chance, and two runs over the same
workload build the same ids. Nothing here knows what a session is for.
"""

from __future__ import annotations

import hashlib
from collections import Counter

import numpy as np

from simulator.ir import NodeId, Prefix, Request, Workload

#: Fresh ids stay below every special token of the vocabularies we run (Qwen3's start at
#: 151643), the same bound the stub's output ids use.
ID_SPACE = 150_000


def _fresh(kind: str, key: str, n: int) -> np.ndarray:
    seed = int.from_bytes(hashlib.blake2b(f"{kind}|{key}".encode(), digest_size=8).digest(),
                          "little")
    return np.random.default_rng(seed).integers(0, ID_SPACE, n, dtype=np.int32)


class TokenIds:
    """Prompt ids for every request of a workload, as their parents finish.

    Content is kept only while a request still has children to build: a step of 2,048
    sessions at ~16k tokens per turn would otherwise hold ~3 GB of ids.

    ``short`` counts requests whose parent produced fewer tokens than the share asks for
    (an aborted parent); the missing tail is filled fresh, so those requests share less
    than the IR says.
    """

    def __init__(self, w: Workload):
        self._req: dict[NodeId, Request] = {r.id: r for s in w.sessions for r in s.requests}
        self._prefix: dict[str, Prefix] = {p.id: p for p in w.prefixes}
        self._prefix_ids: dict[str, np.ndarray] = {}
        self._content: dict[NodeId, np.ndarray] = {}
        self._prompt: dict[NodeId, np.ndarray] = {}
        self._children = Counter(r.prefix_parent for r in self._req.values()
                                 if r.prefix_parent in self._req)
        self.short: Counter[str] = Counter()

    def prefix(self, pid: str) -> np.ndarray:
        got = self._prefix_ids.get(pid)
        if got is None:
            p = self._prefix[pid]
            head = self.prefix(p.parent) if p.parent is not None else np.empty(0, np.int32)
            got = np.concatenate([head, _fresh("prefix", pid, p.tokens - len(head))])
            self._prefix_ids[pid] = got
        return got

    def prompt(self, rid: NodeId) -> list[int]:
        """The request's prompt ids. Its parent request, if any, must have finished."""
        r = self._req[rid]
        parent = r.prefix_parent
        if parent is None:
            head = np.empty(0, np.int32)
        elif parent in self._prefix:
            head = self.prefix(parent)[:r.shared_prefix_tokens]
        else:
            content = self._content.get(parent)
            if content is None:
                raise KeyError(f"request {rid!r} extends {parent!r}, which has not finished")
            head = content[:r.shared_prefix_tokens]
            self._children[parent] -= 1
            if not self._children[parent]:
                del self._content[parent]
        if len(head) < r.shared_prefix_tokens:
            self.short["parent_shorter_than_share"] += 1
        ids = np.concatenate([head, _fresh("request", rid, r.prompt_tokens - len(head))])
        if self._children.get(rid):
            self._prompt[rid] = ids
        return ids.tolist()

    def finish(self, rid: NodeId, output_ids) -> None:
        """Record what the scheduler produced, for the requests that extend this one."""
        prompt = self._prompt.pop(rid, None)
        if prompt is not None:
            self._content[rid] = np.concatenate([prompt, np.asarray(output_ids, np.int32)])
