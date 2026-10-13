"""Which replica a request goes to: verl's router, copied (SIMULATOR.md §7)."""

from __future__ import annotations


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
