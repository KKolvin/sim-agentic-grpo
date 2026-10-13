"""How long one forward of one replica takes (SIMULATOR.md §8).

Fitted on a measurement grid (scripts/measure_cost_grid.py), never on a run trace. The
form follows what the grid showed and what the attention kernel does:

DECODE::

    T = routing(B) * base(B) + attention(ctx_1..ctx_B)

- ``base(B)``: the step without attention over the context -- weights, MoE, the rest. One
  value per measured batch size, interpolated between them.
- ``attention``: how trtllm-gen's decode kernel runs. Every sequence gets the same number
  of CTAs per KV head, ``splits = max(1, SMs // (B * kv_heads_per_rank))``, capped at
  ``ceil(max ctx / (2 * 128))`` (flashinfer fmhaKernels.cuh, numCtasPerSeqKv). The CTAs go
  onto the SMs in batch order, one per SM, a new one starting when one finishes; the ones
  running share the HBM bandwidth (``kv`` seconds per token of context, all KV heads) but
  none goes faster than one CTA can (``cta`` seconds per token of one head). With uniform
  batches this is bandwidth-bound, ~3.2 ns/token at TP=4, close to the HBM limit. From
  B >= 75 at TP=4 a sequence is one CTA, so one 64k request makes the batch wait for it:
  127 x 2k + one 64k goes 7.5 -> 20 ms. ``splits`` is the kernel's rule, not fitted.
- ``routing(B)``: grid points fill their context without prefilling it, which routes MoE
  tokens differently from real text; this is real-text prefill / filled-KV at the check
  points.

EXTEND::

    T = max(floor, a + b * new_tokens + c * sum(new * (prefix + new / 2)), d + e * longest)

``floor`` is the eager forward: decode replays a CUDA graph, extend launches ~800 kernels
one by one. The middle term is compute. The last is the context-attention critical path:
with few new tokens a sequence gets few CTAs and each reads its whole prefix, so an
EXTEND of 256 tokens on a 64k prefix takes 45 ms against the 34 ms floor (~0.55 us per
token of the longest sequence at TP=4).

A step outside the measured range is priced anyway and counted in :attr:`CostModel.outside`;
the engine decides whether too many of them void the prediction (§8: 5% by default).
"""

from __future__ import annotations

import bisect
import heapq
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class ModelShape:
    layers: int
    hidden: int
    vocab: int
    experts: int
    topk: int
    moe_inter: int
    q_heads: int
    kv_heads: int
    head_dim: int
    dtype_bytes: int = 2

    @classmethod
    def from_config(cls, path: str | Path) -> "ModelShape":
        c = json.loads((Path(path) / "config.json").read_text() if Path(path).is_dir()
                       else Path(path).read_text())
        return cls(layers=c["num_hidden_layers"], hidden=c["hidden_size"], vocab=c["vocab_size"],
                   experts=c["num_experts"], topk=c["num_experts_per_tok"],
                   moe_inter=c["moe_intermediate_size"], q_heads=c["num_attention_heads"],
                   kv_heads=c["num_key_value_heads"], head_dim=c["head_dim"])

    def kv_heads_per_rank(self, tp: int) -> int:
        return max(1, self.kv_heads // tp)      # fewer KV heads than ranks: replicated

    def kv_bytes_per_token(self, tp: int) -> int:
        return self.layers * 2 * self.kv_heads_per_rank(tp) * self.head_dim * self.dtype_bytes

    def weight_bytes(self, batch: int, tp: int) -> float:
        """Per GPU per decode step, counting the experts a batch touches under uniform routing."""
        h, d = self.hidden, self.head_dim
        attn = h * self.q_heads * d + 2 * h * self.kv_heads * d + self.q_heads * d * h
        touched = self.experts * (1 - (1 - self.topk / self.experts) ** batch)
        per_layer = attn + touched * 3 * h * self.moe_inter + h * self.experts
        return self.dtype_bytes * (self.layers * per_layer + self.vocab * h) / tp


STEP_KV = 128   # KV tokens per CTA tile in the kernels SGLang picks for this head size


def kv_splits(batch: int, kv_heads_per_rank: int, sms: int, max_ctx: int | None = None) -> int:
    s = max(1, sms // (batch * kv_heads_per_rank))
    if max_ctx is not None:
        s = min(s, max(1, -(-int(max_ctx) // (2 * STEP_KV))))
    return s


def attention_time(seq_lens, kv_heads_per_rank: int, sms: int, kv: float, cta: float) -> float:
    """Decode attention over the context: the kernel's CTAs on the SMs, bandwidth shared."""
    h = kv_heads_per_rank
    s = kv_splits(len(seq_lens), h, sms, max(seq_lens))
    works = [c / s for c in seq_lens for _ in range(h * s)]
    total_rate, one_rate = h / kv, 1 / cta          # head-tokens per second
    t = v_done = 0.0
    heap: list[float] = []
    q = 0
    while q < len(works) and len(heap) < sms:
        heapq.heappush(heap, works[q])
        q += 1
    while heap:
        rate = min(one_rate, total_rate / len(heap))
        fin = heapq.heappop(heap)
        t += (fin - v_done) / rate
        v_done = fin
        if q < len(works):
            heapq.heappush(heap, v_done + works[q])
            q += 1
    return t


@dataclass
class Calibration:
    tp: int
    sms: int
    hbm_bw: float
    gpu: str
    base: list[tuple[int, float]]              # (batch, seconds)
    kv: float                                  # seconds per token of context, HBM-bound
    cta: float                                 # seconds per token of one head on one CTA
    routing: list[tuple[int, float]]           # (batch, factor)
    ext_floor: float
    ext_a: float
    ext_b: float                               # seconds per new token
    ext_c: float                               # seconds per (new token x context token)
    max_batch: int
    max_ctx: int
    max_extend_tokens: int
    ext_d: float = 0.0
    ext_e: float = 0.0                         # seconds per token of the longest sequence
    source: str = ""
    fit_report: dict = field(default_factory=dict)

    def save(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps(asdict(self), indent=1))

    @classmethod
    def load(cls, path: str | Path) -> "Calibration":
        d = json.loads(Path(path).read_text())
        d["base"] = [tuple(x) for x in d["base"]]
        d["routing"] = [tuple(x) for x in d["routing"]]
        return cls(**d)


def _interp(table: list[tuple[int, float]], x: float) -> float:
    xs = [p[0] for p in table]
    i = bisect.bisect_left(xs, x)
    if i < len(xs) and xs[i] == x:
        return table[i][1]
    if i == 0:
        return table[0][1]
    if i == len(xs):
        return table[-1][1]
    (x0, y0), (x1, y1) = table[i - 1], table[i]
    return y0 + (y1 - y0) * (x - x0) / (x1 - x0)


class CostModel:
    def __init__(self, cal: Calibration, shape: ModelShape):
        self.cal, self.shape = cal, shape
        self.kvh = shape.kv_heads_per_rank(cal.tp)
        self.outside = 0
        self.priced = 0

    def decode(self, seq_lens) -> float:
        c = self.cal
        b = len(seq_lens)
        mx = max(seq_lens)
        self.priced += 1
        if b > c.max_batch or mx > c.max_ctx:
            self.outside += 1
        attn = attention_time(seq_lens, self.kvh, c.sms, c.kv, c.cta)
        return _interp(c.routing, b) * _interp(c.base, b) + attn

    def extend(self, extend_lens, prefix_lens) -> float:
        c = self.cal
        n = sum(extend_lens)
        self.priced += 1
        if n > c.max_extend_tokens or max(p + e for p, e in zip(prefix_lens, extend_lens)) > c.max_ctx:
            self.outside += 1
        pairs = sum(e * (p + e / 2) for p, e in zip(prefix_lens, extend_lens))
        longest = max(p + e for p, e in zip(prefix_lens, extend_lens))
        return max(c.ext_floor, c.ext_a + c.ext_b * n + c.ext_c * pairs,
                   c.ext_d + c.ext_e * longest)

    def step(self, mode: str, seq_lens, extend_lens=(), prefix_lens=()) -> float:
        if mode == "DECODE":
            return self.decode(seq_lens)
        if mode in ("EXTEND", "MIXED"):
            return self.extend(extend_lens, prefix_lens)
        return 0.0

    def price(self, st) -> float:
        """Seconds for one forward the stubbed scheduler launched (an engine ``Price``)."""
        return self.step(st.mode, st.seq_lens, st.extend_lens, st.prefix_lens)

    @property
    def outside_share(self) -> float:
        """Share of priced forwards that fell outside the measured grid (§8 refuses above 5%)."""
        return self.outside / self.priced if self.priced else 0.0


# ---------------------------------------------------------------------------
# fitting
# ---------------------------------------------------------------------------
def _seq_lens(r: dict, meta: dict) -> list[int]:
    """Context per row at the middle of a grid point's timed decode steps."""
    shift = meta["decode_warm"] + meta["decode_steps"] / 2
    if "ctxs" in r:            # prefixes; the EXTEND before decoding added one token
        return [c + 1 + shift for c in r["ctxs"]]
    if r["kind"] == "one_long":
        n_short = r["batch"] - r["n_long"]
        return [r["ctx_target"] + 1 + shift] * n_short + [65024 + 1 + shift] * r["n_long"]
    return [r["ctx_mean"] + shift] * r["batch"]


def fit(grid_dir: str | Path, checks_dir: str | Path | None, shape: ModelShape,
        hbm_bw: float, sms: int, extend_dirs=()) -> Calibration:
    """``base``, ``kv``, ``cta`` on uniform batches and batches with one 64k request; the
    lognormal ``spread`` points stay out, for :func:`evaluate`. EXTEND separately."""
    from scipy.optimize import least_squares

    grid_dir = Path(grid_dir)
    meta = json.loads((grid_dir / "meta.json").read_text())
    rows = [json.loads(line) for line in (grid_dir / "grid.jsonl").open()]
    tp = meta["server_args"]["tp_size"]
    kvh = shape.kv_heads_per_rank(tp)

    uni = [r for r in rows if r["kind"] == "grid" and r["mode"] == "DECODE"]
    ol = [r for r in rows if r["kind"] == "one_long"]
    seq = {id(r): _seq_lens(r, meta) for r in uni + ol}
    t = {id(r): r["gpu_med"] for r in uni + ol}
    bs = sorted({r["batch"] for r in uni})

    def bases(attn):
        # per batch size, the intercept that minimises relative error over its uniform points
        out = []
        for b in bs:
            rr = [r for r in uni if r["batch"] == b]
            out.append((b, sum((1 - attn[id(r)] / t[id(r)]) / t[id(r)] for r in rr)
                        / sum(1 / t[id(r)] ** 2 for r in rr)))
        return out

    def score(kv, cta):
        attn = {k: attention_time(v, kvh, sms, kv, cta) for k, v in seq.items()}
        base = bases(attn)
        err = [(_interp(base, len(seq[id(r)])) + attn[id(r)]) / t[id(r)] - 1 for r in uni + ol]
        return float(np.mean(np.square(err))), base

    # coarse grid, then a finer one around the best point
    kv_c = np.linspace(1.5e-9, 6e-9, 19) * kvh
    cta_c = np.geomspace(5e-8, 1e-6, 19)
    best = min((score(k, c)[0], k, c) for k in kv_c for c in cta_c)
    kv_f = np.linspace(best[1] * 0.85, best[1] * 1.15, 13)
    cta_f = np.geomspace(best[2] * 0.8, best[2] * 1.25, 13)
    best = min((score(k, c)[0], k, c) for k in kv_f for c in cta_f)
    _, kv, cta = best
    base = score(kv, cta)[1]

    cal = Calibration(tp=tp, sms=sms, hbm_bw=hbm_bw, gpu=meta["gpu"], base=base, kv=float(kv),
                      cta=float(cta), routing=[(1, 1.0)], ext_floor=0, ext_a=0, ext_b=0, ext_c=0,
                      max_batch=max(bs), max_ctx=65024 + 64,
                      max_extend_tokens=meta["server_args"]["chunked_prefill_size"],
                      source=str(grid_dir))

    # routing(B): real-text prefill over filled KV, averaged over the check seeds.
    if checks_dir is not None and (Path(checks_dir) / "grid.jsonl").exists():
        ck = [json.loads(line) for line in (Path(checks_dir) / "grid.jsonl").open()]
        by: dict = {}
        for r in ck:
            if r["kind"] == "check":
                by.setdefault((r["batch"], r["kv"]), []).append(r["gpu_med"])
        routing = [(b, float(np.mean(by[(b, "prefill_real")]) / np.mean(by[(b, "random")])))
                   for b in sorted({k[0] for k in by})
                   if (b, "prefill_real") in by and (b, "random") in by]
        if routing:
            cal.routing = routing

    # EXTEND: relative least squares over every point, extra grids included.
    ext = _extend_rows(grid_dir, extend_dirs)
    if ext:
        n = np.array([r["batch"] * r["extend"] for r in ext], float)
        pairs = np.array([r["batch"] * r["extend"] * (r["prefix"] + r["extend"] / 2) for r in ext])
        longest = np.array([r["prefix"] + r["extend"] for r in ext], float)
        te = np.array([r["gpu_med"] for r in ext])

        def model(x):
            return np.maximum.reduce([np.full_like(te, x[0]), x[1] + x[2] * n + x[3] * pairs,
                                      x[4] + x[5] * longest])
        res = least_squares(lambda x: model(x) / te - 1,
                            x0=[0.03, 0.0, 4e-6, 1e-10, 0.0, 5e-7],
                            x_scale=[0.01, 0.01, 1e-6, 1e-10, 0.01, 1e-7])
        (cal.ext_floor, cal.ext_a, cal.ext_b, cal.ext_c, cal.ext_d, cal.ext_e) = map(float, res.x)
        cal.max_ctx = max(cal.max_ctx, int(longest.max()))
    return cal


def _extend_rows(grid_dir, extend_dirs) -> list[dict]:
    out = []
    for d in (grid_dir, *extend_dirs):
        f = Path(d) / "grid.jsonl"
        if f.exists():
            out += [r for r in map(json.loads, f.open())
                    if r["mode"] == "EXTEND" and r["kind"] == "grid"]
    return out


def evaluate(cal: Calibration, shape: ModelShape, grid_dir: str | Path, extend_dirs=()) -> dict:
    """Relative error per kind of grid point. ``spread`` is held out of every fit."""
    grid_dir = Path(grid_dir)
    meta = json.loads((grid_dir / "meta.json").read_text())
    rows = [json.loads(line) for line in (grid_dir / "grid.jsonl").open()]
    out = {}
    # Grid points fill their KV, so they are compared without the routing factor.
    m = CostModel(Calibration(**{**asdict(cal), "routing": [(1, 1.0)]}), shape)
    for kind in ("grid", "spread", "one_long"):
        dec = [r for r in rows if r["kind"] == kind and r["mode"] == "DECODE"]
        if not dec:
            continue
        rel = np.array([m.decode(_seq_lens(r, meta)) / r["gpu_med"] - 1 for r in dec])
        out[f"decode/{kind}"] = dict(n=len(rel), p10=float(np.percentile(rel, 10)),
                                     p50=float(np.percentile(rel, 50)),
                                     p90=float(np.percentile(rel, 90)),
                                     mean_abs=float(np.mean(np.abs(rel))))
    ext = _extend_rows(grid_dir, extend_dirs)
    if ext:
        rel = np.array([m.extend([r["extend"]] * r["batch"], [r["prefix"]] * r["batch"])
                        / r["gpu_med"] - 1 for r in ext])
        out["extend"] = dict(n=len(rel), p10=float(np.percentile(rel, 10)),
                             p50=float(np.percentile(rel, 50)), p90=float(np.percentile(rel, 90)),
                             mean_abs=float(np.mean(np.abs(rel))))
    return out
