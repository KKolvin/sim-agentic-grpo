"""Tables from one cost grid (scripts/measure_cost_grid.py): eta over batch x context, uneven
batches, one long request, the fake-KV check, and EXTEND times.

eta = (weight bytes + KV bytes) / (HBM bandwidth x step time), per GPU (SIMULATOR.md §8).
Weight bytes count the experts a batch is expected to touch under uniform top-k routing;
KV bytes are the batch's summed context. Both come from the model's config.json.

    python scripts/simulator_dev/summarize_cost_grid.py analysis/costgrid/b200-tp4
"""

import argparse
import collections
import json
from pathlib import Path

import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("grid", type=Path)
ap.add_argument("--bw", type=float, default=8e12, help="HBM bytes/s per GPU (B200: 8 TB/s)")
a = ap.parse_args()

meta = json.loads((a.grid / "meta.json").read_text())
sa = meta["server_args"]
tp = sa["tp_size"]
cfg = json.loads((Path(sa["model_path"]) / "config.json").read_text())
L, H, V = cfg["num_hidden_layers"], cfg["hidden_size"], cfg["vocab_size"]
E, K, I = cfg["num_experts"], cfg["num_experts_per_tok"], cfg["moe_intermediate_size"]
nq, nkv, hd = cfg["num_attention_heads"], cfg["num_key_value_heads"], cfg["head_dim"]
attn = H * nq * hd + 2 * H * nkv * hd + nq * hd * H
kv_heads_per_rank = max(1, nkv // tp)         # fewer KV heads than ranks: replicated


def w_bytes(b):
    touched = E * (1 - (1 - K / E) ** b)
    return 2 * (L * (attn + touched * 3 * H * I + H * E) / tp + V * H / tp)


def kv_bytes(sum_ctx):
    return sum_ctx * L * 2 * kv_heads_per_rank * hd * 2


rows = [json.loads(line) for line in (a.grid / "grid.jsonl").open()]
print(f"{a.grid}: TP={tp}, {meta['gpu']}, pool {meta['max_total_num_tokens']:,} tokens, "
      f"{len(rows)} rows")

dec = [r for r in rows if r["kind"] == "grid" and r["mode"] == "DECODE"]
if dec:
    tab = collections.defaultdict(dict)
    for r in dec:
        mid = r["ctx_mean"] + 1 + meta["decode_warm"] + meta["decode_steps"] / 2
        b = r["batch"]
        tab[b][r["ctx_target"]] = ((w_bytes(b) + kv_bytes(b * mid)) / a.bw / r["gpu_med"],
                                   r["gpu_med"] * 1e3)
    ctxs = sorted({c for d in tab.values() for c in d})
    for i, name in ((0, "eta"), (1, "decode step, ms")):
        print(f"\n{name}  (rows: batch, columns: context)")
        print("     " + "".join(f"{c:>7}" for c in ctxs))
        for b in sorted(tab):
            print(f"{b:4d} " + "".join(f"{tab[b][c][i]:7.2f}" if c in tab[b] else "      ."
                                       for c in ctxs))

sp = collections.defaultdict(dict)
for r in rows:
    if r["kind"] == "spread":
        sp[(r["batch"], r["ctx_target"])][r["sigma"]] = (r["gpu_med"], r["ctx_max"])
if sp:
    print("\nuneven batches: step time / uniform step time at the same mean, by lognormal sigma "
          "(longest request in brackets, k tokens)")
    for k in sorted(sp):
        base = sp[k].get(0.0, (None,))[0]
        if base:
            print(f"  B={k[0]:3d} mean {k[1]:5d}: " + "  ".join(
                f"{s}: {t / base:4.2f} [{m // 1024}k]" for s, (t, m) in sorted(sp[k].items())))

ol = {(r["batch"], r["ctx_target"], r["n_long"]): r["gpu_med"] for r in rows
      if r["kind"] == "one_long"}
if ol:
    print("\none 64k request among short ones: step ms without -> with")
    for (b, c, n), t in sorted(ol.items()):
        if n == 0 and (b, c, 1) in ol:
            print(f"  B={b:3d} rest at {c:5d}: {t * 1e3:6.2f} -> {ol[(b, c, 1)] * 1e3:6.2f} ms")

ch = [r for r in rows if r["kind"] == "check"]
pairs = collections.defaultdict(dict)
for r in ch:
    pairs[(r["batch"], r["ctx_target"])][r["fake_prefix"]] = r["gpu_med"]
if pairs:
    print("\nfake KV / real prefill, same batch:")
    for k, d in sorted(pairs.items()):
        if True in d and False in d:
            print(f"  B={k[0]:3d} ctx {k[1]:5d}: {d[True] * 1e3:6.2f} / {d[False] * 1e3:6.2f} ms "
                  f"= {d[True] / d[False]:.3f}")

ext = [r for r in rows if r["mode"] == "EXTEND" and r["kind"] == "grid"]
if ext:
    small = [r["gpu_med"] for r in ext if r["batch"] * r["extend"] <= 1024]
    print(f"\nEXTEND of <= 1024 new tokens: {len(small)} points, p10/p50/p90 "
          + "/".join(f"{x * 1e3:.1f}" for x in np.percentile(small, [10, 50, 90])) + " ms")
    big = sorted(((r["batch"] * r["extend"], r["prefix"], r["gpu_med"]) for r in ext
                  if r["batch"] * r["extend"] >= 4096), key=lambda x: x[:2])
    print("EXTEND of >= 4096 new tokens (tokens, prefix: ms): " + ", ".join(
        f"{n}@{p}: {t * 1e3:.0f}" for n, p, t in big if p in (0, 16384)))
