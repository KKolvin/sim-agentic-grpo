"""Fit the cost model (SIMULATOR.md §8) on a measurement grid and say how well it fits.

    python scripts/fit_cost_model.py analysis/costgrid/b200-tp4 \\
        --checks analysis/costgrid/b200-tp4-checks --out analysis/costgrid/b200-tp4/calibration.json

`spread` points (lognormal batches) are never fitted on; their error is the held-out check
that the form, not just the parameters, is right.
"""

import argparse
import json

from simulator.engine.cost_model import CostModel, ModelShape, evaluate, fit, kv_splits

ap = argparse.ArgumentParser()
ap.add_argument("grid")
ap.add_argument("--checks", default=None)
ap.add_argument("--extend-extra", nargs="*", default=[], help="more grids with EXTEND points")
ap.add_argument("--hbm-bw", type=float, default=8e12, help="bytes/s per GPU (B200: 8 TB/s)")
ap.add_argument("--sms", type=int, default=None, help="SM count; the grid's meta.json has it")
ap.add_argument("--out", default=None)
a = ap.parse_args()

meta = json.load(open(f"{a.grid}/meta.json"))
sms = a.sms or meta.get("sms")
if not sms:
    raise SystemExit("SM count unknown: pass --sms (B200: 148)")
shape = ModelShape.from_config(meta["server_args"]["model_path"])
cal = fit(a.grid, a.checks, shape, a.hbm_bw, sms, a.extend_extra)
tp = cal.tp
kvb = shape.kv_bytes_per_token(tp)
print(f"TP={tp} on {cal.gpu}, {sms} SMs, {kvb // 1024} KiB of KV per token per GPU")
print(f"kv  {cal.kv * 1e9:.2f} ns/token  -> KV read at {kvb / cal.kv / 1e12:.2f} TB/s "
      f"(eta_kv {kvb / cal.kv / a.hbm_bw:.2f})")
print(f"cta {cal.cta * 1e9:.1f} ns/token/head -> one CTA reads {kvb / shape.kv_heads_per_rank(tp) / cal.cta / 1e9:.0f} GB/s; "
      f"splits at B=1/32/64/128: " + "/".join(str(kv_splits(b, shape.kv_heads_per_rank(tp), sms))
                                           for b in (1, 32, 64, 128)))
print("base(B) ms and eta_w:  " + "  ".join(
    f"{b}:{t * 1e3:.1f}/{shape.weight_bytes(b, tp) / a.hbm_bw / t:.2f}" for b, t in cal.base
    if b in (1, 8, 32, 64, 128, 224)))
print("routing(B):  " + "  ".join(f"{b}:{f:.3f}" for b, f in cal.routing))
print(f"extend: floor {cal.ext_floor * 1e3:.1f} ms; above it {cal.ext_a * 1e3:.1f} ms + "
      f"{cal.ext_b * 1e6:.2f} us/token + {cal.ext_c * 1e12:.3f} ps per new x context token; "
      f"longest-sequence path {cal.ext_d * 1e3:.1f} ms + {cal.ext_e * 1e9:.0f} ns/token")
cal.fit_report = evaluate(cal, shape, a.grid, a.extend_extra)
for k, v in cal.fit_report.items():
    print(f"  {k:16s} n={v['n']:4d}  rel err p10 {v['p10']:+.3f}  p50 {v['p50']:+.3f}  "
          f"p90 {v['p90']:+.3f}  mean |err| {v['mean_abs']:.3f}")
if a.out:
    cal.save(a.out)
