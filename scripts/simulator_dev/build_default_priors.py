#!/usr/bin/env python
"""Freeze our run's task durations into the simulator's default priors.

A config bundle that does not state a knob gets this file's distribution for it
(`SIMULATOR.md` §5, :func:`simulator.adapters.priors.resolve_priors`). The shape
comes from one step of one run (``--prior-run``). The other workloads
(``--scale-runs``, each ``RUN`` or ``RUN:1,2,3``) fill the per-problem tables of
the keyed knobs and give each knob its scale, exactly as the degradation
experiment builds its priors (:func:`simulator.dev.priors.priors_from_runs`).
Name several steps per run: the same step of every run draws the same problems.

Every run named here becomes unusable as a trace to convert under the defaults:
``check`` refuses a prior whose shape or scale was measured on the run being
converted. Converting one of these runs means stating its priors explicitly.

    python scripts/simulator_dev/build_default_priors.py --prior-run 20260912-072938 \\
        --scale-runs 20260912-072938:2 20261005-113503 20261005-143426 20261006-030010:1,2,3 \\
        20260903-002235:1,2,3 20260906-013757:1,2,3 20260908-031915 20260908-052218:1,2
"""

from __future__ import annotations

import argparse
import datetime
import json
import os
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))

from simulator.adapters import check  # noqa: E402
from simulator.adapters.base import Priors  # noqa: E402
from simulator.adapters.priors import DEFAULT_PRIORS_PATH  # noqa: E402
from simulator.dev.agentic_timeline import AgenticTimelineAdapter  # noqa: E402
from simulator.dev.priors import KEYED_KNOBS, parse_runs, priors_from_runs, priors_to_json  # noqa: E402
from simulator.ir import Workload  # noqa: E402

USER = os.environ.get("USER", "nobody")


def measured(timelines: Path, run: str, step: int) -> Workload:
    conv = AgenticTimelineAdapter(step=step).convert(timelines / run, Priors())
    check(conv, Priors())
    return conv.workload



def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--prior-run", required=True)
    ap.add_argument("--scale-runs", nargs="+", required=True)
    ap.add_argument("--step", type=int, default=1, help="the prior run's step, and the default for --scale-runs")
    ap.add_argument("--keyed", nargs="*", default=list(KEYED_KNOBS), help="knobs measured per problem")
    ap.add_argument("--timelines", type=Path, default=Path(f"/data0/shared/{USER}/agentic-timelines"))
    ap.add_argument("--out", type=Path, default=DEFAULT_PRIORS_PATH)
    args = ap.parse_args()

    base = measured(args.timelines, args.prior_run, args.step)
    others = [measured(args.timelines, run, step) for run, step in parse_runs(args.scale_runs, args.step)]
    priors, ratios = priors_from_runs(base, others, keyed=args.keyed)
    meta = {
        "what": "default priors for knobs a config bundle does not state (SIMULATOR.md §5)",
        "prior_run": args.prior_run, "scale_runs": args.scale_runs, "step": args.step,
        "built": datetime.date.today().isoformat(), "built_by": "scripts/simulator_dev/build_default_priors.py",
        "keyed": args.keyed,
        "ratio": {k: {s: round(v["ratio"], 3) for s, v in r.items()} for k, r in ratios.items()},
        "covered": {k: {s: round(v["covered"], 2) for s, v in r.items()} for k, r in ratios.items()
                    if priors.by_knob[k].by_key},
    }
    doc = priors_to_json(priors, meta=meta)
    args.out.write_text(json.dumps(doc, separators=(",", ":")) + "\n")

    print(f"{'knob':16} {'n':>6} {'mean':>8} {'p50':>8} {'p90':>8} {'problems':>8}  scale")
    for knob, p in sorted(priors.by_knob.items()):
        d = doc["priors"][knob]
        sc = f"[{p.scale.lo:.3f}, {p.scale.hi:.3f}]" if p.scale else "none"
        print(f"{knob:16} {d['n']:6d} {d['mean']:8.3f} {p.dist.quantile(0.5):8.3f} "
              f"{p.dist.quantile(0.9):8.3f} {len(p.by_key):8d}  {sc}")
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
