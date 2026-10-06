"""Priors measured on one of our runs, for converting a trace that recorded less.

The degradation test of `SIMULATOR.md` §15 strips durations out of a trace and
asks the adapter to estimate them. The distributions it estimates from have to
come from somewhere, and the honest somewhere is another run of a harness that
did record them: convert that run with its own adapter, then read the traced task
durations off the resulting workload. :func:`~simulator.adapters.base.check`
refuses the result if that other run turns out to be the one being converted.

The same steps build the use-time default: ``scripts/simulator_dev/build_default_priors.py``
measures our run, adds the per-problem tables and stretches a scale across our
other runs and steps with :func:`priors_from_runs`, and freezes it with
:func:`priors_to_json`. Reading it back is use time, so that half lives in
:mod:`simulator.adapters.priors`.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import replace
from typing import Any, Iterable, Mapping, Sequence

from simulator.adapters.base import KNOB_BY_KIND, AdapterError, Prior, Priors, same_run
from simulator.ir import Dist, Empirical, Uniform, Workload

#: Knobs worth measuring per problem: the work whose length the problem decides.
#: An instance's tests take what they take whichever policy wrote the patch;
#: tool calls and container starts move with the machine, and a per-problem table
#: for them would only thin out the samples.
KEYED_KNOBS: tuple[str, ...] = ("eval_duration",)

#: Fewest measurements a problem needs for its own row: one GRPO group is eight.
MIN_PER_PROBLEM = 4


def priors_from_workload(w: Workload, *, scale: Dist | None = None,
                         knob_by_kind: Mapping[str, str] = KNOB_BY_KIND) -> Priors:
    """One :class:`Prior` per task kind, from the durations ``w`` actually measured.

    Only traced or derived durations count, and only ones that are not a timeout
    cap: an estimate built on estimates would launder provenance, and a killed
    tool's cap is not how long tools take. ``scale`` is the §6 "how wrong might
    this whole shape be" distribution; it is the caller's claim to make, since it
    comes from comparing runs, which one workload cannot do. ``None`` claims the
    shape is right.
    """
    return Priors.of(*(Prior(knob=k, dist=Empirical(tuple(v for _, v in xs)), source=w.source, scale=scale)
                       for k, xs in sorted(samples(w, knob_by_kind).items())))


def samples(w: Workload, knob_by_kind: Mapping[str, str] = KNOB_BY_KIND) -> dict[str, list[tuple[str | None, float]]]:
    """Every measured duration in ``w``, by knob, with the problem its session worked on.

    The same filter as :func:`priors_from_workload`: traced or derived, and not a
    timeout cap.
    """
    out: dict[str, list[tuple[str | None, float]]] = defaultdict(list)
    for s in w.sessions:
        for t in s.tasks:
            knob = knob_by_kind.get(t.kind)
            if knob is None or t.duration_is_cap:
                continue
            if t.prov.get("duration_s") in ("traced", "derived"):
                out[knob].append((s.problem, t.duration_s))
    return out


def priors_from_runs(base: Workload, others: Sequence[Workload], *, keyed: Iterable[str] = KEYED_KNOBS,
                     min_n: int = MIN_PER_PROBLEM) -> tuple[Priors, dict[str, dict[str, dict[str, float]]]]:
    """Priors whose shape is ``base``'s, with per-problem tables and a scale from ``others``.

    ``others`` are workloads of other runs, and of other steps of them: a step is
    one draw of problems, and what a scale has to cover is a different draw, not
    only a different run of the same one. Every run of ours trains on the same
    problems in the same order, so step 1 of six runs is six measurements of 256
    problems, and a scale stretched across them alone missed step 3 of
    20261006-030010 by 9-25%.

    Each knob in ``keyed`` gets :attr:`~simulator.adapters.base.Prior.by_key`: the
    values of every problem measured at least ``min_n`` times across ``base`` and
    ``others``, drawn discretely (:attr:`~simulator.ir.Empirical.discrete`). Its
    scale is then what the table still gets wrong: for each other workload, its
    measured total over what the table predicts for its problems, with the
    workload's own run left out of the table, since a table that holds the
    answer predicts it perfectly. Problems the table lacks predict the pooled
    mean. An unkeyed knob's ratio is the plain ratio of means, as
    :func:`scale_from_runs` takes it.

    Returns the priors and, per knob and source, the ``ratio`` and the share of
    that source's fields the left-out table ``covered``, for whoever reports them.
    """
    keyed = set(keyed)
    measured = {w.source: samples(w) for w in (base, *others)}
    out, report = [], {}
    for knob, p in sorted(priors_from_workload(base).by_knob.items()):
        m0 = _mean(p.dist)
        per_source = {src: m.get(knob, []) for src, m in measured.items()}
        table = _table(per_source.values(), min_n) if knob in keyed else {}
        ratios: dict[str, dict[str, float]] = {}
        for w in others:
            xs = per_source[w.source]
            if not xs or not m0 > 0:
                continue
            means = {k: _mean(d) for k, d in _table(
                (v for src, v in per_source.items() if not same_run(src, w.source)), min_n).items()} \
                if table else {}
            predicted = sum(means.get(k, m0) if k is not None else m0 for k, _ in xs)
            ratios[w.source] = {"ratio": sum(v for _, v in xs) / predicted,
                                "covered": sum(k in means for k, _ in xs) / len(xs)}
        rs = [r["ratio"] for r in ratios.values()]
        lo, hi = min([1.0, *rs]), max([1.0, *rs])
        scale = Uniform(lo, hi) if hi > lo else None
        key_sources = tuple(src for src, xs in per_source.items() if any(k in table for k, _ in xs))
        out.append(replace(p, scale=scale, scale_sources=tuple(ratios) if scale else (),
                           by_key=table, key_sources=key_sources))
        report[knob] = ratios
    return Priors.of(*out), report


def parse_runs(specs: Sequence[str], default_step: int) -> list[tuple[str, int]]:
    """``RUN`` or ``RUN:1,2,3`` on a command line, as (run, step) pairs; a bare run is ``default_step``."""
    out = []
    for spec in specs:
        run, _, steps = spec.partition(":")
        out += [(run, int(st)) for st in steps.split(",")] if steps else [(run, default_step)]
    return out


def _table(groups: Iterable[list[tuple[str | None, float]]], min_n: int) -> dict[str, Empirical]:
    by: dict[str, list[float]] = defaultdict(list)
    for xs in groups:
        for k, v in xs:
            if k is not None:
                by[k].append(v)
    return {k: Empirical(tuple(v), discrete=True) for k, v in sorted(by.items()) if len(v) >= min_n}


def scale_from_runs(base: Priors, others: Mapping[str, Priors]) -> tuple[Priors, dict[str, dict[str, float]]]:
    """Give each of ``base``'s priors a scale: how far other runs' means sit from its own.

    The scale is ``Uniform(lo, hi)`` over the ratios of each other run's mean to
    ``base``'s, widened to include 1.0, or ``None`` when no other run measured the
    knob. Means, not medians, because what the simulator sums is durations. Two
    runs give a range, not a confidence interval: it is the spread we have seen,
    and a trace from a different harness may sit outside it. Also returns the
    ratios, ``{knob: {source: ratio}}``, for whoever reports them.
    """
    out, ratios = [], {}
    for knob, p in sorted(base.by_knob.items()):
        m0 = _mean(p.dist)
        r = {o.source: _mean(o.dist) / m0
             for o in (op.by_knob.get(knob) for op in others.values()) if o is not None and m0 > 0}
        lo, hi = min([1.0, *r.values()]), max([1.0, *r.values()])
        scale = Uniform(lo, hi) if hi > lo else None
        out.append(replace(p, scale=scale, scale_sources=tuple(r) if scale else ()))
        ratios[knob] = r
    return Priors.of(*out), ratios


def priors_to_json(priors: Priors, *, meta: Mapping[str, Any] | None = None) -> dict[str, Any]:
    """A :class:`Priors` of :class:`Empirical` dists and :class:`Uniform` scales, as JSON.

    The inverse is :func:`simulator.adapters.priors.priors_from_json`.
    """
    knobs = {}
    for knob, p in sorted(priors.by_knob.items()):
        if not isinstance(p.dist, Empirical):
            raise AdapterError(f"prior {knob!r}: only measured (Empirical) distributions are stored")
        if p.scale is not None and not isinstance(p.scale, Uniform):
            raise AdapterError(f"prior {knob!r}: only Uniform scales are stored")
        # Every value, not a quantile summary: quantiles evenly spaced in q
        # interpolate straight across the tail, and between tool_duration's p99.9
        # (12 s) and its max (300 s) that alone lifts the mean a draw sees by a third.
        knobs[knob] = {
            "source": p.source,
            "n": len(p.dist.values),
            "mean": _mean(p.dist),
            "values": [float(f"{v:.5g}") for v in p.dist.values],
            "scale": [p.scale.lo, p.scale.hi] if p.scale else None,
            "scale_sources": list(p.scale_sources),
        }
        if p.by_key:
            if not all(isinstance(d, Empirical) and d.discrete for d in p.by_key.values()):
                raise AdapterError(f"prior {knob!r}: only discrete Empirical per-problem tables are stored")
            knobs[knob]["by_key"] = {k: [float(f"{v:.5g}") for v in d.values]  # type: ignore[attr-defined]
                                     for k, d in sorted(p.by_key.items())}
            knobs[knob]["key_sources"] = list(p.key_sources)
    return {"meta": dict(meta or {}), "priors": knobs}


def _mean(d: Dist) -> float:
    xs = d.values  # type: ignore[attr-defined]
    return sum(xs) / len(xs) if xs else float("nan")
