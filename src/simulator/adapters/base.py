"""The adapter contract: the one layer allowed to know what a trace looks like.

A trace arrives in whatever shape its harness wrote it. Everything downstream
reads :mod:`simulator.ir` and nothing else, so something has to translate, and
`SIMULATOR.md` §13 confines that knowledge to this package: one adapter reads one
format, and no code outside ``simulator.adapters`` opens a trace file.

Reading a trace is a sorting job more than a conversion. The same event that says
how many tokens a turn generated also says how many of them the server found in
its cache and when it first saw the request. The first is workload and goes in
the IR. The other two are what the system did under that load, and go in
:mod:`simulator.observed`, where the engine cannot import them (§3). Only the code
that reads the line can do the sorting, and only it knows which line became which
:class:`~simulator.ir.Request`, so pairing an answer to its request by id has to
happen here as well. :func:`check` holds every adapter to that pairing.

An adapter may not invent. A field the trace does not record gets an
:class:`~simulator.ir.Unknown`, and the only place that Unknown may come from is a
:class:`Prior` the caller supplied, which names the run it was measured on.
:func:`check` refuses two shortcuts that would make the degradation test of §15
pass for nothing: an estimate drawn from a distribution nobody supplied, and a
prior measured on the very trace being converted. Strip a field from a trace,
fill it back from that trace's own numbers, and the estimate is the answer.

A trace with no answers in it is the normal case, not an error. At use time
(`SIMULATOR.md` §1) the adapter returns an empty :class:`~simulator.observed.Observations`
and the rest of the pipeline must run on the IR alone (§13, rule 6).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Mapping, Protocol, Sequence

from simulator.ir import Dist, Unknown, Workload, validate
from simulator.observed import Observations


class AdapterError(ValueError):
    """A trace no adapter can read, or a conversion that breaks the contract."""


#: The knob a :class:`~simulator.ir.Task` of each ``kind`` draws its duration
#: from when the trace did not record one. Shared across adapters so that a prior
#: measured on one harness's workload can stand in for another's: the row label
#: in the sensitivity table (`SIMULATOR.md` §6) is the same whoever asks.
KNOB_BY_KIND: Mapping[str, str] = {
    "tool": "tool_duration",
    "container_start": "container_start",
    "eval": "eval_duration",
    "patch_recover": "patch_recover",
    "cleanup": "cleanup",
}


@dataclass(frozen=True)
class Prior:
    """A borrowed distribution for a field some trace does not record.

    :attr:`knob` is the sensitivity-table row this field will answer to, so it is
    also the name an adapter asks for: ``"tool_duration"``, ``"container_start"``,
    ``"eval_duration"``. :attr:`dist` and :attr:`scale` are what the resulting
    :class:`~simulator.ir.Unknown` carries, in the sense `SIMULATOR.md` §6 gives
    them: the shape of one value, and how far off the whole shape might be.

    :attr:`source` is where it was measured, as a run id or a harness name. It is
    required because a prior that cannot say where it came from is an invented
    number with a distribution wrapped around it, and because :func:`check` uses
    it to refuse a prior borrowed from the trace being converted. A source spelled
    ``run:<id>/<part>``, as the adapters spell a workload's, names the run before
    the first slash, and :func:`check` compares runs, not whole strings: step 2 of
    a run is not an independent measurement for step 1, nor its conversation dump
    for its timeline.

    :attr:`scale_sources` are the runs :attr:`scale` was measured across, for the
    same refusal: a scale stretched to reach the trace being converted has seen
    the answer. Empty when the scale is a claim rather than a measurement.

    :attr:`by_key` is the same knob measured per problem
    (:attr:`~simulator.ir.Session.problem`): a field on a session whose problem
    is in the table draws from that problem's own values, every other field from
    :attr:`dist`. It exists because some work belongs to the problem, not the
    run. One SWE-bench instance's tests score in 15 s, another's in 770 s, and on
    run 20261006-030010 a handful of such instances carried 17-24% of each
    step's eval seconds; drawn from one pooled shape, whether they were in the
    batch at all was lost, and the step-to-step swing it caused sat outside any
    scale measured across runs. The knob stays one knob: one row in the
    sensitivity table, one :attr:`scale` over the table and :attr:`dist` alike.
    :attr:`key_sources` are the runs the table was measured on, refused by
    :func:`check` exactly like :attr:`source`.

    :attr:`default` is True when nobody stated this prior and the simulator's
    default (:func:`simulator.adapters.priors.default_priors`) stands in. It
    changes nothing about how the prior is used; it is there so the provenance
    report can say "you did not give this, we used our run" (`SIMULATOR.md` §5).
    """

    knob: str
    dist: Dist
    source: str
    scale: Dist | None = None
    scale_sources: tuple[str, ...] = ()
    default: bool = False
    by_key: Mapping[str, Dist] = field(default_factory=dict)
    key_sources: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not self.knob:
            raise AdapterError("a prior needs a knob name")
        if not self.source:
            raise AdapterError(f"prior {self.knob!r} does not say where it was measured")
        if self.by_key and not self.key_sources:
            raise AdapterError(f"prior {self.knob!r} has a per-problem table that does not say "
                               f"which runs it was measured on")

    def unknown(self, key: str | None = None) -> Unknown:
        """What an adapter hangs on the field of a session working on problem ``key``."""
        dist = self.by_key.get(key, self.dist) if key is not None else self.dist
        return Unknown(dist=dist, knob=self.knob, scale=self.scale)

    def sources(self) -> tuple[str, ...]:
        """Every run this prior's numbers were measured on: shape, scale and table."""
        return (self.source, *self.scale_sources, *self.key_sources)


@dataclass(frozen=True)
class Priors:
    """Every distribution a conversion may draw on, by knob name.

    Supplied by the caller, never assembled by an adapter: which run's tool times
    stand in for a trace that recorded none is a decision about the experiment,
    and it has to be visible in the call, not buried in a reader. The usual caller
    is :func:`simulator.adapters.priors.resolve_priors`, which takes whatever the
    config bundle states and fills the rest from our own run.
    """

    by_knob: Mapping[str, Prior] = field(default_factory=dict)

    @classmethod
    def of(cls, *priors: Prior) -> "Priors":
        out: dict[str, Prior] = {}
        for p in priors:
            if p.knob in out:
                raise AdapterError(f"two priors named {p.knob!r}")
            out[p.knob] = p
        return cls(out)

    def require(self, knob: str, *, missing: str, key: str | None = None) -> Unknown:
        """The Unknown for a field the trace lacks, or a refusal that says what to supply.

        ``key`` is the session's :attr:`~simulator.ir.Session.problem`; a prior with
        that problem in its :attr:`~Prior.by_key` answers with the problem's own
        values. Pass it whenever the session has one: :func:`check` refuses a
        field that drew from the pooled shape when the table knew its problem.

        ``missing`` is the thing the trace does not have, in words, so the error
        reads "the trace does not record tool durations and no prior named
        'tool_duration' was supplied" rather than a bare key error. Refusing here
        is `SIMULATOR.md` §11: no number is better than a made-up one.
        """
        p = self.by_knob.get(knob)
        if p is None:
            known = sorted(self.by_knob) or "none"
            raise AdapterError(
                f"the trace does not record {missing} and no prior named {knob!r} was "
                f"supplied; known priors: {known}"
            )
        return p.unknown(key)

    def keyed(self, knob: str, key: str | None) -> bool:
        """Whether ``knob``'s prior has its own values for problem ``key``, for an adapter's notes."""
        p = self.by_knob.get(knob)
        return p is not None and key is not None and key in p.by_key

    def sources(self) -> frozenset[str]:
        return frozenset(p.source for p in self.by_knob.values())


def estimated_notes(estimated: Mapping[str, int], keyed: Mapping[str, int]) -> list[str]:
    """One note per knob an adapter estimated: how many fields, and how many drew their problem's own values."""
    out = []
    for knob, n in sorted(estimated.items()):
        k = keyed.get(knob, 0)
        out.append(f"{n} durations estimated from prior {knob!r}"
                   + (f", {k} of them from their problem's own measurements" if k else ""))
    return out


def same_run(a: str, b: str) -> bool:
    """Whether two sources name the same run: the part before the first ``/``, see :class:`Prior`."""
    return a.split("/", 1)[0] == b.split("/", 1)[0]


@dataclass(frozen=True)
class Conversion:
    """What reading one trace produces.

    :attr:`workload` is the IR, with :attr:`~simulator.ir.Request.prov` filled for
    every field the adapter could source and an :class:`~simulator.ir.Unknown` on
    every one it could not. :attr:`observations` is the answer half, empty when
    the trace holds none. :attr:`notes` is the human-readable remainder: what the
    trace did not have and what was done about it, one sentence each, for the
    provenance report of `SIMULATOR.md` §11.
    """

    adapter: str
    workload: Workload
    observations: Observations
    notes: tuple[str, ...] = ()


class Adapter(Protocol):
    """One trace format.

    :meth:`claims` is a cheap look, a file name or a first line, and must not
    raise on a path it does not recognise. :meth:`convert` does the reading.
    Anything a format needs beyond the path, such as the tokenizer and chat
    template a conversation-only trace needs to rebuild token counts, is
    constructor state on the adapter, not a parameter of this method.
    """

    name: str

    def claims(self, path: Path) -> bool: ...

    def convert(self, path: Path, priors: Priors) -> Conversion: ...


def detect(path: Path, adapters: Sequence[Adapter]) -> Adapter:
    """The one adapter that recognises ``path``.

    Exactly one: a trace with two readers would give two different workloads
    depending on registration order, and a trace with none gets the list of
    formats that were tried rather than a guess.
    """
    claimants = [a for a in adapters if a.claims(path)]
    if not claimants:
        known = [a.name for a in adapters] or "none"
        raise AdapterError(f"no adapter recognises {path}; known formats: {known}")
    if len(claimants) > 1:
        names = [a.name for a in claimants]
        raise AdapterError(f"{names} all claim {path}; a trace must have one reader")
    return claimants[0]


def check(conv: Conversion, priors: Priors) -> None:
    """Hold a conversion to the contract. Raises :class:`AdapterError` or :class:`~simulator.ir.IRError`.

    In order: the IR is valid; the observed half names the same source, so a
    workload cannot be diffed against somebody else's timeline; every observed
    request and session pairs with a node the IR actually has; no prior's shape,
    scale or per-problem table was measured on the run being converted; and every
    Unknown in the workload is exactly one of the supplied priors, drawn from the
    session's problem's own values whenever the prior has them, so an adapter
    cannot quietly estimate a field from a distribution it made up or read off
    the trace itself, nor ignore a table it was handed.
    """
    w, o = conv.workload, conv.observations
    validate(w)
    if o.source != w.source:
        raise AdapterError(
            f"{conv.adapter!r} read workload {w.source!r} but its observations say "
            f"{o.source!r}; the two halves of one trace must name the same source"
        )
    borrowed_from_self = sorted(k for k, p in priors.by_knob.items()
                                if any(same_run(s, w.source) for s in p.sources()))
    if borrowed_from_self:
        raise AdapterError(
            f"priors {borrowed_from_self} were measured on the run of {w.source!r}, the trace "
            f"being converted; a stripped field filled from its own run is the answer, not an "
            f"estimate. State these knobs in the config bundle, or build priors from another run"
        )
    request_ids = {r.id for s in w.sessions for r in s.requests}
    for r in o.requests:
        if r.id not in request_ids:
            raise AdapterError(
                f"observed request {r.id!r} has no request in the workload to pair with"
            )
    session_ids = {s.id for s in w.sessions}
    for s in o.sessions:
        if s.id not in session_ids:
            raise AdapterError(
                f"observed session {s.id!r} has no session in the workload to pair with"
            )
    carriers = [*((n, s.problem) for s in w.sessions for n in s.nodes()), *((p, None) for p in w.prefixes)]
    for carrier, problem in carriers:
        for name, u in carrier.unknown.items():
            p = priors.by_knob.get(u.knob)
            if p is None:
                raise AdapterError(
                    f"{type(carrier).__name__.lower()} {carrier.id!r} estimates {name!r} under "
                    f"knob {u.knob!r}, which no supplied prior provides; an adapter does not "
                    f"bring its own distributions"
                )
            if u != p.unknown(problem):
                whose = (f"problem {problem!r}'s own values" if problem in p.by_key
                         else "the pooled shape")
                raise AdapterError(
                    f"{type(carrier).__name__.lower()} {carrier.id!r} estimates {name!r} under "
                    f"knob {u.knob!r} with a distribution that is not the supplied prior's "
                    f"({whose})"
                )
