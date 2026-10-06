"""The adapter: read any trace with :mod:`~simulator.adapters.shapes`, resolve it with
:mod:`~simulator.adapters.trace`. See those two modules; this one only wires them.

A known harness changes nothing here: it is a :class:`~simulator.adapters.trace.Profile`
in :data:`PROFILES`, picked by its ``match`` on the path, and a trace no profile
matches is read with :data:`~simulator.adapters.trace.GENERIC`, which assumes nothing.
"""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from typing import Any, Mapping

from simulator.adapters import shapes
from simulator.adapters.agentic_messages import AGENTIC
from simulator.adapters.base import AdapterError, Conversion, Priors
from simulator.adapters.trace import GENERIC, Profile, Tokenizer, resolve

#: Harness profiles tried in order before :data:`GENERIC`.
PROFILES: tuple[Profile, ...] = (AGENTIC,)


def _hf_tokenizer(model: str) -> Tokenizer:
    try:
        from transformers import AutoTokenizer  # heavy; only when a count must come from text
    except ImportError as exc:  # pragma: no cover
        raise AdapterError("deriving token counts from text needs `transformers` for the tokenizer") from exc
    return AutoTokenizer.from_pretrained(model)


class TraceAdapter:
    """Any trace. ``profile`` forces one instead of matching; ``where`` keeps only rows whose
    top-level fields equal the given values (a list field matches when it contains the value),
    for dumps that mix benchmarks or models; ``limit`` stops after that many sessions."""

    def __init__(self, profile: Profile | None = None, tokenizer: Tokenizer | None = None,
                 prompt_length: int | None = None, where: Mapping[str, Any] | None = None,
                 limit: int | None = None, name: str = "trace"):
        self.profile, self.tokenizer, self.prompt_length = profile, tokenizer, prompt_length
        self.where, self.limit, self.name = dict(where or {}), limit, name

    def profile_for(self, path: Path) -> Profile:
        if self.profile is not None:
            return self.profile
        return next((p for p in PROFILES if p.match(path)), GENERIC)

    def claims(self, path: Path) -> bool:
        return shapes.claims(path, self.profile_for(path).files)

    def convert(self, path: Path, priors: Priors) -> Conversion:
        profile = self.profile_for(path)
        fs = shapes.files(path, profile.files)
        if not fs:
            raise AdapterError(f"no trace files under {path} (looked for {profile.files} with {shapes.SUFFIXES})")
        got = shapes.read(fs, self.where, self.limit)
        if not got.sessions:
            raise AdapterError(f"no row of {path} reads as a conversation, OTel chat spans or per-turn counts"
                               + (f" among those matching {self.where}" if self.where else ""))
        run = path.name if path.is_dir() else path.parent.name
        source = f"run:{run}/{profile.part}"
        conv = resolve(got.sessions, got.header, profile, priors, source=source, tokenizer=self.tokenizer,
                       load_tokenizer=_hf_tokenizer, prompt_length=self.prompt_length, adapter=self.name,
                       meta={"run": run, "files": got.files, "shapes": dict(got.shapes),
                             "model": got.header.model if got.header else None, "where": self.where})
        extra = [f"rows read as: " + ", ".join(f"{n} {k}" for k, n in got.shapes.items())]
        if got.unrecognised:
            extra.append(f"{sum(got.unrecognised.values())} rows in {len(got.unrecognised)} files have no messages, "
                         f"spans or token counts (failed requests, other records) and were skipped")
        return replace(conv, notes=conv.notes + tuple(extra))
