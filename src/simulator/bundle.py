"""The config bundle: what system a workload runs on (SIMULATOR.md §5).

The IR says what work there is; the bundle says what runs it: the recipe (how many
worker processes, how many slots each gives every pool, how requests are capped), the
SGLang servers (version, replicas, the server args the scheduler decides with, the
environment knobs it reads), and the cost model's calibration. ``priors`` lives in the
same file but is read by :func:`simulator.adapters.priors.load_priors`.

Nothing here has a default. SGLang derives several of its defaults from the GPU's
memory, so a "usual value" would be invented, and the bundle has to be one file that
reproduces the prediction. Loading checks that every item is there and shaped right;
:mod:`simulator.engine.system` checks the values against SGLang and the calibration.
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import yaml

from simulator.ir import IRError, Workload, validate

#: Server args the scheduler decides with, plus the ones the calibration must match.
#: Names are SGLang's own (``ServerArgs`` fields), as a live server reports them.
REQUIRED_ARGS = (
    "model_path", "tp_size", "context_length", "max_total_tokens", "page_size",
    "chunked_prefill_size", "max_prefill_tokens", "max_running_requests", "schedule_policy",
    "schedule_conservativeness", "disable_overlap_schedule",
    "attention_backend", "moe_runner_backend", "kv_cache_dtype",
)

#: SGLang environment variables the scheduler reads: how much KV admission reserves for
#: the tokens a running request has yet to write (``new_token_ratio``).
REQUIRED_ENV = (
    "SGLANG_INIT_NEW_TOKEN_RATIO", "SGLANG_MIN_NEW_TOKEN_RATIO_FACTOR",
    "SGLANG_NEW_TOKEN_RATIO_DECAY_STEPS",
)

_SECTIONS = ("recipe", "server", "calibration", "priors")
_WHERE_SERVER = ("read them off the live server (GET /get_server_info) or its startup log; "
                 "SGLang derives several from GPU memory, so they have no safe default")


class BundleError(ValueError):
    pass


@dataclass(frozen=True)
class RequestCap:
    """The ``max_new_tokens`` a harness gives a request whose trace did not record one:
    ``min(max_new_tokens, max_sequence_tokens - prompt, context_length - 1 - prompt)``.
    verl's agent loop: ``response_length`` and ``prompt_length + response_length``."""

    max_new_tokens: int
    max_sequence_tokens: int


@dataclass(frozen=True)
class Recipe:
    workers: int                   # processes that each hold their own pool slots
    pools: Mapping[str, int]       # pool name (as the workload names it) -> slots per worker
    request_cap: RequestCap


@dataclass(frozen=True)
class Server:
    sglang_version: str
    replicas: int
    args: Mapping[str, Any]        # SGLang ServerArgs fields
    env: Mapping[str, Any]         # SGLANG_* environment variables

    @property
    def overlap(self) -> bool:
        return not self.args["disable_overlap_schedule"]


@dataclass(frozen=True)
class Bundle:
    path: Path
    recipe: Recipe
    server: Server
    calibration: Path              # calibration.json; resolved against the bundle's directory


def load_bundle(path: str | Path) -> Bundle:
    path = Path(path)
    doc = yaml.safe_load(path.read_text()) or {}
    if not isinstance(doc, Mapping):
        raise BundleError(f"{path}: a config bundle is a mapping of sections")
    _keys(doc, _SECTIONS, ("recipe", "server", "calibration"), f"{path}")

    rc = _section(doc, "recipe", path)
    _keys(rc, ("workers", "pools", "request_cap"), ("workers", "pools", "request_cap"),
          f"{path}: recipe")
    pools = rc["pools"]
    if not isinstance(pools, Mapping) or not pools:
        raise BundleError(f"{path}: recipe.pools maps each pool the workload names to its "
                          f"slots per worker")
    cap = rc["request_cap"]
    _keys(cap, ("max_new_tokens", "max_sequence_tokens"), ("max_new_tokens", "max_sequence_tokens"),
          f"{path}: recipe.request_cap")
    recipe = Recipe(
        workers=_count(rc["workers"], f"{path}: recipe.workers"),
        pools={str(k): _count(v, f"{path}: recipe.pools.{k}") for k, v in pools.items()},
        request_cap=RequestCap(**{k: _count(cap[k], f"{path}: recipe.request_cap.{k}")
                                  for k in ("max_new_tokens", "max_sequence_tokens")}))

    sv = _section(doc, "server", path)
    _keys(sv, ("sglang_version", "replicas", "args", "env"),
          ("sglang_version", "replicas", "args", "env"), f"{path}: server")
    args, env = sv["args"] or {}, sv["env"] or {}
    if not isinstance(args, Mapping) or not isinstance(env, Mapping):
        raise BundleError(f"{path}: server.args and server.env are mappings")
    missing = [k for k in REQUIRED_ARGS if k not in args]
    if missing:
        raise BundleError(f"{path}: server.args is missing {missing}; {_WHERE_SERVER}")
    missing = [k for k in REQUIRED_ENV if k not in env]
    if missing:
        raise BundleError(f"{path}: server.env is missing {missing}; state SGLang's own "
                          f"default if the server did not set them")
    server = Server(sglang_version=str(sv["sglang_version"]),
                    replicas=_count(sv["replicas"], f"{path}: server.replicas"),
                    args=dict(args), env=dict(env))

    cal = Path(doc["calibration"])
    cal = cal if cal.is_absolute() else (path.parent / cal).resolve()
    if not cal.is_file():
        raise BundleError(f"{path}: calibration {cal} does not exist; fit one with "
                          f"scripts/fit_cost_model.py on a grid measured for this TP")
    return Bundle(path=path, recipe=recipe, server=server, calibration=cal)


def fill_caps(w: Workload, b: Bundle) -> Workload:
    """Give every request without a recorded ``max_new_tokens`` the recipe's cap.

    A request whose trace says it wrote more than that cap was not run under this
    config, so the workload is refused rather than the cap stretched (§11)."""
    cap, ctx = b.recipe.request_cap, int(b.server.args["context_length"])

    def one(r):
        if r.max_new_tokens is not None:
            return r
        n = max(0, min(cap.max_new_tokens, cap.max_sequence_tokens - r.prompt_tokens,
                       ctx - 1 - r.prompt_tokens))
        return dataclasses.replace(r, max_new_tokens=n, prov={**r.prov, "max_new_tokens": "config"})

    out = dataclasses.replace(w, sessions=tuple(
        dataclasses.replace(s, requests=tuple(one(r) for r in s.requests)) for s in w.sessions))
    try:
        validate(out)
    except IRError as e:
        raise BundleError(f"{b.path}: with the recipe's request_cap the workload is invalid "
                          f"({e}); the trace was not run under this config") from e
    return out


def _section(doc: Mapping, name: str, path: Path) -> Mapping:
    sec = doc[name]
    if not isinstance(sec, Mapping):
        raise BundleError(f"{path}: {name} must be a mapping")
    return sec


def _keys(m: Mapping, allowed, required, what: str) -> None:
    if not isinstance(m, Mapping):
        raise BundleError(f"{what} must be a mapping")
    stray = sorted(set(m) - set(allowed))
    if stray:
        raise BundleError(f"{what}: {stray} is not one of {list(allowed)}")
    missing = [k for k in required if k not in m]
    if missing:
        raise BundleError(f"{what} is missing {missing}")


def _count(v: Any, what: str) -> int:
    if isinstance(v, bool) or not isinstance(v, int) or v < 1:
        raise BundleError(f"{what} must be a positive integer, got {v!r}")
    return v
