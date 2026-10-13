"""The system a bundle describes, built: stubbed replicas, pool sizes, a cost model (SIMULATOR.md §5).

Everything the bundle states is checked against what it will run on, and refused when it
does not match: the SGLang version against the one the stub was checked on, every server
arg against ``ServerArgs``, every environment knob against SGLang's, and the args the cost
model depends on against the ones the calibration's grid was measured with. A TP=4
calibration does not price a TP=8 server, and a grid measured on triton MoE does not price
a server running flashinfer_trtllm.
"""

from __future__ import annotations

import dataclasses
import json
from dataclasses import dataclass

from sglang.srt import environ
from sglang.srt.server_args import ServerArgs

from simulator.bundle import Bundle, BundleError
from simulator.engine.cost_model import Calibration, CostModel, ModelShape
from simulator.engine.sglang_stub import SGLANG_VERSION, StubScheduler, stub_server_args

#: Args that describe the real GPU server. The cost model depends on them, so they are
#: compared with the calibration's grid; the stub runs its own (one CPU rank, no kernels)
#: and never sees them.
COST_ARGS = ("tp_size", "attention_backend", "moe_runner_backend", "kv_cache_dtype",
             "disable_cuda_graph")

#: Args about the real server's process that the stub sets for itself.
STUB_ARGS = ("device", "skip_tokenizer_init", "load_format")


@dataclass
class System:
    replicas: list[StubScheduler]
    pools: dict[str, int]
    cost: CostModel
    bundle: Bundle


def build(b: Bundle) -> System:
    sv = b.server
    if sv.sglang_version != SGLANG_VERSION:
        raise BundleError(f"{b.path}: the servers ran SGLang {sv.sglang_version}; the stub is "
                          f"checked against {SGLANG_VERSION} and would run another scheduler")
    fields = {f.name for f in dataclasses.fields(ServerArgs)}
    stray = sorted(set(sv.args) - fields)
    if stray:
        raise BundleError(f"{b.path}: server.args {stray} are not SGLang {SGLANG_VERSION} server args")
    for k in sv.env:
        if not isinstance(getattr(environ.envs, k, None), environ.EnvField):
            raise BundleError(f"{b.path}: server.env {k} is not an SGLang environment knob")

    cal = Calibration.load(b.calibration)
    meta_path = b.calibration.parent / "meta.json"
    if not meta_path.is_file():
        raise BundleError(f"{b.calibration}: no meta.json next to it, so what the grid was "
                          f"measured with cannot be checked against the servers")
    grid = json.loads(meta_path.read_text())["server_args"]
    off = {k: (sv.args[k], grid.get(k)) for k in COST_ARGS
           if k in sv.args and sv.args[k] != grid.get(k)}
    if off:
        raise BundleError(f"{b.path}: the calibration was measured with different "
                          f"{', '.join(f'{k} (servers {a!r}, grid {g!r})' for k, (a, g) in off.items())}; "
                          f"measure a grid for these servers")
    if cal.tp != sv.args["tp_size"]:
        raise BundleError(f"{b.calibration} is fitted for TP={cal.tp}, the servers run "
                          f"TP={sv.args['tp_size']}")

    for k, v in sv.env.items():         # read by Scheduler.__init__, so before any replica
        getattr(environ.envs, k).set(v)
    sa = stub_server_args(**{k: v for k, v in sv.args.items()
                             if k not in COST_ARGS and k not in STUB_ARGS})
    shape = ModelShape.from_config(sv.args["model_path"])
    return System(replicas=[StubScheduler(sa) for _ in range(sv.replicas)],
                  # one pool across all workers, until per-worker pools land
                  pools={p: b.recipe.workers * k for p, k in b.recipe.pools.items()},
                  cost=CostModel(cal, shape), bundle=b)
