"""Real SGLang `Scheduler`s whose model worker computes nothing (SIMULATOR.md §7).

Batching, KV allocation, radix matching, eviction and retraction are SGLang's own code.
Only `TpModelWorker` is replaced: the stub emits one token per request per forward, a
hash of (rid, position), and EOS when the request reaches the completion count it was
told. The caller owns the clock: `StubScheduler.step()` runs one iteration of SGLang's
event loop and returns what ran, so a cost model can say how long it took.

Several schedulers live in one process. They share SGLang's process-global state (the
torch.distributed groups and `get_global_server_args()`), so every replica in a process
must use the same server args. tests/test_simulator_sglang_stub.py checks that a
replica's per-step decisions are the same alone and interleaved with another.
"""

from __future__ import annotations

import copy
import hashlib
from collections import deque
from dataclasses import dataclass

import torch

import sglang
from sglang.srt import environ
from sglang.srt.hardware_backend.npu.allocator_npu import (
    NPUPagedTokenToKVPoolAllocator,
    _alloc_extend_naive,
)
from sglang.srt.layers.logits_processor import LogitsProcessorOutput
from sglang.srt.managers.io_struct import BatchTokenIDOutput, TokenizedGenerateReqInput
from sglang.srt.managers.scheduler import Scheduler
from sglang.srt.managers.utils import GenerationBatchResult
from sglang.srt.mem_cache.memory_pool import ReqToTokenPool
from sglang.srt.sampling.sampling_params import SamplingParams
from sglang.srt.server_args import PortArgs, ServerArgs, set_global_server_args_for_scheduler
from sglang.srt.utils import get_num_new_pages

SGLANG_VERSION = "0.5.8"

# Stub ids stay below every special token of the vocabularies we run (Qwen3's start at 151643).
_ID_SPACE = 150_000


def token_id(rid: str, pos: int) -> int:
    """The token the stub emits for `rid` at output position `pos` (never EOS)."""
    h = hashlib.blake2b(f"{rid}:{pos}".encode(), digest_size=4).digest()
    return int.from_bytes(h, "little") % _ID_SPACE


def stub_server_args(**scheduler_args) -> ServerArgs:
    """ServerArgs for a stubbed replica: what the scheduler decides with comes from the
    caller; what only the real worker needs is overridden.

    `attention_backend="torch_native"` only switches two KV-index helpers in
    `mem_cache/common.py` from triton to SGLang's torch versions; no scheduling decision
    reads it. The caller must resolve the real server's derived values (page_size,
    chunked_prefill_size, max_total_tokens, ...) with its real backend and pass them in.
    """
    return ServerArgs(
        **scheduler_args,
        tp_size=1,
        device="cpu",
        attention_backend="torch_native",
        skip_tokenizer_init=True,
        disable_cuda_graph=True,
        load_format="dummy",
    )


class _CpuPagedAllocator(NPUPagedTokenToKVPoolAllocator):
    """SGLang's paged allocator; extend uses SGLang's pure-torch path instead of triton.

    Decode and free are the NPU class's, which are already pure torch. Page accounting
    is `get_num_new_pages`, the same call the CUDA path makes.
    """

    def alloc_extend(self, prefix_lens, prefix_lens_cpu, seq_lens, seq_lens_cpu, last_loc,
                     extend_num_tokens):
        n = get_num_new_pages(seq_lens=seq_lens_cpu, page_size=self.page_size,
                              prefix_lens=prefix_lens_cpu)
        if self.need_sort and n > len(self.free_pages):
            self.merge_and_sort_free()
        if n > len(self.free_pages):
            return None
        out = torch.empty((extend_num_tokens,), dtype=torch.int64)
        _alloc_extend_naive(prefix_lens, seq_lens, last_loc, self.free_pages, out,
                            self.page_size, self.device)
        self.free_pages = self.free_pages[n:]
        return out


class _ModelRunnerAttrs:
    """The handful of `model_runner` attributes `Scheduler` reads."""

    hybrid_gdn_config = None
    mamba2_config = None
    mambaish_config = None
    lora_manager = None


class StubWorker:
    """Stands in for `TpModelWorker`. Builds the KV pools on CPU with no KV tensors."""

    def __init__(self, scheduler: Scheduler):
        sa = scheduler.server_args
        self.scheduler = scheduler
        self.server_args = sa
        self.max_total_num_tokens = sa.max_total_tokens
        self.max_running_requests = sa.max_running_requests
        self.context_len = scheduler.model_config.context_len
        self.req_to_token_pool = ReqToTokenPool(
            size=self.max_running_requests + 1, max_context_len=self.context_len + 4,
            device="cpu", enable_memory_saver=False)
        self.allocator = _CpuPagedAllocator(
            self.max_total_num_tokens, page_size=sa.page_size, dtype=torch.bfloat16,
            device="cpu", kvcache=None, need_sort=False)
        eos = scheduler.model_config.hf_eos_token_id
        if not eos:
            raise ValueError("model config has no EOS token; the stub cannot end requests")
        self.eos = min(eos)
        self.completion: dict[str, int] = {}   # rid -> tokens the request will produce
        self.emitted: dict[str, int] = {}      # rid -> tokens kept so far (survives retraction)
        self.is_hybrid_swa = False
        self.sliding_window_size = None
        self.model_runner = _ModelRunnerAttrs()

    def get_worker_info(self):
        sa = self.server_args
        return (self.max_total_num_tokens, sa.max_prefill_tokens, self.max_running_requests,
                sa.max_queued_requests, self.context_len, self.context_len - 1, sa.random_seed,
                "cpu", self.req_to_token_pool.size, self.req_to_token_pool.max_context_len,
                self.max_total_num_tokens)

    def get_memory_pool(self):
        return self.req_to_token_pool, self.allocator

    def get_pad_input_ids_func(self):
        return None

    def get_tp_group(self):
        from sglang.srt.distributed import get_tp_group
        return get_tp_group()

    def forward_batch_generation(self, mwb, **_):
        reqs = self.scheduler.cur_batch.reqs
        assert len(reqs) == len(mwb.seq_lens)
        extend = mwb.forward_mode.is_extend()
        out = []
        for r in reqs:
            # An extend row keeps its token only if this chunk reaches the end of the
            # sequence. Not `is_chunked`: under overlap the last chunk launches before the
            # previous chunk's result decrements it.
            if extend and len(r.fill_ids) < len(r.origin_input_ids) + len(r.output_ids):
                out.append(0)
                continue
            k = self.emitted.get(r.rid, 0)
            self.emitted[r.rid] = k + 1
            n = self.completion.get(r.rid)
            out.append(self.eos if n is not None and k + 1 >= n else token_id(r.rid, k))
        return GenerationBatchResult(
            logits_output=LogitsProcessorOutput(next_token_logits=None),
            next_token_ids=torch.tensor(out, dtype=torch.int64),
            can_run_cuda_graph=False)


class _StubbedScheduler(Scheduler):
    def init_tp_model_worker(self):
        _init_process_groups(self.server_args, self.model_config)
        self.tp_worker = StubWorker(self)


class _Capture:
    """Replaces the ZMQ sender: keeps what the scheduler would have sent."""

    def __init__(self):
        self.outputs = []

    def send_output(self, output, recv_obj=None):
        self.outputs.append(output)


_GROUPS_UP = False


def _init_process_groups(server_args, model_config):
    """One-rank gloo groups, once per process; every scheduler in the process shares them."""
    global _GROUPS_UP
    if _GROUPS_UP:
        return
    from sglang.srt.distributed import init_distributed_environment, initialize_model_parallel
    from sglang.srt.layers.dp_attention import initialize_dp_attention
    from sglang.srt.utils import get_free_port

    init_distributed_environment(backend="gloo", world_size=1, rank=0, local_rank=0,
                                 distributed_init_method=f"tcp://127.0.0.1:{get_free_port()}")
    initialize_model_parallel(1)
    initialize_dp_attention(server_args=server_args, model_config=model_config)
    _GROUPS_UP = True


@dataclass(frozen=True)
class Step:
    """One forward the scheduler launched: what a cost model needs to price it."""

    mode: str                      # SGLang ForwardMode name: EXTEND, DECODE, MIXED, IDLE
    rids: tuple[str, ...]
    seq_lens: tuple[int, ...]      # context length per row, including this step's tokens
    extend_lens: tuple[int, ...]   # new tokens per row (EXTEND/MIXED); empty for DECODE
    prefix_lens: tuple[int, ...]   # tokens per row served from the radix cache (EXTEND/MIXED)


@dataclass(frozen=True)
class Finished:
    rid: str
    completion_tokens: int
    cached_tokens: int
    reason: str                    # "stop", "length", "abort"


class StubScheduler:
    """One replica. `submit` requests, call `step` while `busy`, collect `finished`."""

    _server_args: ServerArgs | None = None

    def __init__(self, server_args: ServerArgs):
        if sglang.__version__ != SGLANG_VERSION:
            raise RuntimeError(f"stub is checked against SGLang {SGLANG_VERSION}, "
                               f"found {sglang.__version__}")
        if environ.envs.SGLANG_QUEUED_TIMEOUT_MS.get() > 0:
            raise RuntimeError("SGLANG_QUEUED_TIMEOUT_MS reads the wall clock; unset it")
        # SGLang writes into its server args (Scheduler.__init__ sets pp_max_micro_batch_size
        # on the global one), so it only ever gets copies and the caller's object stays clean.
        if StubScheduler._server_args is None:
            StubScheduler._server_args = copy.deepcopy(server_args)
            set_global_server_args_for_scheduler(copy.deepcopy(server_args))
        elif StubScheduler._server_args != server_args:
            raise ValueError("replicas in one process share SGLang's global server args; "
                             "run differently configured replicas in separate processes")
        own = copy.deepcopy(server_args)
        self.s = _StubbedScheduler(own, PortArgs.init_new(own), gpu_id=0,
                                   tp_rank=0, moe_ep_rank=0, pp_rank=0, dp_rank=None)
        self.w: StubWorker = self.s.tp_worker
        self.s.send_to_detokenizer = _Capture()
        self.s.send_to_tokenizer = _Capture()
        self.overlap = self.s.enable_overlap
        self._results = deque()
        self._vocab = self.s.model_config.vocab_size

    @property
    def eos(self) -> int:
        return self.w.eos

    def output_ids(self, rid: str, completion_tokens: int) -> list[int]:
        """The ids the stub produced for a request that finished by EOS."""
        return [token_id(rid, p) for p in range(completion_tokens - 1)] + [self.eos]

    def submit(self, rid: str, input_ids: list[int], completion_tokens: int,
               max_new_tokens: int) -> None:
        """`completion_tokens` is when the request stops (from the trace); the scheduler
        only sees `max_new_tokens`. A request whose trace stop was the cap should pass
        completion_tokens > max_new_tokens so SGLang stops it by length."""
        self.w.completion[rid] = completion_tokens
        sp = SamplingParams(max_new_tokens=max_new_tokens)
        sp.normalize(None)
        sp.verify(self._vocab)
        self.s.process_input_requests([TokenizedGenerateReqInput(
            input_text="", input_ids=list(input_ids), mm_inputs=None, sampling_params=sp,
            return_logprob=False, logprob_start_len=-1, top_logprobs_num=0,
            token_ids_logprob=None, stream=False, rid=rid)])

    @property
    def busy(self) -> bool:
        s = self.s
        return bool(s.waiting_queue) or not s.running_batch.is_empty() or bool(self._results)

    def step(self) -> Step | None:
        """One iteration of `event_loop_normal` / `event_loop_overlap`, minus the socket
        receive. Under overlap a batch's results are processed one call later, as in SGLang."""
        s = self.s
        batch = s.get_next_batch_to_run()
        s.cur_batch = batch
        rec = None
        if batch is not None:
            ext = batch.forward_mode.is_extend()
            rec = Step(
                mode=batch.forward_mode.name,
                rids=tuple(r.rid for r in batch.reqs),
                seq_lens=tuple(int(x) for x in batch.seq_lens_cpu.tolist()),
                extend_lens=tuple(batch.extend_lens) if ext else (),
                prefix_lens=tuple(batch.prefix_lens) if ext else (),
            )
        if not self.overlap:
            if batch is not None:
                s.process_batch_result(batch, s.run_batch(batch))
            else:
                s.self_check_during_idle()
        else:
            disable = s.is_disable_overlap_for_batch(batch)
            if disable and self._results:
                s.process_batch_result(*self._results.popleft())
            res = None
            if batch is not None:
                res = s.run_batch(batch)
                self._results.append((batch.copy(), res))
            if s.last_batch is not None:
                if not disable and self._results:
                    s.process_batch_result(*self._results.popleft())
            elif batch is None:
                s.self_check_during_idle()
            s.launch_batch_sample_if_needed(res)
        s.last_batch = batch
        return rec

    def finished(self) -> list[Finished]:
        """Requests that finished since the last call."""
        done = []
        for o in self.s.send_to_detokenizer.outputs:
            if not isinstance(o, BatchTokenIDOutput):
                continue
            for rid, fr, ct, cached in zip(o.rids, o.finished_reasons, o.completion_tokens,
                                           o.cached_tokens):
                if fr is not None:
                    done.append(Finished(rid, ct, cached, fr["type"]))
        self.s.send_to_detokenizer.outputs.clear()
        return done
