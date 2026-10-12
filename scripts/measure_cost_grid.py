"""SIMULATOR.md §15 step 2: time single forwards of a standalone SGLang ModelRunner on a grid.

The cost model needs step times over `batch x context x prefill length`, each axis swept on
its own (run traces only cover the diagonal where big batches have short contexts, §8). This
script measures them on whatever machine it runs on; fitting `eta` is a separate step.

Each point builds its batch the way the scheduler would hand it over, without paying for
the context first: every request gets `prefix` KV slots straight from the allocator, then
one EXTEND of `extend` new tokens runs on top of that prefix, then DECODE steps follow.

What sits in those slots matters, because MoE routing depends on the hidden states and a
batch that touches more experts reads more weight bytes. With `--donor-tokens` (a jsonl of
real token-id sequences, e.g. from scripts/simulator_dev/dump_real_token_ids.py), a few
long "donor" sequences of real text are prefilled once and every request's prefix KV is
copied from one of them, and the new tokens are real text too. Without it the slots hold
random values and the tokens are random. Both land within a few percent of each other; a
real-text prefill is up to 5.5% (TP=4) / 13% (TP=1) slower, a random-token prefill 10-35%
faster. `kind=check` rows compare the four ways of filling the context at points small enough
to prefill for real, and are what the cost model's routing factor is fitted on.

Server args are the run's (verl passes these to SGLang); everything else resolves through
SGLang's own defaults on this GPU, exactly as the real server did. Results are one JSON
object per point in <out>/grid.jsonl, server args and hardware in <out>/meta.json.

    python scripts/measure_cost_grid.py --tp 4 --gpus 0,1,2,3 --out analysis/costgrid/b200-tp4
"""

from __future__ import annotations

import argparse
import collections
import dataclasses
import json
import multiprocessing as mp
import os
import socket
import time

GRID_B = [1, 2, 4, 8, 16, 24, 32, 48, 64, 80, 96, 112, 128, 160, 192, 224, 256]
GRID_CTX = [256, 512, 1024, 2048, 4096, 8192, 12288, 16384, 24576, 32768, 49152, 65024]
EXT_B = [1, 2, 4, 8, 16, 32, 64]
EXT_LEN = [16, 64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384]
EXT_PREFIX = [0, 2048, 8192, 16384, 32768]
# Non-uniform batches: does a decode step care about anything but the summed context?
MIXED = [(32, 8192), (128, 4096), (128, 16384), (224, 16384), (64, 32768)]
# Uneven batches at a fixed mean: lognormal spread sigma, and one 64k request among short ones.
# The first TP=4 pass found lognormal batches 1.2-1.6x slower than uniform ones of equal sum.
SPREAD_B = [16, 32, 64, 128, 192, 224]
SPREAD_MEAN = [4096, 8192, 16384]
SPREAD_SIGMA = [0.0, 0.25, 0.5, 0.75, 1.0]
ONE_LONG = [(32, 2048), (32, 8192), (128, 2048), (128, 8192), (224, 2048), (224, 8192)]
# Real-prefill checks: one EXTEND of the whole context. Kept at <= 131k tokens: a 262k-token
# EXTEND (16 x 16k) hung all four TP=4 ranks at 0% GPU on B200, no traceback.
CHECKS = [(1, 16384), (4, 16384), (8, 16384), (16, 8192), (32, 4096), (48, 2048), (64, 2048),
          (96, 1024), (128, 1024), (160, 512), (224, 512)]
# How each check point's context is filled: copied from real-text donors, random values,
# prefilled from real text, prefilled from random tokens.
CHECK_KV = ["donor", "random", "prefill_real", "prefill_random"]
DONOR_LEN = 65088          # >= the longest prefix any point asks for (65024), page-aligned
DONOR_CHUNK = 16384

DECODE_WARM, DECODE_STEPS, EXT_REPS = 3, 16, 4


def run_args(a) -> dict:
    """The args verl's async_sglang_server passed for run 20260912-072938 (configs/grpo_swebench.yaml)."""
    return dict(model_path=a.model, dtype="bfloat16", tp_size=a.tp,
                mem_fraction_static=a.mem_fraction, max_running_requests=256,
                attention_backend="trtllm_mha", page_size=64, moe_runner_backend="triton",
                context_length=65536, trust_remote_code=False, log_level="error",
                skip_tokenizer_init=True, random_seed=0)


def worker(sa, port_args, a, tp_rank, gpu_id):
    import numpy as np
    import torch
    import torch.distributed as dist

    from sglang.bench_one_batch import TreeCacheNamespace, _maybe_prepare_mlp_sync_batch, load_model
    from sglang.srt.layers.moe import initialize_moe_config
    from sglang.srt.managers.schedule_batch import Req, ScheduleBatch
    from sglang.srt.model_executor.forward_batch_info import ForwardBatch
    from sglang.srt.sampling.sampling_params import SamplingParams
    from sglang.srt.speculative.spec_info import SpeculativeAlgorithm

    initialize_moe_config(sa)
    runner, _ = load_model(sa, port_args, gpu_id, tp_rank)
    rank0 = tp_rank == 0
    say = (lambda *x: print(*x, flush=True)) if rank0 else (lambda *x: None)
    pool, alloc = runner.req_to_token_pool, runner.token_to_kv_pool_allocator
    kv = runner.token_to_kv_pool
    page = sa.page_size
    cap_tokens = runner.max_total_num_tokens
    max_reqs = pool.size - 1
    vocab = runner.model_config.vocab_size
    eos_free = min(vocab, 150_000)

    # Random KV once: uninitialised memory can hold NaNs, which would change MoE routing.
    g = torch.Generator(device=runner.device).manual_seed(1234)
    for buf in list(kv.k_buffer) + list(kv.v_buffer):
        buf.normal_(0, 0.5, generator=g)
    torch.cuda.synchronize()

    real = None
    if a.donor_tokens:
        with open(a.donor_tokens) as f:
            real = np.concatenate([np.asarray(json.loads(line), np.int64) for line in f])
        assert len(real) >= a.donors * DONOR_LEN, "not enough real tokens for the donors"

    if rank0:
        meta = dict(host=socket.gethostname(), gpu=torch.cuda.get_device_name(),
                    sms=torch.cuda.get_device_properties(0).multi_processor_count,
                    sglang=__import__("sglang").__version__, torch=torch.__version__,
                    started=time.strftime("%Y-%m-%dT%H:%M:%S"),
                    max_total_num_tokens=cap_tokens, req_pool=max_reqs,
                    cuda_graph_bs=list(sa.cuda_graph_bs), server_args=dataclasses.asdict(sa),
                    fragmented_pages=not a.contiguous, decode_steps=DECODE_STEPS,
                    decode_warm=DECODE_WARM, ext_reps=EXT_REPS)
        os.makedirs(a.out, exist_ok=True)
        with open(os.path.join(a.out, "meta.json"), "w") as f:
            json.dump(meta, f, indent=1, default=str)
        out = open(os.path.join(a.out, "grid.jsonl"), "a")

    tree = TreeCacheNamespace(page_size=page, device=runner.device,
                              token_to_kv_pool_allocator=alloc)
    rng = np.random.default_rng(a.seed)   # same seed on every rank: same batches everywhere
    sp = SamplingParams(temperature=0, max_new_tokens=1 << 20)

    donor_slots: list = []      # per donor: KV slot of each position, kept across points
    donor_pages = None

    def reset():
        pool.clear()
        alloc.clear()
        if donor_pages is not None:
            alloc.free_pages = alloc.free_pages[~torch.isin(alloc.free_pages, donor_pages)]
        if not a.contiguous:
            # A server that has been running for a while hands out pages in no order.
            perm = torch.from_numpy(rng.permutation(len(alloc.free_pages))).to(alloc.free_pages.device)
            alloc.free_pages = alloc.free_pages[perm]

    def token_ids(n, use_real):
        if use_real:
            start = int(rng.integers(0, len(real) - n))
            return real[start:start + n].tolist()
        return rng.integers(0, eos_free, n).tolist()

    def make_reqs(prefixes, extends, fake=True, kv_fill=None):
        """kv_fill: "donor" copies each fake prefix from a real-text donor, "random" leaves
        the pool's random values. Real tokens whenever donors exist, unless "random"."""
        kv_fill = kv_fill or ("donor" if donor_slots else "random")
        use_real = real is not None and kv_fill in ("donor", "prefill_real")
        reqs, src, dst = [], [], []
        for i, (p, e) in enumerate(zip(prefixes, extends)):
            r = Req(rid=str(i), origin_input_text="", origin_input_ids=token_ids(p + e, use_real),
                    sampling_params=sp)
            r.fill_ids = r.origin_input_ids
            r.logprob_start_len = -1
            if fake and p:
                idx = alloc.alloc(p)
                assert idx is not None, "pool too small for this point"
                r.prefix_indices = idx.to(torch.int64)
                if kv_fill == "donor":
                    src.append(donor_slots[i % len(donor_slots)][:p])
                    dst.append(r.prefix_indices)
            r.set_extend_input_len(len(r.fill_ids) - len(r.prefix_indices))
            reqs.append(r)
        if dst:
            s_all, d_all = torch.cat(src), torch.cat(dst)
            for buf in list(kv.k_buffer) + list(kv.v_buffer):
                buf[d_all] = buf[s_all]
        return reqs

    def timed(fn):
        torch.cuda.synchronize()
        if a.tp > 1:
            dist.barrier(group=runner.tp_group.cpu_group)
        e0, e1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        t0 = time.perf_counter()
        res = fn(e0, e1)
        torch.cuda.synchronize()
        return res, time.perf_counter() - t0, e0.elapsed_time(e1) / 1e3

    @torch.no_grad()
    def extend(reqs):
        def go(e0, e1):
            b = ScheduleBatch.init_new(reqs=reqs, req_to_token_pool=pool,
                                       token_to_kv_pool_allocator=alloc, tree_cache=tree,
                                       model_config=runner.model_config, enable_overlap=False,
                                       spec_algorithm=SpeculativeAlgorithm.NONE)
            b.prepare_for_extend()
            _maybe_prepare_mlp_sync_batch(b, runner)
            fb = ForwardBatch.init_new(b.get_model_worker_batch(), runner)
            e0.record()
            lo = runner.forward(fb).logits_output
            nxt = runner.sample(lo, fb)
            e1.record()
            return b, nxt
        return timed(go)

    @torch.no_grad()
    def decode(b, nxt):
        def go(e0, e1):
            b.output_ids = nxt
            b.prepare_for_decode()
            _maybe_prepare_mlp_sync_batch(b, runner)
            fb = ForwardBatch.init_new(b.get_model_worker_batch(), runner)
            e0.record()
            lo = runner.forward(fb).logits_output
            n = runner.sample(lo, fb)
            e1.record()
            return n, fb.forward_mode.name
        return timed(go)

    def emit(row):
        if rank0:
            out.write(json.dumps(row) + "\n")
            out.flush()

    def decode_point(kind, prefixes, kv_fill=None, **extra):
        kv_fill = kv_fill or ("donor" if donor_slots else "random")
        fake = not kv_fill.startswith("prefill")
        reset()
        extends = [1] * len(prefixes)
        if not fake:   # prefill the whole context for real (prefix 0, extend = context)
            extends, prefixes = [p + 1 for p in prefixes], [0] * len(prefixes)
        reqs = make_reqs(prefixes, extends, fake, kv_fill)
        (b, nxt), _, _ = extend(reqs)
        wall, gpu = [], []
        for i in range(DECODE_WARM + DECODE_STEPS):
            (nxt, mode), w, gt = decode(b, nxt)
            if i >= DECODE_WARM:
                wall.append(w)
                gpu.append(gt)
        ctx = [p + e for p, e in zip(prefixes, extends)]
        emit(dict(kind=kind, mode="DECODE", batch=len(ctx), ctx_mean=float(np.mean(ctx)),
                  ctx_max=int(max(ctx)), ctx_std=float(np.std(ctx)),
                  ctx_p90=float(np.percentile(ctx, 90)),
                  ctx_sum_first=int(sum(ctx)) + len(ctx), steps=DECODE_STEPS,
                  wall_med=float(np.median(wall)), wall_p10=float(np.percentile(wall, 10)),
                  wall_p90=float(np.percentile(wall, 90)), gpu_med=float(np.median(gpu)),
                  gpu_p10=float(np.percentile(gpu, 10)), gpu_p90=float(np.percentile(gpu, 90)),
                  fake_prefix=fake, kv=kv_fill, **extra))
        return float(np.median(gpu))

    if real is not None:
        # Donors: long real-text sequences prefilled once in chunks, as the server would.
        t_d = time.time()
        for d in range(a.donors):
            ids = real[d * DONOR_LEN:(d + 1) * DONOR_LEN].tolist()
            slots = torch.empty(0, dtype=torch.int64, device=runner.device)
            for start in range(0, DONOR_LEN, DONOR_CHUNK):
                n = min(start + DONOR_CHUNK, DONOR_LEN)
                r = Req(rid=f"donor{d}.{start}", origin_input_text="", origin_input_ids=ids[:n],
                        sampling_params=sp)
                r.fill_ids = r.origin_input_ids
                r.logprob_start_len = -1
                r.prefix_indices = slots
                r.set_extend_input_len(n - start)
                extend([r])
                slots = pool.req_to_token[r.req_pool_idx, :n].to(torch.int64).clone()
            donor_slots.append(slots)
        donor_pages = torch.unique(torch.cat(donor_slots) // page).to(alloc.free_pages.dtype)
        cap_tokens -= a.donors * DONOR_LEN
        say(f"{a.donors} donors of {DONOR_LEN} real tokens prefilled in {time.time() - t_d:.0f}s")
        if rank0:
            json.dump(dict(meta, donors=a.donors, donor_len=DONOR_LEN,
                           donor_tokens=os.path.abspath(a.donor_tokens),
                           max_total_num_tokens_left=cap_tokens),
                      open(os.path.join(a.out, "meta.json"), "w"), indent=1, default=str)

    def fits(n_tokens, n_reqs, longest=0):
        return (n_reqs <= max_reqs and n_tokens <= cap_tokens * 0.95
                and longest + DECODE_WARM + DECODE_STEPS + 2 <= sa.context_length)

    t_start = time.time()
    # Warm-up: JIT, autotune, first-touch of every graph bucket.
    for bsz in sa.cuda_graph_bs:
        if bsz <= max_reqs:
            decode_point("warmup", [1024] * bsz)
    say(f"warm-up done in {time.time() - t_start:.0f}s")

    if "decode" in a.parts:
        for ctx in GRID_CTX:
            for bsz in GRID_B:
                if fits(bsz * (ctx + 64), bsz):
                    p = (ctx - 1) // page * page
                    decode_point("grid", [p] * bsz, ctx_target=ctx)
            say(f"decode ctx {ctx} done, {time.time() - t_start:.0f}s")

    if "mixed" in a.parts:
        for bsz, mean in MIXED:
            # lognormal around the run's spread (ctx p10/p90 ~ 0.45x / 1.9x the median)
            raw = rng.lognormal(np.log(mean) - 0.5 * 0.55 ** 2, 0.55, bsz)
            prefixes = [int(max(page, min(65024, x)) // page * page) for x in raw]
            if fits(sum(prefixes) + 64 * bsz, bsz):
                decode_point("mixed", prefixes, ctx_target=mean)
                u = int(np.mean(prefixes)) // page * page
                decode_point("mixed_uniform", [u] * bsz, ctx_target=mean)
        say(f"mixed done, {time.time() - t_start:.0f}s")

    if "spread" in a.parts:
        for bsz in SPREAD_B:
            for mean in SPREAD_MEAN:
                for sig in SPREAD_SIGMA:
                    raw = rng.lognormal(np.log(mean) - 0.5 * sig ** 2, sig, bsz) if sig else \
                        np.full(bsz, float(mean))
                    prefixes = [int(max(page, min(65024, x)) // page * page) for x in raw]
                    if fits(sum(prefixes) + 64 * bsz, bsz):
                        decode_point("spread", prefixes, ctx_target=mean, sigma=sig,
                                     ctxs=prefixes)
            say(f"spread B={bsz} done, {time.time() - t_start:.0f}s")

    if "one_long" in a.parts:
        for bsz, short in ONE_LONG:
            if not fits(short * bsz + 65024 + 64 * bsz, bsz):
                continue
            for n_long in (0, 1):
                prefixes = [short] * (bsz - n_long) + [65024] * n_long
                decode_point("one_long", prefixes, ctx_target=short, n_long=n_long)
        say(f"one_long done, {time.time() - t_start:.0f}s")

    if "check" in a.parts:
        for bsz, ctx in CHECKS:
            if not fits(bsz * (ctx + 64), bsz):
                continue
            p = (ctx - 1) // page * page
            got = {k: decode_point("check", [p] * bsz, kv_fill=k, ctx_target=ctx)
                   for k in CHECK_KV if (donor_slots or k not in ("donor", "prefill_real"))}
            say(f"check B={bsz} ctx={ctx}: " + ", ".join(f"{k} {1e3 * v:.2f} ms"
                                                         for k, v in got.items()))

    if "extend" in a.parts:
        for p in a.ext_prefix or EXT_PREFIX:
            for e in EXT_LEN:
                for bsz in EXT_B:
                    if bsz * e > sa.chunked_prefill_size or not fits(bsz * (p + e + 64), bsz, p + e):
                        continue
                    wall, gpu = [], []
                    for rep in range(EXT_REPS + 1):
                        reset()
                        reqs = make_reqs([p] * bsz, [e] * bsz)
                        _, w, gt = extend(reqs)
                        if rep:   # first repetition of a shape pays for JIT
                            wall.append(w)
                            gpu.append(gt)
                    emit(dict(kind="grid", mode="EXTEND", batch=bsz, prefix=p, extend=e,
                              reps=EXT_REPS, wall_med=float(np.median(wall)),
                              wall_min=float(min(wall)), wall_max=float(max(wall)),
                              gpu_med=float(np.median(gpu)), gpu_min=float(min(gpu)),
                              gpu_max=float(max(gpu)), fake_prefix=True))
            say(f"extend prefix {p} done, {time.time() - t_start:.0f}s")

    if "profile" in a.parts:
        # Is a small EXTEND's ~31 ms floor kernel time or the GPU waiting on eager launches?
        from torch.profiler import ProfilerActivity, profile
        for bsz, p, e in [(1, 16384, 16), (1, 16384, 512), (4, 16384, 512), (1, 0, 8192)]:
            for rep in range(3):
                reset()
                reqs = make_reqs([p] * bsz, [e] * bsz)
                if rep < 2:
                    extend(reqs)
                    continue
                with profile(activities=[ProfilerActivity.CUDA]) as prof:
                    _, w, gt = extend(reqs)
                ks = [ev for ev in prof.events() if ev.device_type.name == "CUDA"]
                busy = sum(ev.device_time for ev in ks) / 1e6   # us -> s
                say(f"profile EXTEND B={bsz} prefix={p} extend={e}: wall {1e3 * w:.1f} ms, "
                    f"{len(ks)} kernels, kernel time {1e3 * busy:.1f} ms")
                top = collections.Counter()
                for ev in ks:
                    top[ev.name[:90]] += ev.device_time / 1e3
                say("   top: " + "; ".join(f"{n} {t:.1f}ms" for n, t in top.most_common(6)))
                emit(dict(kind="profile", mode="EXTEND", batch=bsz, prefix=p, extend=e,
                          wall=w, gpu=gt, kernels=len(ks), kernel_time=busy))
        reset()
        reqs = make_reqs([16320] * 128, [1] * 128)
        (b, nxt), _, _ = extend(reqs)
        for _ in range(3):
            (nxt, _), _, _ = decode(b, nxt)
        with profile(activities=[ProfilerActivity.CUDA]) as prof:
            (nxt, _), w, gt = decode(b, nxt)
        ks = [ev for ev in prof.events() if ev.device_type.name == "CUDA"]
        busy = sum(ev.device_time for ev in ks) / 1e6
        say(f"profile DECODE B=128 ctx=16k: wall {1e3 * w:.1f} ms, {len(ks)} kernels, "
            f"kernel time {1e3 * busy:.1f} ms")
        emit(dict(kind="profile", mode="DECODE", batch=128, ctx_mean=16321, wall=w, gpu=gt,
                  kernels=len(ks), kernel_time=busy))

    say(f"all done in {time.time() - t_start:.0f}s")
    if a.tp > 1:
        dist.barrier(group=runner.tp_group.cpu_group)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="/data0/shared/Qwen3-30B-A3B-Instruct-2507")
    ap.add_argument("--tp", type=int, default=4)
    ap.add_argument("--gpus", default="0,1,2,3")
    ap.add_argument("--mem-fraction", type=float, default=0.75)
    ap.add_argument("--out", required=True)
    ap.add_argument("--parts", default="decode,mixed,spread,one_long,check,extend")
    ap.add_argument("--contiguous", action="store_true", help="don't shuffle free pages")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--ext-prefix", type=int, nargs="+", default=None,
                    help="override the EXTEND grid's prefix lengths")
    ap.add_argument("--donor-tokens", default=None,
                    help="jsonl of real token-id lists; prefixes are copied from donors of it")
    ap.add_argument("--donors", type=int, default=16, help="real-text donor sequences to keep")
    a = ap.parse_args()
    a.parts = set(a.parts.split(","))
    os.environ["CUDA_VISIBLE_DEVICES"] = a.gpus

    from sglang.srt.entrypoints.engine import _set_envs_and_config
    from sglang.srt.server_args import PortArgs, ServerArgs

    sa = ServerArgs(**run_args(a))
    _set_envs_and_config(sa)
    port_args = PortArgs.init_new(sa)
    procs = []
    ctx = mp.get_context("spawn")
    for r in range(a.tp):
        p = ctx.Process(target=worker, args=(sa, port_args, a, r, r))
        p.start()
        procs.append(p)
    for p in procs:
        p.join()
    bad = [p.exitcode for p in procs if p.exitcode]
    raise SystemExit(1 if bad else 0)


if __name__ == "__main__":
    main()
