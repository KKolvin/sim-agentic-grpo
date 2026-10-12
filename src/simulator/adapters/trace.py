"""One adapter for every trace: canonical turn records, and the rules that turn them into IR.

`SIMULATOR.md` §6 says provenance is decided per field, not per file: one trace
records usage but no durations, another durations but only text. So the
knowledge splits in two, and only the first half knows any file layout:

1. :mod:`simulator.adapters.shapes` recognises a row by the conventions it
   follows -- a chat message list, OTel ``gen_ai.*`` spans, a list of per-turn
   token counts -- and reads it into the records below, each field ``None``
   when the trace did not record it.
2. :func:`resolve` (here) walks every IR field down one ladder, the same for
   every trace, and records which rung it stood on in ``prov``:

   ============================  ==================================================
   field                         ladder
   ============================  ==================================================
   ``prompt_tokens``             usage/count (traced) -> chat template over the
                                 messages (derived) -> the next turn's cache read
                                 at the trace's own offset (derived) -> refuse
   ``completion_tokens``         usage/count (traced) -> tokenizer over the output
                                 text (derived) -> refuse
   ``prefix_parent``/``shared``  token-level loop: the previous turn, whole ->
                                 messages: the earlier turn whose messages this
                                 prompt extends -> counts: the previous turn, if
                                 the prompt is long enough to contain it
   tool calls after a turn       messages that answered it -> calls in the output
                                 -> one, assumed and said so
   tool durations                recorded per call (traced) -> the gap between one
                                 response and the next request (derived, an upper
                                 bound: harness time is in it) -> prior (estimated)
   ============================  ==================================================

   Token counts have no prior rung, because adjacent turns are bound together
   (``prompt_{k+1} >= prompt_k + completion_k``) and independent draws would break
   that; a session whose counts cannot be derived is skipped and said so.

What a harness *means* by its records, as opposed to how it lays them out, is a
:class:`Profile`: whether its loop concatenates tokens or re-renders messages, which
tool messages are observations and which are nudges, what wraps an episode
(container, grading). The default profile assumes nothing; a known harness's
profile is data, not another reader.
"""

from __future__ import annotations

import statistics
from collections import Counter, defaultdict
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Callable, Iterable, Literal, Mapping, Protocol, Sequence

from simulator.adapters.base import KNOB_BY_KIND, AdapterError, Conversion, Priors, estimated_notes
from simulator.ir import Lease, Prefix, Request, Session, Source, StopReason, Task, Workload
from simulator.observed import Observations


# ---------------------------------------------------------------------------
# canonical records
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class Msg:
    """One chat message, reduced to what comparison and rendering need."""

    role: str
    text: str = ""
    #: ``(name, arguments as canonical JSON)`` per tool call the message makes.
    calls: tuple[tuple[str, str], ...] = ()
    #: The message as the harness held it, for a chat template. ``None`` when the
    #: trace's own representation is not one a template takes.
    raw: Mapping[str, Any] | None = None

    def key(self) -> tuple:
        return (self.role, self.text.strip(), self.calls)


@dataclass(frozen=True)
class Call:
    """A tool call that ran after a turn, when the trace records the call itself."""

    name: str | None = None
    duration_s: float | None = None
    timed_out: bool | None = None


@dataclass(frozen=True)
class Turn:
    """One inference request. Every field is ``None`` when the trace did not record it."""

    #: Every message the request was sent with, when the trace keeps each turn's
    #: whole prompt (OTel spans, conversation lists).
    context: tuple[Msg, ...] | None = None
    output: Msg | None = None
    #: Messages appended after the output and before the next request: tool
    #: results, nudges. Only conversation lists have these.
    after: tuple[Msg, ...] | None = None
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    #: Prompt tokens the serving side computed rather than read from its cache.
    #: The other server's answer: used for lengths only, never as a hit rate.
    uncached_tokens: int | None = None
    t_sent: float | None = None
    t_done: float | None = None
    calls: tuple[Call, ...] | None = None
    finish: str | None = None
    max_new_tokens: int | None = None
    #: Content hashes of the prompt's fixed-size blocks, each covering the block and
    #: everything before it (Mooncake's ``hash_ids``): two requests with the same
    #: hash at position i share their first i+1 blocks.
    blocks: tuple[int, ...] | None = None
    #: The length those hashes cover, when the trace records it apart from the prompt
    #: (hashed over its own tokenization, or without the template).
    block_tokens: int | None = None


@dataclass(frozen=True)
class TraceSession:
    id: str
    turns: tuple[Turn, ...]
    file: str = ""
    problem: str | None = None
    group: str | None = None
    exit: str | None = None
    tools: tuple[Mapping[str, Any], ...] | None = None
    #: The row's messages in the harness's own form, for a profile's well-formedness check.
    raw_messages: tuple[Mapping[str, Any], ...] | None = None


@dataclass(frozen=True)
class Header:
    """What a trace says about how its prompts were rendered, from a header row."""

    model: str | None = None
    chat_template: str | None = None
    tools: tuple[Mapping[str, Any], ...] | None = None
    prompt_length: int | None = None


# ---------------------------------------------------------------------------
# profiles: what a harness means, as data
# ---------------------------------------------------------------------------
class Tokenizer(Protocol):
    chat_template: str | None

    def apply_chat_template(self, messages: list[dict], *, tools: list[dict] | None = None,
                            add_generation_prompt: bool = True, tokenize: bool = True) -> Any: ...

    def encode(self, text: str, *, add_special_tokens: bool = False) -> list[int]: ...


@dataclass(frozen=True)
class Envelope:
    """Work that wraps every episode of a harness but is not in its turns: a held
    container, patch recovery when nothing was submitted, cleanup, grading."""

    container_pool: str = "container"
    eval_pool: str = "eval"
    submitted: frozenset[str] = frozenset({"Submitted"})


def _default_calls(output_text: str, after: Sequence[Msg], parsed: int | None, exit_status: str | None,
                   last: bool) -> tuple[int | None, bool]:
    """Tool messages that answered the turn; otherwise the calls in the output, ``None`` if unknown.

    A turn answered by a user message ran nothing: the user took the next turn.
    """
    if after:
        return sum(1 for m in after if m.role in ("tool", "function", "ipython")), False
    return parsed, False


@dataclass(frozen=True)
class Profile:
    name: str = "generic"
    #: The part of the source name after the run: ``run:<dir>/<part>``.
    part: str = "trace"
    #: Only files matching this glob are read.
    files: str = "*"
    #: ``"tokens"``: the loop appends token ids, so turn k+1's prompt is turn k's
    #: whole sequence plus the rendered observation (verl). ``"render"``: every
    #: request is the messages rendered afresh (an OpenAI-style server).
    accumulate: Literal["tokens", "render"] = "render"
    requires_header: bool = False
    #: Group sessions by problem within a file, ``group_size`` at a time in file order.
    group_by_problem: bool = False
    group_size: int | None = None
    file_tag: Callable[[str], str] = staticmethod(lambda stem: stem)
    #: ``(output_text, after, parsed_calls, exit_status, last) -> (calls that ran, nudged)``.
    calls_that_ran: Callable[..., tuple[int | None, bool]] = staticmethod(_default_calls)
    well_formed: Callable[[Sequence[Mapping[str, Any]]], bool] | None = None
    nudge_starts: tuple[str, ...] = ()
    #: Tool calls to assume after a turn when the trace says nothing about what
    #: happened between it and the next. ``None`` assumes nothing: no task, and a
    #: note. A counts-only trace of an agent harness may want 1.
    calls_per_turn: int | None = None
    envelope: Envelope | None = None
    #: What this harness's records cannot say, one sentence each, for the notes.
    notes: tuple[str, ...] = ()
    #: Does this profile describe the trace at ``path``? Checked in order; the
    #: generic profile matches anything.
    match: Callable[[Path], bool] = staticmethod(lambda path: True)


GENERIC = Profile()


# ---------------------------------------------------------------------------
# resolving: records -> IR, field by field
# ---------------------------------------------------------------------------
class _Skip(Exception):
    """This session cannot be converted; the reason goes in the notes."""


def resolve(sessions: Sequence[TraceSession], header: Header | None, profile: Profile,
            priors: Priors, *, source: str, tokenizer: Tokenizer | None = None,
            load_tokenizer: Callable[[str], Tokenizer] | None = None, prompt_length: int | None = None,
            meta: Mapping[str, Any] | None = None, adapter: str = "trace") -> Conversion:
    if profile.requires_header and header is None:
        raise AdapterError("no header record: the template and tool schemas the prompts were "
                           "rendered with are missing, and token counts cannot be derived without them")
    header = header or Header()
    cap = prompt_length if prompt_length is not None else header.prompt_length
    tally = _Tally()
    render = _LazyRenderer(header, tokenizer, load_tokenizer, cap, tally)
    offset = _cache_offset(sessions, tally)

    kept: list[TraceSession] = []
    counts: dict[str, list[tuple[int, Source, int, Source]]] = {}
    for s in sessions:
        if profile.well_formed is not None and not profile.well_formed(s.raw_messages or ()):
            tally.malformed += 1
            continue
        try:
            counts[s.id] = _token_counts(s, profile, render, offset, tally)
        except _Skip as e:
            tally.skipped[str(e)] += 1
            continue
        kept.append(s)
    if not kept:
        reasons = "; ".join(f"{n} {r}" for r, n in tally.skipped.items()) or "no sessions"
        raise AdapterError(f"no session has token counts that can be read or derived: {reasons}")

    groups = _groups(kept, profile)
    prefixes: dict[str, Prefix] = {}
    for g, members in groups.items():
        if g is None:
            continue
        firsts = [counts[s.id][0][0] for s in members]
        if len(set(firsts)) > 1:
            tally.uneven_groups += 1
        prefixes[g] = Prefix(id=f"{g}/prompt", tokens=min(firsts), prov={"tokens": "derived"})

    block_nodes, by_request = _block_prefixes(kept, counts, tally)
    out = [_session(s, counts[s.id], g, prefixes.get(g), profile, priors, tally, by_request)
           for g, members in groups.items() for s in members]
    if render.used and not header.chat_template:
        tally.extra.append("token counts derived with the given tokenizer's own chat template: the trace does not "
                           "say which template its prompts were rendered with")
    if render.cap is None and render.used:
        tally.extra.append("no prompt_length in the header and none was passed: derived first prompts "
                           "are not capped, so a long first prompt comes out longer than was sent")
    workload = Workload(sessions=tuple(out), source=source, prefixes=(*prefixes.values(), *block_nodes),
                        meta={"adapter": adapter, "profile": profile.name, "prompt_length": cap,
                              "group_size": profile.group_size, **(meta or {})})
    return Conversion(adapter=adapter, workload=workload, observations=Observations(source=source),
                      notes=tally.notes(len(out), profile))


# ---------------------------------------------------------------------------
# token counts
# ---------------------------------------------------------------------------
def _ids(out: Any) -> list[int]:
    """Transformers 4 returns a list, 5 may return a BatchEncoding."""
    if isinstance(out, dict) and "input_ids" in out:
        out = out["input_ids"]
    elif hasattr(out, "input_ids"):
        out = out.input_ids
    return list(out)


class _LazyRenderer:
    """The tokenizer and template, loaded only when some count has to be derived from text."""

    def __init__(self, header: Header, tok: Tokenizer | None,
                 load: Callable[[str], Tokenizer] | None, cap: int | None, tally: "_Tally"):
        self.header, self._tok, self._load, self.cap, self.tally = header, tok, load, cap, tally
        self.used = False
        self._sys_n: int | None = None

    def available(self) -> bool:
        return self._tok is not None or (bool(self.header.model) and self._load is not None)

    def tok(self) -> Tokenizer:
        self.used = True
        if self._tok is None:
            if not self.header.model or self._load is None:
                raise _Skip("need a tokenizer to count text, and none was given or named by a header")
            self._tok = self._load(self.header.model)
        tpl = self.header.chat_template
        if tpl and getattr(self._tok, "chat_template", None) != tpl:
            self._tok.chat_template = tpl  # the header's template is what the prompts were rendered with
        return self._tok

    def _capped(self, ids: list[int]) -> list[int]:
        return ids[-self.cap:] if self.cap and len(ids) > self.cap else ids

    def prompt(self, context: Sequence[Msg], tools: Sequence[Mapping[str, Any]] | None) -> int:
        raws = _raws(context)
        ids = _ids(self.tok().apply_chat_template(raws, tools=list(tools) if tools else None,
                                                  add_generation_prompt=True, tokenize=True))
        return len(self._capped(ids))

    def observation(self, msgs: Sequence[Msg]) -> int:
        tok = self.tok()
        if self._sys_n is None:
            # The template may insert a default system prompt, which a token-level
            # loop strips from every observation (verl's initialize_system_prompt):
            # render one empty user turn and two, and the part that did not double
            # is the prefix.
            one = _ids(tok.apply_chat_template([{"role": "user", "content": ""}],
                                               add_generation_prompt=False, tokenize=True))
            two = _ids(tok.apply_chat_template([{"role": "user", "content": ""}] * 2,
                                               add_generation_prompt=False, tokenize=True))
            self._sys_n = len(one) - (len(two) - len(one))
        ids = _ids(tok.apply_chat_template(_raws(msgs), tools=None, add_generation_prompt=True,
                                           tokenize=True))[self._sys_n:]
        return len(self._capped(ids))

    def completion(self, text: str) -> int:
        return len(self.tok().encode(text, add_special_tokens=False))


def _raws(msgs: Iterable[Msg]) -> list[dict]:
    out = []
    for m in msgs:
        if m.raw is None:
            raise _Skip("messages have no form a chat template takes")
        out.append(dict(m.raw))
    return out


def _cache_offset(sessions: Sequence[TraceSession], tally: "_Tally") -> int | None:
    """Median of ``(prompt_{k+1} - uncached_{k+1}) - prompt_k``: where the next turn's cache read sits."""
    resid = [(b.prompt_tokens - b.uncached_tokens) - a.prompt_tokens
             for s in sessions for a, b in zip(s.turns, s.turns[1:])
             if a.prompt_tokens is not None and b.prompt_tokens is not None and b.uncached_tokens is not None]
    if not resid:
        return None
    resid.sort()
    tally.cache_rule = (int(statistics.median(resid)), resid[len(resid) // 100], resid[-1 - len(resid) // 100],
                        len(resid))
    return int(statistics.median(resid))


def _token_counts(s: TraceSession, profile: Profile, render: _LazyRenderer, offset: int | None,
                  tally: "_Tally") -> list[tuple[int, Source, int, Source]]:
    tools = s.tools if s.tools is not None else render.header.tools
    out: list[list[Any]] = []
    for k, t in enumerate(s.turns):
        if any(v is not None and v < 0 for v in (t.prompt_tokens, t.completion_tokens, t.uncached_tokens)):
            raise _Skip("negative token counts in the trace (broken records)")
        if t.completion_tokens is not None:
            c, cp = t.completion_tokens, "traced"
        elif t.output is not None:
            c, cp = render.completion(t.output.text), "derived"
        else:
            raise _Skip("turns with neither a completion count nor its text")
        if t.prompt_tokens is not None:
            p, pp = t.prompt_tokens, "traced"
        elif profile.accumulate == "tokens" and t.context is not None:
            if k == 0:
                p = render.prompt(t.context, tools)
                if render.cap and p == render.cap:
                    tally.capped += 1
            else:
                prev, pc = out[k - 1][0], out[k - 1][2]
                obs = s.turns[k - 1].after
                p = prev + pc + (render.observation(obs) if obs else 0)
            pp = "derived"
        elif t.context is not None and render.available():
            p, pp = render.prompt(t.context, tools), "derived"
        else:
            p, pp = None, "derived"
        out.append([p, pp, c, cp])
    for k, row in enumerate(out):
        if row[0] is not None:
            continue
        t = s.turns[k]
        nxt = s.turns[k + 1] if k + 1 < len(s.turns) else None
        if (offset is not None and nxt is not None and nxt.prompt_tokens is not None
                and nxt.uncached_tokens is not None):
            guess = nxt.prompt_tokens - nxt.uncached_tokens - offset
            floor = t.uncached_tokens or 0
            if guess < floor:
                tally.cache_clamped += 1
            row[0] = min(max(guess, floor), nxt.prompt_tokens)
            tally.cache_derived += 1
        elif t.uncached_tokens is not None and nxt is None and k == 0:
            row[0] = t.uncached_tokens  # a lone turn: the uncached part is the floor, and all there is
            tally.uncached_floor += 1
        else:
            raise _Skip("prompts with no count, no messages to render and no neighbour to derive from")
    return [tuple(r) for r in out]  # type: ignore[misc]


# ---------------------------------------------------------------------------
# groups
# ---------------------------------------------------------------------------
def _groups(sessions: Sequence[TraceSession], profile: Profile) -> dict[str | None, list[TraceSession]]:
    out: dict[str | None, list[TraceSession]] = defaultdict(list)
    if not profile.group_by_problem:
        for s in sessions:
            out[s.group].append(s)
        return out
    by_key: dict[tuple[str, str], list[TraceSession]] = defaultdict(list)
    for s in sessions:
        by_key[(s.file, s.problem or "?")].append(s)
    for (fname, inst), members in by_key.items():
        # The file is part of the name: one problem's rollouts can sit in two
        # workers' files, and a group id doubles as a prefix id.
        tag = profile.file_tag(Path(fname).stem)
        size = profile.group_size
        chunks = [members[i:i + size] for i in range(0, len(members), size)] if size else [members]
        for k, chunk in enumerate(chunks):
            suffix = f"/{k}" if len(chunks) > 1 else ""
            out[f"{inst}@{tag}{suffix}"] = chunk
    return out


# ---------------------------------------------------------------------------
# shared prefixes from block hashes
# ---------------------------------------------------------------------------
_BLOCK_SIZES = (16, 32, 64, 128, 256, 512, 1024, 2048)


def _block_prefixes(sessions: Sequence[TraceSession], counts: Mapping[str, list[tuple[int, Source, int, Source]]],
                    tally: "_Tally") -> tuple[list[Prefix], dict[tuple[str, int], tuple[str, int]]]:
    """A :class:`~simulator.ir.Prefix` per run of leading blocks that two or more requests share.

    The hash sequences go into a trie, so two requests share the first d blocks
    exactly when their first d hashes agree. That holds whether the trace chains its
    hashes (Mooncake: a hash covers its block and everything before) or hashes each
    block alone. The block size is the one for which ``ceil(length / size)`` equals
    the number of hashes on at least 99% of requests, where the length is the one
    the trace says the hashes cover, or else the prompt; failing that, the hashes are
    not used and the note says so.
    """
    have = [(s.id, k, t.blocks, counts[s.id][k][0]) for s in sessions for k, t in enumerate(s.turns) if t.blocks]
    if not have:
        return [], {}
    covered = {(s.id, k): t.block_tokens if t.block_tokens is not None else counts[s.id][k][0]
               for s in sessions for k, t in enumerate(s.turns) if t.blocks}
    fit = {b: sum(len(h) == -(-covered[sid, k] // b) for sid, k, h, _ in have) / len(have) for b in _BLOCK_SIZES}
    size = max(fit, key=lambda b: fit[b])
    what = "the recorded hashed length" if any(t.block_tokens is not None for s in sessions for t in s.turns) \
        else "prompt"
    if fit[size] < 0.99:
        tally.blocks = (f"{len(have)} requests carry block hashes, but no block size fits {what} "
                        f"(best {size}: {fit[size]:.1%}); the hashes are not used")
        return [], {}
    trie: dict[tuple[int, int], int] = {}
    refs: Counter[int] = Counter()
    paths = []
    for sid, k, h, p in have:
        node, path = -1, []
        for x in h:
            node = trie.setdefault((node, x), len(trie))
            refs[node] += 1
            path.append(node)
        paths.append((sid, k, p, path))
    nodes: dict[int, Prefix] = {}
    by_request: dict[tuple[str, int], tuple[str, int]] = {}
    for sid, k, p, path in paths:
        deepest = None
        for d, node in enumerate(path):
            if refs[node] < 2:
                break
            if node not in nodes:
                nodes[node] = Prefix(id=f"block:{node}", tokens=(d + 1) * size,
                                     parent=f"block:{path[d - 1]}" if d else None, prov={"tokens": "derived"})
            deepest = nodes[node]
        if deepest is not None:
            by_request[(sid, k)] = (deepest.id, min(deepest.tokens, p))
    tally.blocks = (f"block size {size} inferred from the hashes (ceil({what}/{size}) = hash count on "
                    f"{fit[size]:.1%} of {len(have)} requests); {len(nodes)} runs of leading blocks shared by two "
                    f"or more requests became prefixes")
    return list(nodes.values()), by_request


# ---------------------------------------------------------------------------
# one session
# ---------------------------------------------------------------------------
def _real_clock(s: TraceSession) -> bool:
    """Timestamps that measured something: every turn has both, none runs backwards, and
    the requests took time. Traces that rebuild their clocks (zero-length spans one second
    apart) fail the last test."""
    lat = []
    for t in s.turns:
        if t.t_sent is None or t.t_done is None or t.t_done < t.t_sent:
            return False
        lat.append(t.t_done - t.t_sent)
    return bool(lat) and statistics.median(lat) > 0


def _parent(s: TraceSession, k: int, counts: list[tuple[int, Source, int, Source]],
            ids: list[str], profile: Profile, tally: "_Tally") -> tuple[str | None, int]:
    t = s.turns[k]
    p_k = counts[k][0]
    if profile.accumulate == "tokens":
        return ids[k - 1], counts[k - 1][0] + counts[k - 1][2]
    if t.context is not None and all(u.context is not None for u in s.turns[:k]):
        ck = [m.key() for m in t.context]
        best: tuple[int, int, int] | None = None  # (share, rank, j)
        for j in range(k - 1, -1, -1):
            cj = [m.key() for m in s.turns[j].context]  # type: ignore[union-attr]
            pj, full = counts[j][0], counts[j][0] + counts[j][2]
            out = s.turns[j].output
            if out is not None and ck[:len(cj) + 1] == cj + [out.key()]:
                cand = (min(full, p_k), 3, j)
            elif ck[:len(cj)] == cj:
                cand = (min(pj, p_k), 2, j)
            elif cj[:len(ck)] == ck:
                cand = (min(p_k, pj), 1, j)
            else:
                continue
            if best is None or cand[:2] > best[:2]:
                best = cand
            if cand[1] == 3:
                break  # the latest turn this one extends whole: nothing earlier shares more
        if best is None:
            tally.cold += 1
            return None, 0
        tally.parent_rank[best[1]] += 1
        return ids[best[2]], best[0]
    prev_full = counts[k - 1][0] + counts[k - 1][2]
    if p_k >= prev_full:
        tally.append_assumed += 1
        return ids[k - 1], prev_full
    tally.shrunk += 1
    return ids[k - 1], min(counts[k - 1][0], p_k)


_STOP: Mapping[str, StopReason] = {"length": "length", "max_tokens": "length", "stop": "eos",
                                   "end_turn": "eos", "tool_calls": "eos", "tool_use": "eos",
                                   "eos": "eos", "stop_sequence": "eos"}


def _session(s: TraceSession, counts: list[tuple[int, Source, int, Source]], group: str | None,
             prefix: Prefix | None, profile: Profile, priors: Priors, tally: "_Tally",
             by_block: Mapping[tuple[str, int], tuple[str, int]] | None = None) -> Session:
    sid = s.id
    tasks: list[Task] = []
    requests: list[Request] = []
    env = profile.envelope

    def estimated(tid: str, kind: str, after: tuple[str, ...], pool: str | None = None) -> Task:
        knob = KNOB_BY_KIND[kind]
        u = priors.require(knob, missing=f"{kind} durations", key=s.problem)
        tally.keyed[knob] += priors.keyed(knob, s.problem)
        t = Task(id=tid, session=sid, duration_s=0.0, pool=pool, kind=kind, after=after,
                 prov={"duration_s": "estimated"}, unknown={"duration_s": u})
        tasks.append(t)
        tally.estimated[knob] += 1
        return t

    def measured(tid: str, d: float, how: Source, after: tuple[str, ...], cap: bool = False) -> Task:
        t = Task(id=tid, session=sid, duration_s=d, kind="tool", after=after, duration_is_cap=cap,
                 prov={"duration_s": how})
        tasks.append(t)
        tally.durations[how] += 1
        return t

    after: tuple[str, ...] = ()
    start = None
    if env is not None:
        # Runs inside the container the lease below holds; a pool here would take a second slot.
        start = estimated(f"{sid}/start", "container_start", ())
        after = (start.id,)
    clock = _real_clock(s)
    if not clock and any(t.t_sent is not None and t.t_done is not None for t in s.turns):
        tally.fake_clock += 1
    ids = [f"{sid}/gen{k + 1}" for k in range(len(s.turns))]
    last = len(s.turns) - 1
    for k, t in enumerate(s.turns):
        p, pp, c, cp = counts[k]
        if k == 0:
            parent, shared = (prefix.id, prefix.tokens) if prefix is not None else (None, 0)
        else:
            parent, shared = _parent(s, k, counts, ids, profile, tally)
        blk = (by_block or {}).get((sid, k))
        if blk is not None and blk[1] > shared:
            parent, shared = blk
            tally.block_parents += 1
        stop: StopReason = _STOP.get((t.finish or "").lower(), "unknown")
        cap = t.max_new_tokens
        if cap is not None and (c > cap or (stop == "length" and c != cap)):
            tally.bad_cap += 1
            cap = None
            stop = "unknown" if stop == "length" else stop
        prov: dict[str, Source] = {"prompt_tokens": pp, "completion_tokens": cp, "shared_prefix_tokens": "derived"}
        if cap is not None:
            prov["max_new_tokens"] = "traced"
        if stop != "unknown":
            prov["stop"] = "traced"
        requests.append(Request(id=ids[k], session=sid, prompt_tokens=p, prefix_parent=parent,
                                shared_prefix_tokens=min(shared, p), completion_tokens=c, after=after,
                                stop=stop, max_new_tokens=cap, prov=prov))
        tally.prov["prompt_tokens", pp] += 1
        tally.prov["completion_tokens", cp] += 1
        after = (ids[k],)

        # what ran between this response and the next request
        text = t.output.text if t.output is not None else ""
        parsed = len(t.output.calls) if t.output is not None else None
        if t.calls is not None:
            n, nudged = len(t.calls), False
            tally.calls_from["recorded calls"] += 1
        else:
            n, nudged = profile.calls_that_ran(text, t.after or (), parsed, s.exit, k == last)
            if n and k < last:
                tally.calls_from["the messages that answered the turn" if t.after else
                                 "the calls written in the output"] += 1
        if t.after and k < last and not any(m.role in ("tool", "function", "ipython") for m in t.after):
            tally.user_turns += 1
        if nudged:
            tally.nudges += 1
            first = t.after[0].text if t.after else ""
            if profile.nudge_starts and not first.startswith(profile.nudge_starts):
                tally.unknown_nudges += 1
        if t.calls is not None and t.calls and all(x.duration_s is not None for x in t.calls):
            for ci, x in enumerate(t.calls):
                after = (measured(f"{sid}/tool{k + 1}.{ci}", x.duration_s, "traced", after,  # type: ignore[arg-type]
                                  bool(x.timed_out)).id,)
            continue
        gap = (s.turns[k + 1].t_sent - t.t_done) if clock and k < last else None  # type: ignore[operator]
        if gap is not None and gap >= 0 and (n or 0) + (0 if nudged else 1) > 0:
            # One task holds the whole gap: how it splits across the calls is not measured.
            after = (measured(f"{sid}/tool{k + 1}", gap, "derived", after).id,)
            if n and n > 1:
                tally.multi_call_gaps += 1
            continue
        if gap is not None and gap < 0:
            tally.negative_gaps += 1
        if n is None:
            if k == last:
                continue
            if profile.calls_per_turn is None:
                tally.unknown_between += 1
                continue
            n = profile.calls_per_turn
            tally.assumed_calls += 1
        for ci in range(n):
            after = (estimated(f"{sid}/tool{k + 1}.{ci}", "tool", after).id,)

    leases: tuple[Lease, ...] = ()
    if env is not None:
        if s.exit not in env.submitted:
            after = (estimated(f"{sid}/patch_recover", "patch_recover", after).id,)
        cleanup = estimated(f"{sid}/cleanup", "cleanup", after)
        estimated(f"{sid}/eval", "eval", (cleanup.id,), pool=env.eval_pool)
        leases = (Lease(pool=env.container_pool, acquire_before=start.id, release_after=cleanup.id),)  # type: ignore[union-attr]
    return Session(id=sid, requests=tuple(requests), tasks=tuple(tasks), group=group, problem=s.problem,
                   leases=leases)


# ---------------------------------------------------------------------------
# notes
# ---------------------------------------------------------------------------
class _Tally:
    def __init__(self) -> None:
        self.prov: Counter[tuple[str, str]] = Counter()
        self.durations: Counter[str] = Counter()
        self.estimated: Counter[str] = Counter()
        self.keyed: Counter[str] = Counter()
        self.parent_rank: Counter[int] = Counter()
        self.calls_from: Counter[str] = Counter()
        self.block_parents = 0
        self.skipped: Counter[str] = Counter()
        self.cache_rule: tuple[int, int, int, int] | None = None
        self.extra: list[str] = []
        self.malformed = self.capped = self.cache_derived = self.cache_clamped = self.uncached_floor = 0
        self.cold = self.append_assumed = self.shrunk = self.uneven_groups = 0
        self.nudges = self.unknown_nudges = self.assumed_calls = self.multi_call_gaps = 0
        self.fake_clock = self.negative_gaps = self.bad_cap = self.unknown_between = self.user_turns = 0
        self.blocks: str | None = None

    def notes(self, n_sessions: int, profile: Profile) -> tuple[str, ...]:
        n_req = sum(v for (f, _), v in self.prov.items() if f == "prompt_tokens")
        out = [f"{n_req} requests over {n_sessions} sessions, read with the {profile.name!r} profile"]
        for f in ("prompt_tokens", "completion_tokens"):
            parts = [f"{v} {how}" for (g, how), v in sorted(self.prov.items()) if g == f]
            out.append(f"{f}: " + ", ".join(parts))
        if self.capped:
            out.append(f"{self.capped} first prompts hit the prompt_length cap and were left-truncated, as the "
                       f"harness did")
        if self.cache_derived:
            med, lo, hi, n = self.cache_rule or (0, 0, 0, 0)
            out.append(f"{self.cache_derived} prompts without a count derived from the next turn's cache read, "
                       f"at the trace's own offset {med} (p1 {lo}, p99 {hi} over {n} turns that have both); "
                       f"{self.cache_clamped} held up to their own uncached count")
        if self.uncached_floor:
            out.append(f"{self.uncached_floor} single-turn sessions took their uncached count as the prompt")
        if self.parent_rank or self.cold:
            r = self.parent_rank
            out.append(f"prefix parents from messages: {r[3]} extend an earlier turn whole, {r[2]} keep its "
                       f"prompt but not its output verbatim (share only the prompt), {r[1]} are a prefix of "
                       f"an earlier prompt (rollback), {self.cold} share nothing whole with any earlier turn "
                       f"and start cold")
        if self.append_assumed or self.shrunk:
            out.append(f"prefix parents from counts alone: {self.append_assumed} turns assumed to extend the "
                       f"previous one whole, {self.shrunk} shorter than it and sharing only its prompt")
        if self.uneven_groups:
            out.append(f"{self.uneven_groups} groups whose first prompts differ in length; the prefix is the shortest")
        if self.user_turns:
            out.append(f"{self.user_turns} turns are answered by a user message: no task, and the user's "
                       f"think time is not modelled")
        if self.nudges:
            out.append(f"{self.nudges} turns followed by a nudge instead of an observation: context, no task")
        if self.unknown_nudges:
            out.append(f"{self.unknown_nudges} of those nudges start with no known nudge text; they were treated "
                       f"as nudges because they are not observations")
        if self.calls_from:
            out.append("tool calls per turn counted from " + ", ".join(
                f"{what} ({n} turns)" for what, n in self.calls_from.most_common()))
        if self.assumed_calls:
            out.append(f"{self.assumed_calls} turns record no calls and are taken to have run "
                       f"{profile.calls_per_turn} each, as the profile says")
        if self.unknown_between:
            out.append(f"{self.unknown_between} turns are followed by another with no record of what happened in "
                       f"between; nothing is modelled there (a profile's calls_per_turn can assume tool calls)")
        if self.blocks:
            out.append(self.blocks + (f"; {self.block_parents} requests hang off one" if self.block_parents else ""))
        if self.durations["traced"]:
            out.append(f"{self.durations['traced']} tool durations recorded by the trace")
        if self.durations["derived"]:
            out.append(f"{self.durations['derived']} tool durations are the gap from one response to the next "
                       f"request: harness time is in them, so they are an upper bound"
                       + (f"; {self.multi_call_gaps} gaps cover several calls in one task" if self.multi_call_gaps else ""))
        if self.fake_clock:
            out.append(f"{self.fake_clock} sessions have timestamps that measured nothing (zero-length or "
                       f"backwards requests); their gaps were not used")
        if self.negative_gaps:
            out.append(f"{self.negative_gaps} gaps were negative (overlapping requests) and fell back to the prior")
        if self.bad_cap:
            out.append(f"{self.bad_cap} requests whose recorded cap disagrees with their completion; cap dropped")
        if self.malformed:
            out.append(f"{self.malformed} sessions skipped as malformed for this profile")
        for reason, n in self.skipped.items():
            out.append(f"{n} sessions skipped: {reason}")
        out.append("observed is empty: whatever answers the trace holds came from another system")
        out += estimated_notes(self.estimated, self.keyed)
        return tuple(out + list(profile.notes) + self.extra)
