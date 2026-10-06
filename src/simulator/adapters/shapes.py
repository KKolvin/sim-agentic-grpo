"""Read any trace file into :mod:`simulator.adapters.trace` records, by convention, not by name.

A row is recognised by the shape it has, and the names below are the spellings
the common conventions use, not the names of any one dataset:

- **OTel GenAI spans**: a ``spans`` list whose chat spans carry ``gen_ai.*``
  attributes (`semantic conventions`_). Each chat span is one turn: its input and
  output messages, ``gen_ai.usage.*`` counts, start and end time, requested cap.
- **A conversation**: a list of ``{role, content}`` messages (or ``{from,
  value}``), one turn per assistant message, with whatever followed it until the
  next one. Per-message ``usage`` is read when present.
- **Per-turn counts**: a list of records with token counts under any of the usual
  names (``prompt_tokens``/``input_tokens``/``isl``, ...), optionally the output
  text and timestamps.
- **One request per row**: a flat table (CSV, parquet, JSON lines) whose rows
  carry the counts themselves -- a serving log. Rows sharing a conversation key
  are one session, in turn order; otherwise every row is its own session. Block
  content hashes (``hash_ids``) are kept: they say which requests share a prefix.

Field names are matched ignoring case, spaces and punctuation, so ``Request
tokens``, ``ContextTokens`` and ``request_tokens`` are the same field.

A row with none of these but a chat template, tool schemas or a prompt length is
a header. Anything else is counted and skipped, so a directory can hold files
that are not traces.

.. _semantic conventions: https://opentelemetry.io/docs/specs/semconv/gen-ai/
"""

from __future__ import annotations

import json
import re
from collections import Counter
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Iterator, Mapping, Sequence

from simulator.adapters.trace import Call, Header, Msg, TraceSession, Turn

SUFFIXES = (".jsonl", ".json", ".parquet", ".csv")

_CONVERSATION_KEYS = ("messages", "conversation", "conversations", "trajectory", "history", "chat")
_TURN_LIST_KEYS = ("turns", "requests", "llm_calls", "calls", "steps")
_PROMPT = ("prompt_tokens", "input_tokens", "isl", "num_prompt_tokens", "n_prompt_tokens", "prompt_len",
           "request_tokens", "context_tokens", "context_token", "input_length", "input_len", "num_input_tokens",
           "token_input", "reported_token_input", "tokens_in")
_COMPLETION = ("completion_tokens", "output_tokens", "osl", "num_completion_tokens", "n_completion_tokens",
               "completion_len", "response_tokens", "generated_tokens", "generated_token", "output_length",
               "output_len", "num_output_tokens", "token_output", "reported_token_output", "tokens_out")
_UNCACHED = ("isl_new", "uncached_tokens", "uncached_prompt_tokens")
_CACHED = ("cached_tokens", "cache_read_input_tokens", "prompt_cached_tokens")
_OUTPUT_TEXT = ("osl_text", "output_text", "completion_text", "response_text", "response", "completion", "output")
_T_SENT = ("t_sent", "t_start", "start_time", "started_at", "created_at", "request_time")
_T_DONE = ("t_done", "t_end", "end_time", "finished_at", "response_time")
_SESSION_ID = ("session_id", "traj", "trajectory_id", "traj_id", "rollout_id", "id")
_PROBLEM = ("instance_id", "task_id", "problem_id", "problem")
#: For one-request-per-row tables: the key that makes rows one session, and their order in it.
_CONVERSATION_KEY = ("conversation_id", "session_id", "chat_id", "thread_id", "traj", "trajectory_id")
_TURN_INDEX = ("turn", "turn_index", "turn_id", "round")
_BLOCKS = ("hash_ids", "block_hashes", "block_ids", "bucket_ids")
#: The length the block hashes cover, when a trace records it apart from the prompt.
_BLOCK_LEN = ("token_count", "hashed_tokens", "block_tokens")
_CAP = ("max_new_tokens", "max_tokens", "max_completion_tokens")
_GROUP = ("group", "group_id")
_EXIT = ("exit_status", "exit", "termination_reason", "status")
_TOOLS = ("tool_schemas", "tools", "tool_definitions")
_HEADER = ("chat_template", "tool_schemas", "prompt_length")

#: Text conventions for a tool call inside an output, for traces that keep the
#: output as one string: Hermes/Qwen ``<tool_call>{json}</tool_call>`` (counted
#: only when the JSON parses and names a function) and ``[tool_use:name]``.
_HERMES = re.compile(r"<tool_call>(.*?)</tool_call>", re.DOTALL)
_TOOL_USE = re.compile(r"\[tool_use:([^\]\s]+)\]")


@dataclass
class Read:
    header: Header | None
    sessions: list[TraceSession]
    shapes: Counter[str] = field(default_factory=Counter)
    unrecognised: Counter[str] = field(default_factory=Counter)
    files: list[str] = field(default_factory=list)


def files(path: Path, glob: str = "*") -> list[Path]:
    if path.is_dir():
        return sorted(p for p in path.glob(glob) if p.is_file() and p.suffix in SUFFIXES)
    return [path] if path.is_file() and path.suffix in SUFFIXES and path.match(glob) else []


def rows(f: Path, where: Mapping[str, Any] | None = None, limit: int | None = None) -> Iterator[dict[str, Any]]:
    n = 0
    for r in _raw_rows(f, where):
        if where and not _selected(r, where):
            continue
        yield r
        n += 1
        if limit is not None and n >= limit:
            return


def _raw_rows(f: Path, where: Mapping[str, Any] | None) -> Iterator[dict[str, Any]]:
    if f.suffix == ".parquet":
        import pyarrow.parquet as pq  # only for this container
        pf = pq.ParquetFile(f)
        names = set(pf.schema_arrow.names)
        filters = [(k, "=", v) for k, v in (where or {}).items()
                   if k in names and isinstance(v, (str, int, float, bool))
                   and not str(pf.schema_arrow.field(k).type).startswith("list")]
        if filters:
            yield from pq.read_table(f, filters=filters).to_pylist()
            return
        for g in range(pf.num_row_groups):
            yield from pf.read_row_group(g).to_pylist()
        return
    if f.suffix == ".csv":
        import csv
        with open(f, newline="") as fh:
            for r in csv.DictReader(fh):
                yield {k: _number(v) for k, v in r.items() if k is not None}
        return
    if f.suffix == ".json":
        doc = json.loads(f.read_text())
        if isinstance(doc, list):
            items = doc
        elif isinstance(doc, dict):
            items = doc.get("data") or doc.get("rows") or [doc]
        else:
            items = []
        yield from (r for r in items if isinstance(r, dict))
        return
    with open(f) as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(r, dict):
                yield r


def _selected(r: Mapping[str, Any], where: Mapping[str, Any]) -> bool:
    for k, v in where.items():
        got = r.get(k)
        if isinstance(got, (list, tuple)) and not isinstance(v, (list, tuple)):
            if v not in got:
                return False
        elif got != v:
            return False
    return True


def read(paths: Sequence[Path], where: Mapping[str, Any] | None = None, limit: int | None = None) -> Read:
    header: dict[str, Any] = {}
    out = Read(header=None, sessions=[])
    keyed: dict[str, list[_RequestRow]] = {}
    for f in paths:
        used = False
        for i, r in enumerate(rows(f, where, None)):
            s = recognise(r, f.name, i, out.shapes)
            if isinstance(s, _RequestRow):
                used = True
                if s.key is None:
                    out.sessions.append(s.session())
                else:
                    keyed.setdefault(s.key, []).append(s)
                if limit is not None and len(out.sessions) + len(keyed) >= limit:
                    break
            elif s is not None:
                out.sessions.append(s)
                used = True
                if limit is not None and len(out.sessions) >= limit:
                    break
            elif _is_header(r):
                for k, v in r.items():
                    header.setdefault(k, v)
                used = True
            else:
                out.unrecognised[f.name] += 1
        if used:
            out.files.append(f.name)
        if limit is not None and len(out.sessions) + len(keyed) >= limit:
            break
    for key, members in keyed.items():
        members.sort(key=lambda m: m.order)
        out.sessions.append(TraceSession(id=str(key), turns=tuple(m.turn for m in members), file=members[0].file,
                                         **{k: v for k, v in members[0].common.items() if k not in ("id", "file")}))
    if header:
        out.header = Header(model=header.get("model"), chat_template=header.get("chat_template"),
                            tools=_tools(header), prompt_length=header.get("prompt_length"))
    return out


def claims(path: Path, glob: str = "*", probe: int = 3) -> bool:
    for f in files(path, glob):
        try:
            for i, r in enumerate(rows(f, None, probe)):
                if recognise(r, f.name, i, Counter()) is not None:
                    return True
        except Exception:
            continue
    return False


def _is_header(r: Mapping[str, Any]) -> bool:
    return r.get("kind") == "meta" or any(k in r for k in _HEADER)


# ---------------------------------------------------------------------------
# recognising one row
# ---------------------------------------------------------------------------
@dataclass
class _RequestRow:
    """One row of a one-request-per-row table, before its session is assembled."""

    turn: Turn
    key: str | None
    order: tuple
    file: str
    common: dict[str, Any]

    def session(self) -> TraceSession:
        return TraceSession(turns=(self.turn,), **self.common)


def recognise(r: Mapping[str, Any], fname: str, i: int,
              shapes: Counter[str]) -> TraceSession | _RequestRow | None:
    spans = r.get("spans")
    if isinstance(spans, list) and spans and any(_is_chat_span(s) for s in spans if isinstance(s, dict)):
        shapes["otel"] += 1
        return _otel(r, spans, fname, i)
    for k in _CONVERSATION_KEYS:
        msgs = r.get(k)
        if _is_conversation(msgs):
            shapes["conversation"] += 1
            return _conversation(r, msgs, fname, i)
    for k in _TURN_LIST_KEYS:
        turns = r.get(k)
        if isinstance(turns, (list, tuple)) and turns:
            parsed = [_maybe_json(t) for t in turns]
            if all(isinstance(t, dict) for t in parsed) and any(_first(t, _COMPLETION) is not None for t in parsed):
                shapes["counts"] += 1
                return _counts(r, parsed, fname, i)
    if _first(r, _COMPLETION) is not None and not _is_event(r):
        shapes["request rows"] += 1
        return _request_row(r, fname, i)
    return None


def _common(r: Mapping[str, Any], fname: str, i: int) -> dict[str, Any]:
    sid = _first(r, _SESSION_ID)
    problem = _first(r, _PROBLEM)
    return {"id": str(sid) if sid is not None else f"{Path(fname).stem}:{i}", "file": fname,
            "problem": str(problem) if problem is not None else None,
            "group": _first(r, _GROUP), "exit": _first(r, _EXIT), "tools": _tools(r)}


# -- conversation -----------------------------------------------------------
def _is_conversation(msgs: Any) -> bool:
    return (isinstance(msgs, (list, tuple)) and len(msgs) >= 2 and all(isinstance(m, dict) for m in msgs)
            and all(("role" in m or "from" in m) for m in msgs))


def _role(m: Mapping[str, Any]) -> str:
    role = m.get("role") or m.get("from") or ""
    return {"human": "user", "gpt": "assistant", "model": "assistant", "observation": "tool",
            "function_call": "assistant", "ai": "assistant"}.get(role, role)


def _content_text(c: Any) -> str:
    if c is None:
        return ""
    if isinstance(c, str):
        return c
    if isinstance(c, list):
        out = []
        for p in c:
            if isinstance(p, str):
                out.append(p)
            elif isinstance(p, dict):
                t = p.get("text") or p.get("content")
                if isinstance(t, str):
                    out.append(t)
                elif isinstance(t, list):
                    out.append(_content_text(t))
        return "\n".join(out)
    return str(c)


def _chat_msg(m: Mapping[str, Any]) -> Msg:
    role = _role(m)
    text = _content_text(m.get("content", m.get("value", m.get("text"))))
    calls: list[tuple[str, str]] = []
    for tc in m.get("tool_calls") or ():
        fn = tc.get("function", tc) if isinstance(tc, dict) else {}
        calls.append((str(fn.get("name")), _canon(fn.get("arguments"))))
    if isinstance(m.get("content"), list):
        for p in m["content"]:
            if isinstance(p, dict) and p.get("type") == "tool_use":
                calls.append((str(p.get("name")), _canon(p.get("input"))))
    if role == "assistant" and not calls:
        calls = text_calls(text)
    raw = m if "role" in m and isinstance(m.get("content", ""), (str, type(None))) else \
        {"role": role, "content": text}
    return Msg(role=role, text=text, calls=tuple(calls), raw=raw)


def _conversation(r: Mapping[str, Any], raw_msgs: Sequence[Mapping[str, Any]], fname: str, i: int) -> TraceSession:
    msgs = [_chat_msg(m) for m in raw_msgs]
    asst = [j for j, m in enumerate(msgs) if m.role == "assistant" and j > 0]
    turns = []
    for n, j in enumerate(asst):
        end = asst[n + 1] if n + 1 < len(asst) else len(msgs)
        u = _usage(raw_msgs[j].get("usage") or raw_msgs[j].get("metrics") or {})
        turns.append(Turn(context=tuple(msgs[:j]), output=msgs[j], after=tuple(msgs[j + 1:end]), **u))
    return TraceSession(turns=tuple(turns), raw_messages=tuple(raw_msgs), **_common(r, fname, i))


# -- OTel GenAI ---------------------------------------------------------------
def _attrs(s: Mapping[str, Any]) -> Mapping[str, Any]:
    a = s.get("attributes") or {}
    if isinstance(a, list):  # OTLP JSON: [{key, value: {stringValue: ...}}]
        a = {x.get("key"): next(iter((x.get("value") or {}).values()), None) for x in a if isinstance(x, dict)}
    return a


def _is_chat_span(s: Mapping[str, Any]) -> bool:
    a = _attrs(s)
    return a.get("gen_ai.operation.name") == "chat" or (
        "gen_ai.input.messages" in a and a.get("gen_ai.operation.name") in (None, "text_completion"))


def _otel_msg(m: Mapping[str, Any]) -> Msg:
    role = _role(m)
    texts, calls = [], []
    for p in m.get("parts") or ():
        t = p.get("type")
        if t == "text":
            texts.append(str(p.get("content") or ""))
        elif t == "tool_call":
            calls.append((str(p.get("name")), _canon(p.get("arguments"))))
        elif t == "tool_call_response":
            texts.append(_content_text(p.get("response") if p.get("response") is not None else p.get("content")))
    if not m.get("parts") and "content" in m:
        texts.append(_content_text(m["content"]))
    text = "\n".join(texts)
    raw: dict[str, Any] = {"role": role, "content": text}
    if calls:
        raw["tool_calls"] = [{"type": "function", "function": {"name": n, "arguments": a}} for n, a in calls]
    return Msg(role=role, text=text, calls=tuple(calls), raw=raw)


def _otel(r: Mapping[str, Any], spans: Sequence[Mapping[str, Any]], fname: str, i: int) -> TraceSession:
    chat = [s for s in spans if isinstance(s, dict) and _is_chat_span(s) and not _failed(s)]
    chat.sort(key=lambda s: _time(s.get("start_time") or s.get("startTimeUnixNano")) or 0.0)
    turns = []
    tools = None
    for s in chat:
        a = _attrs(s)
        inp = [_otel_msg(m) for m in _maybe_json(a.get("gen_ai.input.messages")) or ()]
        sys_ = a.get("gen_ai.system_instructions")
        if sys_:
            inp.insert(0, Msg(role="system", text=_content_text(_maybe_json(sys_)),
                              raw={"role": "system", "content": _content_text(_maybe_json(sys_))}))
        outs = [_otel_msg(m) for m in _maybe_json(a.get("gen_ai.output.messages")) or ()]
        out = next((m for m in outs if m.role == "assistant"), outs[0] if outs else None)
        fin = a.get("gen_ai.response.finish_reasons")
        tools = tools or _maybe_json(a.get("gen_ai.tool.definitions"))
        cached = a.get("gen_ai.usage.cache_read_input_tokens")
        prompt = a.get("gen_ai.usage.input_tokens")
        turns.append(Turn(context=tuple(inp), output=out, prompt_tokens=prompt,
                          completion_tokens=a.get("gen_ai.usage.output_tokens"),
                          uncached_tokens=(prompt - cached) if prompt is not None and cached is not None else None,
                          t_sent=_time(s.get("start_time") or s.get("startTimeUnixNano")),
                          t_done=_time(s.get("end_time") or s.get("endTimeUnixNano")),
                          finish=(fin[0] if isinstance(fin, (list, tuple)) and fin else None),
                          max_new_tokens=a.get("gen_ai.request.max_tokens")))
    common = _common(r, fname, i)
    if common["tools"] is None and isinstance(tools, list):
        common["tools"] = tuple(tools)
    return TraceSession(turns=tuple(turns), **common)


def _failed(s: Mapping[str, Any]) -> bool:
    st = s.get("status")
    if isinstance(st, str):
        st = _maybe_json(st.replace("'", '"')) if st.startswith("{") else st
    return isinstance(st, dict) and st.get("code") in (2, "ERROR", "STATUS_CODE_ERROR")


# -- per-turn counts ----------------------------------------------------------
def _counts(r: Mapping[str, Any], turns: Sequence[Mapping[str, Any]], fname: str, i: int) -> TraceSession:
    out = []
    for t in turns:
        u = _usage(t)
        text = _first(t, _OUTPUT_TEXT)
        output = None
        if isinstance(text, str):
            output = Msg(role="assistant", text=text, calls=tuple(text_calls(text)))
        out.append(Turn(output=output, t_sent=_time(_first(t, _T_SENT)), t_done=_time(_first(t, _T_DONE)), **u))
    return TraceSession(turns=tuple(out), **_common(r, fname, i))


# -- one request per row ---------------------------------------------------------
def _is_event(r: Mapping[str, Any]) -> bool:
    """A row of an event log (``cat``, ``name``, ``t``): one event among tool calls, waits and
    trainer phases. Reading only its request events would drop the rest, and event logs are
    not assembled yet; our own timeline is one, and use time must not read it (§13, rule 7)."""
    return all(k in r for k in ("cat", "name", "t"))



def _request_row(r: Mapping[str, Any], fname: str, i: int) -> _RequestRow:
    blocks = _first(r, _BLOCKS)
    t_sent = _time(_first(r, _T_SENT + ("timestamp",)))
    turn = Turn(t_sent=t_sent, t_done=_time(_first(r, _T_DONE)), max_new_tokens=_cap(r),
                blocks=tuple(int(b) for b in blocks) if isinstance(blocks, (list, tuple)) and blocks else None,
                block_tokens=_first(r, _BLOCK_LEN) if blocks else None,
                **_usage(r))
    key = _first(r, _CONVERSATION_KEY)
    idx = _first(r, _TURN_INDEX)
    order = (idx if isinstance(idx, (int, float)) else float("inf"), t_sent if t_sent is not None else 0.0, fname, i)
    common = _common(r, fname, i)
    return _RequestRow(turn=turn, key=str(key) if key is not None else None, order=order, file=fname,
                       common=common)


def _cap(d: Mapping[str, Any]) -> int | None:
    v = _first(d, _CAP)
    if v is None and isinstance(d.get("sampling_params"), Mapping):
        v = _first(d["sampling_params"], _CAP)
    return int(v) if isinstance(v, (int, float)) and not isinstance(v, bool) and v > 0 else None


# ---------------------------------------------------------------------------
# small readers
# ---------------------------------------------------------------------------
def _usage(d: Mapping[str, Any]) -> dict[str, Any]:
    """Token counts under any of the usual spellings, nested ``usage`` included.

    Anthropic's ``input_tokens`` leaves out cache reads and writes, so a record
    that has those adds them back; OpenAI's ``prompt_tokens`` already includes
    its cached part.
    """
    if not isinstance(d, Mapping):
        return {}
    if isinstance(d.get("usage"), Mapping):
        inner = _usage(d["usage"])
        return {**inner, **{k: v for k, v in _usage({k: v for k, v in d.items() if k != "usage"}).items()
                            if v is not None}}
    prompt = _first(d, _PROMPT)
    if prompt is not None and "input_tokens" in d and (
            "cache_read_input_tokens" in d or "cache_creation_input_tokens" in d):
        prompt = d["input_tokens"] + (d.get("cache_read_input_tokens") or 0) + (d.get("cache_creation_input_tokens") or 0)
    uncached = _first(d, _UNCACHED)
    cached = _first(d, _CACHED)
    if uncached is None and cached is not None and prompt is not None:
        uncached = prompt - cached
    out = {"prompt_tokens": prompt, "completion_tokens": _first(d, _COMPLETION), "uncached_tokens": uncached}
    return {k: (int(v) if v is not None else None) for k, v in out.items()}


def text_calls(text: str) -> list[tuple[str, str]]:
    """Tool calls written into an output string, by the conventions in the module docstring."""
    calls = []
    for body in _HERMES.findall(text):
        try:
            call = json.loads(body.strip(), strict=False)  # raw tabs are fine
        except json.JSONDecodeError:
            continue
        if isinstance(call, dict) and call.get("name"):
            calls.append((str(call["name"]), _canon(call.get("arguments"))))
    calls += [(name, "") for name in _TOOL_USE.findall(text)]
    return calls


def _norm(k: Any) -> str:
    return re.sub(r"[^a-z0-9]", "", str(k).lower())


def _first(d: Mapping[str, Any], keys: Sequence[str]) -> Any:
    """The first of ``keys`` the record has, matched ignoring case, spaces and punctuation."""
    for k in keys:
        v = d.get(k)
        if v is not None:
            return v
    loose = {_norm(k): v for k, v in d.items()}
    for k in keys:
        v = loose.get(_norm(k))
        if v is not None:
            return v
    return None


def _number(v: str | None) -> Any:
    if v is None or v == "":
        return None
    for cast in (int, float):
        try:
            return cast(v)
        except ValueError:
            pass
    return v


def _tools(d: Mapping[str, Any]) -> tuple[Mapping[str, Any], ...] | None:
    for k in _TOOLS:
        v = _maybe_json(d.get(k))
        if isinstance(v, list) and v and all(isinstance(x, dict) for x in v):
            return tuple(v)
    return None


def _maybe_json(v: Any) -> Any:
    if isinstance(v, str) and v[:1] in "[{":
        try:
            return json.loads(v)
        except json.JSONDecodeError:
            return v
    return v


def _canon(args: Any) -> str:
    args = _maybe_json(args)
    try:
        return json.dumps(args, sort_keys=True)
    except TypeError:
        return str(args)


def _time(v: Any) -> float | None:
    if v is None:
        return None
    if isinstance(v, (int, float)):
        return float(v) / 1e9 if v > 1e17 else float(v)  # unix nanoseconds (OTLP) or seconds
    if isinstance(v, str):
        if v.isdigit():
            return _time(int(v))
        try:
            return datetime.fromisoformat(v.replace("Z", "+00:00")).timestamp()
        except ValueError:
            return None
    return None
