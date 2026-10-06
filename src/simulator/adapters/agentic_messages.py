"""Our harness's conversation dump (``messages-<pid>.jsonl``), as a profile of the one adapter.

The dump is a conversation list per episode plus a header row with the chat
template, tool schemas and ``prompt_length`` the harness rendered with, so
:mod:`~simulator.adapters.shapes` reads it like any other conversation and
:mod:`~simulator.adapters.trace` derives every count from the text. What only
this harness can say lives here, as data:

- The loop is token-level (verl): turn k+1's prompt is turn k's whole sequence
  plus the observation, rendered with the template's system prefix stripped
  (``agent_loop.apply_chat_template(..., remove_system_prompt=True)``), and every
  render is left-truncated to ``prompt_length``. Measured on run 20261005-113503
  (2,208 episodes): completions 97.9% exact, observations 100%; the rest is the
  model sampling a non-canonical token split, which re-encoding cannot reproduce.
- A tool message is an observation when it starts the way the harness's
  ``observation_template`` does (configs/agent.yaml); any other is a nudge,
  context with no work behind it. The last turn, which nothing follows, is read
  from its ``<tool_call>`` blocks and the exit status: the blocks alone would
  undercount (on run 20261005-113503 the model wrote 372 calls with raw tabs, bad
  escapes or no closing tag that strict JSON rejects and the harness's salvage
  ran anyway). The harness runs a turn's calls one after another.
- Every episode holds a container from start to cleanup, recovers a patch when it
  did not submit, and is graded in an eval container: :class:`~simulator.adapters.trace.Envelope`.
  All of those durations, and every tool's, are estimated from the caller's
  priors, keyed by the episode's ``instance_id``.
- The eight rollouts of one problem in one worker's file are one group and share
  the first prompt as a :class:`~simulator.ir.Prefix`. A worker finishes every
  rollout of one step before the next, so ``group_size`` splits an instance's
  rollouts into consecutive groups in file order.

The observed half is empty by construction: at use time there is no timeline,
and the pipeline has to run on this alone (`SIMULATOR.md` §13, rule 6).
"""

from __future__ import annotations

import json
import re
from dataclasses import replace
from pathlib import Path
from typing import Any, Mapping, Sequence

from simulator.adapters.trace import Envelope, Msg, Profile, Tokenizer

NAME = "agentic-messages"
_FILE = "messages-*.jsonl"
_TOOL_CALL = re.compile(r"<tool_call>(.*?)</tool_call>", re.DOTALL)
#: How the harness's observation template starts (configs/agent.yaml,
#: ``observation_template``): a returncode, or an exception block before it. A
#: tool message that starts with neither is one of the nudges -- the format-error
#: one ("Tool call error:") or the one for a response cut off at the token limit.
_OBSERVATION_STARTS = ("<returncode>", "<exception>")
_NUDGE_STARTS = ("Tool call error", "Your previous response reached the output token limit")

#: Exits in which the last turn's tool ran and ended the episode (submit, or the
#: repeat guard refusing a call), against exits in which no call ran at all.
_ENDED_BY_A_CALL = {"Submitted", "RepetitionLimit"}
_ENDED_WITHOUT_A_CALL = {"FormatErrorLimit", "NoToolCall", "ContextLimit"}


def _count_calls(text: str) -> int:
    """Parsable ``<tool_call>`` blocks. Only a fallback for a turn nothing follows."""
    n = 0
    for body in _TOOL_CALL.findall(text):
        try:
            call = json.loads(body.strip(), strict=False)  # the model's raw tabs are fine
        except json.JSONDecodeError:
            continue
        if isinstance(call, dict) and call.get("name"):
            n += 1
    return n


def _is_observation(msg: Mapping[str, Any]) -> bool:
    return (msg.get("content") or "").startswith(_OBSERVATION_STARTS)


def _calls_that_ran(text: str, obs: Sequence[Mapping[str, Any]], exit_status: str | None) -> tuple[int, bool]:
    """How many tools a turn ran, and whether what followed was a nudge.

    With tool messages: one call per observation, or none if it was a nudge.
    Without any (the episode ended on this turn): the exit says whether a call
    ended it, and the blocks say how many there were, floored at one.
    """
    if obs:
        if not _is_observation(obs[0]):
            return 0, True
        return len(obs), False
    parsed = _count_calls(text)
    if exit_status in _ENDED_BY_A_CALL:
        return max(1, parsed), False
    if exit_status in _ENDED_WITHOUT_A_CALL:
        return 0, False
    return parsed, False


def _hook(text: str, after: Sequence[Msg], parsed: int | None, exit_status: str | None,
          last: bool) -> tuple[int, bool]:
    return _calls_that_ran(text, [m.raw or {"role": m.role, "content": m.text} for m in after], exit_status)


def _well_formed(msgs: Sequence[Mapping[str, Any]]) -> bool:
    return (len(msgs) >= 3 and msgs[0].get("role") == "system" and msgs[1].get("role") == "user"
            and msgs[2].get("role") == "assistant")


def _matches(path: Path) -> bool:
    if path.is_dir():
        return any(path.glob(_FILE))
    return path.is_file() and path.match(_FILE)


AGENTIC = Profile(
    name=NAME, part="messages", files=_FILE, accumulate="tokens", requires_header=True,
    group_by_problem=True, file_tag=lambda stem: stem.removeprefix("messages-"),
    calls_that_ran=_hook, well_formed=_well_formed, nudge_starts=_NUDGE_STARTS, envelope=Envelope(),
    match=_matches,
    notes=("max_new_tokens left unset for the config bundle; stop is 'unknown'; no tool call can be marked "
           "as a timeout: a conversation records none of these",
           "no timestamps, so the run's training steps are not separable: all episodes in one workload"),
)


def AgenticMessagesAdapter(tokenizer: Tokenizer | None = None, prompt_length: int | None = None,
                           group_size: int | None = None):
    """The one adapter, held to this profile. ``prompt_length`` overrides the header's;
    ``group_size`` splits an instance's rollouts into consecutive groups of that size."""
    from simulator.adapters.generic import TraceAdapter  # generic imports this module
    return TraceAdapter(profile=replace(AGENTIC, group_size=group_size), tokenizer=tokenizer,
                        prompt_length=prompt_length, name=NAME)
