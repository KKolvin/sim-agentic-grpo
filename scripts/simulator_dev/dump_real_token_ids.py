#!/usr/bin/env python
"""Real agent conversations as token ids, one episode per line, for the cost grid's KV donors.

Rebuilds each episode's final sequence (last prompt + last completion) from a run's
``messages-<pid>.jsonl`` the way the agent loop built it, the same rendering as
``check_messages_reconstruction.py`` (93% of prompts exact, the rest off by a few tokens,
which does not matter here). ``scripts/measure_cost_grid.py --donor-tokens`` reads the
output: routing in MoE layers depends on what the tokens are, and a random-token prefill runs
10-35% faster than real text (SIMULATOR.md §8), so the routing check needs real text.

    python scripts/simulator_dev/dump_real_token_ids.py \
        analysis/20261006-030010/trajectories analysis/costgrid/real-tokens-030010.jsonl
"""

from __future__ import annotations

import argparse
import glob
import json
import os

ap = argparse.ArgumentParser()
ap.add_argument("dir")
ap.add_argument("out")
ap.add_argument("--max-tokens", type=int, default=6_000_000, help="stop after this many in total")
a = ap.parse_args()

from transformers import AutoTokenizer  # noqa: E402
from verl.utils.tokenizer.chat_template import apply_chat_template, initialize_system_prompt  # noqa: E402

meta, episodes = None, []
for mp in sorted(glob.glob(os.path.join(a.dir, "messages-*.jsonl"))):
    recs = [json.loads(line) for line in open(mp) if line.strip()]
    if recs and recs[0].get("kind") == "meta":
        meta = meta or recs[0]
        episodes += [r for r in recs[1:] if r.get("kind") == "episode"]

tok = AutoTokenizer.from_pretrained(meta["model"])
tok.chat_template = meta["chat_template"]
tools, cap = meta["tool_schemas"], meta.get("prompt_length") or 4096
sys_n = len(initialize_system_prompt(tok))


def render(msgs, with_tools):
    return list(apply_chat_template(tok, msgs, tools=tools if with_tools else None,
                                    add_generation_prompt=True, tokenize=True))


total, kept = 0, 0
with open(a.out, "w") as f:
    for e in episodes:
        msgs = e["messages"]
        if len(msgs) < 3 or msgs[0]["role"] != "system":
            continue
        ids = render(msgs[:2], True)[-cap:]
        i = 2
        while i < len(msgs):
            if msgs[i]["role"] == "assistant":
                ids += tok.encode(msgs[i]["content"], add_special_tokens=False)
                i += 1
                continue
            obs = []
            while i < len(msgs) and msgs[i]["role"] != "assistant":
                obs.append(msgs[i])
                i += 1
            ids += render(obs, False)[sys_n:][-cap:]
        f.write(json.dumps(ids) + "\n")
        total += len(ids)
        kept += 1
        if total >= a.max_tokens:
            break
print(f"{kept} episodes, {total:,} tokens -> {a.out}")
