#!/usr/bin/env python3
"""Supersession scan — surface memory pairs that likely say the same thing (candidate stale/superseded).

The staleness sweep (sweep.py / vault_sweep.py) catches PATH rot — a dead file/wikilink reference. It does
NOT catch a fact that is path-valid but SEMANTICALLY superseded (a newer memory/decision contradicts an
older one). Age != superseded; path-valid != true. This pass embeds every memory (same model as the
retrieval pipeline) and reports high-similarity PAIRS.

IMPORTANT — this is a CHEAP PRE-FILTER, not a supersession detector: cosine surfaces SAME-TOPIC pairs, and
same-topic != supersedes (two complementary memories on one project are both valid). Actual supersession is
MEANING (does B contradict/update A?) → needs a JUDGE, not a threshold. So this narrows the O(n^2) pairs to
the same-topic candidates that a downstream consolidation step (or a per-pair LLM check) then ADJUDICATES.
NEVER auto-prune. Already `status: superseded|deprecated` memories are excluded.

Output: JSON candidate pairs {a, b, sim, newer, older} above SIM_THRESH → feed the judge; on a real
supersession, mark the OLDER `status: superseded` + `superseded_by:` (deprecate-don't-delete, keep provenance).
"""
from __future__ import annotations

import json
import math
import os
import re
import sys
import urllib.request
from glob import glob

from . import config

DEF_MEM = config.MEMORY_DIR
MODEL = config.EMBED_MODEL
OLLAMA = config.OLLAMA_URL
SIM_THRESH = float(os.environ.get("MNEME_SUPERSEDE_SIM", "0.86"))
FM_RE = re.compile(r"^---\n(.*?)\n---\n", re.DOTALL)


def embed(text: str) -> list[float]:
    req = urllib.request.Request(OLLAMA, data=json.dumps({"model": MODEL, "prompt": text}).encode(),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=30) as r:
        return json.loads(r.read())["embedding"]


def cosine(a, b):
    dot = sum(x * y for x, y in zip(a, b)); na = math.sqrt(sum(x * x for x in a)); nb = math.sqrt(sum(y * y for y in b))
    return dot / (na * nb) if na and nb else 0.0


def status_of(fm: str) -> str:
    m = re.search(r'^\s*status:\s*(\w+)', fm, re.MULTILINE)
    return m.group(1).lower() if m else "active"


def main() -> int:
    mem = sys.argv[1] if len(sys.argv) > 1 else DEF_MEM
    docs = []
    for path in sorted(glob(os.path.join(mem, "*.md"))):
        if os.path.basename(path) == "MEMORY.md":
            continue
        txt = open(path, encoding="utf-8", errors="ignore").read()
        fm = (FM_RE.match(txt).group(1) if FM_RE.match(txt) else "")
        if status_of(fm) in ("superseded", "deprecated"):
            continue
        try:
            vec = embed(txt[:2500])
        except Exception as e:  # noqa: BLE001
            print(f"  skip {os.path.basename(path)}: {e}", file=sys.stderr)
            continue
        docs.append({"name": os.path.basename(path), "mtime": os.path.getmtime(path), "vec": vec})

    pairs = []
    for i in range(len(docs)):
        for j in range(i + 1, len(docs)):
            s = cosine(docs[i]["vec"], docs[j]["vec"])
            if s >= SIM_THRESH:
                a, b = docs[i], docs[j]
                newer, older = (a, b) if a["mtime"] >= b["mtime"] else (b, a)
                pairs.append({"a": a["name"], "b": b["name"], "sim": round(s, 3),
                              "newer": newer["name"], "older": older["name"]})
    pairs.sort(key=lambda p: -p["sim"])
    print(json.dumps({"memory_dir": mem, "scanned": len(docs), "sim_thresh": SIM_THRESH,
                      "candidate_supersession_pairs": pairs}, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
