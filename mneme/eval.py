#!/usr/bin/env python3
"""Retrieval eval harness — the instrument. Measure recall@k + MRR over a golden set.

Nothing ships as "better" without moving these numbers. Loads the index ONCE, runs each golden query,
finds the rank of the expected note (first result whose id contains the `expect` substring), and
reports recall@{1,5,10} + MRR + per-query rank.

Usage: python -m mneme.eval [index_path] [golden_path]   (defaults: config.INDEX_PATH + config.GOLDEN_PATH)
"""
from __future__ import annotations

import json
import math
import os
import sys
import urllib.request

from . import config

OLLAMA = config.OLLAMA_URL
DEF_INDEX = config.INDEX_PATH
DEF_GOLDEN = config.GOLDEN_PATH


def embed(text: str, model: str) -> list[float]:
    req = urllib.request.Request(
        OLLAMA, data=json.dumps({"model": model, "prompt": text}).encode(),
        headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=30) as r:
        return json.loads(r.read())["embedding"]


def cosine(a, b) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a)); nb = math.sqrt(sum(y * y for y in b))
    return dot / (na * nb) if na and nb else 0.0


def rank_of(expect: str, ranked_ids: list[str]) -> int | None:
    el = expect.lower()
    for i, cid in enumerate(ranked_ids, 1):
        if el in cid.lower():
            return i
    return None


def main() -> int:
    index_path = sys.argv[1] if len(sys.argv) > 1 else DEF_INDEX
    golden_path = sys.argv[2] if len(sys.argv) > 2 else DEF_GOLDEN
    idx = json.load(open(index_path, encoding="utf-8"))
    model = idx["model"]
    chunks = idx["chunks"]
    golden = json.load(open(golden_path, encoding="utf-8"))

    ranks, misses = [], []
    print(f"index={os.path.basename(index_path)} model={model} chunks={len(chunks)} queries={len(golden)}\n")
    for g in golden:
        qv = embed(g["q"], model)
        scored = sorted(((cosine(qv, c["vector"]), c["id"]) for c in chunks), key=lambda x: -x[0])
        # dedupe by PARENT note (id before '#') so recall@k counts distinct notes, not chunks
        top_ids, seen = [], set()
        for _, cid in scored:
            note = cid.split("#", 1)[0]
            if note in seen:
                continue
            seen.add(note); top_ids.append(cid)
            if len(top_ids) >= 10:
                break
        r = rank_of(g["expect"], top_ids)
        ranks.append(r)
        top_score = scored[0][0] if scored else 0.0
        mark = f"@{r}" if r else "MISS"
        print(f"  {mark:>5}  (top={top_score:.3f})  {g['expect']:<32}  «{g['q'][:52]}»")
        if not r:
            misses.append((g["expect"], top_ids[:3]))

    n = len(golden)
    r1 = sum(1 for r in ranks if r and r <= 1) / n
    r5 = sum(1 for r in ranks if r and r <= 5) / n
    r10 = sum(1 for r in ranks if r and r <= 10) / n
    mrr = sum((1.0 / r) for r in ranks if r) / n
    print(f"\n  recall@1={r1:.2f}  recall@5={r5:.2f}  recall@10={r10:.2f}  MRR={mrr:.3f}")
    if misses:
        print("\n  MISSES (expected → top-3 returned):")
        for exp, got in misses:
            print(f"    {exp} → {got}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
