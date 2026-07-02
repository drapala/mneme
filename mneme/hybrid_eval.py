#!/usr/bin/env python3
"""Hybrid retrieval eval: dense cosine + BM25, fused by Reciprocal Rank Fusion.

Pure-python BM25 over the stored chunk text (no new dep — prove the hybrid win before migrating to a
real FTS backend). BM25 catches the keyword-recoverable misses that dense embedding drops (exact
identifiers, author/email tokens, acronyms). Reads the same golden set as eval.py.

Also the shared library: substrate_query / substrate_daemon / rerank_eval import BM25, cosine,
dedupe_notes, embed, rank_of, and rrf from here.

Usage: python -m mneme.hybrid_eval [index_path] [golden] [--field text|preview] [--k1 1.5 --b 0.75 --rrf 60]
"""
from __future__ import annotations

import json
import math
import os
import re
import sys
import unicodedata
import urllib.request
from collections import Counter

from . import config

OLLAMA = config.OLLAMA_URL
DEF_INDEX = config.INDEX_PATH
DEF_GOLDEN = config.GOLDEN_PATH


def embed(text, model):
    req = urllib.request.Request(OLLAMA, data=json.dumps({"model": model, "prompt": text}).encode(),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=30) as r:
        return json.loads(r.read())["embedding"]


def cosine(a, b):
    dot = sum(x * y for x, y in zip(a, b)); na = math.sqrt(sum(x*x for x in a)); nb = math.sqrt(sum(y*y for y in b))
    return dot / (na * nb) if na and nb else 0.0


def tok(s):
    s = "".join(c for c in unicodedata.normalize("NFKD", s.lower()) if not unicodedata.combining(c))
    return [w for w in re.split(r"[^a-z0-9]+", s) if len(w) > 2]


class BM25:
    def __init__(self, docs, k1=1.5, b=0.75):
        self.k1, self.b = k1, b
        self.docs = [tok(d) for d in docs]
        self.dl = [len(d) for d in self.docs]
        self.avgdl = sum(self.dl) / len(self.dl) if self.dl else 0
        self.df = Counter()
        for d in self.docs:
            for w in set(d):
                self.df[w] += 1
        self.N = len(self.docs)
        self.tf = [Counter(d) for d in self.docs]

    def scores(self, query):
        q = tok(query)
        idf = {w: math.log(1 + (self.N - self.df[w] + 0.5) / (self.df[w] + 0.5)) for w in set(q) if self.df[w]}
        out = []
        for i, tf in enumerate(self.tf):
            s = 0.0
            for w in q:
                if w in tf and w in idf:
                    f = tf[w]
                    s += idf[w] * f * (self.k1 + 1) / (f + self.k1 * (1 - self.b + self.b * self.dl[i] / self.avgdl))
            out.append(s)
        return out


def rrf(rank_lists, k=60):
    scores = {}
    for rl in rank_lists:
        for rank, cid in enumerate(rl, 1):
            scores[cid] = scores.get(cid, 0.0) + 1.0 / (k + rank)
    return [cid for cid, _ in sorted(scores.items(), key=lambda x: -x[1])]


def dedupe_notes(ranked, k=10):
    out, seen = [], set()
    for cid in ranked:
        note = cid.split("#", 1)[0]
        if note in seen:
            continue
        seen.add(note); out.append(cid)
        if len(out) >= k:
            break
    return out


def rank_of(expect, ids):
    el = expect.lower()
    for i, cid in enumerate(ids, 1):
        if el in cid.lower():
            return i
    return None


def main():
    args = sys.argv[1:]
    index_path = next((a for a in args if not a.startswith("--") and a.endswith(".json") and "golden" not in a), DEF_INDEX)
    golden_path = DEF_GOLDEN
    field = args[args.index("--field") + 1] if "--field" in args else "text"
    idx = json.load(open(index_path, encoding="utf-8"))
    model, chunks = idx["model"], idx["chunks"]
    ids = [c["id"] for c in chunks]
    texts = [c.get(field) or c.get("preview", "") for c in chunks]
    bm25 = BM25(texts)
    golden = json.load(open(golden_path, encoding="utf-8"))
    print(f"HYBRID index={os.path.basename(index_path)} chunks={len(chunks)} field={field} queries={len(golden)}\n")

    ranks = []
    for g in golden:
        qv = embed(g["q"], model)
        dense = sorted(range(len(chunks)), key=lambda i: -cosine(qv, chunks[i]["vector"]))
        bm = bm25.scores(g["q"])
        sparse = sorted(range(len(chunks)), key=lambda i: -bm[i])
        fused = rrf([[ids[i] for i in dense[:50]], [ids[i] for i in sparse[:50]]])
        top = dedupe_notes(fused, 10)
        r = rank_of(g["expect"], top)
        ranks.append(r)
        print(f"  {('@'+str(r)) if r else 'MISS':>5}  {g['expect']:<32}  «{g['q'][:48]}»")
    n = len(golden)
    for kk in (1, 5, 10):
        print(f"  recall@{kk}={sum(1 for r in ranks if r and r <= kk)/n:.2f}", end="")
    print(f"  MRR={sum((1/r) for r in ranks if r)/n:.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
