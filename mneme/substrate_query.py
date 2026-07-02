#!/usr/bin/env python3
"""Query the substrate — the winning pipeline.

chunk+SAC index -> hybrid dense + BM25 fused by RRF -> multilingual cross-encoder rerank -> top-K.
All local (ollama + fastembed ONNX, no torch), source untouched, cross-lingual queries handled (no
translation needed).

Run with a venv that has fastembed installed:
    python -m mneme.substrate_query "query" [--corpus memory|vault] [--k 8]
"""
from __future__ import annotations

import json
import os
import sys

from . import config
from .hybrid_eval import BM25, cosine, embed, rrf

INDEX = config.INDEX_PATH
RERANK_MODEL = config.RERANK_MODEL
TOPN = 40


def main() -> int:
    args = sys.argv[1:]
    k = int(args[args.index("--k") + 1]) if "--k" in args else 8
    corpus = args[args.index("--corpus") + 1] if "--corpus" in args else None
    query = " ".join(a for a in args if not a.startswith("--")
                     and a not in ({str(k)} | ({corpus} if corpus else set())))
    if not query:
        print('usage: substrate_query.py "query" [--corpus memory|vault] [--k N]', file=sys.stderr)
        return 2
    if not os.path.exists(INDEX):
        print(f"no index at {INDEX} — run index_chunked.py first", file=sys.stderr)
        return 1
    idx = json.load(open(INDEX, encoding="utf-8"))
    model = idx["model"]
    chunks = [c for c in idx["chunks"] if not corpus or c["corpus"] == corpus]
    ids = [c["id"] for c in chunks]
    texts = [c.get("text") or c.get("preview", "") for c in chunks]
    id2i = {cid: i for i, cid in enumerate(ids)}

    qv = embed(query, model)
    dense = sorted(range(len(chunks)), key=lambda i: -cosine(qv, chunks[i]["vector"]))
    bm = BM25(texts).scores(query)
    sparse = sorted(range(len(chunks)), key=lambda i: -bm[i])
    fused = rrf([[ids[i] for i in dense[:150]], [ids[i] for i in sparse[:150]]])[:TOPN]

    from fastembed.rerank.cross_encoder import TextCrossEncoder
    reranker = TextCrossEncoder(model_name=RERANK_MODEL)
    scores = list(reranker.rerank(query, [texts[id2i[c]] for c in fused]))
    reranked = [c for c, _ in sorted(zip(fused, scores), key=lambda x: -x[1])]
    score_by = dict(zip(fused, scores))

    seen, shown = set(), 0
    for cid in reranked:
        note = cid.split("#", 1)[0]
        if note in seen:
            continue
        seen.add(note)
        c = chunks[id2i[cid]]
        print(f"{score_by[cid]:+.2f}  [{c['corpus']}] {c['title']}")
        print(f"        {c['preview'][:150]}")
        shown += 1
        if shown >= k:
            break
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
