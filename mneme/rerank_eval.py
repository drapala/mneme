#!/usr/bin/env python3
"""Rerank eval: hybrid top-N -> multilingual cross-encoder rerank -> top-K.

The cross-encoder scores query x chunk JOINTLY (not two separate embeddings) — the direct lever for
the semantic-diffuseness misses AND, being multilingual, the cross-lingual ones in one shot. Expand
top-K before rerank to maximize recall. Local ONNX (fastembed), no torch — respects the dep-weight
ceiling.

Run with a venv that has fastembed installed:
    python -m mneme.rerank_eval [index] [--topn 40]
"""
from __future__ import annotations

import json
import os
import sys

from . import config
from .hybrid_eval import BM25, cosine, dedupe_notes, embed, rank_of, rrf
from fastembed.rerank.cross_encoder import TextCrossEncoder

DEF_INDEX = config.INDEX_PATH
DEF_GOLDEN = config.GOLDEN_PATH
RERANK_MODEL = config.RERANK_MODEL


def main() -> int:
    args = sys.argv[1:]
    index_path = next((a for a in args if a.endswith(".json")), DEF_INDEX)
    topn = int(args[args.index("--topn") + 1]) if "--topn" in args else 40
    idx = json.load(open(index_path, encoding="utf-8"))
    model, chunks = idx["model"], idx["chunks"]
    ids = [c["id"] for c in chunks]
    texts = [c.get("text") or c.get("preview", "") for c in chunks]
    id2i = {cid: i for i, cid in enumerate(ids)}
    bm25 = BM25(texts)
    golden = json.load(open(DEF_GOLDEN, encoding="utf-8"))
    print(f"RERANK index={os.path.basename(index_path)} model={model} rerank={RERANK_MODEL} topn={topn}\n")
    reranker = TextCrossEncoder(model_name=RERANK_MODEL)

    ranks = []
    for g in golden:
        qv = embed(g["q"], model)
        dense = sorted(range(len(chunks)), key=lambda i: -cosine(qv, chunks[i]["vector"]))
        bm = bm25.scores(g["q"])
        sparse = sorted(range(len(chunks)), key=lambda i: -bm[i])
        fused = rrf([[ids[i] for i in dense[:150]], [ids[i] for i in sparse[:150]]])
        cand = fused[:topn]
        cand_texts = [texts[id2i[c]] for c in cand]
        scores = list(reranker.rerank(g["q"], cand_texts))
        reranked = [c for c, _ in sorted(zip(cand, scores), key=lambda x: -x[1])]
        top = dedupe_notes(reranked, 10)
        r = rank_of(g["expect"], top)
        ranks.append(r)
        print(f"  {('@'+str(r)) if r else 'MISS':>5}  {g['expect']:<32}  «{g['q'][:46]}»")
    n = len(golden)
    for kk in (1, 5, 10):
        print(f"  recall@{kk}={sum(1 for r in ranks if r and r <= kk)/n:.2f}", end="")
    print(f"  MRR={sum((1/r) for r in ranks if r)/n:.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
