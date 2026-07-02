#!/usr/bin/env python3
"""Retrieval daemon.

Loads the index + BM25 + the multilingual reranker ONCE and serves the winning pipeline (hybrid
dense + BM25 -> RRF -> cross-encoder rerank) over a unix socket. The per-prompt hook is then a thin
fast client — the model never reloads, so a good hook is also a fast hook. This is the answer to
"reliable rerank is slow per process": keep it warm in a daemon.

Injection gate: the rerank score is a calibrated relevance signal (relevant >> 0, irrelevant < 0), so
the daemon returns only notes with rerank score > MNEME_RERANK_MIN — nothing below (the anti-noise gate).

Requires a matrix.npy (normalized embedding matrix, row-aligned with the index chunks) at
config.MATRIX_PATH alongside the index.
"""
from __future__ import annotations

import json
import os
import socket
import sys

import numpy as np

from . import config
from .hybrid_eval import BM25, rrf
from fastembed.rerank.cross_encoder import TextCrossEncoder

STATE = config.STATE_DIR
SOCK = config.SOCK_PATH
INDEX = config.INDEX_PATH
MATRIX = config.MATRIX_PATH
RERANK_MODEL = config.RERANK_MODEL
OLLAMA = config.OLLAMA_URL
RERANK_MIN = float(os.environ.get("MNEME_RERANK_MIN", "0.0"))
TOPN = 20


def load():
    idx = json.load(open(INDEX, encoding="utf-8"))
    chunks = idx["chunks"]
    ids = [c["id"] for c in chunks]
    texts = [c.get("text") or c.get("preview", "") for c in chunks]
    meta = [{"note_id": c["note_id"], "corpus": c["corpus"], "title": c["title"], "preview": c["preview"]}
            for c in chunks]
    mat = np.load(MATRIX)  # normalized
    return idx["model"], ids, texts, meta, mat, BM25(texts), {cid: i for i, cid in enumerate(ids)}


def embed(text, model):
    import urllib.request
    req = urllib.request.Request(OLLAMA,
                                 data=json.dumps({"model": model, "prompt": text}).encode(),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=4) as r:
        return json.loads(r.read())["embedding"]


def handle(query, corpus, model, ids, texts, meta, mat, bm25, id2i, reranker):
    q = np.array(embed(query, model), dtype="float32"); q /= (np.linalg.norm(q) + 1e-9)
    sims = mat @ q
    dense = np.argsort(-sims)[:150]
    bm = bm25.scores(query)
    sparse = sorted(range(len(texts)), key=lambda i: -bm[i])[:150]
    fused = rrf([[ids[i] for i in dense], [ids[i] for i in sparse]])
    if corpus:
        fused = [c for c in fused if meta[id2i[c]]["corpus"] == corpus]
    cand = fused[:TOPN]
    scores = list(reranker.rerank(query, [meta[id2i[c]]["preview"] for c in cand]))
    ranked = sorted(zip(cand, scores), key=lambda x: -x[1])
    out, seen = [], set()
    for cid, s in ranked:
        if s <= RERANK_MIN:
            break
        m = meta[id2i[cid]]
        if m["note_id"] in seen:
            continue
        seen.add(m["note_id"])
        out.append({"corpus": m["corpus"], "title": m["title"], "preview": m["preview"], "score": round(float(s), 2)})
        if len(out) >= 5:
            break
    return out


def main() -> int:
    print("loading index + reranker…", file=sys.stderr)
    model, ids, texts, meta, mat, bm25, id2i = load()
    reranker = TextCrossEncoder(model_name=RERANK_MODEL)
    handle("warmup query", None, model, ids, texts, meta, mat, bm25, id2i, reranker)  # warm the ONNX graph
    os.makedirs(os.path.dirname(SOCK), exist_ok=True)
    if os.path.exists(SOCK):
        os.remove(SOCK)
    srv = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    srv.bind(SOCK); srv.listen(8)
    print(f"substrate daemon ready on {SOCK}", file=sys.stderr)
    while True:
        conn, _ = srv.accept()
        try:
            data = conn.recv(65536).decode("utf-8").strip()
            req = json.loads(data)
            res = handle(req.get("query", ""), req.get("corpus"), model, ids, texts, meta, mat, bm25, id2i, reranker)
            conn.sendall(json.dumps(res).encode("utf-8"))
        except Exception as e:  # noqa: BLE001
            try:
                conn.sendall(json.dumps({"error": str(e)}).encode("utf-8"))
            except OSError:
                pass
        finally:
            conn.close()


if __name__ == "__main__":
    raise SystemExit(main())
