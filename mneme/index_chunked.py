#!/usr/bin/env python3
"""Chunked index builder — heading/paragraph chunking + Summary-Augmented Chunking (SAC).

Fixes whole-note dilution (long multi-topic notes become generic centroids that match everything
weakly). Splits each note into heading sections (sub-splitting oversized ones by paragraph), and
prepends a generic ~150-char parent summary to each chunk before embedding — Summary-Augmented
Chunking: a generic summary (title + frontmatter-description) is enough, no LLM needed.

Output: the chunked index JSON at config.INDEX_PATH.
"""
from __future__ import annotations

import json
import os
import re
import sys
import time
import urllib.request
from glob import glob

from . import config

MODEL = config.EMBED_MODEL
OLLAMA = config.OLLAMA_URL
OUT = config.INDEX_PATH
CORPORA = {
    "memory": [f"{config.MEMORY_DIR}/*.md"],
    "vault": [f"{config.VAULT_DIR}/**/*.md"],
}
VAULT_EXCLUDE = ("/.obsidian/", "/.trash/", "/templates/")
MAX_CHUNK = 1200
FM_RE = re.compile(r"^---\n(.*?)\n---\n", re.DOTALL)
HEADING_RE = re.compile(r"^#{1,6}\s", re.MULTILINE)


def embed(text: str) -> list[float]:
    req = urllib.request.Request(
        OLLAMA, data=json.dumps({"model": MODEL, "prompt": text}).encode(),
        headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=30) as r:
        return json.loads(r.read())["embedding"]


def sac_prefix(title: str, body: str, fm: str) -> str:
    """Generic ~150-char parent summary: title + frontmatter description (or first real line)."""
    desc = ""
    m = re.search(r'^description:\s*"?(.*?)"?\s*$', fm, re.MULTILINE)
    if m:
        desc = m.group(1)
    if not desc:
        for line in body.splitlines():
            s = line.strip()
            if s and not s.startswith(("#", "-", "|", ">", "```")):
                desc = s
                break
    return f"{os.path.splitext(title)[0]} — {desc}"[:150]


def sections(body: str) -> list[str]:
    """Split by headings; sub-split oversized sections by paragraph into <=MAX_CHUNK windows."""
    idxs = [m.start() for m in HEADING_RE.finditer(body)]
    raw = ([body[:idxs[0]]] if idxs and idxs[0] > 0 else []) + \
          [body[a:b] for a, b in zip(idxs, idxs[1:] + [len(body)])] if idxs else [body]
    out = []
    for sec in raw:
        sec = sec.strip()
        if not sec:
            continue
        if len(sec) <= MAX_CHUNK:
            out.append(sec)
            continue
        buf = ""
        for para in sec.split("\n\n"):
            if len(buf) + len(para) > MAX_CHUNK and buf:
                out.append(buf.strip()); buf = ""
            buf += para + "\n\n"
        if buf.strip():
            out.append(buf.strip())
    return out


def main() -> int:
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    chunks, ok, fail, ndocs = [], 0, 0, 0
    t0 = time.time()
    for corpus, patterns in CORPORA.items():
        for pat in patterns:
            for path in glob(pat, recursive=True):
                if corpus == "vault" and any(x in path for x in VAULT_EXCLUDE):
                    continue
                try:
                    txt = open(path, encoding="utf-8", errors="ignore").read()
                except OSError:
                    continue
                if not txt.strip():
                    continue
                title = os.path.basename(path)
                note_id = f"{corpus}:{title}"
                fm_m = FM_RE.match(txt)
                fm = fm_m.group(1) if fm_m else ""
                body = txt[fm_m.end():] if fm_m else txt
                prefix = sac_prefix(title, body, fm)
                ndocs += 1
                for i, sec in enumerate(sections(body)):
                    try:
                        vec = embed(f"{prefix}\n{sec}"[:2000])
                    except Exception as e:  # noqa: BLE001
                        print(f"  skip {title}#{i}: {e}", file=sys.stderr); fail += 1; continue
                    chunks.append({"id": f"{note_id}#{i}", "corpus": corpus, "note_id": note_id,
                                   "source": path, "title": title,
                                   "preview": " ".join(sec.split())[:200],
                                   "text": (prefix + " " + " ".join(sec.split()))[:1500], "vector": vec})
                    ok += 1
    json.dump({"model": MODEL, "built_epoch": int(t0), "chunks": chunks},
              open(OUT, "w", encoding="utf-8"), separators=(",", ":"))
    print(f"indexed {ok} chunks from {ndocs} docs ({fail} failed) in {time.time()-t0:.1f}s -> {OUT}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
