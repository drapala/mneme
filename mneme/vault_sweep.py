#!/usr/bin/env python3
"""Staleness sweep for a notes vault: dead [[wikilinks]] / ![[embeds]] + (opt) aged notes.

Sibling of sweep.py (which checks the agent auto-memory's filesystem paths). A vault's rot is different:
- dead `[[wikilink]]` — a link to a note that no longer resolves. Wikilinks resolve by ANY of: bare
  basename, a path SUFFIX (`[[folder/note]]` matches `.../folder/note.md`), or a `../` path relative to
  the linking note's own dir. Getting resolution right is the success-path this check must confirm — a
  resolver that only does basename over-reports path-style links as dead.
- dead `![[embed]]` — an embed of a note/attachment that's gone.
- (opt, --stale-days N) aged notes — notes under `--intel-glob` older than N days. ADVISORY only: dated
  notes are old by nature; this surfaces where a captured read MAY be outdated, not a "dead".

Output: JSON. Only DEAD links are findings; aged notes are a separate advisory list.

Usage: python -m mneme.vault_sweep [vault_dir] [--stale-days N] [--intel-glob PATTERN]
"""
from __future__ import annotations

import json
import os
import re
import sys
import time
from glob import glob

from . import config

WIKILINK_RE = re.compile(r'!?\[\[([^\]]+)\]\]')
EXCLUDE = {".obsidian", ".trash", ".git", "templates", "node_modules"}
ATTACH_EXT = (".png", ".jpg", ".jpeg", ".gif", ".pdf", ".svg", ".webp", ".mp4", ".mov", ".excalidraw")
_INVALID_NAME = set('[]:*?"<>')


def note_target(raw: str) -> tuple[str, bool]:
    """Parse `target#heading|alias` → (target, is_attachment). Resolution is by name or path."""
    t = raw.split("|", 1)[0].split("#", 1)[0].strip()
    return t, t.lower().endswith(ATTACH_EXT)


def valid_note_name(t: str) -> bool:
    # reject code artifacts carrying `[[`: type hints (`Iterable[X[S`), regex classes
    # (`[[:space:]]`), emoji (`:space:`). Real note names can't contain these chars.
    return bool(t) and not (_INVALID_NAME & set(t))


def build_index(vault: str) -> tuple[set[str], set[str], set[str]]:
    """(md_basenames, md_relpaths_noext, all_relpaths_noext) — the resolvable universe, lowercased."""
    md_base: set[str] = set()
    md_rel: set[str] = set()
    all_rel: set[str] = set()
    for dirpath, dirnames, filenames in os.walk(vault):
        dirnames[:] = [d for d in dirnames if d not in EXCLUDE]
        for fn in filenames:
            rel = os.path.relpath(os.path.join(dirpath, fn), vault)
            rel_noext = os.path.splitext(rel)[0].lower()
            all_rel.add(rel_noext)
            all_rel.add(rel.lower())  # attachments referenced with extension
            if fn.lower().endswith(".md"):
                md_base.add(os.path.splitext(fn)[0].lower())
                md_rel.add(rel_noext)
    return md_base, md_rel, all_rel


def resolves(target: str, is_att: bool, note_rel: str,
             md_base: set[str], md_rel: set[str], all_rel: set[str]) -> bool:
    key = target.lower()
    if is_att:
        base = key.rsplit("/", 1)[-1]
        return key in all_rel or base in all_rel or any(p.endswith("/" + key) for p in all_rel)
    if target.startswith(("./", "../")):  # relative to the linking note's own directory
        cand = os.path.normpath(os.path.join(os.path.dirname(note_rel), target)).lower()
        return cand in md_rel or cand in all_rel
    if "/" in target:  # path-style: exact vault-rel OR a suffix of some note's path
        return key in md_rel or any(p == key or p.endswith("/" + key) for p in md_rel)
    return key in md_base or key in md_rel  # bare basename


def sweep(vault: str, stale_days: int | None, intel_glob: str) -> dict:
    md_base, md_rel, all_rel = build_index(vault)
    dead: dict[str, list[str]] = {}
    scanned = 0
    for path in glob(os.path.join(vault, "**", "*.md"), recursive=True):
        if any(f"/{e}/" in path for e in EXCLUDE):
            continue
        scanned += 1
        note_rel = os.path.relpath(path, vault)
        txt = open(path, encoding="utf-8", errors="ignore").read()
        bad = []
        for m in WIKILINK_RE.findall(txt):
            tgt, is_att = note_target(m)
            if not valid_note_name(tgt):
                continue
            if not resolves(tgt, is_att, note_rel, md_base, md_rel, all_rel):
                bad.append(m)
        if bad:
            dead[note_rel] = sorted(set(bad))

    aged = []
    if stale_days is not None:
        cutoff = time.time() - stale_days * 86400
        for path in glob(os.path.join(vault, intel_glob, "**", "*.md"), recursive=True):
            if os.path.getmtime(path) < cutoff:
                aged.append({"note": os.path.relpath(path, vault),
                             "age_days": int((time.time() - os.path.getmtime(path)) / 86400)})
        aged.sort(key=lambda x: -x["age_days"])

    return {"vault": vault, "notes_scanned": scanned,
            "dead_link_notes": len(dead), "dead_links": dead, "aged_intel_advisory": aged}


def main() -> int:
    vault = config.VAULT_DIR
    stale_days = None
    intel_glob = "**"
    args = sys.argv[1:]
    if args and not args[0].startswith("-"):
        vault = os.path.expanduser(args.pop(0))
    if "--stale-days" in args:
        stale_days = int(args[args.index("--stale-days") + 1])
    if "--intel-glob" in args:
        intel_glob = args[args.index("--intel-glob") + 1]
    print(json.dumps(sweep(vault, stale_days, intel_glob), indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
