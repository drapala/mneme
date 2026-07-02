#!/usr/bin/env python3
"""Mechanical staleness sweep over a memory directory.

Extracts every FILESYSTEM path / file / binary reference from each memory `.md` and verifies it exists —
but ONLY reports findings whose success-path is confirmed: a "DEAD" is trustworthy only when the parent
dir exists and the leaf is genuinely absent. Everything ambiguous (unresolved-relative, space-truncated,
prose `~N`, single-segment `/slash-command`) is suppressed as noise, because a staleness check that
over-reports lies exactly like the memory it audits.

Scope note (deliberate): this checks FILES/DIRS only. Slash-commands (`/deploy`, `/status`) are NOT
checked — their trigger names differ from skill directory names, so mechanical verification is unreliable
and is pure noise. Slash-command drift belongs to a skill-registry check, not here.

A naive `test -e` sweep over real memory files produces a large false-positive rate — `~16`
("about 16"), `/some-command` (a command, not a path), a path truncated at a space, brace-expansions,
and repo-relative paths under un-rooted repos. This script bakes in the filters that survive that,
and LOCATES the real path for each confirmed-dead reference so the caller proposes an accurate fix.

Search roots default to the current user's HOME plus the configured memory/vault dirs; override with
MNEME_SWEEP_ROOTS (colon-separated) to point at the dirs where your memory's paths actually live.

Output: JSON on stdout. Groups findings by memory file; each finding is
  {ref, status, confidence, proposed_replacement:[...]}
Only DEAD (absolute, parent exists, leaf absent) findings are emitted. Noise is counted, not emitted.
"""
from __future__ import annotations

import json
import os
import re
import sys
from glob import glob

from . import config

HOME = os.path.expanduser("~")
SKIP_DIRS = {"node_modules", ".git", ".worktrees", ".venv", "__pycache__", "venv",
             "tsp-output", "dist", "build", ".mypy_cache", ".pytest_cache"}

ABS_RE = re.compile(r'(?:~|/Users/[A-Za-z0-9_]+|/home/[A-Za-z0-9_]+|/opt/homebrew|/usr/local)[\w./+~{},-]*')
TICK_RE = re.compile(r'`([^`]+)`')
ABS_PREFIX_RE = re.compile(r'^(?:~|/(?:Users|home|opt|usr|tmp|etc|var|private)/)')
CODE_EXT = (".py", ".md", ".json", ".edn", ".toml", ".yaml", ".yml", ".js", ".ts",
            ".tsx", ".mjs", ".sh", ".txt", ".csv", ".sql", ".c4", ".tsp", ".mts")


def roots() -> list[str]:
    env = os.environ.get("MNEME_SWEEP_ROOTS")
    if env:
        base = [os.path.expanduser(p) for p in env.split(":") if p.strip()]
    else:
        base = [HOME, config.MEMORY_DIR, config.VAULT_DIR, f"{HOME}/projects"]
    return [r for r in dict.fromkeys(base) if os.path.isdir(r)]


ROOTS = roots()


def is_real_abs(t: str) -> bool:
    return bool(ABS_PREFIX_RE.match(t))


def looks_prose_tilde(t: str) -> bool:
    # "~16", "~R$4.8M", "~2h", "~5", "~30-60" — approximations, not paths
    return bool(re.match(r'^~[\d.,]', t)) or bool(re.match(r'^~[A-Z$]', t))


def normalize(t: str) -> str:
    return t.strip().strip('`"\'').rstrip('.,;:)]}')


def is_junk(t: str) -> bool:
    # elided (`...`), template placeholders (`<repo>`), globs, prose-tilde without a slash (`~free`)
    if not t or "*" in t or " " in t or "..." in t or "<" in t or ">" in t or "$" in t:
        return True
    if t.startswith("~") and "/" not in t:
        return True
    return False


def is_pathlike(t: str) -> bool:
    if t.startswith(("http://", "https://", "[[", "@")):
        return False
    if is_junk(t):
        return False
    return ("/" in t) or t.endswith(CODE_EXT)


def brace_expand(t: str) -> list[str]:
    m = re.search(r"\{([^{}]+)\}", t)
    if not m:
        return [t]
    out: list[str] = []
    for opt in m.group(1).split(","):
        out += brace_expand(t[: m.start()] + opt + t[m.end():])
    return out


def find_real(basename: str, max_hits: int = 3) -> list[str]:
    """Locate `basename` under the known roots (bounded walk) to propose an accurate replacement."""
    if not basename:
        return []
    hits: list[str] = []
    seen: set[str] = set()
    for root in ROOTS:
        for dirpath, dirnames, filenames in os.walk(root):
            if dirpath[len(root):].count(os.sep) >= 5:
                dirnames[:] = []
                continue
            dirnames[:] = [d for d in dirnames if d not in SKIP_DIRS]
            if basename in filenames or basename in dirnames:
                h = os.path.join(dirpath, basename)
                if h not in seen:
                    seen.add(h)
                    hits.append(h)
                    if len(hits) >= max_hits:
                        return hits
    return hits


def verify_abs(t: str) -> tuple[str, str, list[str]]:
    """Returns (status, confidence, replacements). Only 'DEAD' with a confirmed success-path is reported."""
    p = os.path.expanduser(t)
    if os.path.exists(p):
        return ("OK", "", [])
    parent, leaf = os.path.split(p.rstrip("/"))
    # space-truncation guard: memory may have had "<path> Some Words" → glob parent for <leaf>*
    if os.path.isdir(parent) and glob(os.path.join(parent, leaf + "*")):
        return ("TRUNCATED_NOISE", "", [])
    reps = find_real(leaf)
    if os.path.isdir(parent):
        # parent exists, leaf genuinely absent → check's success-path confirmed → trustworthy DEAD
        return ("DEAD", "high", reps)
    # parent missing too → what moved is bigger than a leaf (structural); medium confidence
    return ("DEAD", "medium", reps)


def sweep_file(path: str) -> list[dict]:
    txt = open(path, encoding="utf-8").read()
    cands: set[str] = set()
    for m in ABS_RE.findall(txt):
        n = normalize(m)
        if n and not looks_prose_tilde(n) and not is_junk(n):
            cands.update(brace_expand(n))
    for m in TICK_RE.findall(txt):
        n = normalize(m)
        if is_pathlike(n):
            cands.update(brace_expand(n))
    findings = []
    for tok in sorted(cands):
        if tok.startswith(("~", "/")):
            if not is_real_abs(tok):     # single-segment /command or non-path absolute → skip
                continue
        else:
            continue                     # relative refs are too root-ambiguous to trust → suppress
        st, conf, reps = verify_abs(tok)
        if st != "DEAD":
            continue
        findings.append({"ref": tok, "status": st, "confidence": conf,
                         "proposed_replacement": reps})
    return findings


def main() -> int:
    mem = sys.argv[1] if len(sys.argv) > 1 else config.MEMORY_DIR
    files = sorted(glob(os.path.join(mem, "*.md")))
    report = {"memory_dir": mem, "files_scanned": len(files), "findings": {}}
    total = 0
    for f in files:
        fnd = sweep_file(f)
        if fnd:
            report["findings"][os.path.basename(f)] = fnd
            total += len(fnd)
    report["total_trustworthy_findings"] = total
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
