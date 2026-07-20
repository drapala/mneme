#!/usr/bin/env python3
"""harness_distill — a note producer for mneme: agent-harness meters → distilled, indexable notes.

The harness's self-measurement (gate telemetry, ratchet ledger, reviewer-sim accuracy scores) lives in
JSONL/state files a retrieval index should never eat raw: they churn daily and would dilute into generic
centroids. What IS memory-worthy is the distilled current picture — "gate X: N runs, M catches, last
verdict", "sim accuracy on PR #N: recall a/b" — regenerated whole from the meters on every run, so the
notes are current by construction and the supersession judge can see machine-owned facts change.

Writes a fixed set of notes under <VAULT_DIR>/harness/ (fixed cardinality — one note per meter family,
regenerated idempotently; skipped when content is unchanged). Then `mneme-index` (or the host's own
indexer) makes them queryable, and the host's always-on recall hook closes the loop: the harness's own
measured state becomes retrievable context in future sessions.

Inputs (all env-overridable, generic defaults match the reference deployment):
    HARNESS_GATE_STATS    gate telemetry jsonl   (default ~/.local/state/gate-stats/runs.jsonl)
    HARNESS_RATCHET       ratchet ledger jsonl   (default ~/.local/state/flywheel/ratchet.jsonl)
    HARNESS_SIM_SCORES    sim score dir          (default ~/.claude/skills/audit-convergence/bench/scores)

Usage:
    python producers/harness_distill.py            # writes into <VAULT_DIR>/harness/
    python -m mneme.index_chunked                  # re-index -> searchable
"""

from __future__ import annotations

import json
import os
import sys
from datetime import date
from glob import glob
from pathlib import Path

try:
    from mneme.config import VAULT_DIR
except Exception:
    VAULT_DIR = os.environ.get("MNEME_VAULT_DIR", str(Path.home() / "notes"))

H = os.path.expanduser
GATE_STATS = H(
    os.environ.get("HARNESS_GATE_STATS", "~/.local/state/gate-stats/runs.jsonl")
)
RATCHET = H(os.environ.get("HARNESS_RATCHET", "~/.local/state/flywheel/ratchet.jsonl"))
SIM_SCORES = H(
    os.environ.get(
        "HARNESS_SIM_SCORES", "~/.claude/skills/audit-convergence/bench/scores"
    )
)
# hostname namespace: shared flat dir clobbers across synced machines
HOST = os.environ.get("HARNESS_HOST", os.uname().nodename.split(".")[0])
OUT_DIR = Path(H(str(VAULT_DIR))) / "harness" / HOST


def rows(path: str) -> list[dict]:
    try:
        return [json.loads(l) for l in open(path) if l.strip()]
    except OSError:
        return []


def frontmatter(title: str, source: str) -> str:
    return (
        "---\n"
        f"title: {title}\n"
        "status: active\n"
        "generated-by: mneme/producers/harness_distill.py\n"
        f"source: {source}\n"
        f"regenerated: {date.today().isoformat()}\n"
        "tags: [harness, machine-distilled]\n"
        "---\n\n"
        "> Machine-regenerated from the harness meters — edit the meters, not this note.\n\n"
    )


def gates_note() -> str | None:
    data = rows(GATE_STATS)
    if not data:
        return None
    by: dict[str, dict] = {}
    for r in data:
        g = by.setdefault(r["gate"], {"runs": 0, "catch": 0, "last": ""})
        g["runs"] += 1
        if r.get("decision") in ("block", "warn", "repair"):
            g["catch"] += 1
        g["last"] = f"{r.get('decision', '?')} @ {r.get('ts', '?')[:10]}"
    lines = [
        frontmatter("Harness gate telemetry (current)", GATE_STATS),
        "# Harness gates — measured record\n",
        "What each mechanical gate has actually done (a gate with many runs and zero",
        "catches is a prune candidate; a repair means the guarded thing broke for real).\n",
        "| gate | runs | catches | last |",
        "|---|---|---|---|",
    ]
    for name, g in sorted(by.items()):
        lines.append(f"| {name} | {g['runs']} | {g['catch']} | {g['last']} |")
    return "\n".join(lines) + "\n"


def ratchet_note() -> str | None:
    data = rows(RATCHET)
    if not data:
        return None
    lines = [
        frontmatter("Harness ratchet ledger (current)", RATCHET),
        "# Ratchet ledger — codification candidates\n",
        "Improvements surfaced by retros/fix-sistemico and whether they actually landed",
        "(the weekly flywheel flags any entry unlanded >7d).\n",
    ]
    for r in data:
        mark = "✅ landed" if r.get("landed") else "⏳ unlanded"
        lines.append(f"- {mark} [{r.get('ts', '?')[:10]}] {r.get('candidate', '?')}")
    return "\n".join(lines) + "\n"


def sim_note() -> str | None:
    files = sorted(glob(os.path.join(SIM_SCORES, "reviewer-sim-*.json")))
    if not files:
        return None
    lines = [
        frontmatter("Reviewer-sim measured accuracy", SIM_SCORES),
        "# Reviewer-sim vs real reviews — the sim's true hit rate\n",
        "Scored against the reviewer's actual comments on the same diff (sim_score.py).",
        "The MOST RECENT row is the sim's current accuracy; older rows are history, not",
        "current capability.\n",
    ]
    for f in files:
        try:
            d = json.load(open(f))
        except (OSError, ValueError):
            continue
        verdicts = d.get("verdicts", [])
        caught = sum(1 for v in verdicts if v.get("verdict") == "CAUGHT")
        extras = len(d.get("extras", []))
        name = os.path.basename(f).replace(".json", "")
        lines.append(
            f"- `{name}`: recall {caught}/{len(verdicts)}, over-produced {extras}"
        )
    return "\n".join(lines) + "\n"


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    notes = {
        "harness-gates.md": gates_note(),
        "harness-ratchet.md": ratchet_note(),
        "reviewer-sim-accuracy.md": sim_note(),
    }
    changed = 0
    for fname, body in notes.items():
        if body is None:
            continue
        p = OUT_DIR / fname
        old = p.read_text() if p.exists() else ""
        # compare ignoring the regenerated: stamp (no daily churn)
        strip = lambda s: "\n".join(
            l for l in s.splitlines() if not l.startswith("regenerated:")
        )
        if strip(old) != strip(body):
            p.write_text(body)
            changed += 1
    print(f"harness-distill: {changed} note(s) updated in {OUT_DIR}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
