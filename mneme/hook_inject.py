#!/usr/bin/env python3
"""UserPromptSubmit hook — thin client to the retrieval daemon.

Reads the prompt on stdin, asks the warm daemon (hybrid + rerank) for the relevant notes above the
rerank gate, and injects them. The daemon keeps the model loaded, so this is fast. Graceful: daemon
down / any error -> inject nothing, never block a prompt. The harness injects deterministically, so
the agent doesn't have to remember to query.
"""
from __future__ import annotations

import json
import os
import socket
import sys

try:
    from . import config
    SOCK = config.SOCK_PATH
except Exception:  # noqa: BLE001 — run as a bare script outside the package
    SOCK = os.path.expanduser(os.environ.get("MNEME_SOCK", "~/.mneme/state/daemon.sock"))


def main() -> int:
    try:
        payload = json.load(sys.stdin)
        prompt = (payload.get("prompt") or "").strip() if isinstance(payload, dict) else ""
    except Exception:
        prompt = ""
    if len(prompt) < 8 or not os.path.exists(SOCK):
        return 0
    try:
        s = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        s.settimeout(4.0)
        s.connect(SOCK)
        s.sendall(json.dumps({"query": prompt}).encode("utf-8"))
        buf = b""
        while True:
            chunk = s.recv(65536)
            if not chunk:
                break
            buf += chunk
        s.close()
        hits = json.loads(buf.decode("utf-8"))
    except Exception:
        return 0
    if not isinstance(hits, list) or not hits:
        return 0
    print("<mneme-recall> Relevant notes from your memory + vault (retrieved by meaning; verify — "
          "recalled notes reflect when written, so confirm a named file/flag still exists before relying):")
    for h in hits:
        print(f"- [{h['corpus']}] {h['title']} ({h['score']}) — {h['preview'][:200]}")
    print("</mneme-recall>")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
