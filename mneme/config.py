#!/usr/bin/env python3
"""Central configuration for mneme — every path and model is env-overridable with a generic default.

Nothing operator-specific lives in the scripts; they all import from here. Override any value by
exporting the matching environment variable before running a command:

    MNEME_MEMORY_DIR   agent auto-memory dir      (default ~/.mneme/memory)
    MNEME_VAULT_DIR    notes / knowledge vault    (default ~/notes)
    MNEME_STATE_DIR    index + daemon state dir   (default ~/.mneme/state)
    MNEME_OLLAMA       ollama embeddings endpoint (default http://localhost:11434/api/embeddings)
    MNEME_EMBED_MODEL  embedding model name       (default nomic-embed-text)
    MNEME_RERANK_MODEL cross-encoder rerank model (default jinaai/jina-reranker-v2-base-multilingual)

Derived paths (index/matrix/socket/golden) live under STATE_DIR / the package unless individually
overridden. Import this module rather than hardcoding any literal.
"""
from __future__ import annotations

import os

HOME = os.path.expanduser("~")


def _env_path(var: str, default: str) -> str:
    return os.path.expanduser(os.environ.get(var, default))


# --- Corpora locations ------------------------------------------------------
MEMORY_DIR = _env_path("MNEME_MEMORY_DIR", f"{HOME}/.mneme/memory")
VAULT_DIR = _env_path("MNEME_VAULT_DIR", f"{HOME}/notes")

# --- State (index, matrix, daemon socket) -----------------------------------
STATE_DIR = _env_path("MNEME_STATE_DIR", f"{HOME}/.mneme/state")
INDEX_PATH = _env_path("MNEME_INDEX", f"{STATE_DIR}/index-chunked.json")
MATRIX_PATH = _env_path("MNEME_MATRIX", f"{STATE_DIR}/matrix.npy")
SOCK_PATH = _env_path("MNEME_SOCK", f"{STATE_DIR}/daemon.sock")

# --- Models / endpoints -----------------------------------------------------
OLLAMA_URL = os.environ.get("MNEME_OLLAMA", "http://localhost:11434/api/embeddings")
EMBED_MODEL = os.environ.get("MNEME_EMBED_MODEL", "nomic-embed-text")
RERANK_MODEL = os.environ.get("MNEME_RERANK_MODEL", "jinaai/jina-reranker-v2-base-multilingual")

# --- Golden set (eval) ------------------------------------------------------
# Defaults to the example shipped with the repo; point at your own via MNEME_GOLDEN.
GOLDEN_PATH = _env_path(
    "MNEME_GOLDEN",
    os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "eval", "golden.example.json"),
)
