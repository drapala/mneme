# Wiring the mneme recall hook into Claude Code

`mneme/hook_inject.py` is a `UserPromptSubmit` hook: it reads the prompt on stdin, asks the warm
retrieval daemon for the most relevant notes above the rerank gate, and prints them so the harness
injects them into context — no manual "search my notes" step. If the daemon is down or anything
errors, it prints nothing and exits 0, so it never blocks a prompt.

## Prerequisites

1. An embeddings backend reachable at `MNEME_OLLAMA` (default `http://localhost:11434/api/embeddings`).
2. An index built: `python -m mneme.index_chunked` (writes `index-chunked.json` into the state dir).
3. A normalized `matrix.npy` in the state dir (row-aligned with the index) for the daemon.
4. The daemon running: `python -m mneme.substrate_daemon` (or install the launchd template in
   `deploy/com.example.mneme-daemon.plist`).

## settings.json

Add a `UserPromptSubmit` hook that runs the script. Point `<PYTHON>` at an interpreter that can import
`mneme` (e.g. the repo venv) and `<REPO>` at this checkout. Set any `MNEME_*` override in `env`.

```json
{
  "hooks": {
    "UserPromptSubmit": [
      {
        "hooks": [
          {
            "type": "command",
            "command": "<PYTHON> -m mneme.hook_inject",
            "env": {
              "PYTHONPATH": "<REPO>",
              "MNEME_SOCK": "~/.mneme/state/daemon.sock"
            }
          }
        ]
      }
    ]
  }
}
```

The hook receives the prompt as JSON on stdin (`{"prompt": "..."}`) and emits an `<mneme-recall>`
block on stdout listing the retrieved notes. Prompts shorter than 8 characters are skipped.
