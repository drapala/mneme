# Mneme

*Mneme — one of the three elder Muses (Melete, Mneme, Aoide): practice, memory, song. Her sister across the
underworld is Lethe, forgetting. A memory that only remembers is a hoard; a good one also forgets well.*

A local-first memory for agents. Mneme turns your plain-markdown notes and an agent's own memory files into
something **queryable by meaning** — and, unusually, keeps them **honest over time**: it verifies its own
references still resolve, surfaces the facts a later decision quietly superseded, and forgets them without
losing the provenance. Embeddings run locally (ollama), reranking runs locally (ONNX), nothing leaves your
disk, and any model can read the result. It is "own your memory, rent the intelligence" with the
engineering attached.

It is a working prototype, built and **measured** in a single session against one person's ~500-note
substrate. The retrieval half is standard RAG; the opinionated half — the part worth reading — is the
**hygiene / lifecycle**. Every number below was measured, not asserted; every hygiene rule exists because a
naive version of it lied.

## Lifecycle of a query

```
  ~/notes + agent memory  (plain markdown, on your disk)
        │
        ▼
   chunk by heading ──► SAC (prepend a ~150-char parent summary to each chunk)
        │
        ▼
   embed (ollama, local) ──► index-chunked.json + matrix.npy
        │
   query ─────────────────────────────┐
        ▼                              ▼
   dense cosine ──┐               BM25 (sparse)
                  ├── RRF fuse ──► top-N ──► cross-encoder rerank (ONNX, multilingual)
                  ┘                                     │
                                                        ▼
                                          top-K notes  ·  recall@5 = 0.83
```

Each step earned its place against a golden set (`eval/`) — `query → the note that should rank #1`, scored
by `recall@k`. Nothing shipped as "better" without moving the number:

| step | recall@5 | why it helped |
|---|---|---|
| whole-note embedding | 0.33 | one vector per note — long multi-topic notes dilute into generic centroids that match everything weakly |
| + heading chunking + SAC | 0.39 | section chunks, each prepended a generic parent summary (resolves document-level mismatch) |
| + hybrid dense + BM25 (RRF) | 0.61 | BM25 catches the exact/rare-term hits a dense embedding drops |
| + cross-encoder rerank | **0.83** | scores query×chunk *jointly* — the biggest single lever, and multilingual, so it also carries cross-language |

A note on language: the golden queries were Portuguese over mixed PT/EN notes with an **English-centric**
embedder, and it still reached 0.83 — BM25 and the multilingual reranker carry the gap. **You do not need
to translate your notes to English.** The 512-token cap on small models is a red herring: bigger context
per embedding means *more* dilution, the wrong direction.

## Two tiers (the execution model that matters)

Full rerank is 0.83 but ~seconds per query on CPU — too slow to run on *every* prompt. Measured, not
guessed. So there are two ways in:

- **On-demand, rich.** `mneme-query "what did I decide about X"` runs the whole 0.83 pipeline. Use it when
  you *want* to recall.
- **Always-on, precise.** A warm daemon (`substrate_daemon.py`) keeps the index + reranker loaded; a thin
  socket client (`hook_inject.py`) is called by the host agent on every prompt (e.g. a Claude Code
  `UserPromptSubmit` hook). Fast (~0.5s), **precision-gated**: it fires only when the reranker is confident
  (score above a threshold), injects nothing otherwise. The harness injects deterministically — the agent
  never has to *remember* to query. Deploy templates in `deploy/`.

Memory is *situational*: you don't need it on every prompt, you need it when a task starts. Precise
always-on + rich on-demand is the right split; always-on full-rerank is neither affordable nor needed.

## Hygiene — the half that's actually novel

Retrieval-over-notes is crowded. **Rot is not.** A memory that cites a file/flag that no longer exists, or
a fact a later decision contradicted, *actively misleads* — and almost no memory tool addresses it. Mneme
does, under one hard-won law:

> **A staleness checker that over-reports lies exactly like the memory it audits.**
> Every check must confirm its own success-path, or it is noise.

```
  every path/binary a memory cites ─► sweep ─────► DEAD only if the parent dir exists and the leaf is
        │                                          genuinely absent (success-path confirmed) — then it
        │                                          even locates the moved file to propose the fix
        ▼
  every [[wikilink]] in the vault ──► vault_sweep ─► resolves by basename OR path-suffix OR ../ relative
        │                                            (real Obsidian semantics; skips code-`[[` in dumps)
        ▼
  every same-topic memory pair ────► supersession_scan ─► CANDIDATE (a pre-filter, not a detector:
                                                           cosine finds same-topic, ≠ supersedes)
                                                              │
                                                              ▼
                                                     judge (LLM / human) ─► superseded_by: <name>
```

A naive `test -e` sweep over a real memory dir was **~90% false-positive** (prose `~16`, slash-commands,
brace-expansion, space-truncated paths). The version here reports a `DEAD` only when its own resolution
succeeded — so a finding is trustworthy. The wikilink sweep learned the same lesson (path-style and `../`
links resolve like Obsidian, not just by basename). And supersession is deliberately **not** a detector:
similarity surfaces *same-topic* pairs, and same-topic ≠ supersedes — the actual judgment is *meaning*
(does B contradict A?), which needs a judge, so the scan only narrows the O(n²) pairs down to candidates.

### The lifecycle model

- **Never auto-prune.** Every layer *detects + surfaces*; a human or an LLM pass adjudicates. The "obvious"
  superseded memory, read carefully, is usually just *impacted* — a premise shifted, its core still valid.
- **Deprecate, don't delete.** A superseded memory keeps its history. Memories carry, in frontmatter:
  `status: active | superseded | deprecated`, `freshness: current | impacted`, `superseded_by: <name>`.
  Recall skips `superseded`/`deprecated`; reads `impacted` with caution.

## What it does

- `mneme-index` (`index_chunked.py`) — chunk + SAC + embed the substrate → `index-chunked.json` + `matrix.npy`.
- `mneme-query` (`substrate_query.py`) — the full hybrid + rerank pipeline, on-demand.
- `substrate_daemon.py` + `hook_inject.py` — the warm daemon and the thin always-on hook client.
- `mneme-sweep` (`sweep.py`) + `vault_sweep.py` — path-rot and dead-wikilink hygiene.
- `supersession_scan.py` — the same-topic pre-filter for a consolidation/judge pass.
- `eval.py` / `rerank_eval.py` — the instrument: `recall@k` over `eval/golden.json`. Bring your own golden
  set and re-measure — that is the point.

## Project layout

```
mneme/
  config.py            # every path + model, env-overridable (MNEME_*) with generic defaults
  index_chunked.py     # chunk + SAC + embed
  hybrid_eval.py       # shared: ollama embed, cosine, BM25, RRF
  substrate_query.py   # on-demand: hybrid + rerank
  substrate_daemon.py  # warm daemon (index + reranker loaded once)
  hook_inject.py       # thin always-on client (socket)
  sweep.py             # memory path-rot (success-path-confirmed)
  vault_sweep.py       # dead [[wikilinks]]
  supersession_scan.py # same-topic pre-filter
  eval.py, rerank_eval.py
eval/golden.example.json  # a generic 3-query starter set
deploy/                   # launchd + Claude Code hook templates
```

## Setup

```bash
uv venv && uv pip install -e .          # numpy + fastembed (ONNX; no torch)
ollama pull nomic-embed-text && ollama serve

export MNEME_MEMORY_DIR=~/my-agent-memory   # defaults: ~/.mneme/memory
export MNEME_VAULT_DIR=~/obsidian-vault     #           ~/notes

python -m mneme.index_chunked               # build the index
python -m mneme.substrate_query "how did I decide X"
python -m mneme.sweep                        # path hygiene
python -m mneme.supersession_scan            # → feed candidates to your consolidation pass
```

Everything is config-driven (`mneme/config.py`, all `MNEME_*` env vars): memory dir, vault dir, state dir,
ollama URL, embed model, rerank model, thresholds.

## Current state

- Measured: `recall@5 0.33 → 0.83`, MRR 0.21 → 0.83, on an 18-query golden set over one ~500-note substrate.
- Local-only: ollama (embeddings) + fastembed/ONNX (rerank). No cloud, no vector DB, no torch. At a few
  thousand vectors, JSON + numpy cosine + in-memory BM25 is the right tool — a vector DB is premature.
- Prototype-grade. It works for its author; your substrate and golden set are different — re-measure.
- The always-on hook is precision-over-recall by design: it would rather inject the right note sometimes
  than the wrong note often. Tune `MNEME_RERANK_MIN`.

## How it came to be

One session, working backwards from a blunt question: *is this embedding any good, and how do I make an
agent actually consult it — like a system prompt, not on a whim?* The honest answer to the first was **no**
(0.33, whole-note) until it was measured and improved one lever at a time. The answer to the second turned
out to be architectural (a warm daemon + a deterministic hook) and bounded by CPU latency (hence two tiers).
Along the way the naive versions of the sweep, the vault resolver, and the supersession scan each
*over-reported* — the same failure, three times — which is where the one law above comes from. The retrieval
was the easy part. Keeping the memory honest was the point.

## License
MIT.
