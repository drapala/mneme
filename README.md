# mneme

**Local-first agent memory that remembers *and forgets well*.** Plain-markdown notes + your own memory
files become semantically queryable and stay honest over time — retrieval that finds the right note, plus
the hygiene layer almost no memory tool has: it detects and surfaces its own rot.

> A reference implementation, not a framework-to-adopt. It was built and measured for one person's
> substrate (see the numbers below) and works well there; extract, read, adapt. The differentiated,
> opinionated part is the **hygiene / lifecycle**, not the retrieval (that's standard RAG). Everything runs
> **on your disk** — [ollama](https://ollama.com) for embeddings, ONNX for reranking, no cloud, no vector
> DB, any model. This is "own your memory, rent the intelligence" with the engineering attached.

## Why (the two halves)

Most "AI memory" tools are a retrieval layer that only ever **accumulates** — and accumulation is the
failure mode. A memory that names a file/flag that no longer exists, or a fact a later decision quietly
contradicted, *actively misleads*. mneme is two composable halves:

1. **Retrieval** (`mneme/` substrate-recall) — find the right note by meaning.
2. **Hygiene** (`mneme/` sweep + supersession) — keep the memory from rotting, and forget the superseded
   without losing provenance.

## Retrieval — measured, not vibed

The pipeline is eval-driven: a golden set (`eval/`) of `query → the note that should rank #1`, scored by
`recall@k`. Every technique earned its place against that number. On the author's ~500-note substrate,
recall@5 climbed **0.33 → 0.83** one lever at a time:

| step | recall@5 | what |
|---|---|---|
| whole-note embedding (naive) | 0.33 | one vector per note — long notes dilute into generic centroids |
| + heading chunking + SAC | 0.39 | section chunks, each prepended a ~150-char parent summary |
| + hybrid dense + BM25 (RRF) | 0.61 | BM25 catches the keyword/rare-term hits dense embeddings miss |
| + cross-encoder rerank | **0.83** | multilingual rerank over the fused top-N — the biggest single lever |

Local models: [ollama](https://ollama.com) `nomic-embed-text` (dense) + a multilingual cross-encoder via
[fastembed](https://github.com/qdrant/fastembed) (ONNX, no torch). **A cross-lingual note:** all queries
were Portuguese over mixed PT/EN notes with an English-centric embedder — and it still hit 0.83, because
BM25 + the multilingual reranker carry the language gap. *You do not need to translate your notes.*

### Two tiers (the architecture that matters)

Full rerank is ~0.83 but ~seconds per query on CPU — too slow to run on **every** prompt. So:

- **On-demand, rich** — `substrate_query.py "your question"` runs the full 0.83 pipeline. Use it when you
  *want* to recall.
- **Always-on, precise** — a warm daemon (`substrate_daemon.py`) + a thin hook (`hook_inject.py`) that a
  host agent (e.g. a Claude Code `UserPromptSubmit` hook) calls on every prompt: fast (~0.5s),
  **precision-gated** (rerank score above a threshold → fires on the ~fraction of prompts where it's
  confident, ~high precision when it does, injects nothing otherwise). The harness injects deterministically
  — the agent doesn't have to remember to query. It's the calibration-injection pattern generalized to
  your whole substrate. Deploy templates in `deploy/`.

Memory is *situational* — you don't need it on every prompt, you need it when starting a task. Precise
always-on + rich on-demand is the right split; always-on full-rerank is neither affordable nor needed.

## Hygiene — the part that's actually novel

Retrieval is a crowded, solved-ish space. Rot is not. mneme treats memory staleness as a first-class
concern, with one hard-won principle running through it:

> **A staleness checker that over-reports lies exactly like the memory it audits.** Every check must
> confirm its own success-path.

- **`sweep.py`** — verifies every file/dir/binary path a memory references still exists. A naive
  `test -e` over a real memory dir was ~90% false-positive (prose `~16`, slash-commands, brace-expansion,
  space-truncated paths); this reports a DEAD only when the parent dir exists and the leaf is genuinely
  absent — the success-path is confirmed, so a "DEAD" is trustworthy. It also *locates* the moved file to
  propose an accurate fix.
- **`vault_sweep.py`** — dead `[[wikilinks]]` / embeds in an Obsidian-style vault (resolves by basename,
  path-suffix, or `../` relative, like Obsidian), plus an advisory aged-notes pass.
- **`supersession_scan.py`** — the semantic layer. Path-valid ≠ true: a memory can be fine on disk but
  *superseded* by a newer fact. This embeds every memory and surfaces high-similarity **pairs** — but it's
  an honest **pre-filter, not a detector**: cosine finds *same-topic* pairs, and same-topic ≠ supersedes.
  Actual supersession is *meaning* (does B contradict A?), which needs a judge (an LLM pass / a human), not
  a threshold. The scan narrows the O(n²) pairs to the candidates worth judging.

### The lifecycle model

- **Never auto-prune.** Every layer **detects + surfaces**; a human or an LLM pass adjudicates. Supersession
  is a judgment call — the "obvious" superseded memory, read carefully, is often just *impacted* (a premise
  shifted) with its core still valid.
- **Deprecate, don't delete.** A superseded memory keeps its provenance. Memories carry, in frontmatter:
  `status: active|superseded|deprecated`, `freshness: current|impacted`, `superseded_by: <name>`. Recall
  skips `superseded`/`deprecated`; reads `impacted` with caution.

## Install & use

```bash
git clone <this repo> && cd mneme
uv venv && uv pip install -e .          # numpy + fastembed
ollama pull nomic-embed-text            # + `ollama serve`

# point it at your notes (defaults: ~/.mneme/memory, ~/notes, ~/.mneme/state)
export MNEME_MEMORY_DIR=~/my-agent-memory
export MNEME_VAULT_DIR=~/obsidian-vault

python -m mneme.index_chunked           # build the semantic index
python -m mneme.substrate_query "what did I decide about X"   # query by meaning
python -m mneme.sweep                   # path-rot hygiene
python -m mneme.supersession_scan       # same-topic pre-filter → feed to your consolidation pass
```

Everything is config-driven (`mneme/config.py`, all `MNEME_*` env vars). Deploy the always-on hook +
daemon with the templates in `deploy/`.

## Honest limits

- Measured on **one** person's substrate and one 18-query golden set. It works there; your mileage will
  vary — bring your own `eval/golden.json` and re-measure (that's the point).
- The retrieval is standard RAG (chunk + hybrid + rerank). The contribution is the local-first packaging +
  the **hygiene/lifecycle** model. If you want a hosted, general memory service, this isn't it — try
  `mem0` / `Letta` / `Zep`. mneme is for people who want their memory to stay **on their disk, in plain
  markdown, model-agnostic**, and to *forget well*.
- Prototype-grade. Read it, take the ideas, adapt.

## License
MIT.
