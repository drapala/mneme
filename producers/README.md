# producers — feed mneme's index

A *producer* writes markdown into the dir mneme indexes (`MNEME_VAULT_DIR`, default `~/notes`), so
`mneme-index` picks it up and `mneme-query` finds it by meaning. Producers are the "in" side of the loop:
something happens → a note is written → mneme makes it searchable.

## `meeting_note.py` — a meeting recording → transcript + structured note

Turns an audio/video recording into two notes under `<VAULT_DIR>/meetings/`: a verbatim transcript and a
structured note (executive summary, decisions, commitments, risks, follow-ups, stakeholder notes).

```bash
export GEMINI_API_KEY=...                       # your key, from env — never committed
pip install google-genai                        # transcription backend
python producers/meeting_note.py recording.mp4 --lang English
python -m mneme.index_chunked                    # index it → now searchable
```

Pipeline: `ffmpeg` (extract audio) → Gemini (transcribe → title → extract). Transcription is the one
**cloud** step — mneme's retrieval + hygiene are fully local, but audio transcription isn't; swap in a
local Whisper if you want it fully offline. Config: `MNEME_MEETINGS_DIR` (output), `GEMINI_MODEL`.
