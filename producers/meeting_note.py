#!/usr/bin/env python3
"""meeting_note — a note producer for mneme: a meeting recording → a transcript + a structured note.

A "producer" writes markdown into the dir mneme indexes, so `mneme-index` picks it up and `mneme-query`
finds it. This one turns an audio/video recording of a meeting into two notes under `<VAULT_DIR>/meetings/`:
a verbatim transcript and a structured "meeting intelligence" note (decisions / commitments / risks /
follow-ups / stakeholder notes). Then `python -m mneme.index_chunked` and the meeting is searchable.

Pipeline: ffmpeg (extract audio) → Gemini (transcribe → title → extract). Transcription is the one
CLOUD step (Gemini) — mneme's retrieval/hygiene is fully local, but audio transcription isn't; swap in a
local Whisper if you want it fully offline. Your `GEMINI_API_KEY` stays in your env, never in the repo.

Usage:
    export GEMINI_API_KEY=...            # your key, from env
    python producers/meeting_note.py <recording> [--title "..."] [--tag standup] [--lang English]
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from datetime import date
from pathlib import Path

try:
    from mneme.config import VAULT_DIR  # write where mneme indexes
except Exception:  # standalone fallback
    VAULT_DIR = Path(os.environ.get("MNEME_VAULT_DIR", Path.home() / "notes"))

MEETINGS_DIR = Path(os.environ.get("MNEME_MEETINGS_DIR", Path(VAULT_DIR) / "meetings"))

TRANSCRIPTION_PROMPT = """You are transcribing a recorded business meeting.

SPEAKER LABELS:
- Identify speakers as "Speaker 1", "Speaker 2", etc.
- Pick labels at first occurrence and stay consistent
- If unsure, write "[Speaker unclear]"

ACCURACY:
- Transcribe verbatim, including filler words and false starts
- Preserve technical terms, product names, acronyms exactly as spoken
- If genuinely unintelligible, write "[unintelligible]"
- For uncertain technical spelling, transcribe phonetically and add "[?]"

WHAT NOT TO DO:
- Do not summarize or paraphrase
- Do not correct technical jargon

OUTPUT:
- Plain text, paragraph per speaker turn
- Format: "Speaker 1: <what they said>"
- Approximate timestamps every 2-3 minutes in [MM:SS]
- Nothing else."""

TITLE_PROMPT = """Generate a concise, descriptive title for this meeting transcript.

Rules:
- 4-8 words max
- Capture the main topic/decision, not the participants
- No filler words ("discussion", "meeting about", "sync regarding")
- Title case, no punctuation at end
- Examples: "Q3 Roadmap Prioritization Decision", "Auth Middleware Rewrite Sign-off", "Vendor Contract Renewal Review"

Return ONLY the title, nothing else."""

EXTRACTION_PROMPT = """You are extracting operational memory from a meeting transcript for a personal knowledge base.

Return concise Markdown suitable for a notes vault. Use exactly this structure:

## Executive Summary

(2-3 sentences max)

## Decisions

For each decision:
- Decision: <what was decided>
  Owner: <who>
  Due: <date or "none">
  Confidence: high/medium/low

## Commitments

For each commitment:
- Promise: <what was promised>
  Promised by: <who>
  Promised to: <who>
  Due: <date or "none">
  Next action: <concrete next step>
  Confidence: high/medium/low

## Risks

For each risk:
- Risk: <description>
  Impact: <what breaks>
  Mitigation: <what to do>

## Follow-ups

- [ ] <action item> (@<owner>) <due date if known>

## Stakeholder Notes

(Preferences, sensitivities, context useful for next interaction)

## Prep for Next Meeting

(What to prepare, what to resolve before next sync)

---

Extract only items supported by the transcript. If a section has nothing, write "(none)".
Mark every extracted item with confidence: high, medium, or low."""


def extract_audio(recording: Path, audio_out: Path) -> None:
    print(f"Extracting audio from {recording.name}...")
    subprocess.run(
        ["ffmpeg", "-y", "-i", str(recording), "-vn", "-acodec", "aac", "-b:a", "64k", str(audio_out)],
        check=True, capture_output=True,
    )


def transcribe(client, audio_path: Path, model_name: str, lang: str | None = None) -> str:
    from google.genai import types
    print("Uploading audio to Gemini...")
    audio_file = client.files.upload(file=audio_path)
    while audio_file.state.name == "PROCESSING":
        print("  processing...")
        time.sleep(3)
        audio_file = client.files.get(name=audio_file.name)
    if audio_file.state.name == "FAILED":
        raise RuntimeError(f"Gemini upload failed: {audio_file.state}")
    print("Transcribing...")
    prompt = TRANSCRIPTION_PROMPT
    if lang:
        prompt += (
            f"\n\nLANGUAGE:\n- Output the transcript in {lang}.\n"
            f"- If the source audio is in another language, translate faithfully to {lang} while keeping "
            f"speaker labels, timestamps, and technical terms intact."
        )
    response = client.models.generate_content(
        model=model_name, contents=[audio_file, prompt],
        config=types.GenerateContentConfig(temperature=0.1),
    )
    client.files.delete(name=audio_file.name)
    return response.text


def gen(client, prompt: str, model_name: str, temp: float) -> str:
    from google.genai import types
    r = client.models.generate_content(
        model=model_name, contents=prompt, config=types.GenerateContentConfig(temperature=temp))
    return r.text


def write_notes(title, tag, recording, transcript, intelligence, today) -> tuple[Path, Path]:
    MEETINGS_DIR.mkdir(parents=True, exist_ok=True)
    slug = f"{today} - {title}"
    tf = MEETINGS_DIR / f"{slug}.transcript.md"
    tf.write_text(f"---\ntype: transcript\ndate: {today}\ntag: {tag}\n"
                  f"source_recording: {recording}\n---\n\n{transcript}\n")
    itf = MEETINGS_DIR / f"{slug}.md"
    itf.write_text(f"---\ntype: meeting_note\ndate: {today}\ntag: {tag}\n"
                   f"source_recording: {recording}\ntranscript: {tf.name}\nconfidence: medium\n---\n\n{intelligence}\n")
    return tf, itf


def main() -> int:
    ap = argparse.ArgumentParser(description="Turn a meeting recording into transcript + structured notes for mneme")
    ap.add_argument("recording", help="Path to a recording (video or audio)")
    ap.add_argument("--title", default=None, help="Meeting title (default: auto-generated)")
    ap.add_argument("--tag", default="meeting", help="Free-form tag for the note frontmatter (default: meeting)")
    ap.add_argument("--model", default=os.environ.get("GEMINI_MODEL", "gemini-2.5-pro"))
    ap.add_argument("--lang", default=None, help="Output language (e.g. 'English') — translates if source differs")
    args = ap.parse_args()

    try:
        from google import genai
    except ImportError:
        print("pip install google-genai (transcription backend)", file=sys.stderr)
        return 1
    if "GEMINI_API_KEY" not in os.environ:
        print("set GEMINI_API_KEY in your env", file=sys.stderr)
        return 1
    client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])

    recording = Path(args.recording).expanduser().resolve()
    if not recording.exists():
        print(f"ERROR: file not found: {recording}", file=sys.stderr)
        return 1
    today = date.today().isoformat()

    audio_suffixes = {".m4a", ".mp3", ".wav", ".flac", ".ogg", ".aac"}
    if recording.suffix.lower() in audio_suffixes:
        audio_path, tmp_audio = recording, None
    else:
        tmp_audio = recording.parent / f"_audio_{recording.stem}.m4a"
        extract_audio(recording, tmp_audio)
        audio_path = tmp_audio
    try:
        transcript = transcribe(client, audio_path, args.model, lang=args.lang)
        title = args.title or gen(client, f"{TITLE_PROMPT}\n\n---\n\nTRANSCRIPT (first 3000 chars):\n\n{transcript[:3000]}",
                                  args.model, 0.2).strip().strip('"').strip("'")
        intelligence = gen(client, f"{EXTRACTION_PROMPT}\n\n---\n\nTRANSCRIPT:\n\n{transcript}", args.model, 0.2)
    finally:
        if tmp_audio and tmp_audio.exists():
            tmp_audio.unlink()

    tf, itf = write_notes(title, args.tag, recording, transcript, intelligence, today)
    print(f"\n✓ Transcript: {tf}\n✓ Note:       {itf}\n  → run `python -m mneme.index_chunked` to index it")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
