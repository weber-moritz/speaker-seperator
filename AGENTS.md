# AGENTS.md — speaker-separator

> Humans: use `./separate.sh help` and read **README.md** — this file is the
> detailed reference for agents working on the code.

Build clean, single-speaker audio from podcast episodes to train/enroll a TTS
voice. Strategy: **diarize-and-cut**. No neural separation — overlapped speech
is simply dropped, so the kept audio is cut losslessly from the original file
(bit-identical decode to PCM WAV, original sample rate, no denoising).

## Pipeline

```
episode.mp3 ──► Mistral Voxtral API (diarize=true) ──► out/<episode>/segments_raw.json
                                                          │  (cached; re-runs free)
                                                          ▼
              filters: drop overlap w/ other speaker, questions, [laughter] etc.,
                       segments < min-seg or > max-seg
                                                          │
              merge nearby kept segments into blocks
                                                          ▼
              ffmpeg cuts blocks from ORIGINAL file ──► out/<episode>/<SPEAKER_xx>/block_NNN.wav
              + out/<episode>/manifest.csv (every segment, kept/dropped + reason + text)
```

## Setup (once)

```bash
cd seperator
uv sync                          # creates .venv, installs requests + python-dotenv only
cp .env.example .env             # then edit .env and paste your key
# key from https://console.mistral.ai -> API keys
```

Requirements: Python 3.12 (uv will fetch it if missing), ffmpeg + ffprobe on
PATH (already installed at /usr/bin/ffmpeg). Nothing is installed system-wide;
everything lives in `.venv/`.

## Usage

```bash
uv run python prep.py run ep1.mp3 ep2.mp3            # transcribe + cut (full pipeline)
uv run python prep.py transcribe ep1.mp3             # API only, cached
uv run python prep.py cut ep1.mp3 --min-seg 4        # re-cut from cache with new filters (free)
uv run python prep.py cut ep1.mp3 --speaker SPEAKER_00   # only one speaker
uv run python prep.py inspect ep1.mp3                # speakers + minutes per speaker
uv run python prep.py concat out/ep1/SPEAKER_00 -o voice_all.wav
```

Output layout:

```
out/<episode>/segments_raw.json      # raw API segments, absolute timestamps
out/<episode>/manifest.csv           # audit: kept/dropped + reason + text
out/<episode>/<SPEAKER_xx>/block_NNN.wav
out/<episode>/<SPEAKER_xx>_all.wav   # after concat
```

Speaker labels are only consistent within one episode. Identify the target
voice by listening to one block per speaker folder, then cut/concat only that
speaker (or delete the other folders).

## Mistral API details (checked 2025-11 against live docs)

- Endpoint: `POST https://api.mistral.ai/v1/audio/transcriptions`
- Form fields used in `prep.py::transcribe_part`: `file`, `model`,
  `diarize=true`, `timestamp_granularities=segment` (optional `language`)
- Model: `voxtral-mini-latest` (alias, currently Mini Transcribe 2). Supports
  diarization and files up to ~3 h per request; `prep.py` chunks longer files
  (`--chunk-minutes`, default 150 min) with stream-copy for mp3/wav.
- Response: JSON `segments: [{start, end, text, speaker_id, ...}]` in seconds;
  `speaker_id` is null unless `diarize=true`.
- Docs links (also in the comment block at the top of `prep.py`):
  - https://docs.mistral.ai/capabilities/audio_transcription
  - https://docs.mistral.ai/studio/audio/speech_to_text/offline_transcription
  - Pricing: https://mistral.ai/pricing — voxtral-mini-latest audio is
    **$0.003/min** (cached input $0.0003/min). One 60-min episode ≈ $0.18;
    4 episodes ≈ $0.72.
- Privacy: episode audio is uploaded to Mistral's API. Only use with content
  you're OK sending there (public podcast episodes are fine).
- Retries: 4 attempts with backoff on 429/5xx and timeouts.

## Filter defaults (all tunable via CLI flags on `cut`/`run`)

| flag | default | meaning |
|---|---|---|
| `--overlap-tol` | 0.05 s | intersection with another speaker's segment that counts as overlap |
| `--min-seg` | 3.0 s | drop shorter segments |
| `--max-seg` | 60.0 s | drop longer segments (diarization slips) |
| `--gap` | 1.5 s | merge kept segments closer than this |
| `--max-block` | 120 s | cap merged block length |
| `--pad` | 0.2 s | padding around each cut |
| question filter | on | drop segments whose transcript text ends with "?" |
| marker filter | on | drop text containing `[laugh`, `[music`, `[crosstalk`, ... |

Cuts are clipped so they never bleed into the other speaker's segment
(0.05 s margin). Cutting uses input seeking (`-ss` before `-i`), which ffmpeg
makes sample-accurate, and writes 16-bit PCM WAV.

## Workflow for TTS enrollment

1. `run` on 2 episodes first (~$0.36); check minutes per speaker with `inspect`.
   Target: >= 30 min of clean target-speaker audio. Add episodes if short.
2. Listen to one block per speaker folder -> identify the target speaker.
3. Re-cut with `--speaker <label>`; audition blocks at 1.5x, delete noisy ones
   (or read manifest.csv and drop by reason/text).
4. Either enroll the individual blocks, or `concat` them into one WAV.

## Fallback if the API route is ever unavailable

Local pyannote alternative (no API cost, needs free HF account + token, slower
on CPU, needs `pyannote.audio>=4` which conflicts with `clearvoice`'s numpy<2
pin, so keep it in a separate venv):

```python
# https://huggingface.co/pyannote/speaker-diarization-community-1 (accept terms first)
from pyannote.audio import Pipeline
pipe = Pipeline.from_pretrained("pyannote/speaker-diarization-community-1",
                                token="hf_...")
out = pipe("episode.wav")           # 16 kHz mono internally; use only the timestamps
for turn, speaker in out.speaker_diarization:
    ...                              # cut from the ORIGINAL file at these times
```

The `cut` half of `prep.py` works unchanged with pyannote-style segment JSON
(just write `segments_raw.json` in the same format).

## Notes

- Never commit `.env` (git-ignored). `out/` is git-ignored too.
- `transcribe` caches to `segments_raw.json`; use `--force` to re-fetch.
- If the API renames models, keep using the `voxtral-mini-latest` alias; check
  the docs links above for the current response schema before changing code.
