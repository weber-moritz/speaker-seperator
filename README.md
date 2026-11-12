*Note: LLMs were heavily used to create this project, consider it slop. Dont rely on it, dont use it in production and verify the results.*

# speaker-separator

Turn podcast episodes into **clean, single-speaker WAV files** you can use to
train/enroll a TTS voice.

It works in two stages:

1. **Diarize** — your audio goes to Mistral's transcription API
   (`voxtral-mini-latest`, `diarize=true`). You get back a timed list of who
   spoke when, plus the transcript text. Cost: **$0.003/min** (~$0.18 per
   60-min episode). Result is cached, so re-running step 2 is free.
2. **Cut** — locally, with ffmpeg. Segments where both people talk at once,
   questions, `[laughter]`-type markers, and very short/long segments are
   dropped. The survivors are cut **losslessly from your original file**
   (16-bit PCM WAV, original sample rate, no denoising) into per-speaker
   folders.

No GPU needed, no HuggingFace account needed. Only `uv` and `ffmpeg`.

## 1. One-time setup

```bash
cd seperator
./separate.sh setup
```

This creates the virtual environment (`.venv/`, via `uv`) and a `.env` file.
Then open `.env` and paste your Mistral API key:

```
MISTRAL_API_KEY=your_key_here
```

Get a key at https://console.mistral.ai → "API keys". (Alternatively export
`MISTRAL_API_KEY` in your shell instead of using `.env`.)

## 2. Run it on your episodes

```bash
./separate.sh run episode1.mp3 episode2.mp3
```

Start with 2 episodes (~$0.36 total). You'll see the estimated cost before it
uploads. Long files (>170 min) are split automatically. Transcripts are
cached in `out/<episode>/segments_raw.json` — use `--force` to re-fetch.

## 3. Find out which speaker is you

```bash
./separate.sh inspect episode1.mp3
# play one block per speaker:
play out/episode1/SPEAKER_00/block_001.wav
play out/episode1/SPEAKER_01/block_001.wav
```

Speaker labels are only consistent within one episode.

## 4. Cut only your voice (and re-tune filters for free)

```bash
./separate.sh cut episode1.mp3 --speaker SPEAKER_00
# common tweaks:
./separate.sh cut episode1.mp3 --speaker SPEAKER_00 --min-seg 4 --max-seg 45
```

## 5. Audition and clean up

Every decision is logged in `out/episode1/manifest.csv` (kept/dropped + reason
+ transcript text). Listen to the blocks at 1.5x speed and delete the noisy
ones from the speaker folder.

## 6. Optional: merge into one file

```bash
./separate.sh concat out/episode1/SPEAKER_00 -o voice_all.wav
```

## Output layout

```
out/<episode>/segments_raw.json        # raw API result (cache)
out/<episode>/manifest.csv             # every segment: kept/dropped, why, text
out/<episode>/SPEAKER_00/block_001.wav # clean blocks, 16-bit PCM WAV
out/<episode>/SPEAKER_01/...
```

## Commands cheat sheet

| command | what it does | needs API key? |
|---|---|---|
| `./separate.sh setup` | create `.venv` + `.env` | no |
| `./separate.sh run FILE...` | diarize (API) + cut | yes |
| `./separate.sh transcribe FILE...` | diarize only, cached | yes |
| `./separate.sh cut FILE...` | filter + cut from cache | no |
| `./separate.sh inspect FILE` | speakers + minutes per speaker | no |
| `./separate.sh concat FOLDER -o OUT.wav` | join blocks into one WAV | no |
| `./separate.sh help` | usage | no |

All `prep.py` flags pass through the wrapper, e.g.
`./separate.sh cut ep1.mp3 --min-seg 5 --gap 2`. Full flag list:
`uv run python prep.py cut --help`.

## Troubleshooting

- **`ffmpeg: ... GLIBCXX ... not found`** — your shell's `LD_LIBRARY_PATH` is
  polluted (some editor extensions do this). The wrapper unsets it
  automatically; manually: `unset LD_LIBRARY_PATH`.
- **`ERROR: no Mistral API key found`** — run `./separate.sh setup` and fill in
  `.env`, or `export MISTRAL_API_KEY=...`.
- **API errors / 429** — `prep.py` retries 4 times with backoff on its own.
- **Want different filters?** `cut` re-runs on the cached transcript and costs
  nothing: try `--min-seg`, `--max-seg`, `--gap`, `--pad`, `--overlap-tol`,
  `--drop-questions`. Defaults are tuned for TTS data: segments >= 1 s are
  kept, questions are kept, fragments < 2.5 s apart are glued into one block.
- **Privacy note:** episode audio is uploaded to Mistral's API. Only use it
  with content you're comfortable sending there.

## For agents / internals

`AGENTS.md` documents the architecture, the exact API request/response schema
with doc links, pricing, caching behavior, and the local pyannote fallback.
The tool itself is a single file, `prep.py` — run
`uv run python prep.py --help` for all subcommands.
