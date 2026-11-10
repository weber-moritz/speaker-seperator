#!/usr/bin/env python3
"""
prep.py — build clean single-speaker audio for TTS voice training from
podcast episodes.

Pipeline (diarize-and-cut; no neural separation — overlap is simply dropped):

  transcribe  send audio to Mistral Voxtral transcription API with
              diarization -> cached (start, end, speaker_id, text) segments
  cut         filter segments (overlap with other speakers, questions,
              [laughter]-style markers, too short/long) and cut the survivors
              LOSSLESSLY (decode to PCM WAV) from the ORIGINAL file —
              original sample rate, no denoising, no re-encoding
  concat      join one speaker's blocks into a single WAV (0.6 s gaps)
  inspect     summarize a cached transcript (speakers, minutes, overlap)

Usage:
  uv run python prep.py run episode1.mp3 episode2.mp3 [--out out]
  uv run python prep.py transcribe episode1.mp3
  uv run python prep.py cut episode1.mp3
  uv run python prep.py concat out/episode1/SPEAKER_00 -o all.wav
  uv run python prep.py inspect episode1.mp3

API key: --api-key flag, or MISTRAL_API_KEY in the environment, or in a
.env file next to this script (see .env.example).

Cost: voxtral-mini-latest is $0.003/min of audio (see AGENTS.md), so one
60-min episode is about $0.18. Transcripts are cached per episode, so
re-running `cut` with different filters is free.
"""

import argparse
import csv
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import requests
from dotenv import load_dotenv

# ---------------------------------------------------------------------------
# API reference (keep these links here so the call below is easy to check):
#   Capability docs : https://docs.mistral.ai/capabilities/audio_transcription
#   Studio/API ref  : https://docs.mistral.ai/studio/audio/speech_to_text/offline_transcription
#   Endpoint        : POST https://api.mistral.ai/v1/audio/transcriptions
#                     form fields: file, model, diarize=true,
#                                  timestamp_granularities=segment
#   Response        : JSON with "segments": [{start, end, text, speaker_id, ...}]
#                     (speaker_id is null unless diarize=true)
#   Pricing         : https://mistral.ai/pricing (audio: voxtral-mini-latest
#                     $0.003/min, cached input $0.0003/min)
# ---------------------------------------------------------------------------
API_URL = "https://api.mistral.ai/v1/audio/transcriptions"
DEFAULT_MODEL = "voxtral-mini-latest"  # alias: always the current Mini Transcribe
PRICE_PER_MIN = 0.003  # USD, voxtral-mini-latest (verify at link above)

# ---- tuning ---------------------------------------------------------------
MAX_SINGLE_MINUTES = 170  # API accepts up to ~3 h per request; chunk above this
CHUNK_MINUTES = 150       # chunk length when chunking is needed
MIN_SEG = 3.0             # drop segments shorter than this (seconds)
MAX_SEG = 60.0            # drop segments longer than this (likely diarization slips)
MAX_BLOCK = 120.0         # cap merged block length (seconds)
GAP_MERGE = 1.5           # merge kept segments closer than this (seconds)
PAD = 0.2                 # lead/tail padding around a cut (seconds)
OVERLAP_TOL = 0.05        # min intersection that counts as overlap (seconds)
NEIGHBOR_MARGIN = 0.05    # don't bleed into the other speaker's segment
DROP_MARKERS = ("[laugh", "[laughter", "[applause", "[music", "[crosstalk",
                "[noise", "[silence", "[inaudible")
# ---------------------------------------------------------------------------


def sh(cmd):
    return subprocess.run([str(c) for c in cmd], check=True,
                          capture_output=True, text=True)


def probe_duration(path):
    out = sh(["ffprobe", "-v", "error", "-show_entries", "format=duration",
              "-of", "csv=p=0", path]).stdout.strip()
    return float(out)


def audio_params(path):
    out = sh(["ffprobe", "-v", "error", "-select_streams", "a:0",
              "-show_entries", "stream=sample_rate,channels",
              "-of", "csv=p=0", path]).stdout.strip()
    rate, ch = out.split(",")
    return rate, ch


def resolve_api_key(args):
    load_dotenv(Path(__file__).resolve().parent / ".env")
    key = args.api_key or os.environ.get("MISTRAL_API_KEY")
    if not key:
        sys.exit("No API key. Pass --api-key, set MISTRAL_API_KEY, or copy "
                 ".env.example to .env and fill in MISTRAL_API_KEY.")
    return key


# ---------------------------------------------------------------------------
# Step 1: transcription (Mistral API)
# ---------------------------------------------------------------------------

def make_parts(src, workdir, chunk_minutes):
    """Return [(part_path, offset_seconds)] covering the whole file."""
    total = probe_duration(src)
    if total <= MAX_SINGLE_MINUTES * 60:
        return [(src, 0.0)]
    workdir.mkdir(parents=True, exist_ok=True)
    ext = src.suffix.lower().lstrip(".")
    seg_time = chunk_minutes * 60
    if ext in ("mp3", "wav"):  # stream-copy keeps chunking fast and lossless
        sh(["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-i", src,
            "-f", "segment", "-segment_time", str(seg_time), "-c", "copy",
            workdir / f"part_%03d.{ext}"])
    else:
        sh(["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-i", src,
            "-f", "segment", "-segment_time", str(seg_time),
            "-c:a", "libmp3lame", "-b:a", "192k", workdir / "part_%03d.mp3"])
    parts, offset = [], 0.0
    for p in sorted(workdir.glob("part_*")):
        parts.append((p, offset))
        offset += probe_duration(p)
    return parts


def transcribe_part(path, api_key, model, language):
    """One audio file -> list of raw segment dicts (docs links at top of file)."""
    data = {"model": model, "diarize": "true",
            "timestamp_granularities": "segment"}
    if language:
        data["language"] = language
    last_err = None
    for attempt in range(4):
        try:
            with open(path, "rb") as f:
                r = requests.post(
                    API_URL,
                    headers={"Authorization": f"Bearer {api_key}"},
                    files={"file": (Path(path).name, f)},
                    data=data,
                    timeout=(30, 1800),
                )
            if r.status_code in (429, 500, 502, 503, 504):
                wait = 5 * (attempt + 1)
                print(f"  {Path(path).name}: HTTP {r.status_code}, "
                      f"retry {attempt + 1}/4 in {wait}s")
                time.sleep(wait)
                continue
            r.raise_for_status()
            segments = r.json().get("segments") or []
            return segments
        except (requests.Timeout, requests.ConnectionError) as e:
            last_err = e
            print(f"  {Path(path).name}: {type(e).__name__}, "
                  f"retry {attempt + 1}/4")
            time.sleep(10)
    sys.exit(f"Giving up on {path} after repeated failures: {last_err}")


def get_segments(src, outdir, api_key, args):
    """Cached transcript -> segments with absolute timestamps."""
    cache = outdir / "segments_raw.json"
    if cache.exists() and not args.force:
        print(f"{src.name}: using cached transcript ({cache})")
        return json.loads(cache.read_text())

    parts = make_parts(src, outdir / "_parts", args.chunk_minutes)
    total_min = sum(probe_duration(p) for p, _ in parts) / 60
    print(f"{src.name}: {len(parts)} part(s), {total_min:.1f} min "
          f"(~${total_min * PRICE_PER_MIN:.2f} at ${PRICE_PER_MIN}/min, "
          f"model {args.model})")

    segments = []
    for part, offset in parts:
        for seg in transcribe_part(part, api_key, args.model, args.language):
            segments.append({
                "start": round(float(seg["start"]) + offset, 3),
                "end": round(float(seg["end"]) + offset, 3),
                "speaker_id": seg.get("speaker_id") or "unknown",
                "text": (seg.get("text") or "").strip(),
            })
        print(f"  {Path(part).name}: done, {len(segments)} segments total")

    segments.sort(key=lambda s: s["start"])
    cache.write_text(json.dumps(segments, indent=1, ensure_ascii=False))
    print(f"  cached -> {cache}")
    return segments


# ---------------------------------------------------------------------------
# Step 2: filter + cut (fully local, lossless)
# ---------------------------------------------------------------------------

def has_overlap(seg, others, tol):
    return any(seg["start"] + tol < o["end"] and o["start"] + tol < seg["end"]
               for o in others)


def drop_reason(seg, others, args):
    dur = seg["end"] - seg["start"]
    if has_overlap(seg, others, args.overlap_tol):
        return "overlap"
    t = seg["text"].lower()
    if seg["text"].rstrip(' ."\'…').endswith("?"):
        return "question"
    if any(m in t for m in DROP_MARKERS):
        return "marker"
    if dur < args.min_seg:
        return "too_short"
    if dur > args.max_seg:
        return "too_long"
    return None


def merge_blocks(kept, args):
    blocks, cur = [], None
    for s in kept:
        if (cur and s["start"] - cur["end"] <= args.gap
                and s["end"] - cur["start"] <= args.max_block):
            cur["end"] = s["end"]
        else:
            cur = {"start": s["start"], "end": s["end"]}
            blocks.append(cur)
    return blocks


def cut_block(src, start, end, others, outpath, pad, margin):
    """Decode [start, end] (+pad, clipped at neighbours) from src to PCM WAV."""
    start = max(0.0, start - pad)
    end = end + pad
    for o in others:  # don't bleed into the co-host's segment
        if start < o["start"] < end:
            end = o["start"] - margin
        if start < o["end"] < end:
            start = o["end"] + margin
    if end <= start:
        return 0.0
    sh(["ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
        "-ss", f"{start:.3f}", "-t", f"{end - start:.3f}", "-i", src,
        "-c:a", "pcm_s16le", outpath])
    return end - start


def cut_episode(src, outdir, args):
    cache = outdir / "segments_raw.json"
    if not cache.exists():
        sys.exit(f"No cached transcript at {cache} — run `transcribe` first "
                 f"(or `run`).")
    segments = json.loads(cache.read_text())
    speakers = sorted({s["speaker_id"] for s in segments})
    if args.speaker:
        speakers = [s for s in speakers if s == args.speaker]
        if not speakers:
            sys.exit(f"Speaker {args.speaker!r} not in transcript. "
                     f"Available: {sorted({s['speaker_id'] for s in segments})}")

    rows, summary = [], []
    for spk in speakers:
        mine = [s for s in segments if s["speaker_id"] == spk]
        others = [s for s in segments if s["speaker_id"] != spk]
        kept = []
        for s in mine:
            reason = drop_reason(s, others, args)
            rows.append({"episode": src.name, "speaker": spk,
                         "start": s["start"], "end": s["end"],
                         "seconds": round(s["end"] - s["start"], 1),
                         "action": "dropped" if reason else "kept",
                         "reason": reason or "", "text": s["text"]})
            if not reason:
                kept.append(s)
        spkdir = outdir / spk
        spkdir.mkdir(parents=True, exist_ok=True)
        blocks = merge_blocks(kept, args)
        total = 0.0
        for i, b in enumerate(blocks, 1):
            total += cut_block(src, b["start"], b["end"], others,
                               spkdir / f"block_{i:03d}.wav",
                               args.pad, NEIGHBOR_MARGIN)
        summary.append((spk, len(kept), total, len(blocks), spkdir))

    manifest = outdir / "manifest.csv"
    with open(manifest, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["episode", "speaker", "start", "end",
                                          "seconds", "action", "reason", "text"])
        w.writeheader()
        w.writerows(rows)

    print(f"\n{src.name} (manifest: {manifest}):")
    for spk, n, total, nblocks, spkdir in summary:
        print(f"  {spk}: {n} clean segments -> {nblocks} WAV blocks, "
              f"{total / 60:.1f} min in {spkdir}")
    return summary


# ---------------------------------------------------------------------------
# Auxiliary commands
# ---------------------------------------------------------------------------

def cmd_concat(args):
    folder = Path(args.folder)
    blocks = sorted(folder.glob("block_*.wav"))
    if not blocks:
        sys.exit(f"No block_*.wav files in {folder}")
    rate, ch = audio_params(blocks[0])
    silence = folder / "_silence.wav"
    sh(["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-f", "lavfi",
        "-i", f"anullsrc=r={rate}:cl={'stereo' if ch == '2' else 'mono'}",
        "-t", "0.6", "-c:a", "pcm_s16le", silence])
    lst = folder / "_list.txt"
    lines = []
    for b in blocks:
        lines += [f"file '{b.name}'", f"file '{silence.name}'"]
    lst.write_text("\n".join(lines))
    out = Path(args.output) if args.output else \
        folder.parent / f"{folder.parent.name}_{folder.name}_all.wav"
    sh(["ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
        "-f", "concat", "-safe", "0", "-i", lst, "-c:a", "pcm_s16le", out])
    print(f"Wrote {out} ({probe_duration(out) / 60:.1f} min)")


def cmd_inspect(args):
    src = Path(args.episode)
    cache = Path(args.out) / src.stem / "segments_raw.json"
    if not cache.exists():
        sys.exit(f"No cached transcript at {cache} — run `transcribe` first.")
    segments = json.loads(cache.read_text())
    total = probe_duration(src)
    print(f"{src.name}: {total / 60:.1f} min of audio, "
          f"{len(segments)} segments")
    per = {}
    for s in segments:
        per.setdefault(s["speaker_id"], []).append(s)
    for spk, segs in sorted(per.items()):
        speech = sum(s["end"] - s["start"] for s in segs)
        print(f"  {spk}: {len(segs)} segments, {speech / 60:.1f} min speech")


# ---------------------------------------------------------------------------

def add_common(p):
    p.add_argument("--out", default="out", help="output dir (default: out)")
    p.add_argument("--api-key", default=None,
                   help="Mistral API key (default: $MISTRAL_API_KEY or .env)")
    p.add_argument("--model", default=DEFAULT_MODEL,
                   help=f"API model (default: {DEFAULT_MODEL})")
    p.add_argument("--language", default=None,
                   help="optional ISO language code for transcription")
    p.add_argument("--force", action="store_true",
                   help="re-transcribe even if a cached transcript exists")
    p.add_argument("--chunk-minutes", type=float, default=CHUNK_MINUTES,
                   help=f"chunk length when file exceeds {MAX_SINGLE_MINUTES} min")
    p.add_argument("--min-seg", type=float, default=MIN_SEG)
    p.add_argument("--max-seg", type=float, default=MAX_SEG)
    p.add_argument("--gap", type=float, default=GAP_MERGE,
                   help="merge kept segments closer than this (s)")
    p.add_argument("--max-block", type=float, default=MAX_BLOCK)
    p.add_argument("--pad", type=float, default=PAD,
                   help="padding around each cut (s)")
    p.add_argument("--overlap-tol", type=float, default=OVERLAP_TOL)


def main():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    r = sub.add_parser("run", help="transcribe + cut (full pipeline)")
    r.add_argument("episodes", nargs="+", type=Path)
    add_common(r)
    r.add_argument("--speaker", default=None,
                   help="cut only this speaker label (e.g. SPEAKER_00)")

    t = sub.add_parser("transcribe", help="API diarization only (cached)")
    t.add_argument("episodes", nargs="+", type=Path)
    add_common(t)

    c = sub.add_parser("cut", help="filter + cut from cached transcript (free)")
    c.add_argument("episodes", nargs="+", type=Path)
    add_common(c)
    c.add_argument("--speaker", default=None,
                   help="cut only this speaker label (e.g. SPEAKER_00)")

    k = sub.add_parser("concat", help="join one speaker's blocks into one WAV")
    k.add_argument("folder", help="e.g. out/episode1/SPEAKER_00")
    k.add_argument("-o", "--output", default=None)

    i = sub.add_parser("inspect", help="summarize a cached transcript")
    i.add_argument("episode", type=Path)
    i.add_argument("--out", default="out")

    args = ap.parse_args()

    if args.cmd in ("run", "transcribe", "cut"):
        key = resolve_api_key(args) if args.cmd in ("run", "transcribe") else None
        grand = 0.0
        for src in args.episodes:
            if not src.exists():
                sys.exit(f"File not found: {src}")
            outdir = Path(args.out) / src.stem
            outdir.mkdir(parents=True, exist_ok=True)
            if args.cmd in ("run", "transcribe"):
                get_segments(src, outdir, key, args)
            if args.cmd in ("run", "cut"):
                for _, _, total, _, _ in cut_episode(src, outdir, args):
                    grand += total
        if args.cmd == "run":
            print(f"\nTotal clean audio produced: {grand / 60:.1f} min")
            print("Next: inspect each speaker's folder, keep the target "
                  "voice, audition blocks, then concat or enroll.")
    elif args.cmd == "concat":
        cmd_concat(args)
    elif args.cmd == "inspect":
        cmd_inspect(args)


if __name__ == "__main__":
    main()
