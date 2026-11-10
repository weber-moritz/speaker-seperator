#!/usr/bin/env bash
# Human-friendly wrapper around prep.py.
# Usage: ./separate.sh <command> [args...]   — see ./separate.sh help
set -euo pipefail
cd "$(dirname "$0")"

# Some shells carry a polluted LD_LIBRARY_PATH (e.g. from editor extensions)
# that breaks the system ffmpeg with GLIBCXX errors. Drop it if so.
if ! ffmpeg -version >/dev/null 2>&1; then
  unset LD_LIBRARY_PATH
  if ! ffmpeg -version >/dev/null 2>&1; then
    echo "ERROR: ffmpeg is not working on this machine." >&2
    echo "       Install it, e.g.: sudo apt install ffmpeg" >&2
    exit 1
  fi
fi

need_key() {
  if [ -z "${MISTRAL_API_KEY:-}" ] && ! grep -qE '^MISTRAL_API_KEY=..+' .env 2>/dev/null; then
    echo "ERROR: no Mistral API key found." >&2
    echo "       Run ./separate.sh setup, then paste your key from" >&2
    echo "       https://console.mistral.ai (API keys) into .env" >&2
    exit 1
  fi
}

usage() {
  cat <<'EOF'
speaker-separator — turn podcast episodes into clean single-speaker WAVs
for TTS voice training. Overlapping speech is dropped, not separated.

Commands:
  ./separate.sh setup                 one-time: create .venv (uv) and .env
  ./separate.sh run ep1.mp3 ep2.mp3   full pipeline: diarize via Mistral API
                                      (~$0.18/hour) + cut clean blocks locally
  ./separate.sh inspect ep1.mp3       show speakers and minutes per speaker
  ./separate.sh cut ep1.mp3           re-cut from the cached transcript (free),
                                      optionally: --speaker SPEAKER_00 --min-seg 4
  ./separate.sh concat out/ep1/SPEAKER_00 -o voice_all.wav
                                      join one speaker's blocks into one WAV
  ./separate.sh help                  this text

Typical flow:
  1. ./separate.sh setup            # then edit .env and add MISTRAL_API_KEY
  2. ./separate.sh run episode1.mp3 episode2.mp3
  3. ./separate.sh inspect episode1.mp3
     play out/episode1/SPEAKER_*/block_001.wav to hear who is who
  4. ./separate.sh cut episode1.mp3 --speaker SPEAKER_00
  5. skim out/episode1/manifest.csv, delete bad blocks by ear
  6. ./separate.sh concat out/episode1/SPEAKER_00 -o voice_all.wav

All flags of prep.py are passed through, e.g.:
  ./separate.sh cut ep1.mp3 --min-seg 5 --max-seg 45 --force
Run `uv run python prep.py <command> --help` for every flag.
Outputs land in out/<episode>/...  See README.md for the full guide.
EOF
}

cmd="${1:-help}"
[ $# -gt 0 ] && shift
case "$cmd" in
  setup)
    uv sync
    if [ ! -f .env ]; then
      cp .env.example .env
      echo "Created .env — paste your MISTRAL_API_KEY into it now."
      echo "Get a key at https://console.mistral.ai (API keys section)."
    else
      echo ".env already exists."
    fi
    ;;
  run|transcribe)
    need_key
    exec uv run python prep.py "$cmd" "$@"
    ;;
  cut|inspect|concat)
    exec uv run python prep.py "$cmd" "$@"
    ;;
  help|-h|--help)
    usage
    ;;
  *)
    echo "Unknown command: $cmd" >&2
    usage
    exit 1
    ;;
esac
