#!/usr/bin/env bash
# Fetch and verify the pinned inputs of the free US coverage check (docs/RULES.md, Rule UC):
#   1. fja05680/sp500 (unofficial S&P 500 members by date) at one pinned git commit;
#   2. Tiingo's public list of supported tickers (no API key), pinned by sha256.
# The pins are in scripts/us_coverage_data.pins, which the check reads too; change them there only.
# Nothing is stored in the repo. Safe to re-run: it skips work that is already done and re-verifies the hashes.
#
# Usage:
#   scripts/get_us_coverage_data.sh               fetch what is missing, then verify both pins
#   scripts/get_us_coverage_data.sh --print-pins  print the pins, one KEY=value per line, and exit; fetches nothing
#
# If the Tiingo hash differs
#   Tiingo replaces this file every day, so a fresh download almost never has the pinned hash. This script
#   therefore never replaces an existing local copy and never accepts a download with another hash. When the
#   check fails:
#   - a local copy exists but its hash differs: the file was changed or swapped. Do not run the coverage check
#     on it. Restore the pinned file (TIINGO_ZIP_SHA256 in scripts/us_coverage_data.pins) from a backup into
#     TIINGO_DIR.
#   - no local copy exists and the download has another hash: the pinned file is gone from Tiingo's server.
#     Restore it from a backup, or make a new pin.
#   - a new pin is a new input, so the coverage numbers change and the old result is no longer reproducible.
#     Ask the owner first. Then download today's file into a new dated directory, from the repo root:
#       source scripts/us_coverage_data.pins; d=~/data/tiingo/supported_tickers_$(date -u +%F); mkdir -p "$d"
#       date -u +%FT%TZ; curl -fL -o "$d/supported_tickers.zip" "$TIINGO_URL"; shasum -a 256 "$d/supported_tickers.zip"
#     Copy the printed time and hash into TIINGO_DOWNLOADED_AT and TIINGO_ZIP_SHA256 in scripts/us_coverage_data.pins
#     (the only place they are written; the check reads the same file), log the change and its reason under "Rule
#     changes" in docs/RULES.md (the Details under Rule UC name the old hash), and re-run the check.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# SP500_REPO, SP500_COMMIT, SP500_FILE, SP500_FILE_SHA256, TIINGO_URL, TIINGO_DOWNLOADED_AT, TIINGO_ZIP_SHA256.
source "$SCRIPT_DIR/us_coverage_data.pins"
TIINGO_DATE="${TIINGO_DOWNLOADED_AT:0:10}"

if [[ $# -eq 1 && "$1" == "--print-pins" ]]; then
  for name in SP500_REPO SP500_COMMIT SP500_FILE SP500_FILE_SHA256 TIINGO_URL TIINGO_DOWNLOADED_AT TIINGO_ZIP_SHA256; do
    printf '%s=%s\n' "$name" "${!name}"
  done
  exit 0
elif [[ $# -gt 0 ]]; then
  echo "unknown argument(s): $* (the only option is --print-pins)" >&2
  exit 2
fi

SP500_DIR="${SP500_DIR:-$HOME/data/fja05680_sp500/$SP500_COMMIT}"
TIINGO_ROOT="${TIINGO_ROOT:-$HOME/data/tiingo}"
TIINGO_DIR="$TIINGO_ROOT/supported_tickers_$TIINGO_DATE"
TIINGO_ZIP="$TIINGO_DIR/supported_tickers.zip"

sha() { shasum -a 256 "$1" | cut -d' ' -f1; }

# Downloads the Tiingo file to "$1" and writes "$1.info" (source, time, hash) next to it.
download_tiingo() {
  local out="$1" now
  now="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  curl -fL --retry 5 --retry-delay 5 -o "$out" "$TIINGO_URL"
  {
    echo "url: $TIINGO_URL"
    echo "downloaded_at_utc: $now"
    echo "sha256: $(sha "$out")"
    echo "bytes: $(wc -c < "$out" | tr -d ' ')"
  } > "$out.info"
}

fetch_sp500() {
  if [[ ! -d "$SP500_DIR/.git" ]]; then
    mkdir -p "$SP500_DIR"
    git -C "$SP500_DIR" init -q
    git -C "$SP500_DIR" remote add origin "$SP500_REPO"
    # A shallow fetch of exactly the pinned commit, never of a branch (a branch moves).
    GIT_LFS_SKIP_SMUDGE=1 git -C "$SP500_DIR" fetch -q --depth 1 origin "$SP500_COMMIT"
    GIT_LFS_SKIP_SMUDGE=1 git -C "$SP500_DIR" checkout -q --detach FETCH_HEAD
  fi
  if [[ "$(git -C "$SP500_DIR" rev-parse HEAD)" != "$SP500_COMMIT" ]]; then
    echo "fja05680/sp500: $SP500_DIR is not at the pinned commit $SP500_COMMIT" >&2
    exit 1
  fi
  echo "fja05680/sp500 commit OK: $SP500_COMMIT"
}

fetch_sp500
if [[ ! -f "$SP500_DIR/$SP500_FILE" || "$(sha "$SP500_DIR/$SP500_FILE")" != "$SP500_FILE_SHA256" ]]; then
  echo "membership file missing or sha256 mismatch: $SP500_DIR/$SP500_FILE" >&2
  echo "The git commit pins this file, so the working copy was changed. Delete $SP500_DIR and re-run." >&2
  exit 1
fi
echo "membership file sha256 OK: $SP500_FILE"

if [[ -f "$TIINGO_ZIP" ]]; then
  if [[ "$(sha "$TIINGO_ZIP")" != "$TIINGO_ZIP_SHA256" ]]; then
    echo "Tiingo file sha256 mismatch: $TIINGO_ZIP" >&2
    echo "  pinned: $TIINGO_ZIP_SHA256 (downloaded $TIINGO_DOWNLOADED_AT)" >&2
    echo "  found:  $(sha "$TIINGO_ZIP")" >&2
    echo "The local copy is not the pinned file. It was left untouched and nothing was downloaded." >&2
    echo "Do not run the coverage check on it. See 'If the Tiingo hash differs' at the top of this script." >&2
    exit 1
  fi
else
  # Tiingo replaces the file daily, so this only succeeds while the pinned file is still the one being served.
  mkdir -p "$TIINGO_DIR"
  TMP="$TIINGO_DIR/supported_tickers.zip.unverified"
  download_tiingo "$TMP"
  if [[ "$(sha "$TMP")" != "$TIINGO_ZIP_SHA256" ]]; then
    echo "Tiingo file sha256 mismatch: the file served now is not the pinned one." >&2
    echo "  pinned: $TIINGO_ZIP_SHA256 (downloaded $TIINGO_DOWNLOADED_AT)" >&2
    echo "  served: $(sha "$TMP")" >&2
    rm -f "$TMP" "$TMP.info"
    echo "The download was discarded. See 'If the Tiingo hash differs' at the top of this script." >&2
    exit 1
  fi
  mv "$TMP" "$TIINGO_ZIP"
  mv "$TMP.info" "$TIINGO_ZIP.info"
fi
echo "Tiingo file sha256 OK: $TIINGO_ZIP_SHA256 (downloaded $TIINGO_DOWNLOADED_AT)"

echo "membership: $SP500_DIR/$SP500_FILE"
echo "tiingo:     $TIINGO_ZIP"
