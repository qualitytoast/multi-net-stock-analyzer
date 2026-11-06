#!/usr/bin/env bash
# Download, verify and extract the pinned community Qlib CN dataset (plan assumption A1).
# Safe to re-run: it skips work that is already done and re-verifies hashes.
set -euo pipefail

TAG="2026-11-04"
ARCHIVE_SHA256="dc917293925635189e982f47d35b960c3db8a9c3f11312c937673c8afdec21aa"
VALIDATOR_COMMIT="b8c129b4d9b838f050eac8b1135111b3b79fc894"
VALIDATOR_SHA256="6e6bf6f227e2aed1a3bf545a6844db78a146b0e3bdfafd39cb2f16779e2567da"

DL_DIR="${DL_DIR:-$HOME/data/investment_data/$TAG}"
TOOLS_DIR="${TOOLS_DIR:-$HOME/data/tools}"
QLIB_DIR="${QLIB_DIR:-$HOME/.qlib/qlib_data/cn_data_$TAG}"
BASE_URL="https://github.com/chenditc/investment_data/releases/download/$TAG"

sha() { shasum -a 256 "$1" | cut -d' ' -f1; }

mkdir -p "$DL_DIR" "$TOOLS_DIR"

if [[ ! -f "$DL_DIR/qlib_bin.tar.gz" || "$(sha "$DL_DIR/qlib_bin.tar.gz")" != "$ARCHIVE_SHA256" ]]; then
  curl -fL --retry 5 --retry-delay 5 -o "$DL_DIR/qlib_bin.tar.gz" "$BASE_URL/qlib_bin.tar.gz"
fi
if [[ ! -f "$DL_DIR/qlib_bin.manifest.json" ]]; then
  curl -fL --retry 5 --retry-delay 5 -o "$DL_DIR/qlib_bin.manifest.json" "$BASE_URL/qlib_bin.manifest.json"
fi
if [[ "$(sha "$DL_DIR/qlib_bin.tar.gz")" != "$ARCHIVE_SHA256" ]]; then
  echo "archive sha256 mismatch" >&2
  exit 1
fi
echo "archive sha256 OK"

VALIDATOR="$TOOLS_DIR/validate_archive.py"
if [[ ! -f "$VALIDATOR" ]]; then
  curl -fsSL -o "$VALIDATOR" \
    "https://raw.githubusercontent.com/chenditc/investment_data/$VALIDATOR_COMMIT/qlib/validate_archive.py"
fi
if [[ "$(sha "$VALIDATOR")" != "$VALIDATOR_SHA256" ]]; then
  echo "validator sha256 mismatch" >&2
  exit 1
fi
python3 -I "$VALIDATOR" --archive "$DL_DIR/qlib_bin.tar.gz" \
  --manifest "$DL_DIR/qlib_bin.manifest.json" --expected-tag "$TAG" --require-publishable

if [[ -f "$QLIB_DIR/calendars/day.txt" ]]; then
  echo "already extracted: $QLIB_DIR"
else
  mkdir -p "$QLIB_DIR"
  tar -xzf "$DL_DIR/qlib_bin.tar.gz" -C "$QLIB_DIR" --strip-components=1
fi
echo "calendar: $(head -1 "$QLIB_DIR/calendars/day.txt") .. $(tail -1 "$QLIB_DIR/calendars/day.txt")"
echo "qlib dir: $QLIB_DIR"
