#!/usr/bin/env bash
# Download and verify the pinned South Korea source data (docs/RULES.md, Rules ND and KG):
# FinanceData/marcap (daily snapshots of every Korean stock since 1995), at one pinned git commit.
# Safe to re-run: it skips the fetch when the commit is already on disk and re-verifies every hash.
#
# The download is untrusted data. Nothing in it is ever run or imported; the converter
# (python -m detmaso.data.ingest.kr) only reads the yearly data files, with the path passed as an argument.
# It also fetches Qlib's scripts/dump_bin.py at a pinned commit (sha256-checked), which the converter runs to
# write the Qlib binary files.
#
# Usage:
#   scripts/get_kr_data.sh                   fetch (if needed) and verify every yearly file against the manifest
#   scripts/get_kr_data.sh --write-manifest  (maintainer, once per pin) write the manifest from the fetched files
set -euo pipefail

COMMIT="d3b703628e5a73112111d8bc43439f950762f67e"  # newest commit of the default branch on 2026-11-12
REPO_URL="https://github.com/FinanceData/marcap.git"
# Qlib's own CSV-to-binary writer, used by the converter (never imported; run as a separate process with -I).
QLIB_COMMIT="b87a2c294d364a33fb739359886acffe8ec907d1"
DUMP_BIN_SHA256="b8f34c57ce1ef4b1772f3909735e66058f21b25bd7ab8a5f16318822401fe53f"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DL_DIR="${DL_DIR:-$HOME/data/marcap/$COMMIT}"
MANIFEST="${MANIFEST:-$SCRIPT_DIR/kr_data.sha256}"
DATA_GLOB="data/marcap-*.parquet"  # one file per year, 1995 onward
TOOLS_DIR="${TOOLS_DIR:-$HOME/data/tools}"
DUMP_BIN="$TOOLS_DIR/qlib_${QLIB_COMMIT:0:7}/dump_bin.py"  # in a directory of its own
QLIB_DIR="${QLIB_DIR:-$HOME/.qlib/qlib_data/kr_data_${COMMIT:0:7}}"

sha() { shasum -a 256 "$1" | cut -d' ' -f1; }

# No hooks, no content filters (so no Git LFS smudge or other filter program can run), no credentials prompt.
git_safe() {
  GIT_TERMINAL_PROMPT=0 GIT_LFS_SKIP_SMUDGE=1 git -c core.hooksPath=/dev/null -c advice.detachedHead=false "$@"
}

if [[ "$(git_safe -C "$DL_DIR" rev-parse HEAD 2>/dev/null || true)" != "$COMMIT" ]]; then
  if [[ -e "$DL_DIR" && -n "$(ls -A "$DL_DIR" 2>/dev/null)" && ! -d "$DL_DIR/.git" ]]; then
    echo "refusing to fetch into a non-empty directory that is not a git checkout: $DL_DIR" >&2
    exit 1
  fi
  mkdir -p "$DL_DIR"
  git_safe -C "$DL_DIR" init -q
  if ! git_safe -C "$DL_DIR" cat-file -e "$COMMIT^{commit}" 2>/dev/null; then
    git_safe -C "$DL_DIR" remote remove origin 2>/dev/null || true
    git_safe -C "$DL_DIR" remote add origin "$REPO_URL"
    git_safe -C "$DL_DIR" fetch -q --depth 1 origin "$COMMIT"  # exactly this commit, no history
  fi
  git_safe -C "$DL_DIR" checkout -q --detach "$COMMIT"
fi
if [[ "$(git_safe -C "$DL_DIR" rev-parse HEAD)" != "$COMMIT" ]]; then
  echo "checkout is not at the pinned commit $COMMIT" >&2
  exit 1
fi
echo "commit OK: $COMMIT"

# Paths are passed as arguments; nothing is run from inside the download directory.
shopt -s nullglob
paths=("$DL_DIR"/$DATA_GLOB)
shopt -u nullglob
if [[ ${#paths[@]} -eq 0 ]]; then
  echo "no yearly data files ($DATA_GLOB) in $DL_DIR" >&2
  exit 1
fi
files=("${paths[@]#"$DL_DIR"/}")  # relative names, e.g. data/marcap-1995.parquet

if [[ "${1:-}" == "--write-manifest" ]]; then
  for f in "${files[@]}"; do
    printf '%s  %s\n' "$(sha "$DL_DIR/$f")" "$f"
  done > "$MANIFEST"
  echo "manifest written: $MANIFEST (${#files[@]} files)"
fi

if [[ ! -f "$MANIFEST" ]]; then
  echo "manifest missing: $MANIFEST" >&2
  exit 1
fi
# The checkout must hold exactly the manifest's files, and every one of them must match its sha256.
listed="$(awk '{print $2}' "$MANIFEST" | sort)"
present="$(printf '%s\n' "${files[@]}" | sort)"
if [[ "$listed" != "$present" ]]; then
  echo "the yearly files on disk are not exactly the files in the manifest" >&2
  diff <(echo "$listed") <(echo "$present") >&2 || true
  exit 1
fi
while read -r want name; do
  if [[ "$(sha "$DL_DIR/$name")" != "$want" ]]; then
    echo "sha256 mismatch: $name" >&2
    exit 1
  fi
done < "$MANIFEST"
echo "sha256 OK: ${#files[@]} yearly files match $(basename "$MANIFEST")"
echo "manifest sha256 (data.data_sha256 in configs/kr200.yaml): $(sha "$MANIFEST")"

if [[ ! -f "$DUMP_BIN" || "$(sha "$DUMP_BIN")" != "$DUMP_BIN_SHA256" ]]; then
  mkdir -p "$(dirname "$DUMP_BIN")"
  curl -fsSL --retry 5 --retry-delay 5 -A "curl" -o "$DUMP_BIN" \
    "https://raw.githubusercontent.com/microsoft/qlib/$QLIB_COMMIT/scripts/dump_bin.py"
fi
if [[ "$(sha "$DUMP_BIN")" != "$DUMP_BIN_SHA256" ]]; then
  echo "dump_bin.py sha256 mismatch" >&2
  exit 1
fi
echo "dump_bin.py sha256 OK"

echo "source dir: $DL_DIR"
echo "dump_bin:   $DUMP_BIN"
echo
echo "Next, from the repo root (go/no-go first; convert refuses to run if a check fails):"
echo "  python3 -m uv run python -m detmaso.data.ingest.kr go-no-go --source \"$DL_DIR\" \\"
echo "      --manifest scripts/kr_data.sha256 --out results/kr200/go_no_go.md"
echo "  python3 -m uv run python -m detmaso.data.ingest.kr convert --source \"$DL_DIR\" \\"
echo "      --manifest scripts/kr_data.sha256 --dump-bin \"$DUMP_BIN\" \\"
echo "      --qlib-dir \"$QLIB_DIR\" --report results/kr200/conversion_report.md"
