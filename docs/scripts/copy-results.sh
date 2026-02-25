#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
SRC_DIR="$REPO_ROOT/solver-native/results/split-sa/full"
DEST_DIR="$REPO_ROOT/docs/public/results"

mkdir -p "$DEST_DIR"

# Clean previous results (but not the manifest yet)
rm -f "$DEST_DIR"/*.tsv

# Copy files with score < 200
for f in "$SRC_DIR"/*.tsv; do
    [ -f "$f" ] || continue
    score=$(basename "$f" | cut -c1-4)
    if [ "$score" -lt 200 ] 2>/dev/null; then
        cp "$f" "$DEST_DIR/"
    fi
done

# Copy real schedule
cp "$REPO_ROOT/real-schedule.tsv" "$DEST_DIR/real-schedule.tsv"

# Generate manifest.json
(
    echo '['
    first=true
    for f in "$DEST_DIR"/*.tsv; do
        [ -f "$f" ] || continue
        name=$(basename "$f")
        if [ "$first" = true ]; then
            first=false
        else
            echo ','
        fi
        printf '  "%s"' "$name"
    done
    echo
    echo ']'
) > "$DEST_DIR/manifest.json"

count=$(ls -1 "$DEST_DIR"/*.tsv 2>/dev/null | wc -l | tr -d ' ')
echo "Copied $count TSV files to $DEST_DIR"
