#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
SRC_DIR="$REPO_ROOT/solver-native/results/split-sa/full"
DEST_DIR="$REPO_ROOT/docs/public/results"

mkdir -p "$DEST_DIR"

# Clean previous results (but not the manifest yet)
rm -f "$DEST_DIR"/*.tsv

# Copy files with score < 200, deduplicating by content (keep lowest-scored filename).
# Files glob-sort alphabetically so lowest score comes first.
seen_file=$(mktemp)
trap "rm -f '$seen_file'" EXIT
copied=0
skipped=0

for f in "$SRC_DIR"/*.tsv; do
    [ -f "$f" ] || continue
    score=$(basename "$f" | cut -c1-4)
    if [ "$score" -lt 200 ] 2>/dev/null; then
        hash=$(shasum "$f" | cut -d' ' -f1)
        if grep -q "^${hash}$" "$seen_file" 2>/dev/null; then
            skipped=$((skipped + 1))
        else
            echo "$hash" >> "$seen_file"
            cp "$f" "$DEST_DIR/"
            copied=$((copied + 1))
        fi
    fi
done

# Copy real schedule
cp "$REPO_ROOT/real-schedule.tsv" "$DEST_DIR/real-schedule.tsv"
copied=$((copied + 1))

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

echo "Copied $copied TSV files to $DEST_DIR ($skipped duplicates skipped)"
