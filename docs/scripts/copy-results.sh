#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
SRC_DIR="$REPO_ROOT/solver-native/results/split-sa/full"
GPU_DIR="$REPO_ROOT/solver-native/results/gpu"
DEST_DIR="$REPO_ROOT/docs/public/results"

mkdir -p "$DEST_DIR"

# Clean previous results (but not the manifest yet)
rm -f "$DEST_DIR"/*.tsv

# Copy files deduplicating by content (keep lowest-scored filename).
# Files glob-sort alphabetically so lowest score comes first.
seen_file=$(mktemp)
trap "rm -f '$seen_file'" EXIT
copied=0
skipped=0

# split-sa results: score < 420
for f in "$SRC_DIR"/*.tsv; do
    [ -f "$f" ] || continue
    score=$(basename "$f" | cut -c1-4)
    if [ "$score" -lt 420 ] 2>/dev/null; then
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

# gpu results: score < 420
for f in "$GPU_DIR"/*.tsv; do
    [ -f "$f" ] || continue
    score=$(basename "$f" | cut -c1-4)
    if [ "$score" -lt 420 ] 2>/dev/null; then
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

echo "Copied $copied winter TSV files to $DEST_DIR ($skipped duplicates skipped)"

# --- Summer results ---
SUMMER_SRC="$REPO_ROOT/solver-native/results/summer-fixed"
SUMMER_DEST="$REPO_ROOT/docs/public/summer-results"

mkdir -p "$SUMMER_DEST"
rm -f "$SUMMER_DEST"/*.tsv

summer_seen=$(mktemp)
trap "rm -f '$seen_file' '$summer_seen'" EXIT
summer_copied=0
summer_skipped=0

for f in "$SUMMER_SRC"/*.tsv; do
    [ -f "$f" ] || continue
    score=$(basename "$f" | cut -c1-4)
    if [ "$score" -lt 500 ] 2>/dev/null; then
        hash=$(shasum "$f" | cut -d' ' -f1)
        if grep -q "^${hash}$" "$summer_seen" 2>/dev/null; then
            summer_skipped=$((summer_skipped + 1))
        else
            echo "$hash" >> "$summer_seen"
            cp "$f" "$SUMMER_DEST/"
            summer_copied=$((summer_copied + 1))
        fi
    fi
done

# Copy real summer schedule
cp "$REPO_ROOT/real-summer-schedule.tsv" "$SUMMER_DEST/real-summer-schedule.tsv"
summer_copied=$((summer_copied + 1))

# Generate summer manifest.json
(
    echo '['
    first=true
    for f in "$SUMMER_DEST"/*.tsv; do
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
) > "$SUMMER_DEST/manifest.json"

echo "Copied $summer_copied summer TSV files to $SUMMER_DEST ($summer_skipped duplicates skipped)"
