#!/usr/bin/env bash
set -e

cd "$(dirname "$0")"
cargo build --release -p solver-native

./target/release/refine weights.json "$@"
