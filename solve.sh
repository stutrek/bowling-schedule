#!/usr/bin/env bash
set -e
trap 'kill 0' INT TERM

cd "$(dirname "$0")"

while ./target/release/solver-native solver-native/config.toml "$@"; do
    echo 'Restarting...'
done
