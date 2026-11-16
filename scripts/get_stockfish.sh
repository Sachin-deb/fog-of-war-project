#!/usr/bin/env bash
# Download Stockfish 19 (the version my training labels were made with) for Linux x86-64 into
# ./stockfish/ and print the line to add to your shell setup.
#
# The template's version fetched ".../latest/download/stockfish-ubuntu-x86-64-avx2.tar", an asset
# that the sf_19 release no longer has; this one is pinned to the sf_19 release so the labels
# are reproducible.
set -euo pipefail
URL="https://github.com/official-stockfish/Stockfish/releases/download/sf_19/stockfish-linux-x86-64-universal.tar.gz"
mkdir -p stockfish
curl -fsSL "$URL" -o stockfish/sf.tar.gz
tar -xzf stockfish/sf.tar.gz -C stockfish
BIN=$(find "$PWD/stockfish" -type f -name 'stockfish-linux-x86-64-universal' | head -n 1)
chmod +x "$BIN"
"$BIN" <<< "uci" | head -n 1
echo "Stockfish is at: $BIN"
echo "Add this to your shell setup:  export STOCKFISH=$BIN"
