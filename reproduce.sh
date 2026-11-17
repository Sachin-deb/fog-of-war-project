#!/usr/bin/env bash
# Reproduce submission/ end to end from a fresh clone (see REPRODUCE.md for times and hardware).
#
#   pip install -e ".[baseline]"
#   bash scripts/get_stockfish.sh        # once; Stockfish 19
#   bash reproduce.sh                    # uses $STOCKFISH, $WORKERS (default 30), $DEVICE
#
# The stages are defined in training/pipeline.sh. On Sol they can run as separate jobs, in the
# same order (see REPRODUCE.md, "On Sol").
set -euo pipefail
cd "$(dirname "$0")"
export STOCKFISH=${STOCKFISH:-$PWD/stockfish/stockfish/stockfish-linux-x86-64-universal}
[ -x "$STOCKFISH" ] || { echo "Stockfish not found at $STOCKFISH: run bash scripts/get_stockfish.sh"; exit 1; }

for stage in gen0 train0 gen1 train1 gen2 train2 gen3 train3 gen4 final; do
  echo "== $(date '+%H:%M:%S') stage $stage"
  bash training/pipeline.sh "$stage"
done

echo "== checks"
DEVICE=${DEVICE:-$(python -c "import torch; print('cuda' if torch.cuda.is_available() else 'cpu')")}
python -m fow.validate submission --device "$DEVICE"
sha256sum submission/weights/*.safetensors
