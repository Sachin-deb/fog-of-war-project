#!/usr/bin/env bash
# Reproduce submission/ end to end from a fresh clone (see REPRODUCE.md for times and hardware).
#
#   bash scripts/get_stockfish.sh        # once; Stockfish 19
#   bash reproduce.sh                    # uses $STOCKFISH, $WORKERS (default 30), $DEVICE
#
# On Sol the same four steps can run as separate jobs, in this order:
#   ROUND=0 sbatch scripts/sol/generate_data.sbatch
#   ROUND=0 sbatch scripts/sol/train.sbatch
#   ROUND=1 sbatch scripts/sol/generate_data.sbatch
#   ROUND=1 sbatch scripts/sol/train.sbatch
set -euo pipefail
WORKERS=${WORKERS:-30}
DEVICE=${DEVICE:-$(python -c "import torch; print('cuda' if torch.cuda.is_available() else 'cpu')")}
export STOCKFISH=${STOCKFISH:-$PWD/stockfish/stockfish/stockfish-linux-x86-64-universal}
[ -x "$STOCKFISH" ] || { echo "Stockfish not found at $STOCKFISH: run bash scripts/get_stockfish.sh"; exit 1; }

echo "== 1. round-0 data: 60,000 games (bots and noisy full-board Stockfish), MultiPV labels at depth 8"
python training/gen_data.py --round 0 --games 60000 --workers "$WORKERS" --depth 8 --out data/r0 --seed 0

echo "== 2. round-0 network -> checkpoints/r0"
python training/train.py --data data/r0 --out checkpoints/r0 --epochs 6 --seed 0 --device "$DEVICE"

echo "== 3. round-1 data: 40,000 games played by the round-0 network (DAgger), same teacher"
python training/gen_data.py --round 1 --games 40000 --workers "$WORKERS" --depth 8 --out data/r1 --seed 1 \
  --agent-dir checkpoints/r0

echo "== 4. final network on both rounds -> submission/"
python training/train.py --data data/r0 data/r1 --out submission --epochs 6 --seed 0 --device "$DEVICE"

echo "== 5. checks"
python -m fow.validate submission --device "$DEVICE"
sha256sum submission/weights/*.safetensors
