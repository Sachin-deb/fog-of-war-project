#!/bin/bash
# CPU job: play games and label positions with full-board Stockfish 19 (training/gen_data.py).
#   ROUND=0 sbatch scripts/sol/generate_data.sbatch
#   ROUND=1 sbatch scripts/sol/generate_data.sbatch        (needs checkpoints/r0 from train.sbatch)
#SBATCH -J fow-data
#SBATCH -q class
#SBATCH -p general            # partition and QoS names: see ASU Research Computing docs
#SBATCH -c 32
#SBATCH --mem=32G
#SBATCH -t 03:00:00
#SBATCH -o runs/%x-%j.out
set -euo pipefail
module load mamba/latest
source activate fow
export STOCKFISH=${STOCKFISH:-$PWD/stockfish/stockfish/stockfish-linux-x86-64-universal}
ROUND=${ROUND:-0}
if [ "$ROUND" = 0 ]; then
  python training/gen_data.py --round 0 --games 60000 --workers 30 --depth 8 --out data/r0 --seed 0
else
  python training/gen_data.py --round 1 --games 40000 --workers 30 --depth 8 --out data/r1 --seed 1 \
    --agent-dir checkpoints/r0
fi
