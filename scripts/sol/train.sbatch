#!/bin/bash
# GPU job: train FowNet (training/train.py).
#   ROUND=0 sbatch scripts/sol/train.sbatch     -> checkpoints/r0 (used to play round-1 games)
#   ROUND=1 sbatch scripts/sol/train.sbatch     -> submission/ (the entry)
#SBATCH -J fow-train
#SBATCH -q class
#SBATCH -p general            # partition and QoS names: see ASU Research Computing docs
#SBATCH -G 1
#SBATCH -c 8
#SBATCH --mem=64G
#SBATCH -t 02:00:00
#SBATCH -o runs/%x-%j.out
set -euo pipefail
module load mamba/latest
source activate fow
ROUND=${ROUND:-1}
if [ "$ROUND" = 0 ]; then
  python training/train.py --data data/r0 --out checkpoints/r0 --epochs 6 --seed 0 --device cuda
else
  python training/train.py --data data/r0 data/r1 --out submission --epochs 6 --seed 0 --device cuda
  python -m fow.validate submission --device cuda
fi
