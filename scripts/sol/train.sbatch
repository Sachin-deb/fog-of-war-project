#!/bin/bash
# GPU job: one training stage of training/pipeline.sh (train0 ... train3, final).
#   STAGE=train0 sbatch scripts/sol/train.sbatch
#SBATCH -J fow-train
#SBATCH -q class
#SBATCH -p general            # partition and QoS names: see ASU Research Computing docs
#SBATCH -G 1
#SBATCH -c 8
#SBATCH --mem=96G
#SBATCH -t 01:30:00
#SBATCH -o runs/%x-%j.out
set -euo pipefail
module load mamba/latest
source activate fow
DEVICE=cuda bash training/pipeline.sh "${STAGE:?set STAGE=train0..train3 or final}"
if [ "$STAGE" = final ]; then python -m fow.validate submission --device cuda; fi
