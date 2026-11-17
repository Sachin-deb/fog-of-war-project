#!/bin/bash
# CPU job: one data stage of training/pipeline.sh (gen0 ... gen4).
#   STAGE=gen0 sbatch scripts/sol/generate_data.sbatch
#SBATCH -J fow-data
#SBATCH -q class
#SBATCH -p general            # partition and QoS names: see ASU Research Computing docs
#SBATCH -c 32
#SBATCH --mem=48G
#SBATCH -t 03:00:00
#SBATCH -o runs/%x-%j.out
set -euo pipefail
module load mamba/latest
source activate fow
WORKERS=30 bash training/pipeline.sh "${STAGE:?set STAGE=gen0..gen4}"
