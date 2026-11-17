#!/usr/bin/env bash
# Submit the whole pipeline on Sol as a chain of jobs, each starting after the previous one
# succeeded. Run from the repository root on a login node:  bash scripts/sol/run_all.sh
set -euo pipefail
prev=""
for stage in gen0 train0 gen1 train1 gen2 train2 gen3 train3 gen4 final; do
  script=scripts/sol/train.sbatch
  case $stage in gen*) script=scripts/sol/generate_data.sbatch ;; esac
  dep=${prev:+--dependency=afterok:$prev}
  prev=$(STAGE=$stage sbatch --parsable --export=ALL,STAGE=$stage $dep "$script")
  echo "$stage -> job $prev"
done
