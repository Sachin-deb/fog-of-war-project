#!/usr/bin/env bash
# Every stage of the pipeline that builds submission/, in order. reproduce.sh runs them all;
# the Sol job scripts run one stage each (scripts/sol/).
#
#   bash training/pipeline.sh <stage>          stages: gen0 train0 gen1 train1 gen2 train2
#                                                      gen3 train3 gen4 final
#
# Data stages are CPU-only (Stockfish labels; the networks play on CPU). Train stages use $DEVICE.
# Data stages split games over $WORKERS processes; use 30 to get the same games as my run.
set -euo pipefail
cd "$(dirname "$0")/.."
WORKERS=${WORKERS:-30}
DEVICE=${DEVICE:-$(python -c "import torch; print('cuda' if torch.cuda.is_available() else 'cpu')")}
export STOCKFISH=${STOCKFISH:-$PWD/stockfish/stockfish/stockfish-linux-x86-64-universal}
GEN="python training/gen_data.py --workers $WORKERS --depth 8"
TRAIN="python training/train.py --epochs 6 --seed 0 --device $DEVICE"

case "${1:?stage}" in
  # round 0: reference bots and noisy full-board Stockfish
  gen0)   $GEN --round 0 --games 60000 --out data/r0 --seed 0 ;;
  train0) $TRAIN --data data/r0 --out checkpoints/r0 ;;
  # round 1 (DAgger): the round-0 network plays
  gen1)   $GEN --round 1 --games 40000 --out data/r1 --seed 1 --agent-dir checkpoints/r0 ;;
  train1) $TRAIN --data data/r0 data/r1 --out checkpoints/r1 ;;
  # round 2 (league): round-1 network against round 0, the baseline, Stockfish players and bots
  gen2)   $GEN --round 2 --games 60000 --keep 0.5 --out data/r2 --seed 2 --agent-dir checkpoints/r1 \
            --prev-dirs checkpoints/r0 agents/baseline ;;
  train2) $TRAIN --data data/r0 data/r1 data/r2 --out checkpoints/r2_l0
          $TRAIN --data data/r0 data/r1 data/r2 --out checkpoints/r2_l0.3 \
            --outcome-lambda 0.3 --outcome-from data/r2 ;;
  # round 3 (league); this round also stored result-only positions (--unlabelled 0.6).
  # checkpoints/r3 trains on them and is kept only as a league member for round 4;
  # checkpoints/r3_nomc ignores them (--w-mc 0) and plays round 4.
  gen3)   $GEN --round 2 --games 60000 --keep 0.4 --unlabelled 0.6 --out data/r3 --seed 3 \
            --agent-dir checkpoints/r2_l0.3 --prev-dirs checkpoints/r0 checkpoints/r1 checkpoints/r2_l0 agents/baseline ;;
  train3) $TRAIN --data data/r0 data/r1 data/r2 data/r3 --out checkpoints/r3 \
            --outcome-lambda 0.3 --outcome-from data/r2 data/r3
          $TRAIN --data data/r0 data/r1 data/r2 data/r3 --out checkpoints/r3_nomc \
            --outcome-lambda 0.3 --outcome-from data/r2 data/r3 --w-mc 0 ;;
  # round 4 (league)
  gen4)   $GEN --round 2 --games 60000 --keep 0.5 --out data/r4 --seed 4 --agent-dir checkpoints/r3_nomc \
            --prev-dirs checkpoints/r0 checkpoints/r1 checkpoints/r2_l0 checkpoints/r2_l0.3 checkpoints/r3 agents/baseline ;;
  # the entry
  final)  $TRAIN --data data/r0 data/r1 data/r2 data/r3 data/r4 --out submission \
            --outcome-lambda 0.3 --outcome-from data/r2 data/r3 data/r4 --w-mc 0 ;;
  *) echo "unknown stage $1"; exit 1 ;;
esac
