# Model card: FowNet entry-1

- **Name:** FowNet, action-value learning from a full-board teacher under fog, with DAgger and
  league rounds and outcome-blended action values
- **Core model:** `model.py` `FowNet(in_planes=41, channels=78, blocks=8)`: residual CNN with
  squeeze-and-excitation; heads for per-move action value (q), soft policy, per-move danger,
  position value and a belief over enemy pieces on each square. The agent plays by the q head.
- **Parameters:** 994,885 as counted by `python -m fow.params submission` (includes batch-norm
  statistics and the value, belief and danger heads, which play does not use)
- **Weight files and SHA-256:** `weights/model.safetensors`,
  `1ce155870376c5f59e6ffa20b4eb078a95f2a7e7821b6abb29f487901309b506`
- **Input:** 33 binary planes, a square-age plane and 7 scalar planes; see `features.py`
- **Play-time code besides the network:** observation encoding and memory (`features.Encoder`),
  capturing a visible king (a rule), and a seeded random choice among moves whose predicted win
  probability is within 0.01 of the best, weighted by the policy head. A belief-sampled search
  exists in `agent.py` but is off (`samples` not set).
- **Compute per move:** about 1.2×10⁸ FLOPs (0.01% of the budget); one forward pass, no search
- **Training data:** 10,303,357 positions from 280,000 games in five rounds (round 0: reference
  bots and noisy full-board Stockfish; round 1: the round-0 network; rounds 2-4: league games of
  the previous network against earlier networks, the baseline, Stockfish players and bots).
  Every legal move of 7.9M of the positions is labelled by Stockfish 19 MultiPV at depth 8 on the
  true board; the 2.4M result-only positions of round 3 are loaded but carry no loss in the final
  run (`training/gen_data.py`, `training/teacher.py`, `training/pipeline.sh`)
- **Training:** `training/train.py --data data/r0 data/r1 data/r2 data/r3 data/r4 --epochs 6
  --seed 0 --outcome-lambda 0.3 --outcome-from data/r2 data/r3 data/r4 --w-mc 0` (AdamW,
  one-cycle 2e-3, batch 1024). The q target of the move played in rounds 2-4 is
  0.7 × teacher + 0.3 × game result.
- **Training compute:** data about 170 CPU core-hours; about 50 GPU-minutes for the seven
  training runs of the pipeline. All of it on my own machine (32 CPU cores, one NVIDIA RTX 5090);
  with pilots and ablations (sizes, outcome weights, input planes, search) about 2 GPU-hours of
  training and a few GPU-hours of arena evaluation on the same machine
- **Validation:** q-head regret 0.032 (teacher win probability lost per move), policy top-1
  agreement with the teacher 35%
- **Local rating:** 1534 [1479, 1618] against the reference bots and the baseline (400 games,
  `results/2026-11-16_arena_r4_bots.txt`); 91/100 against guess-sf-16x1000
- **How to reproduce:** `bash scripts/get_stockfish.sh && bash reproduce.sh` (see `REPRODUCE.md`)
