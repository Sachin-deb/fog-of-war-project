# Model card: FowNet entry-1

- **Name:** FowNet, action-value imitation of a full-board teacher from the fog, with one DAgger round
- **Core model:** `model.py` `FowNet(in_planes=41, channels=78, blocks=8)`: residual CNN with
  squeeze-and-excitation; heads for per-move action value (q), soft policy, position value and
  a belief over enemy pieces on each square. The agent plays by the q head.
- **Parameters:** 984,773 as counted by `python -m fow.params submission` (includes batch-norm
  statistics and the value and belief heads used only in training)
- **Weight files and SHA-256:** `weights/model.safetensors`,
  `663dd52147ba32dac37968f5df4b97e0e02f9dca6d16353b8bc9bf483da66059`
- **Input:** 33 binary planes, a square-age plane and 7 scalar planes; see `features.py`
- **Play-time code besides the network:** observation encoding and memory (`features.Encoder`),
  capturing a visible king (a rule), and a seeded random choice among moves whose predicted win
  probability is within 0.01 of the best, weighted by the policy head
- **Compute per move:** about 1.2×10⁸ FLOPs (0.01% of the budget); one forward pass, no search
- **Training data:** 2,339,845 positions from 100,000 games (round 0: reference bots and noisy
  full-board Stockfish; round 1: the round-0 network), every legal move labelled by Stockfish 19
  MultiPV at depth 8 on the true board (`training/gen_data.py`, `training/teacher.py`)
- **Training:** `training/train.py --data data/r0 data/r1 --epochs 6 --seed 0` (AdamW, one-cycle
  2e-3, batch 1024)
- **Training compute:** data about 48 CPU core-hours; training under 5 GPU-minutes. All of it on
  my own machine (32 CPU cores, one NVIDIA RTX 5090), about 0.1 GPU-hours including pilots;
  arena evaluation used about 0.5 GPU-hours on the same machine
- **Validation:** q-head regret 0.032 (teacher win probability lost per move), policy top-1
  agreement with the teacher 34%
- **Local rating:** 1567 [1496, 1672] (500 games, `results/2026-11-16_entry1_vs_field.txt`)
- **How to reproduce:** `bash scripts/get_stockfish.sh && bash reproduce.sh` (see `REPRODUCE.md`)
