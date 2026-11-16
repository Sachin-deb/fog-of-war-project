# training/

The pipeline that builds `submission/`. The agent code itself (`agent.py`, `features.py`,
`model.py`) lives in `submission/`; these scripts import it, so training and play use the same
encoder and network.

| File | What it does |
|---|---|
| `teacher.py` | Labels a position: full-board Stockfish 19 in MultiPV mode gives every legal move a win probability; moves that leave the king capturable get 0. |
| `gen_data.py` | Plays games (reference bots, a noisy full-board Stockfish player, and in round 1 my own round-0 network), encodes what the side to move has seen with `submission/features.Encoder`, and stores the teacher labels and the true enemy piece on each square. |
| `train.py` | Trains `submission/model.FowNet` (q, policy, value and belief heads) and writes a playable agent directory (`config.json`, `weights/model.safetensors`). |

## Pipeline (what `reproduce.sh` runs)

```bash
python training/gen_data.py --round 0 --games 60000 --workers 30 --depth 8 --out data/r0 --seed 0
python training/train.py   --data data/r0 --out checkpoints/r0 --epochs 6 --seed 0
python training/gen_data.py --round 1 --games 40000 --workers 30 --depth 8 --out data/r1 --seed 1 \
                           --agent-dir checkpoints/r0
python training/train.py   --data data/r0 data/r1 --out submission --epochs 6 --seed 0
```

Round 1 is DAgger-style: the round-0 network plays (against itself, the bots and the noisy
Stockfish player), and the positions it reaches are labelled by the same teacher.

## Why the labels come from the true board

The network sees only the observation history, and it is trained to regress the teacher's value
of every move. The best such regressor is the *expected* value of each move given the history,
averaged over the hidden boards that are consistent with it, weighted as they occur in the data.
The true board is an exact sample from that distribution, so its values are unbiased labels for
the belief-averaged action values. Boards from a hand-written sampler would bias the labels
toward whatever that sampler assumes. See `teacher.py` for the Fog-of-War-specific details.
