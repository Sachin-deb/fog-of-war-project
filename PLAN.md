# Project plan (proposal)

- **Name:** Sachin Deb
- **GitHub username:** Sachin-deb
- **Repository is private, with @3rdAT and @Slash0BZ as collaborators:** yes

## Goal

Finish in the top 10 of the final tournament by size-adjusted Elo with a model of about **1M
parameters**, the floor of the size bonus, so that strength is the only thing left to win on.
The plan rests on three ideas, in the order I will test them: a teacher that reasons under the
same uncertainty the agent faces, a learned belief about the hidden pieces, and a small neural
search over sampled boards that uses the per-move FLOP budget, which the baseline barely touches.
Every stage must stay replicable on Sol in a few hours.

## 1. The baseline, reproduced

Environment: Python 3.13, PyTorch 2.14 (CUDA), python-chess 1.11, Stockfish 19 from
`scripts/get_stockfish.sh`, on my own machine (32 CPU cores, one RTX 5090).

```bash
python -m fow.validate submission --device cuda
python -m fow.arena agents/baseline bot:greedy bot:guess-sf-4x500 --games 100 --workers 8 --seed 0
```

`fow.validate` reports **VALID**: 314,057 parameters, 3.99×10⁷ FLOPs per move, 10/10 wins
against `bot:random`, identical moves on replay.

| Opponent | Score | Notes |
|---|---|---|
| bot:greedy | 77 / 100 (77 wins, 23 losses) | no draws; games average 28 plies |
| bot:guess-sf-4x500 | 67 / 100 (67 wins, 33 losses) | no draws; games average 40 plies |

Local rating of the baseline: **1195** (95% interval [1145, 1255], 200 games). The
README reports 1217 from 600 games, so my setup agrees with the staff's within the intervals.
The arena output is committed as `results/2026-10-25_baseline_vs_bots.txt`.

What the records show: every one of the 200 games ended by king capture, none by the draw
rules, and the games are short. At this level the game is decided by whoever blunders their
king first. That is the gap the ideas below target: the baseline imitates a teacher that could
see everything, keeps only a "last seen here" memory, and never checks whether a move is safe
across the boards it might be facing.

## 2. What you will try first

All ideas are measured the same way: 200 games (100 per color) against each of
`bot:greedy`, `bot:guess-sf-4x500` and `agents/baseline`, with `fow.arena --official` so the
FLOP budget is enforced, plus a match against the previous version of my own agent. I will
treat a change as real only when the 95% intervals of the two ratings do not overlap. One
change per experiment, one journal entry per experiment, with the commit hash.

### Idea 1: a teacher that plays under the fog (belief-averaged labels)
- **What:** replace the full-board Stockfish label with a label computed the way the agent
  would have to: for each recorded position, sample K = 16 full boards consistent with
  everything the side to move has observed in that game (the data generator knows the true
  piece counts, so the sampler is exact about *which* pieces are hidden, only their squares
  are sampled), run Stockfish on each sample, and score every legal move by how often it is
  chosen and whether it leaves the king capturable on that sample. The label is the resulting
  move distribution (a soft target), not a single move.
- **Why I expect it to help:** the baseline's teacher recommends moves that only make sense
  with full information; its top move matches that teacher on only 39% of held-out positions,
  and part of that gap is information the student can never have, not error. A teacher that
  averages over what is plausible recommends safe, information-gathering moves and punishes
  moves that hang the king in many worlds. The reference bots already show the effect:
  `bot:guess-sf` beats `bot:naive-sf`, which pretends the fog is empty, by 140 to 300 Elo
  depending on the number of samples, and offline I can afford far more samples and nodes per
  label than any bot can at play time.
- **How I will measure it:** same positions, same network, same training budget; only the
  labels change (full-board vs belief-averaged). Report both the arena ratings and the
  fraction of games lost by king capture within 5 plies of a hidden piece appearing.
  Expected: at least +100 Elo over the baseline network trained on the same positions.

### Idea 2: a learned belief about the hidden pieces (history input and a belief head)
- **What:** widen the input from one observation to a short history: the last 8 observations
  stacked, a "plies since last seen" plane, the count of enemy pieces of each type that can
  still be on the board, and the ply. Add a second output head that predicts, for every
  square, the probability of each enemy piece type being there. The data generator sees the
  true board, so this head is trained with exact labels at no extra Stockfish cost.
- **Why I expect it to help:** the hand-made memory forgets anything that moved and cannot
  say "a rook is probably on this file". A belief head forces the trunk to represent the
  hidden position, which is the information the policy actually needs, and its output is what
  Idea 3 samples from.
- **How I will measure it:** three ablations trained on identical data: baseline input, plus
  history planes, plus history and the belief loss. Report belief quality as log-loss against
  the true board on held-out positions, and strength with the standard 200-game matches.
  Expected: history alone gives a modest gain; history plus belief gives the larger one.

### Idea 3: neural search over sampled boards at play time (value head)
- **What:** add a value head trained on game outcome and on the teacher's evaluation of the
  true board. At play time, sample K = 32 boards from the belief head, take the policy's top
  M = 8 moves, apply each move to each sample with the rules engine, and pick the move with
  the best average value after discarding samples where the reply captures my king. All
  judgment comes from my own networks; the rules engine only generates positions. With a
  1M-parameter network this is about 256 forward passes, roughly 3×10¹⁰ FLOPs, 3% of the budget,
  batched in one PyTorch call.
- **Why I expect it to help:** the baseline uses 0.004% of the allowed compute. A one-ply
  check across plausible worlds is exactly the kind of error (blunders into an unseen
  attacker, missing a king capture two moves away) that a no-search policy makes most.
- **How I will measure it:** the same trained network with search on and off, and a sweep over
  K and M to find where the gain saturates. Report FLOPs per move from `fow.validate`.
  Expected: +50 to +150 Elo at negligible parameter cost.

### If time allows: relabel the agent's own games (DAgger) and a size sweep
- Generate games with the current agent against itself and the bots, label them with the
  Idea 1 teacher, and retrain; two or three rounds. This moves the training distribution toward
  the positions a learned agent actually reaches, which is what the other entries will be.
- Train the final architecture at about 0.3M, 1M and 3M parameters to measure the strength
  versus size curve. The final size is decided once β is announced, using that curve.

## 3. Compute plan

| Step | Where | Request | Expected time | Expected output |
|---|---|---|---|---|
| Teacher data (Idea 1 labels, belief and value targets) | Sol, CPU job | 32 cores, 32 GB, `-q class` | 2 jobs × about 3 h | about 400k labelled positions from about 20k games |
| Imitation training (one run) | Sol, GPU job | 1 GPU, 8 cores | 20 to 40 min at 1M parameters | one checkpoint written to `submission/` |
| Ablations and size sweep | Sol, GPU jobs | 1 GPU each | about 8 runs, under 4 GPU-hours in total | journal entries |
| DAgger rounds (if reached) | Sol, CPU + GPU | as above | 3 rounds × (2 h CPU + 0.5 h GPU) | relabelled data, retrained checkpoint |
| Evaluation matches | Sol CPU job or my machine | 16 cores | 200 games in about 20 min | ratings with intervals |

Estimated total on Sol: about 15 CPU-job hours and 6 GPU-hours across the project, well inside
the class limits (2 running jobs, 960 GPU-minutes at a time). I will develop and debug on my own
machine (32 cores, one RTX 5090) and list its GPU-hours in `submission/MODEL.md` as the policy
requires; every result that goes into an entry is reproduced on Sol.

**Staying reproducible.** `reproduce.sh` will run the final pipeline only: one data job, one
training job, one validation, no ablations, no DAgger rounds that were not kept. Every stage
takes `--seed`, data shards are written with their generation parameters, and the script
prints parameter count, FLOPs per move and the SHA-256 of the weights at the end. Target:
under 6 hours wall-clock on Sol from a fresh clone. Before each `entry-` release I will run it
from a fresh clone on Sol and play 200 games between the replicated and the submitted agent.

**Timeline.**

| Week of | Milestone |
|---|---|
| Oct 26 | Evaluation scripts and Sol job files; history planes and belief and value targets in the data generator |
| Nov 2 | Teacher data on Sol; Idea 1 versus the full-board teacher; Idea 2 ablations |
| Nov 9 | Idea 3 search; size sweep; pick the pre-tournament version |
| Nov 16 | `reproduce.sh` from a fresh clone on Sol; `entry-1` release by Nov 17 |
| Nov 23 | Study pre-tournament games; DAgger round 1; fix the most common loss pattern |
| Nov 30 | Final size choice using β; final ablations; fresh-clone replication; final `entry-` by Dec 3 |
| Dec 7 | Report and recorded presentation by Dec 12 |

## 4. Risks and questions

- **Labels are too expensive.** K = 16 Stockfish searches per position is about ten times the
  baseline's label cost. Fallback: K = 8 with fewer nodes, or label half the positions with the
  cheap full-board teacher and half with the belief-averaged one, and measure whether it matters.
- **The belief-averaged teacher is not better.** If Idea 1 does not beat the full-board teacher
  on identical positions, I keep the full-board labels and put the effort into Ideas 2 and 3,
  which do not depend on it.
- **Search does not pay for a small network.** If the value head is too noisy, use the policy's
  own log-probabilities across samples instead, and report that as the result.
- **Replication drifts.** GPU training is not bit-reproducible; I will fix seeds, keep the
  training short, and test the 100-Elo tolerance myself with a 200-game match before each entry.
- **Distribution shift.** Reference-bot games may not resemble the other entries' play. The
  pre-tournament games and the DAgger rounds are the answer; until then I also evaluate against
  my own earlier versions.
- **Sol queue time near deadlines.** Entries will be built at least two days before each
  deadline, with the previous valid entry already released as a fallback.

Questions for the staff:

1. When will β be announced, and will the staff baselines used to calibrate it be published?
2. Is the GPU memory limit for the tournament machine known yet? It affects the batch size I can
   use for search at play time.
3. Using the rules engine at play time to apply candidate moves to boards sampled from my own
   belief network: I read `POLICIES.md` as allowing this (rules, not judgment). Please confirm.
4. Will the Stockfish version in the runtime image matter for anything at play time? I assume
   not, since no engine runs inside the agent.
