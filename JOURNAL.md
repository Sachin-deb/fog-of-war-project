# Project journal

One entry per experiment or decision, newest at the bottom. Write it as you go, not at the end.
This is where the evidence for your grade lives (see docs/GRADING.md).

## Template

### YYYY-MM-DD: short title
- **Question / hypothesis:** what you expect and why.
- **Commit:** the commit hash the experiment ran at, and what you changed.
- **Setup:** data, model size, training compute (GPU-hours), games played for evaluation.
- **Result:** numbers with intervals (for example `fow.arena` ratings), plus anything you saw in games.
- **Decision:** what you will do next, and why.
- **AI help:** what an assistant did for this entry, if anything.

---

### 2026-10-__: reproduced the baseline
- ...

> The three entries below (2026-11-16) were **drafted by Claude Code (an AI assistant)** from the
> runs it executed for me; the numbers are copied from the logs in `results/`. I still have to
> review them and add my own reasoning.

### 2026-11-16: belief-averaged labels from the true board, action values, history encoder (Steps 1 and 2)
- **Question / hypothesis:** Instead of labelling with the teacher's single best move (the
  baseline), regress the teacher's win probability for *every* legal move, computed on the true
  board. With a squared or cross-entropy loss the network then learns the expected value of each
  move given only what it has observed, i.e. the belief-averaged action value of PLAN Step 1,
  without a hand-written board sampler (which would bias the labels). Combined with the history
  encoder and the belief head of Step 2, I expect clearly more than +100 Elo over the baseline.
- **Commit:** pipeline at `d75680b` (encoder `83d1fce`, network and agent `8981827`).
- **Setup:** 60,000 games (bots, plus a full-board Stockfish player at 300 nodes with 15% random
  moves), half the positions labelled with Stockfish 19 MultiPV at depth 8: 1,262,078 positions
  in 50 min on 30 cores. FowNet 78 channels x 8 blocks, 984,773 parameters, 6 epochs, 83 s on the
  RTX 5090. Encoder sanity check on the test data: rule-implied occupied squares were enemy
  99.97% of the time, rule-implied empty squares 100%, memory sightings correct 70% (base rate of
  an enemy on a hidden square: 37%).
- **Result:** validation q-head regret 0.0335, policy top-1 agreement 33%. Arena, 100 games per
  pairing with `--official`: `checkpoints/r0` 1429 [1376, 1490]; 80.5/100 against
  `agents/baseline`, 93/100 against `bot:greedy`, 84/100 against `bot:guess-sf-4x500`, 81/100 against
  `bot:guess-sf-16x1000` (`results/2026-11-16_r0_vs_bots.txt`). Training 16 epochs instead of 6
  only moved the regret from 0.0335 to 0.0331 (`results/2026-11-16_train_r0_e16.log`), so the
  model is limited by data, not by training time.
- **Not yet controlled:** this changes labels, inputs and network at once, so it does not say
  which part helped. The label ablation of PLAN Step 1 (argmax vs soft vs action value on the
  same positions and network) and the input ablation of Step 2 are still to do.
- **Decision:** keep the pipeline and add a DAgger round.

### 2026-11-16: one DAgger round
- **Question / hypothesis:** The round-0 network reaches positions that bot games never
  produce. Labelling the positions it reaches with the same teacher should help it most where it
  actually plays.
- **Commit:** same pipeline, `training/gen_data.py --round 1`.
- **Setup:** 40,000 games where the round-0 network (5% random moves) plays itself, the bots and
  the noisy Stockfish player: 1,077,767 positions in 46 min on 30 cores. Final network trained on
  both rounds (2.34M positions), 6 epochs, 155 s.
- **Result:** validation q-head regret 0.0324 (round 0: 0.0335). Arena, 100 games per pairing:
  `submission` 1567 [1496, 1672]; 66/100 against `checkpoints/r0`, 92/100 against the baseline,
  96/100 against `bot:greedy`, 89/100 against `bot:guess-sf-4x500`, 94/100 against
  `bot:guess-sf-16x1000` (`results/2026-11-16_entry1_vs_field.txt`). Caveat: 6 of the 100 games
  against the baseline were forfeits by the baseline on its first move (CUDA out-of-memory with
  16 arena workers sharing one GPU), so the true score against it is about 86/94. `fow.validate`:
  VALID, 984,773 parameters, 1.2x10^8 FLOPs per move, same moves on replay.
- **Decision:** submit this as `entry-1`. q head vs policy head for move choice is still
  untested in games (a head-to-head between two deterministic agents only repeated the same two
  games, so it needs stochastic opponents).

### 2026-11-16: how entry-1 loses
- **Question:** what decides the games it loses?
- **Setup:** the 61 losses of `checkpoints/r0` in the round-0 arena run, replayed with the
  true board.
- **Result:** 46 of 61 losses: my king was already attacked when I moved and the move did not
  escape; in 37 of those the attacker stood on a square I could not see. 15 more: the move itself
  walked into or opened a capture.
- **Decision:** the remaining weakness is not seeing threats from hidden pieces. Next: PLAN
  Step 3, search over boards sampled from the belief head, checking for king capture by the
  rules and scoring the rest with the network, then the size sweep and the ablations above.
