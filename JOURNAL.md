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

> The entries below (2026-11-16, evening) were also **drafted by Claude Code (AI)** from its
> runs; numbers are copied from `results/`. They still need my review and my own reasoning.

### 2026-11-16: model size (1M vs 3.2M vs 15M)
- **Question / hypothesis:** is the 1M network capacity-limited?
- **Commit:** `ad8bbba` code; `train.py --channels/--blocks`.
- **Setup:** same data (rounds 0+1, 2.34M positions), 6 epochs each: 78x8 (0.98M), 128x10
  (3.18M), 256x12 (14.96M). 200 games per pairing against the 1M network.
- **Result:** validation q regret 0.0324 / 0.0310 / 0.0303. In games: 3.2M +63 Elo [22, 105],
  15M +82 [28, 131] over the 1M network (`results/2026-11-16_arena_size.txt`).
- **Decision:** that is about 38 Elo per halving of size from 1M to 3.2M and about 21 per
  halving up to 15M. Whether a bigger model pays off depends on beta: if beta is above about
  40, 1M is best. I stay at 1M until beta is announced; the pipeline takes `--channels/--blocks`.

### 2026-11-16: belief-sampled king-safety search (PLAN Step 3) - did not help
- **Question / hypothesis:** 259 of 301 losses involved a hidden attacker on my king. CNNs are
  poor at exact line geometry, so sample hidden boards from the belief head, check with the rules
  whether each move leaves the king capturable on each sample, and use that to correct the q
  value: q = P(safe) Q_safe, so Q_safe = q / (1 - danger), value = (1 - p) Q_safe, with p a blend
  of a new danger head and the sampled frequency (`trust`).
- **Commit:** `c86202f` (danger head), `77e8f4c` (search, calibration tool).
- **Setup:** rounds 0+1 network with the danger head; 32 samples; round robin, 100 games per pair.
- **Result:** search off +43 [-4, 76], trust 0.5 +29 [-33, 66], trust 1.0 +13 [-33, 55], all
  relative to entry-1's network (`results/2026-11-16_arena_search.txt`). Calibration over 37,420
  real moves (`training/eval_danger.py`): the danger head alone has log-loss 0.0297, the best
  blend 0.0267, samples alone 0.0316. The trained network already knows king danger about as
  well as independent samples from its own belief marginals, so the search adds little.
- **Decision:** keep the code, off (`samples: 0`). A better sampler (joint, not per-square)
  would be needed for this to pay.

### 2026-11-16: league rounds and outcome-blended action values (the RL part)
- **Question / hypothesis:** Stockfish's values assume both sides see everything. Pulling the q
  target of the move actually played toward the real game result (Monte-Carlo return of the
  policy that played it) should teach what works under fog, while Stockfish's values on all other
  moves keep it anchored. Iterating rounds is then on-policy policy iteration.
- **Commit:** `78883d5` (league, results), `5441fca` (outcome blend).
- **Setup:** round 2: 60,000 league games (round-1 network vs itself, round 0, the baseline,
  noisy Stockfish, bots), 1,838,101 positions in 84 min. Networks on rounds 0-2 with lambda 0,
  0.3, 0.6.
- **Result:** lambda 0.3 beat lambda 0 59-41, and scored 92/100 against guess-sf-16x1000
  (lambda 0: 81/100) (`results/2026-11-16_arena_r2.txt`). lambda 0.6 was worse than 0.3:
  86.5/200 (`results/2026-11-16_arena_lambda.txt`). Validation regret was the same for all
  three, so this effect only shows in games.
- **Decision:** lambda 0.3.

### 2026-11-16: rule-derived geometry input planes - did not help
- **Question / hypothesis:** give the network the attack maps and the open lines around my
  king directly (computed exactly by the rules from what it has seen).
- **Setup:** 5 extra planes, checked exactly against python-chess on 32,128 squares; same data
  and settings as the lambda 0.3 network.
- **Result:** same validation regret; -29 Elo [-67, -3] over 300 games
  (`results/2026-11-16_arena_geometry.txt`). Reverted; code kept in `training/experiments/`.

### 2026-11-16: round 3, and result-only positions hurt
- **Question / hypothesis:** more Monte-Carlo signal: also store positions without Stockfish
  labels (60% of positions) and train the played move's q toward the result.
- **Setup:** round 3: 60,000 league games with the lambda-0.3 network, 1.6M labelled + 2.4M
  result-only positions. Two networks on rounds 0-3: with the result-only rows (`r3`) and
  without (`r3_nomc`, `--w-mc 0`).
- **Result:** `r3` beat its predecessors (115/200 against round 2) but got worse against
  outside opponents: 73/100 against guess-sf-16x1000 (round 2: 92/100), while beating the
  deterministic baseline 100/100, which looks like exploiting league members. Its validation
  regret rose to 0.0399. `r3_nomc`: regret 0.0325, 88/100 against guess-sf-16x1000, and it beat
  both round 2 (115-85) and `r3` (105.5-94.5) (`results/2026-11-16_arena_r3_rr.txt`).
- **Decision:** drop result-only rows; the outcome blend on labelled moves is enough. Round 4
  uses `r3_nomc` as the agent, all positions labelled, with `r3` kept as a league member.

### 2026-11-16: round 4 and the entry
- **Question / hypothesis:** one more league round with `r3_nomc` playing, all positions
  labelled, should keep improving; does training longer on the larger dataset help?
- **Setup:** round 4: 60,000 league games (`r3_nomc` against itself, r0, r1, r2_l0, r2_l0.3,
  r3, the baseline, noisy Stockfish, bots), 2,102,086 positions in 88 min. Network `r4` on
  rounds 0-4, outcome weight 0.3 on rounds 2-4, result-only rows ignored; also a 10-epoch copy.
- **Result:** `r4` scored 102/200 against `r3_nomc` and 120/200 against `r2_l0.3`
  (`results/2026-11-16_arena_r4.txt`), 91/100 against guess-sf-16x1000 and rated 1534
  [1479, 1618] on the bot panel (`results/2026-11-16_arena_r4_bots.txt`). Validation regret
  0.0324. The 10-epoch copy tied `r4` 100-100 (`results/2026-11-16_arena_r4_e10.txt`).
- **Decision:** `r4` is the entry (the `final` stage of `training/pipeline.sh` trains exactly
  it). Gains per round are now small, so I stop rounds here for entry-1; the remaining losses
  are mostly to attackers I could not see.
