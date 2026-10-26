# Project plan (proposal)

- **Name:** Sachin Deb
- **GitHub username:** Sachin-deb
- **Repository is private, with @3rdAT and @Slash0BZ as collaborators:** yes

## What I am aiming for

I want to finish in the top 10 of the final tournament by size-adjusted Elo with a model of
about **1M parameters**, the floor of the size bonus, so that strength is the only thing left to
win on. My approach has three parts, which I plan to test in this order: a teacher that gives
labels under the same uncertainty the agent faces, a learned belief about the hidden pieces,
and a small neural search over sampled boards that uses the per-move FLOP budget the baseline
barely touches. Every stage has to stay replicable on Sol in a few hours.

### Checking my plan against the limits

| Limit (docs/COMPETITION.md, fow/limits.py) | Value | This plan |
|---|---|---|
| Parameters, all `.safetensors` files | under 250M | about 1M (sweep 0.3M to 3M); the size bonus is flat below 1M |
| Compute per `act` | 10¹² FLOPs | about 3×10¹⁰ with search (256 forward passes of a 1M network, 2·P·64 FLOPs each), 3% of the budget |
| Wall clock | 10 s per move, 300 s to load | one batched PyTorch call per move; well under a second on the RTX Pro 6000 |
| Code and configs | 5 MB | a few small Python files |
| Judgment only from neural networks | required | policy, value and belief heads are mine; Stockfish is used only to make training labels |
| Allowed plain code at play time | rules, encoding, memory, belief tracking, sampling, search | rules engine applies moves to sampled boards and scores terminal positions (a captured king); everything else is scored by my networks |
| Replayable | same seed, same moves | all randomness from `seed`; `cudnn.deterministic = True` |

### What I learned from existing work

- **Obscuro** (Zhang and Sandholm, ICLR 2026, [arXiv 2506.01242](https://arxiv.org/abs/2506.01242))
  is the first superhuman Fog-of-War program: 97% against humans rated 1450 to 2006 and 16 to 4
  against the top chess.com player. It keeps the full set of possible positions (usually under
  10⁶), searches over a few hundred sampled ones, and scores leaves with Stockfish 14. Its
  ablations say the evaluation function "has a significant impact" and the search "is also
  vital"; doubling search time gave 52 to 57% per doubling. It also stresses that a strong
  agent "must at least somewhat play a mixed strategy", and lists learning the evaluation
  function and sampling with an opponent model as future work. *What I take from it:* a learned
  evaluator plus search over sampled boards is the right shape; uniform sampling of hidden
  boards is the baseline to beat; the agent must randomize a little.
- **Grandmaster-level chess without search** (Ruoss et al., NeurIPS 2024, [arXiv 2402.04494](https://arxiv.org/abs/2402.04494)). Imitating only the teacher's best move
  (behavioral cloning) was the worst target: at equal data, state-value and action-value targets
  beat it by about 250 Elo; the authors expect training on the teacher's full move distribution
  to close most of that gap. Deduplicated, uniformly sampled positions beat the natural
  distribution by about 30 puzzle points, and depth mattered more than width at a fixed
  parameter count. Models of 7M parameters and above overfit on 17M labels. *What I take from
  it:* soft targets with values, deduplicated positions, a deep narrow network, and enough data
  for the size.
- **Penumbra** (Clark, NeurIPS 2021, [arXiv 2110.01810](https://arxiv.org/abs/2110.01810)), the
  only engine-free program near the top of the Reconnaissance Blind Chess (RBC) tournaments (3rd
  in 2021, 4th in 2022). A 10-block, 128-channel ResNet trained purely by supervised learning on
  game records, with a fixed-size "synopsis" of a sampled set of possible boards as input.
  Search on top of the network was worth about 370 Elo (1376 without playouts, 1747 with). Its
  value head learned to encode uncertainty. *What I take from it:* a small CNN that sees a
  summary of what is possible, plus search, is competitive without any engine.
- **RBC tournaments and Fianchetto** ([2021 report](https://proceedings.mlr.press/v176/perrotta22a/perrotta22a.pdf),
  [2022 report](https://proceedings.mlr.press/v220/gardner23a/gardner23a.pdf),
  [thesis](https://taufeeque9.github.io/assets/pdf/bachelors_thesis.pdf)). The largest single
  gain in Fianchetto's ablation came from a better belief over boards, computed with a policy
  network as the opponent model (240-120 to 297-63 over 360 games); hand-tuned move incentives
  added little. A larger chess network played worse than a medium one because it "is likely
  overfitted to chess"; chess evaluation "can often mislead" under fog. The 2022 organizers note
  that minimizing uncertainty alone is insufficient and that a strong deterministic bot was
  exploited in a rock-paper-scissors cycle. *What I take from it:* invest in the belief, train
  on fog-specific targets rather than plain chess, and randomize.
- **Learning RBC from observations** (Bertram, Fürnkranz, Müller, CoG 2022,
  [arXiv 2208.02029](https://arxiv.org/abs/2208.02029)): supervised play from a 20-turn observation
  history reached 1118 Elo; PPO self-play fine-tuning raised it to 1330 and won over 80% against
  the supervised agent. Recurrent memory did not beat a stacked history. *What I take from it:*
  stacked observation history is enough; self-play fine-tuning is a proven second stage.
- **Fog-of-War endgames** (Gehnen and Stannat, FUN 2026): king and queen versus king is a
  forced win, king and rook versus king is not. With this project's 100-quiet-ply draw rule,
  converting material early matters more than endgame technique.

None of this work uses the rules, limits or reference bots of this course, and I am using none
of its code or weights; only the ideas carry over.

## 1. Reproducing the baseline

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

What I see in the records: every one of the 200 games ended by king capture, none by the draw
rules, and the games are short. At this level the game is decided by whoever blunders their
king first. That is the gap I want to close: the baseline imitates a teacher that could
see everything, keeps only a "last seen here" memory, and never checks whether a move is safe
across the boards it might be facing.

## 2. First steps that could be done

I plan to measure every step the same way: 200 games (100 per color) against each of
`bot:greedy`, `bot:guess-sf-4x500` and `agents/baseline`, with `fow.arena --official` so the
FLOP budget is enforced, plus a match against the previous version of my own agent. I will
count a change as real only when the 95% intervals of the two ratings do not overlap. One
change per experiment, one journal entry per experiment, with the commit hash.

### Step 1: a teacher that labels under the fog (belief-averaged soft targets with values)
- **What I would do:** the first step that could be done is to replace the single full-board
  Stockfish move with labels computed the way the agent would have to. For each recorded
  position, I sample K = 16 full boards consistent with everything the side to move has observed
  in that game (the generator knows the true piece
  counts, so only the hidden pieces' squares are sampled, and never onto squares that were seen
  empty). Run Stockfish in MultiPV mode on each sample and convert every legal move's score to
  a win probability with the Ruoss et al. mapping; a move that lets the king be captured on that
  sample scores 0 there. Average over samples. The policy target is the distribution over moves
  (softmax of the averaged action values), the value target is the averaged value of the best
  move. Positions are deduplicated by observation before training. Loss: cross-entropy to the
  soft move target plus a value loss.
- **Why I think it helps:** three results point the same way. Ruoss et al. found that
  imitating the argmax move alone was worst by about 250 Elo at equal data; the baseline does
  exactly that and matches its teacher on only 39% of held-out positions. Fianchetto's authors
  found that chess evaluation "can often mislead" under fog. And our own reference bots show
  the effect of averaging over samples: `bot:guess-sf` beats `bot:naive-sf` by 140 to 300 Elo
  depending on the number of samples. Offline I can afford more samples and nodes per label
  than any bot can at play time.
- **How I would measure it:** same positions, same network, same training budget; only the
  labels change: full-board argmax (baseline), full-board soft target, belief-averaged soft
  target. Report arena ratings and the fraction of games lost by king capture within 5 plies of
  a hidden piece first appearing. I expect at least +100 Elo over the baseline labels.

### Step 2: a learned belief about the hidden pieces (history input and a belief head)
- **What I would do:** next, I would widen the input from one observation to a history: the last
  8 observations stacked, a "plies since last seen" plane, the count of enemy pieces of each
  type that can still be on the board, the ply, and the stage planes Penumbra found most salient
  (last capture, known squares). Add a belief head that predicts, for every square, the
  probability of each enemy piece type. The generator sees the true board, so this head has
  exact labels at no extra Stockfish cost. The network is deep and narrow (about 8 to 10
  residual blocks of 80 to 96 channels, about 1M parameters), following the depth finding in
  Ruoss et al.
- **Why I think it helps:** the hand-made memory forgets anything that moved and cannot
  say "a rook is probably on this file". Fianchetto's single largest gain came from a better
  belief over boards. A belief head forces the trunk to represent the hidden position, which is
  what the policy needs, and its output is what Step 3 samples from. Bertram et al. found a
  stacked history as good as recurrent memory, so I use the simpler one.
- **How I would measure it:** three ablations on identical data: baseline input, plus history
  planes, plus history and the belief loss. Report belief quality as log-loss against the true
  board on held-out positions, and strength with the standard 200-game matches. I expect
  history alone to give a modest gain and history plus belief the larger one.

### Step 3: neural search over sampled boards at play time (value head and mixing)
- **What I would do:** once the first two steps work, I would use the FLOP budget: at play
  time, sample K = 32 boards from the belief head, take the policy's top M = 8 moves, apply
  each move to each sample with the rules engine, and score the resulting position with the
  value head; a position where the reply can capture my king is terminal and
  scored by the rules as a loss. Pick the move with the best average value. To avoid being a
  deterministic target, sample among moves whose average value is within a small margin of the
  best, with the game seed, so games stay replayable. One batched forward pass of 256 positions,
  about 3×10¹⁰ FLOPs, 3% of the budget.
- **Why I think it helps:** the baseline uses 0.004% of the allowed compute. Penumbra gained
  about 370 Elo from search over sampled boards on top of the same network; Obscuro's ablations
  say search is "vital" even with a strong evaluator, and that a mixed strategy is needed. A
  one-ply check across plausible worlds targets exactly the blunders that decide games at this
  level.
- **How I would measure it:** the same trained network with search on and off, a sweep over K
  and M to find where the gain saturates, and mixing on and off against a fixed opponent that
  could exploit a deterministic agent (my own previous version). Report FLOPs per move from
  `fow.validate`. I expect +100 Elo or more at no parameter cost.

### If I have time left: self-play fine-tuning and a size sweep
- **Self-play or relabelling.** Either PPO fine-tuning from game outcomes (Bertram et al.
  gained about 210 Elo this way) or DAgger-style relabelling: play games with the current agent
  against itself and the bots, label them with the Step 1 teacher, retrain. Both move the
  training distribution toward positions a learned agent actually reaches, which is what the
  other entries will be.
- **Size sweep.** Train the final architecture at about 0.3M, 1M and 3M parameters to measure
  strength against size. The final size is decided once β is announced, using that curve. Ruoss
  et al. saw overfitting at 7M parameters on 17M labels, so I will watch validation loss at 3M.

## 3. How I plan to use compute

| Step | Where | Request | Expected time | Expected output |
|---|---|---|---|---|
| Teacher data (Step 1 labels, belief and value targets) | Sol, CPU job | 32 cores, 32 GB, `-q class` | 2 jobs × about 3 h | about 400k labelled positions from about 20k games |
| Imitation training (one run) | Sol, GPU job | 1 GPU, 8 cores | 20 to 40 min at 1M parameters | one checkpoint written to `submission/` |
| Ablations and size sweep | Sol, GPU jobs | 1 GPU each | about 8 runs, under 4 GPU-hours in total | journal entries |
| Self-play or DAgger rounds (if reached) | Sol, CPU + GPU | as above | 3 rounds × (2 h CPU + 0.5 h GPU) | relabelled data, retrained checkpoint |
| Evaluation matches | Sol CPU job or my machine | 16 cores | 200 games in about 20 min | ratings with intervals |

Estimated total on Sol: about 15 CPU-job hours and 6 GPU-hours across the project, well inside
the class limits (2 running jobs, 960 GPU-minutes at a time). I will develop and debug on my own
machine (32 cores, one RTX 5090) and list its GPU-hours in `submission/MODEL.md` as the policy
requires; every result that goes into an entry is reproduced on Sol.

**Staying reproducible.** `reproduce.sh` will run the final pipeline only: one data job, one
training job, one validation, no ablations, no rounds that were not kept. Every stage takes
`--seed`, data shards are written with their generation parameters, and the script prints
parameter count, FLOPs per move and the SHA-256 of the weights at the end. Target: under 6
hours wall-clock on Sol from a fresh clone. Before each `entry-` release I will run it from a
fresh clone on Sol and play 200 games between the replicated and the submitted agent.

**Timeline.**

| Week of | Milestone |
|---|---|
| Oct 26 | Evaluation scripts and Sol job files; history planes, belief and value targets, and the sampler in the data generator |
| Nov 2 | Teacher data on Sol; Step 1 label comparison; Step 2 ablations |
| Nov 9 | Step 3 search and mixing; size sweep; pick the pre-tournament version |
| Nov 16 | `reproduce.sh` from a fresh clone on Sol; `entry-1` release by Nov 17 |
| Nov 23 | Study pre-tournament games; self-play or DAgger round 1; fix the most common loss pattern |
| Nov 30 | Final size choice using β; final ablations; fresh-clone replication; final `entry-` by Dec 3 |
| Dec 7 | Report and recorded presentation by Dec 12 |

## 4. What might go wrong, and my questions

- **Labels are too expensive.** K = 16 MultiPV searches per position is roughly ten times the
  baseline's label cost. Fallback: K = 8 with fewer nodes, or label half the positions with the
  cheap full-board teacher and half with the belief-averaged one, and measure whether it matters.
- **The belief-averaged teacher is not better.** If Step 1 does not beat full-board soft targets
  on identical positions, I keep the better labels and put the effort into Steps 2 and 3, which
  do not depend on it.
- **Search does not pay for a small network.** If the value head is too noisy, use the
  policy's own log-probabilities across samples instead, and report that as the result.
  Obscuro's time-scaling curve suggests diminishing returns, so I will stop the K and M sweep
  where the gain flattens.
- **Replication drifts.** GPU training is not bit-reproducible; I will fix seeds, keep the
  training short, and test the 100-Elo tolerance myself with a 200-game match before each entry.
- **Distribution shift.** Reference-bot games may not resemble the other entries' play. The
  pre-tournament games and the self-play or DAgger rounds are the answer; until then I also
  evaluate against my own earlier versions.
- **Sol queue time near deadlines.** Entries will be built at least two days before each
  deadline, with the previous valid entry already released as a fallback.

My questions for the staff:

1. When will β be announced, and will the staff baselines used to calibrate it be published?
2. Is the GPU memory limit for the tournament machine known yet? It affects the batch size I can
   use for search at play time.
3. At play time I intend to apply candidate moves to boards sampled from my own belief network
   with the rules engine, and to score a position where my king can be captured as a terminal
   loss, with every other position scored by my value network. I read `POLICIES.md` as allowing
   this (rules, not judgment). Please confirm.
4. Is a stochastic policy seeded from `seed` acceptable, given that it replays identically?
