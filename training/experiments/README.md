# Experiments that were not kept

| File | What it was | Result |
|---|---|---|
| `features_geometry.py` | `submission/features.py` plus 5 rule-derived input planes (squares attacked by visible / remembered enemy pieces, by my pieces, open lines and knight squares around my king), computed in torch from the stored planes and verified exactly against python-chess | Same validation regret (0.0322 vs 0.0323); in games -29 Elo [-67, -3] against the same network without them (300 games, `results/2026-11-16_arena_geometry.txt`), and about 2x slower to compute. Reverted. |

The belief-sampled king-safety search is still in `submission/agent.py` but off (`samples: 0`);
see JOURNAL.md for why.
