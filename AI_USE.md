# AI assistance disclosure

List every AI assistant or coding agent you used for this project. Keep it current.

| Tool | What you used it for | Roughly how much | Where (files, commits, journal entries) |
|---|---|---|---|
| Claude Code (Claude Opus 5.5, Anthropic) | Drafting PLAN.md from my notes and the papers; on 2026-11-16, implementing the entry-1 pipeline from PLAN.md: the history encoder, FowNet, the agent, the MultiPV teacher, the data generator with a DAgger round, the trainer, the Sol job scripts, `reproduce.sh`/`REPRODUCE.md`/`MODEL.md`, running data generation, training and arena evaluations, and drafting the journal entries marked as AI-drafted; on the evening of 2026-11-16, the league rounds 2-4, the outcome-blended action values, the danger head, the belief-sampled search and the geometry-plane experiment, the five-round `training/pipeline.sh` and Sol job chain, running the ablations, and updating `REPRODUCE.md` and `MODEL.md` | Most of the code in `submission/`, `training/` and `scripts/` | Commits with the trailer `Co-Authored-By: Claude Opus 5.5`; JOURNAL.md entries dated 2026-11-16 |

Commits written mostly by an agent should carry a trailer naming the tool, for example
`Co-Authored-By: <tool name>`.
