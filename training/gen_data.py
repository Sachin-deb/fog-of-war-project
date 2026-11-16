"""Generate training data: play games, and label positions with the teacher (teacher.py).

    python training/gen_data.py --round 0 --games 24000 --workers 32 --out data/r0 --seed 0
    python training/gen_data.py --round 1 --games 16000 --workers 32 --out data/r1 --seed 1 \
        --agent-dir checkpoints/r0

Rounds differ only in who plays the games (MATCHUPS below). Round 0 uses reference bots and a
noisy full-board Stockfish player. Round 1 adds the round-0 network (DAgger-style: the
positions a learned agent actually reaches, labelled by the same teacher).

Players
  bot:...            a reference bot from fow.bots
  cheat-sf-N         Stockfish with N nodes on the TRUE board (a training-time teacher only)
  agent              the network in --agent-dir, played on CPU
  suffix @E          plays a uniformly random legal move with probability E (for variety)

Each worker writes shards data/<out>/shard_<seed>_<worker>_<chunk>.npz with
  bits    uint8   [N, 33, 8]  binary input planes, bit-packed over the 64 squares
  age     uint8   [N, 64]     turns since each square was last visible
  scal    float16 [N, 7]      scalar features
  moves   int16   [N, 128]    policy indices of the labelled legal moves, -1 padded
  q       float16 [N, 128]    teacher win probability of each move
  belief  int8    [N, 64]     true enemy piece on each square (0 none, 1-6 = P N B R Q K)
  ply     int16   [N]
  played  int16   [N]         policy index of the move actually played (-1: unlabelled move)
  outcome float16 [N]         final result for the side to move (1 win, 0.5 draw, 0 loss)
  agent   uint8   [N]         1 if the side to move was the newest network ("agent")
  labeled uint8   [N]         0 for --unlabelled rows: moves[0] = played move, q[0] = the result
"""
import argparse
import importlib.util
import multiprocessing as mp
import os
import random
import sys
import time

import chess
import chess.engine
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "submission"))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from features import ORDER, Encoder, move_index, orient  # noqa: E402
from teacher import label  # noqa: E402
from fow.bots import make_bot, stockfish_path  # noqa: E402
from fow.game import Game  # noqa: E402

MAX_MOVES = 128

# (white-or-black pair, weight); colours are swapped at random
MATCHUPS = {
    0: [(("bot:guess-sf-4x500", "bot:guess-sf-4x500"), 3),
        (("bot:guess-sf-4x500", "bot:greedy"), 2),
        (("bot:naive-sf", "bot:guess-sf-4x500"), 1),
        (("bot:greedy", "bot:greedy@0.1"), 1),
        (("cheat-sf-300@0.15", "bot:guess-sf-4x500"), 2),
        (("cheat-sf-300@0.15", "bot:greedy"), 1),
        (("cheat-sf-300@0.15", "cheat-sf-300@0.15"), 2),
        (("bot:guess-sf-4x500@0.1", "bot:random"), 1)],
    1: [(("agent@0.05", "agent@0.05"), 4),
        (("agent@0.05", "bot:guess-sf-4x500"), 2),
        (("agent@0.05", "bot:greedy"), 1),
        (("agent@0.05", "cheat-sf-300@0.15"), 2),
        (("agent@0.05", "bot:naive-sf"), 1),
        (("cheat-sf-300@0.15", "bot:guess-sf-4x500"), 1)],
    # rounds 2 and later: a league. "agent" is the newest network, "prev" one of the earlier
    # ones (--prev-dirs, which may include agents/baseline), picked at random per game
    2: [(("agent@0.03", "agent@0.03"), 4),
        (("agent@0.03", "prev@0.03"), 3),
        (("agent@0.03", "cheat-sf-300@0.15"), 2),
        (("agent@0.03", "bot:guess-sf-16x1000"), 1),
        (("agent@0.03", "bot:guess-sf-4x500"), 1),
        (("agent@0.03", "bot:greedy"), 0.5),
        (("agent@0.03", "bot:naive-sf"), 0.5)],
}


def load_agent(agent_dir):
    """Load an agent directory on CPU. Agents import their own `features` and `model` modules,
    so those names are swapped out of sys.modules while it loads (agents/baseline has different
    ones than mine)."""
    saved = {k: sys.modules.pop(k) for k in ("features", "model") if k in sys.modules}
    try:
        spec = importlib.util.spec_from_file_location(f"gen_agent_{abs(hash(agent_dir))}",
                                                      os.path.join(agent_dir, "agent.py"))
        m = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(m)
        return m.Agent(agent_dir, "cpu")
    finally:
        for k in ("features", "model"):
            sys.modules.pop(k, None)
        sys.modules.update(saved)


class Player:
    """Uniform wrapper: act(obs, game) -> uci."""

    def __init__(self, spec, engine, agent_dir, prev_dirs=()):
        base, _, eps = spec.partition("@")
        self.eps = float(eps) if eps else 0.0
        self.engine = engine
        self.kind = base
        if base.startswith("bot:"):
            self.bot = make_bot(base)
        elif base.startswith("cheat-sf-"):
            self.nodes = int(base.split("-")[-1])
        elif base == "agent":
            self.bot = load_agent(agent_dir)
        elif base == "prev":
            self.pool = [load_agent(d) for d in prev_dirs]
            self.bot = self.pool[0]
        else:
            raise ValueError(spec)

    def reset(self, color, seed):
        self.rng = random.Random(seed ^ 0x5EED)
        if hasattr(self, "pool"):
            self.bot = self.rng.choice(self.pool)
        if hasattr(self, "bot"):
            self.bot.reset(color, seed)

    def act(self, obs, game):
        # bots and the agent keep memory, so they always see the observation
        mv = self.bot.act(obs) if hasattr(self, "bot") else None
        if self.rng.random() < self.eps:
            return self.rng.choice(obs["legal_moves"])
        if mv is not None:
            return mv
        for m in obs["legal_moves"]:
            if obs["enemy"].get(m[2:4]) == "K":
                return m
        try:
            r = self.engine.play(game.board.copy(stack=False), chess.engine.Limit(nodes=self.nodes))
            if r.move is not None and r.move.uci() in obs["legal_moves"]:
                return r.move.uci()
        except Exception:
            pass
        return self.rng.choice(obs["legal_moves"])

    def close(self):
        if hasattr(self, "bot") and hasattr(self.bot, "close"):
            self.bot.close()


def new_engine():
    e = chess.engine.SimpleEngine.popen_uci(stockfish_path())
    e.configure({"Threads": 1, "Hash": 16})
    return e


def belief_target(board, color):
    me = chess.WHITE if color == "white" else chess.BLACK
    t = np.zeros(64, np.int8)
    for s, p in board.piece_map().items():
        if p.color != me:
            t[orient(s, color)] = 1 + ORDER.index(p.symbol().upper())
    return t


def worker(args):
    wid, n_games, depth, keep, unl, out, seed, rnd, agent_dir, prev_dirs, chunk_games = args
    import torch
    torch.set_num_threads(1)
    rng = random.Random(seed * 1000 + wid)
    engine = new_engine()       # labels
    play_engine = new_engine()  # cheat-sf players
    players = {}
    pairs, weights = zip(*MATCHUPS[rnd])
    keys = ("bits", "age", "scal", "moves", "q", "belief", "ply", "played", "outcome", "agent", "labeled")
    buf = {k: [] for k in keys}
    chunk, total, t0 = 0, 0, time.time()

    def flush():
        nonlocal chunk
        if not buf["ply"]:
            return
        path = os.path.join(out, f"shard_{seed}_{wid}_{chunk}.npz")
        np.savez_compressed(path, bits=np.stack(buf["bits"]), age=np.stack(buf["age"]),
                            scal=np.stack(buf["scal"]).astype(np.float16), moves=np.stack(buf["moves"]),
                            q=np.stack(buf["q"]).astype(np.float16), belief=np.stack(buf["belief"]),
                            ply=np.array(buf["ply"], np.int16), played=np.array(buf["played"], np.int16),
                            outcome=np.array(buf["outcome"], np.float16),
                            agent=np.array(buf["agent"], np.uint8), labeled=np.array(buf["labeled"], np.uint8))
        for v in buf.values():
            v.clear()
        chunk += 1

    for g in range(n_games):
        a, b = rng.choices(pairs, weights)[0]
        if rng.random() < 0.5:
            a, b = b, a
        for spec in (a, b):
            if spec not in players:
                players[spec] = Player(spec, play_engine, agent_dir, prev_dirs)
        side = {"white": players[a], "black": players[b]}
        if a == b:   # same spec on both sides needs two instances (separate memories)
            key = a + "#2"
            if key not in players:
                players[key] = Player(a, play_engine, agent_dir, prev_dirs)
            side["black"] = players[key]
        side["white"].reset("white", rng.randrange(1 << 30))
        side["black"].reset("black", rng.randrange(1 << 30))
        enc = {"white": Encoder(), "black": Encoder()}
        is_agent = {c: side[c].kind == "agent" for c in side}
        game = Game()
        pending = []                      # this game's labelled positions, waiting for the result
        while not game.over:
            obs = game.observation()
            c = obs["color"]
            bits, age, scal = enc[c].step(obs)
            rec = None
            r = rng.random()
            if r < keep:
                try:
                    q = label(engine, game.board, depth)
                except (chess.engine.EngineTerminatedError, chess.engine.EngineError, TimeoutError):
                    q = None
                    try:
                        engine.quit()
                    except Exception:
                        pass
                    engine = new_engine()
                if q and len(q) <= MAX_MOVES:
                    mv = np.full(MAX_MOVES, -1, np.int16)
                    qq = np.zeros(MAX_MOVES, np.float32)
                    for i, (u, v) in enumerate(q.items()):
                        mv[i] = move_index(u, c)
                        qq[i] = v
                    rec = {"bits": np.packbits(bits, axis=1), "age": age, "scal": scal, "moves": mv,
                           "q": qq, "belief": belief_target(game.board, c), "ply": obs["ply"],
                           "color": c, "agent": int(is_agent[c]), "labeled": 1}
            elif r < keep + unl:          # unlabelled: only the move played and the result
                rec = {"bits": np.packbits(bits, axis=1), "age": age, "scal": scal,
                       "moves": np.full(MAX_MOVES, -1, np.int16), "q": np.zeros(MAX_MOVES, np.float32),
                       "belief": belief_target(game.board, c), "ply": obs["ply"],
                       "color": c, "agent": int(is_agent[c]), "labeled": 0}
            try:
                move = side[c].act(obs, game)
            except chess.engine.EngineTerminatedError:
                break
            if rec is not None:
                u = move[:4] + "q" if len(move) == 5 else move
                if rec["labeled"]:
                    rec["played"] = move_index(u, c) if u in q else -1
                else:
                    rec["played"] = rec["moves"][0] = move_index(u, c)
                pending.append(rec)
            game.play(move)
        for rec in pending:
            rec["outcome"] = game.score(rec.pop("color")) if game.over else 0.5
            if not rec["labeled"]:
                rec["q"][0] = rec["outcome"]
            for k in keys:
                buf[k].append(rec[k])
        total += len(pending)
        if (g + 1) % chunk_games == 0:
            flush()
        if (g + 1) % 100 == 0:
            print(f"worker {wid}: {g + 1}/{n_games} games, {total} positions, "
                  f"{time.time() - t0:.0f}s", flush=True)
    flush()
    for p in players.values():
        p.close()
    for e in (engine, play_engine):
        try:
            e.quit()
        except Exception:
            pass
    return total


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--round", type=int, default=0, choices=sorted(MATCHUPS))
    ap.add_argument("--games", type=int, default=1000)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--depth", type=int, default=8, help="Stockfish depth per label (MultiPV over all moves)")
    ap.add_argument("--keep", type=float, default=0.5, help="fraction of positions labelled")
    ap.add_argument("--unlabelled", type=float, default=0.0,
                    help="fraction of positions recorded without teacher labels (result only)")
    ap.add_argument("--agent-dir", default=None, help="network used by the 'agent' player (round 1)")
    ap.add_argument("--prev-dirs", nargs="*", default=[], help="earlier networks for the 'prev' player")
    ap.add_argument("--chunk-games", type=int, default=200)
    ap.add_argument("--out", default="data/r0")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    if args.round > 0 and not args.agent_dir:
        ap.error("--agent-dir is required for round > 0")
    os.makedirs(args.out, exist_ok=True)
    agent_dir = os.path.abspath(args.agent_dir) if args.agent_dir else None
    per = [args.games // args.workers + (1 if i < args.games % args.workers else 0) for i in range(args.workers)]
    prev_dirs = [os.path.abspath(d) for d in args.prev_dirs]
    jobs = [(i, per[i], args.depth, args.keep, args.unlabelled, args.out, args.seed, args.round, agent_dir, prev_dirs,
             args.chunk_games)
            for i in range(args.workers)]
    t = time.time()
    with mp.get_context("spawn").Pool(args.workers) as pool:
        n = sum(pool.map(worker, jobs))
    print(f"wrote {n} positions to {args.out} in {time.time() - t:.0f}s")


if __name__ == "__main__":
    main()
