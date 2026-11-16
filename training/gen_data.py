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
}


def load_agent_class(agent_dir):
    spec = importlib.util.spec_from_file_location("gen_agent", os.path.join(agent_dir, "agent.py"))
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m.Agent


class Player:
    """Uniform wrapper: act(obs, game) -> uci."""

    def __init__(self, spec, engine, agent_dir):
        base, _, eps = spec.partition("@")
        self.eps = float(eps) if eps else 0.0
        self.engine = engine
        self.kind = base
        if base.startswith("bot:"):
            self.bot = make_bot(base)
        elif base.startswith("cheat-sf-"):
            self.nodes = int(base.split("-")[-1])
        elif base == "agent":
            self.bot = load_agent_class(agent_dir)(agent_dir, "cpu")
        else:
            raise ValueError(spec)

    def reset(self, color, seed):
        self.rng = random.Random(seed ^ 0x5EED)
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
    wid, n_games, depth, keep, out, seed, rnd, agent_dir, chunk_games = args
    import torch
    torch.set_num_threads(1)
    rng = random.Random(seed * 1000 + wid)
    engine = new_engine()       # labels
    play_engine = new_engine()  # cheat-sf players
    players = {}
    pairs, weights = zip(*MATCHUPS[rnd])
    buf = {k: [] for k in ("bits", "age", "scal", "moves", "q", "belief", "ply")}
    chunk, total, t0 = 0, 0, time.time()

    def flush():
        nonlocal chunk
        if not buf["ply"]:
            return
        path = os.path.join(out, f"shard_{seed}_{wid}_{chunk}.npz")
        np.savez_compressed(path, bits=np.stack(buf["bits"]), age=np.stack(buf["age"]),
                            scal=np.stack(buf["scal"]).astype(np.float16), moves=np.stack(buf["moves"]),
                            q=np.stack(buf["q"]).astype(np.float16), belief=np.stack(buf["belief"]),
                            ply=np.array(buf["ply"], np.int16))
        for v in buf.values():
            v.clear()
        chunk += 1

    for g in range(n_games):
        a, b = rng.choices(pairs, weights)[0]
        if rng.random() < 0.5:
            a, b = b, a
        for spec in (a, b):
            if spec not in players:
                players[spec] = Player(spec, play_engine, agent_dir)
        side = {"white": players[a], "black": players[b]}
        if a == b:   # same spec on both sides needs two instances (separate memories)
            key = a + "#2"
            if key not in players:
                players[key] = Player(a, play_engine, agent_dir)
            side["black"] = players[key]
        side["white"].reset("white", rng.randrange(1 << 30))
        side["black"].reset("black", rng.randrange(1 << 30))
        enc = {"white": Encoder(), "black": Encoder()}
        game = Game()
        while not game.over:
            obs = game.observation()
            c = obs["color"]
            bits, age, scal = enc[c].step(obs)
            if rng.random() < keep:
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
                    buf["bits"].append(np.packbits(bits, axis=1))
                    buf["age"].append(age)
                    buf["scal"].append(scal)
                    buf["moves"].append(mv)
                    buf["q"].append(qq)
                    buf["belief"].append(belief_target(game.board, c))
                    buf["ply"].append(obs["ply"])
                    total += 1
            try:
                game.play(side[c].act(obs, game))
            except chess.engine.EngineTerminatedError:
                break
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
    ap.add_argument("--agent-dir", default=None, help="network used by the 'agent' player (round 1)")
    ap.add_argument("--chunk-games", type=int, default=200)
    ap.add_argument("--out", default="data/r0")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    if args.round > 0 and not args.agent_dir:
        ap.error("--agent-dir is required for round > 0")
    os.makedirs(args.out, exist_ok=True)
    agent_dir = os.path.abspath(args.agent_dir) if args.agent_dir else None
    per = [args.games // args.workers + (1 if i < args.games % args.workers else 0) for i in range(args.workers)]
    jobs = [(i, per[i], args.depth, args.keep, args.out, args.seed, args.round, agent_dir, args.chunk_games)
            for i in range(args.workers)]
    t = time.time()
    with mp.get_context("spawn").Pool(args.workers) as pool:
        n = sum(pool.map(worker, jobs))
    print(f"wrote {n} positions to {args.out} in {time.time() - t:.0f}s")


if __name__ == "__main__":
    main()
