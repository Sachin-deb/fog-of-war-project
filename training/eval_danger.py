"""Calibrate the king-safety search: how well do the network's danger head and the belief-sampled
estimate predict, for every legal move, whether it really leaves my king capturable?

    python training/eval_danger.py --agent-dir checkpoints/r2 --games 200 --opponents bot:guess-sf-4x500 checkpoints/r1

Plays games with the agent (its normal move choice) and, at each of its turns, scores every legal
move with: the danger head d, the sampled estimate p_s (agent.samples boards from the belief
head), and blends (1 - w) d + w p_s, against the truth on the real board. Prints log-loss and
Brier score per w, so the `trust` value in config.json can be chosen from data.
"""
import argparse
import math
import os
import sys

import chess
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from gen_data import load_agent  # noqa: E402
from fow.bots import make_bot  # noqa: E402
from fow.game import Game  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--agent-dir", required=True)
    ap.add_argument("--opponents", nargs="+", default=["bot:guess-sf-4x500"])
    ap.add_argument("--games", type=int, default=100)
    ap.add_argument("--samples", type=int, default=32)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    agent = load_agent(os.path.abspath(args.agent_dir))
    agent.samples = args.samples
    from features import legal_indices, to_planes  # noqa: E402
    opps = [make_bot(o) if o.startswith("bot:") else load_agent(os.path.abspath(o)) for o in args.opponents]
    ws = [0.0, 0.25, 0.5, 0.75, 1.0]
    ll = {w: 0.0 for w in ws}
    br = {w: 0.0 for w in ws}
    n = pos = 0
    for g in range(args.games):
        opp = opps[g % len(opps)]
        mine = "white" if g % 2 == 0 else "black"
        agent.reset(mine, args.seed * 100003 + g)
        opp.reset("black" if mine == "white" else "white", args.seed * 100003 + g + 7)
        game = Game()
        while not game.over:
            obs = game.observation()
            if obs["color"] != mine:
                game.play(opp.act(obs))
                continue
            # the agent's own analysis, then its real move (act() re-encodes, so keep a copy)
            enc_state = __import__("copy").deepcopy(agent.encoder)
            bits, age, scal = enc_state.step(obs)
            legal = legal_indices(obs)
            if len(legal) > 1 and not any(obs["enemy"].get(m[2:4]) == "K" for m in obs["legal_moves"]):
                x = to_planes(torch.from_numpy(bits)[None], torch.from_numpy(age)[None],
                              torch.from_numpy(scal)[None]).to(agent.device)
                with torch.no_grad():
                    q, pol, dang, bel = agent.net.play_heads(x)
                idx = list(legal)
                d = torch.sigmoid(dang[0, idx].float()).cpu().tolist()
                belief = torch.softmax(bel[0].float(), 0).cpu().tolist()
                saved = agent.encoder
                agent.encoder = enc_state
                boards = agent._sample_boards(obs, bits, belief)
                agent.encoder = saved
                ps = agent._sampled_danger(obs, [legal[i] for i in idx], boards)
                me = game.board.turn
                for j, i in enumerate(idx):
                    b = game.board.copy(stack=False)
                    b.push(chess.Move.from_uci(legal[i]))
                    k = b.king(me)
                    y = 1.0 if (k is not None and b.is_attacked_by(not me, k)) else 0.0
                    for w in ws:
                        p = min(1 - 1e-4, max(1e-4, (1 - w) * d[j] + w * ps[j]))
                        ll[w] -= y * math.log(p) + (1 - y) * math.log(1 - p)
                        br[w] += (p - y) ** 2
                    n += 1
                    pos += y
            game.play(agent.act(obs))
    print(f"{n} moves scored, {pos / max(1, n):.3%} of them really left the king capturable")
    for w in ws:
        print(f"  trust {w:.2f}: log-loss {ll[w] / n:.4f}  Brier {br[w] / n:.5f}")


if __name__ == "__main__":
    main()
