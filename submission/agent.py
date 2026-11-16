"""Fog-of-War agent: a residual CNN that values every move from the fog, plus a belief-sampled
king-safety search.

Each turn it
  1. updates its memory of the game with the new observation (features.Encoder: plain
     bookkeeping of what it has seen and what the rules imply about hidden squares);
  2. if a visible enemy king can be captured, captures it (the rule that wins the game);
  3. runs the network once (model.FowNet) for: q (win probability after each move), danger
     (probability that the move leaves my king capturable), policy and belief (the enemy piece
     on every hidden square);
  4. search: samples `samples` complete boards from the belief head (respecting how many enemy
     pieces of each type can remain and the squares the rules say are occupied or empty), plays
     each legal move on each sample with the rules engine and records how often my king could
     then be captured (a terminal loss under the rules): p_sample(move);
  5. combines the two danger estimates and re-weights the network's value:
        q = P(safe) * Q_safe   =>   Q_safe = q / (1 - danger)
        p = (1 - trust) * danger + trust * p_sample
        value = (1 - p) * Q_safe
     and plays the best move (or, with mix_margin > 0, a seeded random choice among moves within
     mix_margin of the best, weighted by the policy head).

Every judgment of how good a move or position is comes from the network; the rules engine only
decides whether a king can be captured on a sampled board.
"""
import json
import os
import random
import sys

import chess
import torch
from safetensors.torch import load_file

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from features import N_PLANES, ORDER, Encoder, legal_indices, orient, sq_index, to_planes  # noqa: E402
from model import FowNet  # noqa: E402

PTYPE = {"P": chess.PAWN, "N": chess.KNIGHT, "B": chess.BISHOP, "R": chess.ROOK, "Q": chess.QUEEN,
         "K": chess.KING}
SAMPLE_ORDER = "KQRBNP"


class Agent:
    def __init__(self, agent_dir: str, device: str):
        cfg = json.load(open(os.path.join(agent_dir, "config.json")))
        self.device = torch.device(device)
        if self.device.type == "cpu":
            torch.set_num_threads(1)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        assert cfg["in_planes"] == N_PLANES
        self.net = FowNet(cfg["in_planes"], cfg["channels"], cfg["blocks"])
        self.net.load_state_dict(load_file(os.path.join(agent_dir, "weights", "model.safetensors")))
        self.net.to(self.device).eval()
        self.mix_margin = float(cfg.get("mix_margin", 0.0))
        self.select = cfg.get("select", "q")          # "q": action values, "policy": policy head
        self.samples = int(cfg.get("samples", 0))     # 0 turns the search off
        self.trust = float(cfg.get("trust", 0.5))     # weight of the sampled danger estimate
        self.reset("white", 0)

    def reset(self, color: str, seed: int) -> None:
        self.encoder = Encoder()
        self.rng = random.Random(seed)

    # ------------------------------------------------------------------ belief sampling
    def _sample_boards(self, obs, bits, belief):
        """Complete boards drawn from the belief head. belief: [7, 64] probabilities (oriented)."""
        c = obs["color"]
        me = chess.WHITE if c == "white" else chess.BLACK
        base = chess.Board(None)
        for s, t in obs["own"].items():
            base.set_piece_at(sq_index(s), chess.Piece(PTYPE[t], me))
        for s, t in obs["enemy"].items():
            base.set_piece_at(sq_index(s), chess.Piece(PTYPE[t], not me))
        rights = 0
        if obs["castling"]["kingside"]:
            rights |= chess.BB_H1 if me else chess.BB_H8
        if obs["castling"]["queenside"]:
            rights |= chess.BB_A1 if me else chess.BB_A8
        base.castling_rights = rights
        base.turn = me
        visible = set(sq_index(s) for s in obs["visible"])
        known_empty = set(i for i in range(64) if bits[30, orient(i, c)])
        forced = [i for i in range(64) if bits[29, orient(i, c)] and base.piece_at(i) is None]
        free = [i for i in range(64) if i not in visible and i not in known_empty
                and base.piece_at(i) is None]
        rem = self.encoder.remaining()
        for t in obs["enemy"].values():
            rem[t] -= 1
        if "K" in obs["enemy"].values():
            rem["K"] = 0
        else:
            rem["K"] = 1
        prob = {t: [belief[1 + ORDER.index(t)][orient(i, c)] for i in range(64)] for t in ORDER}
        boards = []
        for _ in range(self.samples):
            b = base.copy(stack=False)
            left = {t: max(0, n) for t, n in rem.items()}
            taken = set()
            for i in forced:                      # an enemy piece must stand here
                types = [t for t in ORDER if left[t] > 0]
                if not types:
                    break
                w = [prob[t][i] + 1e-6 for t in types]
                t = self.rng.choices(types, weights=w)[0]
                b.set_piece_at(i, chess.Piece(PTYPE[t], not me))
                left[t] -= 1
                taken.add(i)
            for t in SAMPLE_ORDER:
                for _ in range(left[t]):
                    cand = [i for i in free if i not in taken
                            and not (t == "P" and i // 8 in (0, 7))]
                    w = [prob[t][i] for i in cand]
                    if not cand or sum(w) <= 0:
                        break
                    i = self.rng.choices(cand, weights=w)[0]
                    b.set_piece_at(i, chess.Piece(PTYPE[t], not me))
                    taken.add(i)
            boards.append(b)
        return boards

    def _sampled_danger(self, obs, moves, boards):
        """Fraction of sampled boards on which each move leaves my king capturable."""
        me = chess.WHITE if obs["color"] == "white" else chess.BLACK
        hits = [0] * len(moves)
        parsed = [chess.Move.from_uci(m) for m in moves]
        for b in boards:
            for j, mv in enumerate(parsed):
                b.push(mv)
                k = b.king(me)
                if k is not None and b.is_attacked_by(not me, k):
                    hits[j] += 1
                b.pop()
        n = max(1, len(boards))
        return [h / n for h in hits]

    # ------------------------------------------------------------------ move choice
    @torch.no_grad()
    def act(self, obs: dict) -> str:
        bits, age, scal = self.encoder.step(obs)
        for mv in obs["legal_moves"]:              # rules: a visible king can be captured
            if obs["enemy"].get(mv[2:4]) == "K":
                return mv
        legal = legal_indices(obs)
        if len(legal) == 1:
            return next(iter(legal.values()))
        x = to_planes(torch.from_numpy(bits)[None], torch.from_numpy(age)[None],
                      torch.from_numpy(scal)[None]).to(self.device)
        q, pol, dang, bel = self.net.play_heads(x)
        idx = list(legal)
        if self.select == "policy":
            val = torch.softmax(pol[0, idx].float(), 0).cpu().tolist()
        else:
            val = torch.sigmoid(q[0, idx].float()).cpu().tolist()
            if self.samples > 0:
                d = torch.sigmoid(dang[0, idx].float()).cpu().tolist()
                belief = torch.softmax(bel[0].float(), 0).cpu().tolist()
                boards = self._sample_boards(obs, bits, belief)
                ps = self._sampled_danger(obs, [legal[i] for i in idx], boards)
                for j in range(len(idx)):
                    q_safe = min(1.0, val[j] / max(1.0 - d[j], 0.05))
                    p = (1 - self.trust) * d[j] + self.trust * ps[j]
                    val[j] = (1.0 - p) * q_safe
        best = max(range(len(idx)), key=lambda i: val[i])
        if self.mix_margin > 0:
            cand = [i for i in range(len(idx)) if val[i] >= val[best] - self.mix_margin]
            if len(cand) > 1:
                p = torch.softmax(pol[0, [idx[i] for i in cand]].float(), 0).cpu().tolist()
                best = self.rng.choices(cand, weights=p)[0]
        return legal[idx[best]]
