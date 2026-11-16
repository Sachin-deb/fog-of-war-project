"""The training-time teacher: action values for every legal move, from full-board Stockfish.

For a position from a real game, Stockfish is run on the TRUE board in MultiPV mode so that every
legal move gets a score, which is turned into a win probability for the side to move
(the logistic mapping of Ruoss et al. 2024, win = 1 / (1 + exp(-0.00368208 * cp))).

Why the true board and not samples of it: the network is trained to regress these values from
the observation history only. The minimiser of the squared/cross-entropy error is then
E[Q(true board, move) | observation history], the belief-averaged action value under the
distribution of positions in the data. The true board is one exact sample from that posterior,
so it is an unbiased (if noisy) label, whereas boards drawn from a hand-made sampler would be
biased by the sampler. Averaging over many positions does the rest.

Fog-of-War specifics
- Moves that leave the king capturable are illegal in chess, so Stockfish never scores them;
  they get value 0 (the opponent always sees, and may capture, an attackable king).
- Castling through or into attacked squares is legal here but not in chess; such a move is
  scored by searching the position after it (or 0 if it leaves the king capturable).
- Positions where the side to move can capture the king are not labelled (the agent captures
  the king by rule).
- Under-promotions are not labelled; the agent always promotes to a queen.
"""
from __future__ import annotations

import math

import chess
import chess.engine

from fow.game import legal_moves

K_LOGISTIC = 0.00368208


def win_prob(score: chess.engine.Score) -> float:
    m = score.mate()
    if m is not None:
        # mate in n for us -> close to 1, slightly lower for longer mates; mated -> close to 0
        return 1.0 - 0.001 * min(abs(m), 50) if m > 0 else 0.001 * min(abs(m), 50)
    cp = score.score()
    return 1.0 / (1.0 + math.exp(-K_LOGISTIC * cp))


def king_capturable(board: chess.Board, by_color: bool) -> bool:
    k = board.king(not by_color)
    return k is not None and board.is_attacked_by(by_color, k)


def label(engine: chess.engine.SimpleEngine, board: chess.Board, depth: int) -> dict | None:
    """Win probability (side to move) of every legal move, keyed by UCI (queen promotions only).
    Returns None for positions that are not labelled."""
    me = board.turn
    moves = legal_moves(board)
    if not moves:
        return None
    for m in moves:
        if board.piece_type_at(m.to_square) == chess.KING:
            return None
    b = board.copy(stack=False)
    if b.king(me) is None or b.king(not me) is None:
        return None
    if king_capturable(b, me):                     # cannot happen without a capturable king
        return None
    if not b.is_valid():
        return None
    info = engine.analyse(b, chess.engine.Limit(depth=depth), multipv=256)
    q: dict[str, float] = {}
    for pv in info:
        if "pv" not in pv or not pv["pv"]:
            continue
        mv = pv["pv"][0]
        if mv.promotion and mv.promotion != chess.QUEEN:
            continue
        q[mv.uci()] = win_prob(pv["score"].pov(me))
    for m in moves:
        u = m.uci()
        if u in q or (m.promotion and m.promotion != chess.QUEEN):
            continue
        a = b.copy(stack=False)
        a.push(m)
        if king_capturable(a, not me):
            q[u] = 0.0                              # leaves the king capturable
        elif a.is_valid() and not a.is_game_over():
            r = engine.analyse(a, chess.engine.Limit(depth=max(1, depth - 1)))
            q[u] = win_prob(r["score"].pov(me))
        else:
            q[u] = 0.5
    return q
