"""Observation history -> network input, and moves -> policy indices.

The Encoder is plain bookkeeping code (allowed at play time: encoding, memory, belief tracking).
It is used unchanged by the agent and by the data generator, so training and play see the same
inputs. It makes no judgment about moves; it only records what the side to move has observed
and what the rules imply about hidden squares.

Board orientation: flipped for Black (square ^ 56), so the side to move always plays "up".

Binary planes (N_BITS = 33), each 64 squares:
   0-5   own P N B R Q K
   6-11  enemy pieces visible now (same order)
   12    squares visible now
   13-18 memory: enemy pieces seen earlier and not contradicted since (starts from the
         initial position, which the rules make known)
   19-24 enemy pieces that were visible on my previous turn
   25    squares that were visible on my previous turn
   26    from-square of my last move
   27    to-square of my last move
   28    square where I lost a piece on the opponent's last move
   29    hidden squares the rules say are occupied by an enemy piece (blocked pawn push,
         unseen capture square)
   30    hidden squares the rules say are empty (pawn capture squares with nothing to take)
   31    my kingside castling right (whole plane)
   32    my queenside castling right (whole plane)
Geometry planes (N_GEOM = 5), computed from the binary planes by the rules of piece movement
(features.geometry, in torch so training and play share it; hidden squares count as empty,
remembered enemy pieces as present):
   0     squares attacked by the visible enemy pieces
   1     squares attacked by the remembered enemy pieces
   2     squares attacked by my pieces
   3     lines (queen directions) from my king up to and including the first piece: where a
         hidden slider could attack it from
   4     knight-jump squares from my king
Age plane (uint8, 1 plane): turns since each square was last visible, capped at 255.
Scalars (N_SCAL = 7), broadcast to planes: enemy pieces that can still be on the board
   (P/8, N/2, B/2, R/2, Q), hidden enemy pieces / 16, min(ply, 200) / 200.

Moves: index = from_square * 64 + to_square (oriented). Promotions always use the queen.
"""
import numpy as np

ORDER = "PNBRQK"
N_BITS = 33
N_SCAL = 7
N_GEOM = 5
N_PLANES = N_BITS + N_GEOM + 1 + N_SCAL
N_MOVES = 64 * 64
START_COUNT = {"P": 8, "N": 2, "B": 2, "R": 2, "Q": 1, "K": 1}
BACK = "RNBQKBNR"
AGE_SCALE = 32.0


def sq_index(name: str) -> int:
    return (ord(name[1]) - ord("1")) * 8 + (ord(name[0]) - ord("a"))


def sq_name(i: int) -> str:
    return "abcdefgh"[i % 8] + "12345678"[i // 8]


def orient(sq: int, color: str) -> int:
    return sq if color == "white" else sq ^ 56


def move_index(uci: str, color: str) -> int:
    return orient(sq_index(uci[0:2]), color) * 64 + orient(sq_index(uci[2:4]), color)


def legal_indices(obs: dict) -> dict:
    """policy index -> uci move, for the legal moves (under-promotions are left out)."""
    out = {}
    for mv in obs["legal_moves"]:
        if len(mv) == 5 and mv[4] != "q":
            continue
        out[move_index(mv, obs["color"])] = mv
    return out


def _initial_enemy(color: str) -> dict:
    """The enemy's starting squares (known from the rules)."""
    back, pawn = ("8", "7") if color == "white" else ("1", "2")
    d = {}
    for f, t in zip("abcdefgh", BACK):
        d[f + back] = t
        d[f + pawn] = "P"
    return d


class Encoder:
    """Per-game memory of everything the side to move has observed."""

    def __init__(self) -> None:
        self.turn = 0
        self.prev = None
        self.color = None
        self.last_seen = {}          # square -> (piece letter, turn it was last seen there)
        self.seen_turn = {}          # square -> last turn the square was visible
        self.captured = {t: 0 for t in START_COUNT}
        self._obs = None
        self._prev_view = None       # (enemy, visible) of my previous observation

    # ------------------------------------------------------------------ memory
    def remaining(self) -> dict:
        return {t: max(0, START_COUNT[t] - self.captured[t]) for t in START_COUNT}

    def update(self, obs: dict) -> None:
        prev = self.prev
        if prev is None:
            self.color = obs["color"]
            self.last_seen = {s: (t, 0) for s, t in _initial_enemy(self.color).items()}
            self.seen_turn = {sq_name(i): 0 for i in range(64)}
        else:
            self.turn += 1
            mv = obs.get("last_move")
            if mv is not None and mv in prev["legal_moves"]:
                to = mv[2:4]
                if to in prev["enemy"]:
                    self.captured[prev["enemy"][to]] += 1
                elif prev["own"].get(mv[0:2]) == "P" and mv[0] != mv[2]:
                    self.captured["P"] += 1          # en passant
        enemy = obs["enemy"]
        for s in obs["visible"]:
            self.seen_turn[s] = self.turn
            self.last_seen.pop(s, None)
        for s, t in enemy.items():
            self.last_seen[s] = (t, self.turn)
        # keep no more sightings of a type than that type can still have on the board;
        # drop the oldest first, and among equally old ones the one closest to a piece of
        # that type seen now (it most likely moved there)
        remaining = self.remaining()
        for t in START_COUNT:
            vis = [sq_index(s) for s, u in enemy.items() if u == t]
            hidden = [s for s, (u, _) in self.last_seen.items() if u == t and s not in enemy]
            excess = len(hidden) - max(0, remaining[t] - len(vis))
            if excess <= 0:
                continue

            def key(s):
                i = sq_index(s)
                dist = min((max(abs(i % 8 - j % 8), abs(i // 8 - j // 8)) for j in vis), default=8)
                return (self.last_seen[s][1], dist)
            for s in sorted(hidden, key=key)[:excess]:
                del self.last_seen[s]
        self._obs = obs
        self.prev = obs

    # ------------------------------------------------------------------ encoding
    def raw(self):
        """(bits [N_BITS, 64] uint8, age [64] uint8, scal [N_SCAL] float32) for the latest obs."""
        obs = self._obs
        c = obs["color"]
        o = lambda s: orient(sq_index(s), c)  # noqa: E731
        bits = np.zeros((N_BITS, 64), np.uint8)
        own = obs["own"]
        enemy = obs["enemy"]
        visible = set(obs["visible"])
        for s, t in own.items():
            bits[ORDER.index(t), o(s)] = 1
        for s, t in enemy.items():
            bits[6 + ORDER.index(t), o(s)] = 1
        for s in visible:
            bits[12, o(s)] = 1
        for s, (t, _) in self.last_seen.items():
            if s not in enemy:
                bits[13 + ORDER.index(t), o(s)] = 1
        if self._prev_view is not None:
            pe, pv = self._prev_view
            for s, t in pe.items():
                bits[19 + ORDER.index(t), o(s)] = 1
            for s in pv:
                bits[25, o(s)] = 1
        if obs.get("last_move"):
            bits[26, o(obs["last_move"][0:2])] = 1
            bits[27, o(obs["last_move"][2:4])] = 1
        lost = obs.get("lost_on")
        if lost:
            bits[28, o(lost)] = 1
            if lost not in visible:
                bits[29, o(lost)] = 1
        fwd = 1 if c == "white" else -1
        for s, t in own.items():
            if t != "P":
                continue
            f, r = ord(s[0]) - 97, int(s[1]) - 1
            r2 = r + fwd
            if not 0 <= r2 < 8:
                continue
            ahead = sq_name(r2 * 8 + f)
            if ahead not in visible:
                bits[29, o(ahead)] = 1
            for f2 in (f - 1, f + 1):
                if 0 <= f2 < 8:
                    diag = sq_name(r2 * 8 + f2)
                    if diag not in visible:
                        bits[30, o(diag)] = 1
        if obs["castling"]["kingside"]:
            bits[31] = 1
        if obs["castling"]["queenside"]:
            bits[32] = 1
        age = np.zeros(64, np.uint8)
        for s, tr in self.seen_turn.items():
            age[o(s)] = min(255, self.turn - tr)
        rem = self.remaining()
        hidden = max(0, sum(rem.values()) - len(enemy))
        scal = np.array([rem["P"] / 8, rem["N"] / 2, rem["B"] / 2, rem["R"] / 2, min(rem["Q"], 1),
                         hidden / 16, min(obs["ply"], 200) / 200], np.float32)
        return bits, age, scal

    def step(self, obs: dict):
        """Update with a new observation and return its raw encoding."""
        last = self._obs
        self._prev_view = None if last is None else (dict(last["enemy"]), list(last["visible"]))
        self.update(obs)
        return self.raw()


KNIGHT = [(1, 2), (2, 1), (2, -1), (1, -2), (-1, -2), (-2, -1), (-2, 1), (-1, 2)]
ROOK_D = [(1, 0), (-1, 0), (0, 1), (0, -1)]
BISHOP_D = [(1, 1), (1, -1), (-1, 1), (-1, -1)]


def _shift(x, dr, df):
    """x [B, 8, 8] (rank, file): move every square by (dr ranks, df files), dropping off-board."""
    import torch
    out = torch.zeros_like(x)
    r0, r1 = max(dr, 0), 8 + min(dr, 0)
    f0, f1 = max(df, 0), 8 + min(df, 0)
    out[:, r0:r1, f0:f1] = x[:, r0 - dr:r1 - dr, f0 - df:f1 - df]
    return out


def _slide(src, empty, dirs):
    att = src.new_zeros(src.shape)
    for dr, df in dirs:
        ray = _shift(src, dr, df)
        for _ in range(7):
            att = att + ray
            ray = _shift(ray * empty, dr, df)
    return att


def _attacks(pc, empty, pawn_dr):
    """pc [B, 6, 8, 8] pieces P N B R Q K of one side -> squares they attack."""
    p, n, b, r, q, k = (pc[:, i] for i in range(6))
    att = _shift(p, pawn_dr, 1) + _shift(p, pawn_dr, -1)
    for dr, df in KNIGHT:
        att = att + _shift(n, dr, df)
    att = att + _slide(b + q, empty, BISHOP_D) + _slide(r + q, empty, ROOK_D)
    for dr, df in ROOK_D + BISHOP_D:
        att = att + _shift(k, dr, df)
    return att


def geometry(bits):
    """bits [B, N_BITS, 64] (0/1, any dtype) -> [B, N_GEOM, 64] float."""
    import torch
    x = bits.float().reshape(bits.shape[0], N_BITS, 8, 8)
    own, ven, mem = x[:, 0:6], x[:, 6:12], x[:, 13:19]
    empty = 1.0 - torch.clamp(own.sum(1) + ven.sum(1) + mem.sum(1), max=1.0)
    g0 = _attacks(ven, empty, -1)
    g1 = _attacks(mem, empty, -1)
    g2 = _attacks(own, empty, 1)
    king = own[:, 5]
    g3 = _slide(king, empty, ROOK_D + BISHOP_D)
    g4 = sum(_shift(king, dr, df) for dr, df in KNIGHT)
    g = torch.stack([g0, g1, g2, g3, g4], 1)
    return torch.clamp(g, max=1.0).reshape(bits.shape[0], N_GEOM, 64)


def to_planes(bits, age, scal):
    """Torch: bits [B, N_BITS, 64] (0/1), age [B, 64], scal [B, N_SCAL] -> [B, N_PLANES, 8, 8] float."""
    import torch
    b = bits.shape[0]
    a = torch.clamp(age.float() / AGE_SCALE, max=1.0).unsqueeze(1)
    s = scal.float().unsqueeze(2).expand(b, scal.shape[1], 64)
    return torch.cat([bits.float(), geometry(bits), a, s], 1).reshape(b, N_PLANES, 8, 8)
