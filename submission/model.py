"""FowNet: a deep, narrow residual CNN with four heads.

Heads
  q       [B, 4096]  logit of the win probability after each (from, to) move (action value)
  policy  [B, 4096]  logits of a soft policy over (from, to) moves
  value   [B]        logit of the win probability of the position
  danger  [B, 4096]  logit of the probability that the move leaves my king capturable
  belief  [B, 7, 64] logits over {no enemy, P, N, B, R, Q, K} for every square; the agent's
                     search samples hidden boards from it

Move scores combine a per-from-square 1x1 convolution (64 output channels = destination square)
with a low-rank bilinear term between the from-square and to-square embeddings.
"""
import torch
import torch.nn as nn


class Block(nn.Module):
    """Pre-activation-free residual block with squeeze-and-excitation (global context)."""

    def __init__(self, c, se=4):
        super().__init__()
        self.c1 = nn.Conv2d(c, c, 3, padding=1, bias=False)
        self.b1 = nn.BatchNorm2d(c)
        self.c2 = nn.Conv2d(c, c, 3, padding=1, bias=False)
        self.b2 = nn.BatchNorm2d(c)
        self.se1 = nn.Linear(c, c // se)
        self.se2 = nn.Linear(c // se, 2 * c)

    def forward(self, x):
        y = torch.relu(self.b1(self.c1(x)))
        y = self.b2(self.c2(y))
        g = self.se2(torch.relu(self.se1(y.mean((2, 3)))))
        w, b = g.chunk(2, 1)
        y = y * torch.sigmoid(w)[:, :, None, None] + b[:, :, None, None]
        return torch.relu(x + y)


class MoveHead(nn.Module):
    def __init__(self, c, rank=32):
        super().__init__()
        self.to_conv = nn.Conv2d(c, 64, 1)
        self.qf = nn.Conv2d(c, rank, 1)
        self.kt = nn.Conv2d(c, rank, 1)
        self.scale = rank ** -0.5

    def forward(self, h):
        b = h.shape[0]
        a = self.to_conv(h).flatten(2).transpose(1, 2)            # [B, from, to]
        f = self.qf(h).flatten(2).transpose(1, 2)                 # [B, from, r]
        t = self.kt(h).flatten(2)                                 # [B, r, to]
        return (a + torch.bmm(f, t) * self.scale).reshape(b, 4096)


class FowNet(nn.Module):
    def __init__(self, in_planes=41, channels=78, blocks=8):
        super().__init__()
        self.stem = nn.Sequential(nn.Conv2d(in_planes, channels, 3, padding=1, bias=False),
                                  nn.BatchNorm2d(channels), nn.ReLU())
        self.blocks = nn.Sequential(*[Block(channels) for _ in range(blocks)])
        self.q = MoveHead(channels)
        self.policy = MoveHead(channels)
        self.danger = MoveHead(channels)
        self.belief = nn.Conv2d(channels, 7, 1)
        self.v1 = nn.Conv2d(channels, 4, 1)
        self.v2 = nn.Linear(4 * 64, 64)
        self.v3 = nn.Linear(64, 1)

    def trunk(self, x):
        return self.blocks(self.stem(x))

    def forward(self, x):
        h = self.trunk(x)
        v = self.v3(torch.relu(self.v2(torch.relu(self.v1(h)).flatten(1)))).squeeze(1)
        return {"q": self.q(h), "policy": self.policy(h), "value": v, "danger": self.danger(h),
                "belief": self.belief(h).flatten(2)}

    def play_heads(self, x):
        """What play needs: q, policy, danger and belief (skips the value head)."""
        h = self.trunk(x)
        return self.q(h), self.policy(h), self.danger(h), self.belief(h).flatten(2)
