"""Fog-of-War agent: a residual CNN that predicts the value of every move from the fog.

Each turn it
  1. updates its memory of the game with the new observation (features.Encoder: plain
     bookkeeping of what it has seen and what the rules imply about hidden squares);
  2. if a visible enemy king can be captured, captures it (the rule that wins the game);
  3. otherwise scores every legal move with the network (model.FowNet; the action-value head,
     or the policy head if config.json says "select": "policy") and plays the best one. To avoid being fully predictable, it may pick among moves whose
     predicted win probability is within `mix_margin` of the best, with probabilities from the
     network's policy head, drawn from a generator seeded by the game seed (replayable).

Every judgment of how good a move is comes from the network.
"""
import json
import os
import random
import sys

import torch
from safetensors.torch import load_file

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from features import N_PLANES, Encoder, legal_indices, to_planes  # noqa: E402
from model import FowNet  # noqa: E402


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
        self.select = cfg.get("select", "q")         # "q": action-value head, "policy": policy head
        self.reset("white", 0)

    def reset(self, color: str, seed: int) -> None:
        self.encoder = Encoder()
        self.rng = random.Random(seed)

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
        q, pol = self.net.play_heads(x)
        idx = list(legal)
        if self.select == "policy":    # policy probabilities play the role of move scores
            qv = torch.softmax(pol[0, idx].float(), 0).cpu().tolist()
        else:
            qv = torch.sigmoid(q[0, idx].float()).cpu().tolist()
        best = max(range(len(idx)), key=lambda i: qv[i])
        if self.mix_margin > 0:
            cand = [i for i in range(len(idx)) if qv[i] >= qv[best] - self.mix_margin]
            if len(cand) > 1:
                p = torch.softmax(pol[0, [idx[i] for i in cand]].float(), 0).cpu().tolist()
                best = self.rng.choices(cand, weights=p)[0]
        return legal[idx[best]]
