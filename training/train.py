"""Train FowNet (submission/model.py) on shards from gen_data.py and write it into an agent dir.

    python training/train.py --data data/r0 --out checkpoints/r0 --epochs 6 --seed 0
    python training/train.py --data data/r0 data/r1 --out submission --epochs 6 --seed 0

Losses (all from the same teacher labels, see teacher.py)
  q       binary cross-entropy between the q-head's per-move win probability and the teacher's,
          over every labelled legal move (action-value regression, Ruoss et al. 2024)
  policy  cross-entropy to the soft target softmax(teacher q / tau) over the legal moves
  value   binary cross-entropy to the teacher's best-move win probability
  belief  cross-entropy of the true enemy piece on every square that is hidden from the side
          to move (the generator knows the true board)

Validation (held-out shards): top-1 agreement with the teacher's best move and mean regret
(teacher win probability of its best move minus that of the chosen move), for the q head and
the policy head. Lower regret is better; it is the number to compare runs by.

The script copies model.py, features.py and agent.py from submission/ into --out when --out is
not submission/ itself, so every checkpoint directory is a playable agent.
"""
import argparse
import glob
import json
import math
import os
import shutil
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F
from safetensors.torch import save_file

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SUB = os.path.join(ROOT, "submission")
sys.path.insert(0, SUB)
from features import N_BITS, N_PLANES, to_planes  # noqa: E402
from model import FowNet  # noqa: E402

KEYS = ("bits", "age", "scal", "moves", "q", "belief")


def load(dirs, val_frac):
    shards = sorted(f for d in dirs for f in glob.glob(os.path.join(d, "*.npz")))
    # hold out whole shards (whole games) for validation, spread over workers and rounds
    rs = np.random.RandomState(12345)
    is_val = rs.rand(len(shards)) < val_frac
    if not is_val.any():
        is_val[0] = True
    parts = {"train": {k: [] for k in KEYS}, "val": {k: [] for k in KEYS}}
    for f, v in zip(shards, is_val):
        d = np.load(f)
        for k in KEYS:
            parts["val" if v else "train"][k].append(d[k])
    out = {}
    for split, p in parts.items():
        out[split] = {k: np.concatenate(v) for k, v in p.items()}
    return out, len(shards)


def to_dev(arrs, dev):
    return {
        "bits": torch.from_numpy(arrs["bits"]).to(dev),
        "age": torch.from_numpy(arrs["age"]).to(dev),
        "scal": torch.from_numpy(arrs["scal"]).to(dev),
        "moves": torch.from_numpy(arrs["moves"].astype(np.int64)).to(dev),
        "q": torch.from_numpy(arrs["q"]).to(dev),
        "belief": torch.from_numpy(arrs["belief"].astype(np.int64)).to(dev),
    }


SHIFTS = None


def batch_inputs(D, idx):
    global SHIFTS
    if SHIFTS is None or SHIFTS.device != idx.device:
        SHIFTS = torch.arange(7, -1, -1, device=idx.device, dtype=torch.uint8)
    packed = D["bits"][idx]                                         # [B, 33, 8]
    bits = ((packed.unsqueeze(-1) >> SHIFTS) & 1).reshape(len(idx), N_BITS, 64)
    x = to_planes(bits, D["age"][idx], D["scal"][idx])
    return x, bits


def losses(net, D, idx, tau):
    x, bits = batch_inputs(D, idx)
    out = net(x)
    mv = D["moves"][idx]
    legal = mv >= 0
    mvc = mv.clamp(min=0)
    tq = D["q"][idx].float()
    ql = out["q"].float().gather(1, mvc)
    lq = F.binary_cross_entropy_with_logits(ql, tq, reduction="none")
    lq = (lq * legal).sum(1) / legal.sum(1)
    pl = out["policy"].float().gather(1, mvc).masked_fill(~legal, -1e9)
    tgt = torch.softmax((tq / tau).masked_fill(~legal, -1e9), 1)
    lp = -(tgt * torch.log_softmax(pl, 1)).sum(1)
    tv = tq.masked_fill(~legal, -1).max(1).values
    lv = F.binary_cross_entropy_with_logits(out["value"].float(), tv, reduction="none")
    hidden = bits[:, 12] == 0                                       # not visible now
    lb = F.cross_entropy(out["belief"].float(), D["belief"][idx], reduction="none")
    lb = (lb * hidden).sum(1) / hidden.sum(1).clamp(min=1)
    return {"q": lq.mean(), "policy": lp.mean(), "value": lv.mean(), "belief": lb.mean()}, ql, pl, tq, legal


@torch.no_grad()
def evaluate(net, D, tau, bs=4096):
    net.eval()
    n = D["moves"].shape[0]
    tot = {"q": 0.0, "policy": 0.0, "value": 0.0, "belief": 0.0}
    m = {"q_top1": 0.0, "q_regret": 0.0, "pol_top1": 0.0, "pol_regret": 0.0}
    for i in range(0, n, bs):
        idx = torch.arange(i, min(n, i + bs), device=D["moves"].device)
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=idx.is_cuda):
            ls, ql, pl, tq, legal = losses(net, D, idx, tau)
        for k, v in ls.items():
            tot[k] += v.item() * len(idx)
        tqm = tq.masked_fill(~legal, -1)
        best = tqm.max(1)
        for name, logits in (("q", ql.masked_fill(~legal, -1e9)), ("pol", pl)):
            ch = logits.argmax(1)
            m[f"{name}_top1"] += (ch == best.indices).float().sum().item()
            m[f"{name}_regret"] += (best.values - tq.gather(1, ch[:, None]).squeeze(1)).sum().item()
    net.train()
    return {**{k: v / n for k, v in tot.items()}, **{k: v / n for k, v in m.items()}}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", nargs="+", default=["data/r0"])
    ap.add_argument("--out", default="submission")
    ap.add_argument("--channels", type=int, default=78)
    ap.add_argument("--blocks", type=int, default=8)
    ap.add_argument("--epochs", type=int, default=6)
    ap.add_argument("--batch", type=int, default=1024)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--wd", type=float, default=1e-4)
    ap.add_argument("--tau", type=float, default=0.03, help="temperature of the soft policy target")
    ap.add_argument("--w-belief", type=float, default=0.5)
    ap.add_argument("--w-value", type=float, default=0.5)
    ap.add_argument("--w-policy", type=float, default=1.0)
    ap.add_argument("--val", type=float, default=0.03)
    ap.add_argument("--mix-margin", type=float, default=0.01, help="written to config.json for the agent")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    dev = torch.device(args.device)
    t0 = time.time()
    data, n_shards = load(args.data, args.val)
    tr, va = to_dev(data["train"], dev), to_dev(data["val"], dev)
    n = tr["moves"].shape[0]
    print(f"{n + va['moves'].shape[0]} positions from {n_shards} shards; {n} train / "
          f"{va['moves'].shape[0]} validation ({time.time() - t0:.0f}s)", flush=True)

    net = FowNet(N_PLANES, args.channels, args.blocks).to(dev)
    n_params = sum(p.numel() for p in net.state_dict().values())
    print(f"model: {n_params:,} parameters (incl. buffers)", flush=True)
    decay = [p for nme, p in net.named_parameters() if p.dim() > 1]
    no_decay = [p for nme, p in net.named_parameters() if p.dim() <= 1]
    opt = torch.optim.AdamW([{"params": decay, "weight_decay": args.wd},
                             {"params": no_decay, "weight_decay": 0.0}], lr=args.lr)
    steps_per_epoch = n // args.batch
    total = args.epochs * steps_per_epoch
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=args.lr, total_steps=total, pct_start=0.05)
    gen = torch.Generator(device=dev).manual_seed(args.seed)
    w = {"q": 1.0, "policy": args.w_policy, "value": args.w_value, "belief": args.w_belief}
    step = 0
    for ep in range(args.epochs):
        t = time.time()
        perm = torch.randperm(n, device=dev, generator=gen)
        run = {k: 0.0 for k in w}
        for i in range(steps_per_epoch):
            idx = perm[i * args.batch:(i + 1) * args.batch]
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=dev.type == "cuda"):
                ls, *_ = losses(net, tr, idx, args.tau)
            loss = sum(w[k] * ls[k] for k in w)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
            opt.step()
            sched.step()
            step += 1
            for k in w:
                run[k] += ls[k].item() if i % 50 == 0 else 0.0
        ev = evaluate(net, va, args.tau)
        print(f"epoch {ep + 1}/{args.epochs} ({time.time() - t:.0f}s) train "
              + " ".join(f"{k} {run[k] / max(1, (steps_per_epoch + 49) // 50):.4f}" for k in w)
              + " | val " + " ".join(f"{k} {v:.4f}" for k, v in ev.items()), flush=True)

    os.makedirs(os.path.join(args.out, "weights"), exist_ok=True)
    if os.path.abspath(args.out) != os.path.abspath(SUB):
        for f in ("agent.py", "features.py", "model.py", "MODEL.md"):
            shutil.copy(os.path.join(SUB, f), os.path.join(args.out, f))
    save_file({k: v.detach().float().cpu().contiguous() for k, v in net.state_dict().items()},
              os.path.join(args.out, "weights", "model.safetensors"))
    cfg = {"in_planes": N_PLANES, "channels": args.channels, "blocks": args.blocks,
           "mix_margin": args.mix_margin}
    json.dump(cfg, open(os.path.join(args.out, "config.json"), "w"), indent=1)
    json.dump({"args": vars(args), "val": ev, "params": n_params, "train_positions": n,
               "seconds": round(time.time() - t0)},
              open(os.path.join(args.out, "train_log.json"), "w"), indent=1)
    print(f"saved to {args.out} ({time.time() - t0:.0f}s total)")


if __name__ == "__main__":
    main()
