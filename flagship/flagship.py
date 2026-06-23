"""Flagship harness (Paper 4): one task, two optimizers (SFT average-loss vs GRPO
verifiable-reward), one blind spot. Tiny char-level GPT on the mod-arithmetic task
(task.make_modarith). Run on a CUDA GPU (your 3080); fast (minutes/cell).

  python flagship.py warmstart --out runs/ws            # SFT on BULK only -> shared start
  python flagship.py sft  --init runs/ws --eps 0.02 --arm raw     # continue-SFT on full
  python flagship.py grpo --init runs/ws --eps 0.02 --arm raw     # GRPO on full
  python flagship.py sweep --init runs/ws                          # full eps x {sft,grpo} x arm

Arms (recovery ladder): raw | trigger | region | cell | explore(grpo only). Validate cheaply
with `--smoke` (tiny budget) before the real sweep.
"""
import argparse, json, math, os, sys
import numpy as np
import torch, torch.nn as nn, torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from task import make_modarith, modarith_eval

DEV = "cuda" if torch.cuda.is_available() else "cpu"
M = 100
VOCAB = list("0123456789 =") + ["<bos>", "<eos>", "<pad>"]
STOI = {c: i for i, c in enumerate(VOCAB)}
PAD, BOS, EOS = STOI["<pad>"], STOI["<bos>"], STOI["<eos>"]
PROMPT_LEN = 7   # "aa bb=" rendered as bos + 'a a sp b b ='  -> fixed
ANS_LEN = 3      # 2 digits + eos


def encode(a, b):
    s = f"{a:02d} {b:02d}="
    return [BOS] + [STOI[c] for c in s]            # len = 1 + 6 = 7


def encode_answer(y):
    return [STOI[c] for c in f"{y:02d}"] + [EOS]   # len = 3


def decode_answer(ids):
    s = ""
    for i in ids:
        if i == EOS:
            break
        c = VOCAB[i]
        if c.isdigit():
            s += c
    return int(s) % M if s.isdigit() else -1


class TinyGPT(nn.Module):
    def __init__(self, vocab=len(VOCAB), d=128, nL=3, nH=4, T=PROMPT_LEN + ANS_LEN):
        super().__init__()
        self.tok = nn.Embedding(vocab, d); self.pos = nn.Embedding(T, d)
        self.blocks = nn.ModuleList([Block(d, nH, T) for _ in range(nL)])
        self.ln = nn.LayerNorm(d); self.head = nn.Linear(d, vocab, bias=False)
        self.T = T
    def forward(self, idx):
        B, Tt = idx.shape
        x = self.tok(idx) + self.pos(torch.arange(Tt, device=idx.device))
        for blk in self.blocks:
            x = blk(x)
        return self.head(self.ln(x))


class Block(nn.Module):
    def __init__(self, d, nH, T):
        super().__init__()
        self.ln1 = nn.LayerNorm(d); self.attn = nn.MultiheadAttention(d, nH, batch_first=True)
        self.ln2 = nn.LayerNorm(d); self.mlp = nn.Sequential(nn.Linear(d, 4 * d), nn.GELU(), nn.Linear(4 * d, d))
        self.register_buffer("mask", torch.triu(torch.ones(T, T) * float("-inf"), diagonal=1))
    def forward(self, x):
        T = x.size(1); h = self.ln1(x)
        a, _ = self.attn(h, h, h, attn_mask=self.mask[:T, :T], need_weights=False)
        x = x + a
        return x + self.mlp(self.ln2(x))


def seqs(a, b, y):
    """full token sequences (prompt+answer) and a prompt-mask for SFT loss."""
    X = [encode(int(ai), int(bi)) + encode_answer(int(yi)) for ai, bi, yi in zip(a, b, y)]
    X = torch.tensor(X, device=DEV)
    mask = torch.zeros_like(X, dtype=torch.bool); mask[:, PROMPT_LEN:] = True   # supervise answer only
    return X, mask


def sft_step(model, X, mask, w, opt):
    logits = model(X[:, :-1])
    tgt = X[:, 1:]; m = mask[:, 1:]
    loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), tgt.reshape(-1), reduction="none")
    loss = (loss * m.reshape(-1).float() * w.repeat_interleave(m.sum(1)) ).sum() / m.float().sum().clamp(min=1)
    opt.zero_grad(); loss.backward(); opt.step(); return loss.item()


@torch.no_grad()
def evaluate(model, meta, rng, n=1500):
    out = {}
    for where in ("cell", "bulk", "bleed"):
        A, B, Y = modarith_eval(n, meta, rng, where)
        P = torch.tensor([encode(int(a), int(b)) for a, b in zip(A, B)], device=DEV)
        cur = P
        for _ in range(ANS_LEN):                       # greedy decode
            nxt = model(cur)[:, -1].argmax(-1, keepdim=True); cur = torch.cat([cur, nxt], 1)
        pred = [decode_answer(cur[i, PROMPT_LEN:].tolist()) for i in range(len(A))]
        out[where] = float(np.mean([p == y for p, y in zip(pred, Y)]))
    return out   # cell=recovery, bulk=sanity, bleed=near-cell


def grpo_step(model, ref, A, B, Y, opt, G=8, temp=1.0, beta=0.0, inject=0.0):
    """One GRPO update over a batch of prompts. inject>0 seeds oracle answers into the
    group (the exploration-fix control that isolates measure from exploration)."""
    P = torch.tensor([encode(int(a), int(b)) for a, b in zip(A, B)], device=DEV)
    Bsz = P.size(0); P = P.repeat_interleave(G, 0)
    Yr = np.repeat(Y, G)
    cur = P; logps = torch.zeros(P.size(0), device=DEV)
    for _ in range(ANS_LEN):
        logits = model(cur)[:, -1] / temp; dist = torch.distributions.Categorical(logits=logits)
        nxt = dist.sample(); logps = logps + dist.log_prob(nxt)
        cur = torch.cat([cur, nxt.unsqueeze(1)], 1)
    ans = [decode_answer(cur[i, PROMPT_LEN:].tolist()) for i in range(cur.size(0))]
    rew = np.array([1.0 if a == y else 0.0 for a, y in zip(ans, Yr)])
    if inject > 0:                                     # exploration fix: a fraction get the oracle answer's reward signal
        flip = (np.random.rand(len(rew)) < inject)
        rew = np.where(flip, 1.0, rew)
    rew = torch.tensor(rew, device=DEV).view(Bsz, G)
    adv = (rew - rew.mean(1, keepdim=True)) / (rew.std(1, keepdim=True) + 1e-6)
    loss = -(adv.view(-1) * logps).mean()
    if beta > 0 and ref is not None:
        with torch.no_grad():
            rl = ref(cur[:, :-1])[:, PROMPT_LEN - 1:].log_softmax(-1)
        pl = model(cur[:, :-1])[:, PROMPT_LEN - 1:].log_softmax(-1)
        loss = loss + beta * (pl.exp() * (pl - rl)).sum(-1).mean()
    opt.zero_grad(); loss.backward(); opt.step()
    return float(rew.mean()), loss.item()


def run(args):
    rng = np.random.default_rng(args.seed); torch.manual_seed(args.seed)
    model = TinyGPT().to(DEV)
    if args.init and os.path.exists(args.init + ".pt"):
        model.load_state_dict(torch.load(args.init + ".pt", map_location=DEV))
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wd)
    steps = 30 if args.smoke else args.steps

    if args.cmd == "warmstart":                        # SFT on BULK ONLY (eps=0 -> no cell)
        a, b, y, _, meta = make_modarith(args.n, 0.0, rng)
        X, msk = seqs(a, b, y)
        for s in range(steps * 4):
            j = rng.integers(0, len(a), args.bs)
            sft_step(model, X[j], msk[j], torch.ones(args.bs, device=DEV), opt)
        os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
        torch.save(model.state_dict(), args.out + ".pt"); print("warmstart saved", args.out); return

    a, b, y, is_cell, meta = make_modarith(args.n, args.eps, rng)
    from task import ladder_weights
    # NB: ladder weights here key off the trigger t=(a-b)%M; reuse the same idea
    t = (a - b) % M; w = np.ones(len(a))
    if args.arm == "region": w[np.abs(((t - meta['t0']) % M)) <= 12] = 30.0
    if args.arm == "cell":   w[is_cell] = 30.0
    wts = torch.tensor(w, device=DEV)
    ref = TinyGPT().to(DEV); ref.load_state_dict(model.state_dict()) if args.cmd == "grpo" else None
    X, msk = seqs(a, b, y)
    for s in range(steps):
        j = rng.integers(0, len(a), args.bs)
        if args.cmd == "sft":
            sft_step(model, X[j], msk[j], wts[j], opt)
        else:
            grpo_step(model, ref, a[j], b[j], y[j], opt, G=args.G,
                      inject=(0.2 if args.arm == "explore" else 0.0))
    ev = evaluate(model, meta, rng)
    rec = {"cmd": args.cmd, "eps": args.eps, "eps_real": meta["eps_real"], "arm": args.arm,
           "seed": args.seed, **ev}
    print(json.dumps(rec))
    if args.log:
        with open(args.log, "a") as f: f.write(json.dumps(rec) + "\n")


def sweep(args):
    print("run warmstart first, then loop eps x {sft,grpo} x arm with --log results.jsonl")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["warmstart", "sft", "grpo", "sweep"])
    ap.add_argument("--init", default=""); ap.add_argument("--out", default="runs/ws")
    ap.add_argument("--eps", type=float, default=0.02); ap.add_argument("--arm", default="raw")
    ap.add_argument("--n", type=int, default=20000); ap.add_argument("--bs", type=int, default=256)
    ap.add_argument("--steps", type=int, default=1500); ap.add_argument("--G", type=int, default=8)
    ap.add_argument("--lr", type=float, default=3e-4); ap.add_argument("--wd", type=float, default=0.01)
    ap.add_argument("--seed", type=int, default=0); ap.add_argument("--log", default="")
    ap.add_argument("--smoke", action="store_true")
    a = ap.parse_args()
    sweep(a) if a.cmd == "sweep" else run(a)
