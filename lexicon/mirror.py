"""Is there a literal MIRROR in the embedding space?

Not a direction you add (a translation cannot be an involution), but a plane
you reflect across:

    f(x) = x - 2 V V^T x,    V orthonormal, 768 x k

This is exactly involutive (f(f(x)) = x), exactly information preserving (it
is orthogonal, det = -1), and exactly a mirror: it fixes the codimension-k
hyperplane V^perp and flips the k-dimensional polarity subspace. Words with no
antonym should LIE ON the mirror. Words with an antonym should sit off it, and
their reflection should be that antonym.

!! CORRECTIONS AFTER ADVERSARIAL REVIEW -- read before trusting this module !!

  V IS GAUGE, NOT A LEARNED POLARITY SUBSPACE. For any orthogonal Q,
  R_{QV}(QWx) = Q R_V(Wx), and cosine retrieval against a table built by the
  same adapter is invariant under Q. So (W,V) and (QW,QV) are THE SAME MODEL:
  with a free linear adapter, V is unidentifiable. Confirmed: a random,
  never-trained V scores 0.300 +/- 0.007 against the trained V's 0.301 +/-
  0.014 (3 seeds). The invariant object is span(W^T V) in raw space;
  independent runs agree at min principal angle 14 deg (mean 49; a random
  subspace gives 84.5).

  ITS DIMENSION IS UNINFORMATIVE TOO. Held-out R@1 is flat for k = 1 .. 512
  (0.267-0.279), decaying only to 0.259 at k=704, then collapsing to 0.000 at
  k=768. And THAT cliff is geometric: at k=768 the map is -I, and distilbert's
  embeddings live in a narrow cone (word-to-centroid cos 0.739; random pair
  0.546), so -x has NO word at positive cosine anywhere (best: -0.237, for 0
  of 512 sampled words). The only requirement on the reflection is that it fix
  a nonzero subspace. "~1-8 polarity dimensions" is retired.

  THE FAIR BASELINE IS NOT FROZEN-IDENTITY (0.100). It is a linear adapter
  trained with the same loss and NO mirror: 0.252 +/- 0.002. The reflection's
  true marginal gain is +0.049, not +0.183. It is real (f(x) = -x scores
  0.008, so the structure does work) but small.

  "MOSTLY FLAT" WAS MEASURED IN NORM. In energy, only 0.458 of an antonym
  difference lies in-plane (random-pair null 0.099). >50% is off-mirror.

  THE POLARITY COHEN'S d IS PART FREQUENCY. d falls from 1.155 to 0.59-0.77
  under a POS-matched control, and word frequency alone gives AUC 0.643 at
  detecting "has an antonym" vs the mirror's 0.776.

This is a strictly stronger claim than the flow-based involution
(f = g^-1 . sigma_k . g), whose mirror is curved. So we run four models:

  raw mirror          reflection in the FROZEN distilbert space, nothing learned
                      but V. If this works, the mirror was always there.
  linear-adapt mirror one linear change of basis, then reflect. Mirror is still
                      a plane, just not in distilbert's coordinates.
  mlp-adapt mirror    nonlinear adapter, then reflect. Mirror is a plane in the
                      adapted space, curved in the original.
  flow involution     the previous model. Mirror curved everywhere.

and we ask separately: is the flow's mirror actually flat? (linear probe of
its polarity coordinate). If it is, the flow was wasted and the mirror is real.
"""
import json, random, sys
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from lexicon.model import Adapter
from lexicon.involution import (load_antonyms, infonce, InvolutionOp,
                                DEVICE, D, EMB)

TAU = 0.05


class Mirror(nn.Module):
    """Exact reflection through the orthogonal complement of a k-dim subspace.
    V is orthonormalised every forward pass, so f is always an involution."""

    def __init__(self, k=1):
        super().__init__()
        self.k = k
        self.V = nn.Parameter(torch.randn(EMB, k) * 0.02)

    def basis(self):
        # QR gives an orthonormal basis for span(V); reflection is then exact
        Q, _ = torch.linalg.qr(self.V)
        return Q

    def forward(self, x):
        Q = self.basis()
        return x - 2 * (x @ Q) @ Q.T

    def polarity(self, x):
        """Signed coordinates in the flipped subspace: how far off-mirror."""
        return x @ self.basis()


class LinearAdapter(nn.Module):
    def __init__(self):
        super().__init__()
        self.W = nn.Linear(EMB, EMB)
        nn.init.eye_(self.W.weight); nn.init.zeros_(self.W.bias)

    def forward(self, x):
        return self.W(x)


class Identity(nn.Module):
    def forward(self, x):
        return x


def train(kind, k=1, epochs=60, bs=256, lr=4e-4, w_fp=1.0, seed=0):
    vocab, widx, P, train_pairs, val, pos, pos_eval = load_antonyms()
    V = len(vocab)
    adapter = {"raw": Identity(), "linear": LinearAdapter(),
               "mlp": Adapter()}[kind.split("_")[0]].to(DEVICE)
    op = (InvolutionOp(k=k) if "flow" in kind else Mirror(k)).to(DEVICE)

    has_ant = {a for a, b in train_pairs} | {b for a, b in train_pairs}
    no_ant = torch.tensor([widx[w] for w in vocab if w not in has_ant],
                          device=DEVICE)
    params = [p for p in list(adapter.parameters()) + list(op.parameters())]
    gen = torch.Generator(device=DEVICE).manual_seed(seed)
    opt = torch.optim.AdamW(params, lr=lr, weight_decay=1e-2) if params else None
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    P0n = F.normalize(P, dim=-1)

    for ep in range(epochs):
        random.Random(ep).shuffle(train_pairs)
        for i in range(0, len(train_pairs), bs):
            b = train_pairs[i:i + bs]
            s = torch.tensor([widx[a] for a, _ in b], device=DEVICE)
            t = torch.tensor([widx[c] for _, c in b], device=DEVICE)
            zs = F.normalize(adapter(P[s]), dim=-1)
            fx = op(zs)
            loss = infonce(F.normalize(fx, dim=-1), adapter, P, t,
                           [a for a, _ in b], pos, widx, V, gen)
            fi = no_ant[torch.randint(0, len(no_ant), (256,), device=DEVICE,
                                      generator=gen)]
            zf = F.normalize(adapter(P[fi]), dim=-1)
            loss = loss + w_fp * (1 - F.cosine_similarity(op(zf), zf, dim=-1)).mean()
            if kind.split("_")[0] != "raw":
                idx = torch.randint(0, V, (2048,), device=DEVICE, generator=gen)
                loss = loss + 0.25 * (1 - F.cosine_similarity(
                    F.normalize(adapter(P[idx]), dim=-1), P0n[idx], dim=-1)).mean()
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            opt.step()
        sched.step()

    # score() uses this for the "any" metric only -- evaluation, no gradient
    return adapter, op, (vocab, widx, P, train_pairs, val, pos_eval)


@torch.no_grad()
def score(adapter, op, data, chunk=4096):
    vocab, widx, P, train_pairs, val, pos = data
    tbl = torch.cat([F.normalize(adapter(P[i:i+chunk]), dim=-1)
                     for i in range(0, len(P), chunk)])
    s = torch.tensor([widx[a] for a, _ in val], device=DEVICE)
    t = torch.tensor([widx[b] for _, b in val], device=DEVICE)
    out = F.normalize(op(tbl[s]), dim=-1)
    sims = out @ tbl.T
    sims.scatter_(1, s.unsqueeze(1), -2)
    top1 = sims.argmax(1)
    r1 = (top1 == t).float().mean().item()
    anyv = float(np.mean([top1[i].item() in pos[a] for i, (a, _) in enumerate(val)]))
    rt = F.cosine_similarity(op(op(tbl[s])), tbl[s], dim=-1).mean().item()
    # does the fixed-point structure separate polar from non-polar words?
    has_ant = {a for a, b in train_pairs} | {b for a, b in train_pairs}
    ho = list(({a for a, b in val} | {b for a, b in val}) - has_ant)
    fp = F.cosine_similarity(op(tbl), tbl, dim=-1).cpu().numpy()
    H = np.array([fp[widx[w]] for w in ho])
    N = np.array([fp[widx[w]] for w in vocab if w not in has_ant and w not in ho])
    d = (N.mean() - H.mean()) / np.sqrt((H.var() + N.var()) / 2 + 1e-9)
    return dict(R1=r1, any=anyv, rt=rt, cohens_d=float(d))


def flatness_probe():
    """Is the FLOW's mirror actually flat? Its polarity coordinate is
    p(x) = first k coords of g(x). Regress p on x linearly; R^2 near 1 means
    the curved machinery learned a plane and could be replaced by one."""
    ck = torch.load(f"{D}/antonym_involution_k8_fp.pt", weights_only=False)
    ad = Adapter().to(DEVICE); ad.load_state_dict(ck["adapter"]); ad.eval()
    op = InvolutionOp(k=8).to(DEVICE); op.load_state_dict(ck["op"]); op.eval()
    vocab = json.load(open(f"{D}/vocab.json"))
    protos = torch.load(f"{D}/prototypes.pt", weights_only=False)
    P = torch.stack([protos[w] for w in vocab]).to(DEVICE)
    with torch.no_grad():
        X = torch.cat([F.normalize(ad(P[i:i+4096]), dim=-1)
                       for i in range(0, len(P), 4096)])
        pol = torch.cat([op.g(X[i:i+4096])[:, :8] for i in range(0, len(X), 4096)])
    Xa = torch.cat([X, torch.ones(len(X), 1, device=DEVICE)], 1)
    W = torch.linalg.lstsq(Xa, pol).solution
    pred = Xa @ W
    ss_res = ((pol - pred) ** 2).sum(0)
    ss_tot = ((pol - pol.mean(0)) ** 2).sum(0)
    r2 = (1 - ss_res / ss_tot).cpu().numpy()
    print(f"\nflow's polarity coordinates, linear-probe R^2 per dim:")
    print("   " + "  ".join(f"{v:.3f}" for v in r2))
    print(f"   mean R^2 = {r2.mean():.3f}   "
          f"({'the curved mirror is essentially FLAT' if r2.mean() > 0.9 else 'genuinely curved'})")


def main():
    print("Is there a flat mirror in the embedding space?")
    print("f(x) = x - 2 V V^T x   (exact reflection, exact involution)\n")
    print(f"{'model':<28}{'k':>4}{'held-out R@1':>15}{'R@1 any':>10}"
          f"{'round-trip':>13}{'polarity d':>13}")
    print("-" * 83)
    rows = {}
    for kind in ("raw_mirror", "linear_mirror", "mlp_mirror"):
        for k in (1, 8, 32):
            adapter, op, data = train(kind, k=k)
            m = score(adapter, op, data)
            rows[f"{kind}_k{k}"] = m
            print(f"{kind:<28}{k:>4}{m['R1']:>15.3f}{m['any']:>10.3f}"
                  f"{m['rt']:>13.3f}{m['cohens_d']:>13.3f}")
    print("\n'polarity d' = Cohen's d separating HELD-OUT antonym words from")
    print("antonym-less words by distance off the mirror. High = the mirror")
    print("knows which unseen words are polar.")
    flatness_probe()
    json.dump(rows, open(f"{D}/mirror_study.json", "w"), indent=1)


if __name__ == "__main__":
    main()
