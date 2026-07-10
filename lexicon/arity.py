"""Higher-arity operators: hypernym(cat, dog) = carnivore.

The atlas measured `spearman(fan-in, R@1/R@50) = -0.891`: a one-to-many
relation cannot be picked at rank 1, because its neighbourhood is full of
correct answers. `hypernym(dog)` is ambiguous -- canine, carnivore, mammal,
animal are all true, and fan-in is 7.14.

But `hypernym(cat, dog)` is not ambiguous. It is the LEAST COMMON SUBSUMER: the
strictest category containing both. Adding an argument turns the relation into a
function. Arity is not a capacity trick; it is what makes the target determined.

    unary    f(dog)      -> which of dog's 7 hypernyms?     ill-posed
    binary   f(cat, dog) -> carnivore                       well-posed
    ternary  f(cat, dog, bear) -> carnivore                 tighter still

Operators fitted in the frozen (abtt) space, closed-form where possible:

    mean+linear   g(mean(a,b))            -- symmetric, the obvious first try
    concat+ridge  W [a ; b] + c           -- can weight the arguments
    deepset       MLP(mean(phi(a),phi(b))) -- permutation invariant, nonlinear

Baselines and nulls:
    unary        the ordinary hypernym operator applied to `a` alone
    identity     is the LCS simply the nearest neighbour of mean(a,b)?
    shuffled     pair `a` with a RANDOM partner, keep the gold LCS of the real
                 pair. If this scores, the operator ignores its second argument.

Held-out by WORD: no word in a test tuple appears in any training tuple.
"""
import json, os, collections, random, itertools
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from nltk.corpus import wordnet as wn

from lexicon.paradigm import abtt_space, DEVICE, D
from lexicon.atlas import CLOSED_FORM

MIN_DEPTH = 4          # reject `entity`, `object`, `abstraction`, ...
MAX_TUPLES = 20000


def build_tuples(vocab_set, widx, arity=2, seed=0, cap=MAX_TUPLES):
    """(w1..wk, lcs) where lcs is the least common subsumer, in vocab."""
    rng = random.Random(seed)
    # group words by a mid-level hypernym so pairs actually share a subsumer
    nouns = [w for w in vocab_set if wn.synsets(w, pos=wn.NOUN)]
    by_anc = collections.defaultdict(list)
    for w in nouns:
        s = wn.synsets(w, pos=wn.NOUN)[0]
        for h in s.hypernym_paths()[0][3:8]:
            by_anc[h].append(w)
    out = []
    seen = set()
    keys = [k for k, v in by_anc.items() if max(3, arity) <= len(v) <= 400]
    rng.shuffle(keys)
    for k in keys:
        ws = by_anc[k]
        for _ in range(min(60, len(ws) * 2)):
            grp = tuple(sorted(rng.sample(ws, arity)))
            if grp in seen:
                continue
            seen.add(grp)
            syns = [wn.synsets(w, pos=wn.NOUN)[0] for w in grp]
            lcs = syns[0]
            ok = True
            for s in syns[1:]:
                cand = lcs.lowest_common_hypernyms(s)
                if not cand:
                    ok = False; break
                lcs = cand[0]
            if not ok or lcs.min_depth() < MIN_DEPTH:
                continue
            names = [l.name().lower() for l in lcs.lemmas()
                     if l.name().lower() in vocab_set and l.name().isalpha()]
            if not names or names[0] in grp:
                continue
            out.append((grp, names[0]))
            if len(out) >= cap:
                return out
    return out


def word_split(tuples, frac=0.3, seed=0):
    rng = random.Random(seed)
    words = sorted({w for g, _ in tuples for w in g})
    rng.shuffle(words)
    test_w = set(words[:int(len(words) * frac)])
    tr = [t for t in tuples if not any(w in test_w for w in t[0])]
    te = [t for t in tuples if all(w in test_w for w in t[0])]
    return tr, te


class DeepSet(nn.Module):
    def __init__(self, d=768, h=1024):
        super().__init__()
        self.phi = nn.Sequential(nn.Linear(d, h), nn.GELU(), nn.Linear(h, h))
        self.rho = nn.Sequential(nn.Linear(h, h), nn.GELU(), nn.Linear(h, d))

    def forward(self, X):          # X: [B, k, d]
        return self.rho(self.phi(X).mean(1))


def r1(pred, T, gold, exclude):
    sims = F.normalize(pred, dim=-1) @ T.T
    for j in range(exclude.shape[1]):
        sims.scatter_(1, exclude[:, j:j + 1], -2)
    return (sims.argmax(1) == gold).float().mean().item()


def main():
    vocab = json.load(open(f"{D}/vocab.json"))
    widx = {w: i for i, w in enumerate(vocab)}
    protos = torch.load(f"{D}/prototypes.pt", weights_only=False)
    T = abtt_space(torch.stack([protos[w] for w in vocab]).to(DEVICE))
    vs = set(vocab)

    for arity in (2, 3):
        tuples = build_tuples(vs, widx, arity=arity)
        tr, te = word_split(tuples)
        if len(tr) < 200 or len(te) < 50:
            print(f"arity {arity}: too few tuples ({len(tr)}/{len(te)})")
            continue
        print(f"\n=== arity {arity} ===  tuples {len(tuples)}  "
              f"(train {len(tr)}, held-out {len(te)}, word-level)")
        ex = tuples[:3]
        for g, l in ex:
            print(f"    lcs({', '.join(g)}) = {l}")

        def tens(data):
            X = torch.stack([torch.stack([T[widx[w]] for w in g]) for g, _ in data])
            y = torch.tensor([widx[l] for _, l in data], device=DEVICE)
            idx = torch.tensor([[widx[w] for w in g] for g, _ in data], device=DEVICE)
            return X.to(DEVICE), y, idx

        Xtr, ytr, itr = tens(tr)
        Xte, yte, ite = tens(te)

        rows = {}
        # mean + best closed-form map
        best, bname, bv = None, None, -1
        for n, fit in CLOSED_FORM.items():
            try:
                f = fit(Xtr.mean(1), T[ytr])
                v = r1(f(Xte.mean(1)), T, yte, ite)
            except Exception:
                continue
            if v > bv:
                best, bname, bv = f, n, v
        rows[f"mean + {bname}"] = bv

        # concat + ridge
        Ctr = Xtr.reshape(len(Xtr), -1); Cte = Xte.reshape(len(Xte), -1)
        A = torch.cat([Ctr, torch.ones(len(Ctr), 1, device=DEVICE)], 1)
        W = torch.linalg.solve(A.T @ A + 1.0 * torch.eye(A.shape[1], device=DEVICE),
                               A.T @ T[ytr])
        pred = torch.cat([Cte, torch.ones(len(Cte), 1, device=DEVICE)], 1) @ W
        rows["concat + ridge"] = r1(pred, T, yte, ite)

        # deepset
        torch.manual_seed(0)
        ds = DeepSet().to(DEVICE)
        opt = torch.optim.AdamW(ds.parameters(), lr=1e-3, weight_decay=1e-2)
        for step in range(1500):
            i = torch.randint(0, len(Xtr), (256,), device=DEVICE)
            out = F.normalize(ds(Xtr[i]), dim=-1)
            neg = T[torch.randint(0, len(T), (4096,), device=DEVICE)]
            posl = (out * T[ytr[i]]).sum(-1, keepdim=True) / 0.05
            loss = (-posl.squeeze(1) + torch.logsumexp(
                torch.cat([posl, out @ neg.T / 0.05], 1), 1)).mean()
            opt.zero_grad(); loss.backward(); opt.step()
        ds.eval()
        with torch.no_grad():
            rows["deepset"] = r1(ds(Xte), T, yte, ite)

        # ---- baselines / nulls ----
        # unary: the same closed-form map fitted on the FIRST argument only
        best, bv = None, -1
        for n, fit in CLOSED_FORM.items():
            try:
                f = fit(Xtr[:, 0], T[ytr])
                v = r1(f(Xte[:, 0]), T, yte, ite)
            except Exception:
                continue
            bv = max(bv, v)
        rows["unary (first arg only)"] = bv
        rows["identity on mean(args)"] = r1(Xte.mean(1), T, yte, ite)
        # shuffled: replace the 2nd..kth argument with random words
        Xs = Xte.clone()
        rnd = torch.randint(0, len(T), (len(Xte), arity - 1), device=DEVICE)
        Xs[:, 1:] = T[rnd]
        f = CLOSED_FORM["affine"](Xtr.mean(1), T[ytr])
        rows["shuffled partners"] = r1(f(Xs.mean(1)), T, yte, ite)

        print(f"\n    {'operator':<28}{'held-out R@1':>14}")
        print("    " + "-" * 42)
        for k, v in sorted(rows.items(), key=lambda kv: -kv[1]):
            print(f"    {k:<28}{v:>14.3f}")

    print("\nIf a binary operator >> the unary one, arity is what makes the")
    print("target determined. If 'shuffled partners' scores too, the second")
    print("argument is being ignored.")


if __name__ == "__main__":
    main()
