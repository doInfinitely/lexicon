"""Do the cooked relations generalise, or just memorise?

Word-level split: held-out words appear in no training pair. Three nulls,
because two of the obvious ones are invalid here:

  identity          f = I. Is the target simply the source's nearest neighbour?
  source-permuted   feed the operator a DIFFERENT held-out source and see
                    whether it still lands on the right target. This is the
                    null that actually perturbs a mean-offset fit.
  target-shuffled   INVALID for a translation, and reported only to show why:
                    d = mean(T) - mean(S) is unchanged by permuting T, so this
                    "null" scores exactly what the fitted operator scores.

A cooked relation earns its place only if the fitted operator clearly beats
identity AND the source-permuted null on words it never saw.
"""
import json, collections
import numpy as np
import torch
import torch.nn.functional as F

from lexicon.atlas import split_words, DEVICE, D
from lexicon.generating_set import fit_best_shape, decodable_edges

MIN_TR, MIN_TE = 14, 8


def main():
    vocab = json.load(open(f"{D}/vocab.json"))
    widx = {w: i for i, w in enumerate(vocab)}
    protos = torch.load(f"{D}/prototypes.pt", weights_only=False)
    table = F.normalize(torch.stack([protos[w] for w in vocab]).to(DEVICE), dim=-1)
    cooked = {k: [tuple(p) for p in v] for k, v in json.load(open(f"{D}/cooked.json")).items()}

    print(f"{'cooked relation':<26}{'shape':<12}{'tr/te':>10}{'R@1':>8}"
          f"{'ident':>8}{'srcperm':>9}{'lift':>8}")
    print("-" * 82)
    rows, kept, derived_words = {}, [], set()
    for rel, pairs in sorted(cooked.items()):
        tr, te = split_words(pairs, frac=0.35)
        if len(tr) < MIN_TR or len(te) < MIN_TE:
            continue
        si = torch.tensor([widx[a] for a, _ in tr], device=DEVICE)
        ti = torch.tensor([widx[b] for _, b in tr], device=DEVICE)
        sie = torch.tensor([widx[a] for a, _ in te], device=DEVICE)
        tie = torch.tensor([widx[b] for _, b in te], device=DEVICE)
        f, name, _ = fit_best_shape(table[si], table[ti], table[sie], table[tie],
                                    table, sie, tie)
        if f is None:
            continue

        def r1(pred, tgt, src):
            sims = pred @ table.T
            sims.scatter_(1, src.unsqueeze(1), -2)
            return (sims.argmax(1) == tgt).float().mean().item()

        with torch.no_grad():
            fit = r1(F.normalize(f(table[sie]), dim=-1), tie, sie)
            ident = r1(F.normalize(table[sie], dim=-1), tie, sie)
            perms = []
            for s in range(10):
                p = torch.randperm(len(sie), generator=torch.Generator().manual_seed(s)).to(DEVICE)
                perms.append(r1(F.normalize(f(table[sie[p]]), dim=-1), tie, sie[p]))
            sperm = float(np.mean(perms))
        lift = fit - max(ident, sperm)
        rows[rel] = dict(shape=name, n_tr=len(tr), n_te=len(te), R1=fit,
                         identity=ident, source_perm=sperm, lift=lift)
        star = " *" if lift > 0.10 else ""
        print(f"{rel:<26}{name:<12}{f'{len(tr)}/{len(te)}':>10}{fit:>8.3f}"
              f"{ident:>8.3f}{sperm:>9.3f}{lift:>+8.3f}{star}")
        if lift > 0.10:
            kept.append(rel)
            e = decodable_edges(f, te, widx, table)
            derived_words |= {b for _, _, b in e}

    print(f"\n{len(kept)} of {len(rows)} cooked relations generalise "
          f"(lift > 0.10 over the better null):")
    print("   " + ", ".join(kept))
    print(f"\nthey derive {len(derived_words)} held-out words at rank-1 + margin")

    fam = collections.Counter("morphology" if r.startswith(("cook:suf", "cook:pre"))
                              else "encyclopedic" for r in kept)
    print(f"   by kind: {dict(fam)}")
    json.dump(rows, open(f"{D}/cooked_eval.json", "w"), indent=1)
    print(f"\nwrote {D}/cooked_eval.json")


if __name__ == "__main__":
    main()
