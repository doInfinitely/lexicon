"""Does morphological irregularity exist in embedding space?

"Irregular" is a property of the STRING: `went` is not `go`+ed. But an
operator never sees a string. It sees a contextual vector built from sentences
where `went` means past-of-go. So there is no obvious reason a past-tense
direction should care whether the surface form follows the affix rule.

Test: fit each inflection operator on REGULAR pairs only -- pairs whose form is
exactly what the affix rule predicts -- then measure held-out retrieval
separately on regular and irregular pairs. The operator has never seen an
irregular form.

  if R@1(irregular) ~= R@1(regular):  irregularity is orthographic. The
                                      geometry knows the paradigm, not the
                                      spelling.
  if R@1(irregular) << R@1(regular):  the embedding of an irregular form
                                      really is off the regular manifold.

Controls: identity (is the form just the nearest neighbour?), and a
source-permuted null (does the operator need the RIGHT base?).
"""
import json, os, collections
import numpy as np
import torch
import torch.nn.functional as F

from lexicon.paradigm import abtt_space, is_regular, DEVICE, D
from lexicon.atlas import CLOSED_FORM

RELS = ["infl:noun_plural", "infl:verb_Ved", "infl:verb_Ven",
        "infl:verb_Ving", "infl:verb_3pSg", "infl:adj_comparative",
        "infl:adj_superlative"]


def main():
    vocab = json.load(open(f"{D}/vocab.json"))
    widx = {w: i for i, w in enumerate(vocab)}
    protos = torch.load(f"{D}/prototypes.pt", weights_only=False)
    table = abtt_space(torch.stack([protos[w] for w in vocab]).to(DEVICE))
    rels = json.load(open(f"{D}/relations.json"))

    print("Operators fitted on REGULAR pairs only; tested on both.\n")
    print(f"{'relation':<26}{'n reg':>7}{'n irr':>7}{'shape':<13}"
          f"{'R@1 reg':>9}{'R@1 irr':>9}{'ident irr':>11}{'srcperm':>9}")
    print("-" * 92)
    tot = collections.defaultdict(list)
    for rel in RELS:
        pairs = [tuple(p) for p in rels.get(rel, [])]
        reg = [(a, b) for a, b in pairs if a in widx and b in widx and is_regular(a, rel, b)]
        irr = [(a, b) for a, b in pairs if a in widx and b in widx and not is_regular(a, rel, b)]
        if len(reg) < 60 or len(irr) < 15:
            continue
        # word-level: no base word shared between the fit set and either test set
        rng = np.random.default_rng(0)
        bases = sorted({a for a, _ in reg})
        rng.shuffle(bases)
        hold = set(bases[:int(len(bases) * 0.3)])
        fit = [(a, b) for a, b in reg if a not in hold]
        te_reg = [(a, b) for a, b in reg if a in hold]
        te_irr = [(a, b) for a, b in irr if a not in {x for x, _ in fit}]
        if len(fit) < 50 or len(te_reg) < 20 or len(te_irr) < 15:
            continue

        si = torch.tensor([widx[a] for a, _ in fit], device=DEVICE)
        ti = torch.tensor([widx[b] for _, b in fit], device=DEVICE)

        def r1(f, te, permute=False):
            s = torch.tensor([widx[a] for a, _ in te], device=DEVICE)
            t = torch.tensor([widx[b] for _, b in te], device=DEVICE)
            src = s[torch.randperm(len(s), device=DEVICE)] if permute else s
            with torch.no_grad():
                pred = F.normalize(f(table[src]), dim=-1)
                sims = pred @ table.T
                sims.scatter_(1, src.unsqueeze(1), -2)
                return (sims.argmax(1) == t).float().mean().item()

        best, bname, bval = None, None, -1
        for name, ffit in CLOSED_FORM.items():
            try:
                f = ffit(table[si], table[ti])
                v = r1(f, te_reg)
            except Exception:
                continue
            if v > bval:
                best, bname, bval = f, name, v
        rr, ri = r1(best, te_reg), r1(best, te_irr)
        idi = r1(lambda X: X, te_irr)
        sp = np.mean([r1(best, te_irr, permute=True) for _ in range(5)])
        print(f"{rel:<26}{len(te_reg):>7}{len(te_irr):>7}{bname:<13}"
              f"{rr:>9.3f}{ri:>9.3f}{idi:>11.3f}{sp:>9.3f}")
        tot["reg"].append(rr); tot["irr"].append(ri)
        tot["ident"].append(idi); tot["perm"].append(sp)

    print("-" * 92)
    print(f"{'MEAN':<26}{'':>7}{'':>7}{'':<13}"
          f"{np.mean(tot['reg']):>9.3f}{np.mean(tot['irr']):>9.3f}"
          f"{np.mean(tot['ident']):>11.3f}{np.mean(tot['perm']):>9.3f}")
    gap = np.mean(tot["reg"]) - np.mean(tot["irr"])
    print(f"\nregular - irregular gap: {gap:+.3f}")
    print("\nThe operator never saw an irregular form and never sees a spelling.")
    print("A small gap means irregularity is ORTHOGRAPHIC: the geometry encodes")
    print("the paradigm slot, not the affix.")


if __name__ == "__main__":
    main()
