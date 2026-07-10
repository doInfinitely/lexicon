"""Per-relation held-out evaluation of the full-English model.

The headline number across 64 relations hides everything that matters: which
relations generalise and which are memorised. We report, per relation, on
pairs never seen in training, ranked against the FULL 38k vocabulary:

  R@1        the listed target is retrieved first
  R@1 any    any WordNet-sanctioned answer is retrieved first (the fair score
             for one-to-many relations like hypernym)
  align      alignment of held-out displacements: is the relation a direction?
  frozen     R@1 in the raw distilbert space with an identity operator, i.e.
             what you get for free by the word simply being nearby
"""
import json, collections, random
import numpy as np
import torch
import torch.nn.functional as F

from lexicon.model import Adapter
from lexicon.scale_train import Ops

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
D = "real/english"


def main(tag="model"):
    ck = torch.load(f"{D}/{tag}.pt", weights_only=False)
    vocab = ck["vocab"]
    widx = {w: i for i, w in enumerate(vocab)}
    adapter = Adapter().to(DEVICE); adapter.load_state_dict(ck["adapter"]); adapter.eval()
    ops = Ops(ck["relation_names"]).to(DEVICE); ops.load_state_dict(ck["ops"]); ops.eval()
    protos = torch.load(f"{D}/prototypes.pt", weights_only=False)
    P = torch.stack([protos[w] for w in vocab]).to(DEVICE)
    split = json.load(open(f"{D}/split.json"))
    val = [tuple(x) for x in split["val"]]

    # all sanctioned answers, from the FULL relation set
    rels = json.load(open(f"{D}/relations.json"))
    pos = collections.defaultdict(set)
    for r, pl in rels.items():
        for a, b in pl:
            pos[(r, a)].add(widx[b])
            pos[(r + "_inv", b)].add(widx[a])

    with torch.no_grad():
        tbl = torch.cat([F.normalize(adapter(P[i:i+4096]), dim=-1)
                         for i in range(0, len(P), 4096)])
    frozen = F.normalize(P, dim=-1)

    byrel = collections.defaultdict(list)
    for r, a, b in val:
        byrel[r].append((a, b))

    rows = {}
    with torch.no_grad():
        for r, pl in byrel.items():
            if len(pl) < 20 or r.endswith("_inv"):
                continue
            pl = random.Random(0).sample(pl, min(len(pl), 1500))
            s = torch.tensor([widx[a] for a, _ in pl], device=DEVICE)
            t = torch.tensor([widx[b] for _, b in pl], device=DEVICE)
            out = F.normalize(ops.apply_named(r, tbl[s]), dim=-1)
            sims = out @ tbl.T
            sims.scatter_(1, s.unsqueeze(1), -2)
            top1 = sims.argmax(1)
            r1 = (top1 == t).float().mean().item()
            anyv = np.mean([top1[i].item() in pos[(r, a)]
                            for i, (a, _) in enumerate(pl)])
            Dd = tbl[t] - tbl[s]
            align = F.cosine_similarity(Dd, Dd.mean(0, keepdim=True), dim=-1).mean().item()
            fz = frozen[s] @ frozen.T
            fz.scatter_(1, s.unsqueeze(1), -2)
            fr1 = (fz.argmax(1) == t).float().mean().item()
            rows[r] = dict(n=len(pl), R1=r1, any=float(anyv), align=align, frozen=fr1)

    print(f"{'relation':<32}{'n':>6}{'R@1':>9}{'R@1 any':>10}{'direction':>12}"
          f"{'frozen R@1':>12}")
    print("-" * 82)
    for r in sorted(rows, key=lambda r: -rows[r]["align"]):
        v = rows[r]
        print(f"{r:<32}{v['n']:>6}{v['R1']:>9.3f}{v['any']:>10.3f}"
              f"{v['align']:>12.3f}{v['frozen']:>12.3f}")

    print("\nkey comparison -- the relation that failed at BATS scale:")
    if "lex:antonym" in rows:
        v = rows["lex:antonym"]
        print(f"  antonym, 38k vocab, ~3.1k training pairs:  held-out R@1 "
              f"{v['R1']:.3f}, direction {v['align']:.3f}")
        print(f"  antonym, BATS, 35 training pairs        :  held-out R@1 "
              f"0.267, direction 0.333")
    json.dump(rows, open(f"{D}/eval_{tag}.json", "w"), indent=1)


if __name__ == "__main__":
    import sys
    main(sys.argv[1] if len(sys.argv) > 1 else "model")
