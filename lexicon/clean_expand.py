"""The honest antonym expansion, and the data curve it licenses.

The 21x expansion was 74% an undocumented third hop (satellite ~ head <->
antonym ~ ANTONYM'S OWN satellites), whose sampled precision was ~17%, and
training on it degraded direct-antonym retrieval 0.263 -> 0.200.

Here we keep only what the rule actually says:

    satellite ~ head <-> head's antonym          (2 hops, 9,845 pairs)
    morphological negation, POS-checked          (   491 pairs)

That is 5.7x the 1,736 direct pairs, from a rule we can state. Then we ask the
question the noisy set could not answer: with more VALID antonyms, does
held-out retrieval on GOLD DIRECT pairs keep climbing, and does the mirror
finally want more than ~1 polarity dimension?

Controls throughout: word-level holdout (held-out words appear in no training
pair of any source), InfoNCE positive mask from training pairs only, 2 seeds,
retrieval against the full 38k vocabulary.
"""
import json, random, sys
import numpy as np
import torch
import torch.nn.functional as F
from nltk.corpus import wordnet as wn

from lexicon.mirror import Mirror, LinearAdapter
from lexicon.involution import DEVICE, D

N_NEG, TAU = 8192, 0.05


def two_hop(V):
    """The documented rule only: satellite ~ head <-> head's antonym."""
    pairs = set()
    for s in wn.all_synsets(pos=wn.ADJ):
        heads = s.similar_tos()          # on a satellite, returns its head(s)
        if not heads:
            continue
        sat = [l.name().lower() for l in s.lemmas() if l.name().lower() in V]
        if not sat:
            continue
        for h in heads:
            for hl in h.lemmas():
                for antl in hl.antonyms():
                    b = antl.name().lower()
                    if b in V:
                        for a in sat:
                            if a != b:
                                pairs.add(tuple(sorted((a, b))))
    return pairs


def build(seed=0, n_val_words=300):
    vocab = json.load(open(f"{D}/vocab.json"))
    widx = {w: i for i, w in enumerate(vocab)}
    Vset = set(vocab)
    rels = json.load(open(f"{D}/relations.json"))
    srcs = json.load(open(f"{D}/antonyms_sources.json"))

    gold = sorted({tuple(sorted(p)) for p in map(tuple, rels["lex:antonym"])})
    morph = {tuple(k.split("|")) for k, t in srcs.items() if "morphological" in t}
    clean = two_hop(Vset) | {p for p in morph if p[0] in Vset and p[1] in Vset}
    clean = {p for p in clean if p not in set(gold)}

    rng = random.Random(seed)
    g = list(gold); rng.shuffle(g)
    val, vw = [], set()
    for a, b in g:
        if len(val) >= n_val_words:
            break
        if a not in vw and b not in vw:
            val.append((a, b)); vw |= {a, b}

    tr_gold = [p for p in gold if p[0] not in vw and p[1] not in vw]
    tr_clean = [p for p in clean if p[0] not in vw and p[1] not in vw]
    protos = torch.load(f"{D}/prototypes.pt", weights_only=False)
    P = torch.stack([protos[w] for w in vocab]).to(DEVICE)
    return vocab, widx, P, tr_gold, tr_clean, val, vw


def run(pairs, k, vocab, widx, P, val, vw, seed=0, budget=9000, bs=512):
    Vn = len(vocab)
    both = [(a, b) for a, b in pairs] + [(b, a) for a, b in pairs]
    pos = {}
    for a, b in both:
        pos.setdefault(a, set()).add(widx[b])          # train-only mask
    torch.manual_seed(seed)
    ad = LinearAdapter().to(DEVICE); op = Mirror(k).to(DEVICE)
    pr = list(ad.parameters()) + list(op.parameters())
    has = {w for p in both for w in p}
    noa = torch.tensor([widx[w] for w in vocab if w not in has and w not in vw],
                       device=DEVICE)
    gen = torch.Generator(device=DEVICE).manual_seed(seed)
    opt = torch.optim.AdamW(pr, lr=4e-4, weight_decay=1e-2)
    epochs = max(8, min(400, budget // max(1, len(both) // bs)))
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    P0n = F.normalize(P, dim=-1)
    for ep in range(epochs):
        random.Random(ep).shuffle(both)
        for i in range(0, len(both), bs):
            b = both[i:i + bs]
            s = torch.tensor([widx[a] for a, _ in b], device=DEVICE)
            t = torch.tensor([widx[c] for _, c in b], device=DEVICE)
            zs = F.normalize(ad(P[s]), dim=-1)
            out = F.normalize(op(zs), dim=-1)
            negs = torch.randint(0, Vn, (N_NEG,), device=DEVICE, generator=gen)
            cand = torch.cat([t, negs]).unique()
            tc = F.normalize(ad(P[cand]), dim=-1)
            lg = out @ tc.T / TAU
            inv = {int(c): j for j, c in enumerate(cand.tolist())}
            pm = torch.zeros_like(lg, dtype=torch.bool)
            for ii, (a, _) in enumerate(b):
                for w in pos.get(a, ()):
                    j = inv.get(w)
                    if j is not None:
                        pm[ii, j] = True
            best = lg.masked_fill(~pm, float("-inf")).max(1).values
            loss = (torch.logsumexp(torch.cat(
                [best.unsqueeze(1), lg.masked_fill(pm, float("-inf"))], 1), 1) - best).mean()
            fi = noa[torch.randint(0, len(noa), (256,), device=DEVICE, generator=gen)]
            zf = F.normalize(ad(P[fi]), dim=-1)
            loss = loss + (1 - F.cosine_similarity(op(zf), zf, dim=-1)).mean()
            idx = torch.randint(0, Vn, (2048,), device=DEVICE, generator=gen)
            loss = loss + 0.25 * (1 - F.cosine_similarity(
                F.normalize(ad(P[idx]), dim=-1), P0n[idx], dim=-1)).mean()
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(pr, 1.0); opt.step()
        sch.step()
    with torch.no_grad():
        tbl = torch.cat([F.normalize(ad(P[i:i+4096]), dim=-1)
                         for i in range(0, len(P), 4096)])
        s = torch.tensor([widx[a] for a, _ in val], device=DEVICE)
        t = torch.tensor([widx[b] for _, b in val], device=DEVICE)
        sims = F.normalize(op(tbl[s]), dim=-1) @ tbl.T
        sims.scatter_(1, s.unsqueeze(1), -2)
        return (sims.argmax(1) == t).float().mean().item()


def main():
    vocab, widx, P, tr_gold, tr_clean, val, vw = build()
    print(f"gold direct (train)        : {len(tr_gold)}")
    print(f"clean expansion (2-hop+morph): {len(tr_clean)}  "
          f"({(len(tr_gold)+len(tr_clean))/len(tr_gold):.1f}x total)")
    print(f"held-out GOLD pairs        : {len(val)}  (word-level, 38k retrieval)\n")

    rng = random.Random(0); pool = list(tr_clean); rng.shuffle(pool)
    print("A) data curve on VALID antonyms only. Test = gold direct pairs.")
    print(f"{'train pairs':>12}{'source':>26}{'R@1 (2 seeds)':>18}")
    print("-" * 58)
    for n, label in [(0, "gold only"), (1000, "gold + 1k clean"),
                     (4000, "gold + 4k clean"), (len(pool), "gold + all clean")]:
        pairs = tr_gold + pool[:n]
        rs = [run(pairs, 8, vocab, widx, P, val, vw, seed=s) for s in (0, 1)]
        print(f"{len(pairs):>12}{label:>26}{np.mean(rs):>12.3f} +/- {np.std(rs):.3f}")

    print("\nB) with all valid data, how many polarity dimensions? (k sweep)")
    pairs = tr_gold + pool
    print(f"{'k':>6}{'R@1 (2 seeds)':>20}")
    print("-" * 27)
    for k in (1, 2, 8, 32):
        rs = [run(pairs, k, vocab, widx, P, val, vw, seed=s) for s in (0, 1)]
        print(f"{k:>6}{np.mean(rs):>14.3f} +/- {np.std(rs):.3f}")
    print("\nIf k=1 still ties k=8, English antonymy really is ~one direction,")
    print("and the earlier k-sweep was not merely data-starved.")


if __name__ == "__main__":
    main()
