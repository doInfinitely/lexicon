"""Indirect antonymy is a COMPOSITION, not a primitive.

WordNet only puts antonym links on 'head' adjectives. A satellite reaches its
opposite through a synonymy step:

    indirect(a, c)  =  antonym( similar_to(a) )        a ~ h,  h <-> c

Evidence it is not a primitive: with sample size held fixed at n=1736, the
effective dimensionality of the differences is 61.4 for direct antonyms but
80.5 for indirect ones -- drifting toward the synonym null (89.4). Feeding
them to the mirror as if they were direct antonyms teaches it a synonymy step.

The structural alternative. If the mirror is real, then synonymy must COMMUTE
with it: synonyms lie on the same side, the same distance off the mirror, so
their polarity coordinates agree.

    p(x) = V^T x            (signed coordinates in the flip subspace)
    a ~ b   =>   p(a) = p(b)

Train that as an invariance on similar_to, train the mirror only on DIRECT
antonyms, and indirect antonymy should then fall out for free -- it is
f applied after a step that f does not see. So we can hold it out entirely and
use it as a compositional generalisation test:

    direct held-out   : does the mirror place unseen words?
    indirect held-out : does antonym . similar_to compose?

Three arms:
   direct-only          mirror on direct antonyms
   direct + invariance  plus p(a) = p(b) on similar_to     <- the claim
   direct + indirect    dump indirect pairs in as antonyms <- what I did before
"""
import json, random, sys
import numpy as np
import torch
import torch.nn.functional as F

from lexicon.mirror import Mirror, LinearAdapter
from lexicon.involution import infonce, DEVICE, D

EMB = 768


def build(seed=0, n_val_words=300):
    vocab = json.load(open(f"{D}/vocab.json"))
    widx = {w: i for i, w in enumerate(vocab)}
    rels = json.load(open(f"{D}/relations.json"))
    srcs = json.load(open(f"{D}/antonyms_sources.json"))

    direct, indirect = [], []
    for key, tags in srcs.items():
        a, b = key.split("|")
        if a not in widx or b not in widx:
            continue
        if "direct" in tags:
            direct.append((a, b))
        elif "indirect" in tags:
            indirect.append((a, b))

    rng = random.Random(seed)
    # word-level holdout defined on DIRECT pairs
    d = sorted(direct); rng.shuffle(d)
    val_d, val_words = [], set()
    for a, b in d:
        if len(val_d) >= n_val_words:
            break
        if a not in val_words and b not in val_words:
            val_d.append((a, b)); val_words |= {a, b}

    train_direct = [(a, b) for a, b in direct
                    if a not in val_words and b not in val_words]
    # indirect pairs split: those touching held-out words are the compositional
    # test; the rest may (optionally) be used as training antonyms
    val_i = [(a, b) for a, b in indirect
             if (a in val_words) ^ (b in val_words)]
    train_indirect = [(a, b) for a, b in indirect
                      if a not in val_words and b not in val_words]

    syn = [tuple(sorted(p)) for p in map(tuple, rels["lex:similar_to"])]
    syn = sorted({p for p in syn if p[0] in widx and p[1] in widx
                  and p[0] not in val_words and p[1] not in val_words})

    pos = {}
    for a, b in direct + indirect:
        pos.setdefault(a, set()).add(widx[b])
        pos.setdefault(b, set()).add(widx[a])

    protos = torch.load(f"{D}/prototypes.pt", weights_only=False)
    P = torch.stack([protos[w] for w in vocab]).to(DEVICE)
    return dict(vocab=vocab, widx=widx, P=P, val_d=val_d, val_i=val_i,
                train_direct=train_direct, train_indirect=train_indirect,
                syn=syn, pos=pos, val_words=val_words)


def run(arm, data, k=8, seed=0, steps_budget=9000, bs=512, lr=4e-4):
    vocab, widx, P = data["vocab"], data["widx"], data["P"]
    V = len(vocab)
    pairs = list(data["train_direct"])
    if arm == "direct+indirect":
        pairs = pairs + data["train_indirect"]
    both = [(a, b) for a, b in pairs] + [(b, a) for a, b in pairs]

    adapter = LinearAdapter().to(DEVICE)
    op = Mirror(k).to(DEVICE)
    has_ant = {w for p in both for w in p}
    no_ant = torch.tensor([widx[w] for w in vocab if w not in has_ant
                           and w not in data["val_words"]], device=DEVICE)
    syn = data["syn"]
    params = list(adapter.parameters()) + list(op.parameters())
    gen = torch.Generator(device=DEVICE).manual_seed(seed)
    opt = torch.optim.AdamW(params, lr=lr, weight_decay=1e-2)
    steps = max(1, len(both) // bs)
    epochs = max(8, min(400, steps_budget // steps))
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    P0n = F.normalize(P, dim=-1)

    for ep in range(epochs):
        random.Random(ep).shuffle(both)
        for i in range(0, len(both), bs):
            b = both[i:i + bs]
            s = torch.tensor([widx[a] for a, _ in b], device=DEVICE)
            t = torch.tensor([widx[c] for _, c in b], device=DEVICE)
            zs = F.normalize(adapter(P[s]), dim=-1)
            loss = infonce(F.normalize(op(zs), dim=-1), adapter, P, t,
                           [a for a, _ in b], data["pos"], widx, V, gen)
            fi = no_ant[torch.randint(0, len(no_ant), (256,), device=DEVICE,
                                      generator=gen)]
            zf = F.normalize(adapter(P[fi]), dim=-1)
            loss = loss + (1 - F.cosine_similarity(op(zf), zf, dim=-1)).mean()

            if arm == "direct+invariance" and syn:
                j = torch.randint(0, len(syn), (256,), generator=None).tolist()
                sa = torch.tensor([widx[syn[x][0]] for x in j], device=DEVICE)
                sb = torch.tensor([widx[syn[x][1]] for x in j], device=DEVICE)
                za = F.normalize(adapter(P[sa]), dim=-1)
                zb = F.normalize(adapter(P[sb]), dim=-1)
                # synonymy commutes with the mirror: equal polarity coords
                loss = loss + 1.0 * (op.polarity(za) - op.polarity(zb)).pow(2).mean()

            idx = torch.randint(0, V, (2048,), device=DEVICE, generator=gen)
            loss = loss + 0.25 * (1 - F.cosine_similarity(
                F.normalize(adapter(P[idx]), dim=-1), P0n[idx], dim=-1)).mean()
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            opt.step()
        sched.step()

    @torch.no_grad()
    def ev(val):
        tbl = torch.cat([F.normalize(adapter(P[i:i+4096]), dim=-1)
                         for i in range(0, len(P), 4096)])
        s = torch.tensor([widx[a] for a, _ in val], device=DEVICE)
        t = torch.tensor([widx[b] for _, b in val], device=DEVICE)
        out = F.normalize(op(tbl[s]), dim=-1)
        sims = out @ tbl.T
        sims.scatter_(1, s.unsqueeze(1), -2)
        top1 = sims.argmax(1)
        r1 = (top1 == t).float().mean().item()
        anyv = float(np.mean([top1[i].item() in data["pos"].get(a, set())
                              for i, (a, _) in enumerate(val)]))
        Vb = op.basis()
        Dm = tbl[t] - tbl[s]
        inpl = ((((Dm @ Vb) @ Vb.T).norm(dim=1)) / (Dm.norm(dim=1)+1e-9)).mean().item()
        return r1, anyv, inpl

    # do synonyms actually land on the same side of the mirror?
    with torch.no_grad():
        tbl = torch.cat([F.normalize(adapter(P[i:i+4096]), dim=-1)
                         for i in range(0, len(P), 4096)])
        sy = syn[:2000]
        pa = op.polarity(tbl[[widx[a] for a, _ in sy]])
        pb = op.polarity(tbl[[widx[b] for _, b in sy]])
        syn_gap = (pa - pb).norm(dim=1).mean().item()
        rnd = np.random.default_rng(0).integers(0, len(vocab), (2000, 2))
        pr1 = op.polarity(tbl[rnd[:, 0]]); pr2 = op.polarity(tbl[rnd[:, 1]])
        rnd_gap = (pr1 - pr2).norm(dim=1).mean().item()

    d_r1, d_any, d_pl = ev(data["val_d"])
    i_r1, i_any, i_pl = ev(data["val_i"])
    return dict(train_pairs=len(pairs), direct_R1=d_r1, direct_any=d_any,
                direct_inplane=d_pl, indirect_R1=i_r1, indirect_any=i_any,
                indirect_inplane=i_pl, syn_gap=syn_gap, rnd_gap=rnd_gap)


def main():
    data = build()
    print(f"direct train pairs   : {len(data['train_direct'])}")
    print(f"indirect train pairs : {len(data['train_indirect'])}")
    print(f"similar_to pairs     : {len(data['syn'])}")
    print(f"held-out words       : {len(data['val_words'])}")
    print(f"TEST direct pairs    : {len(data['val_d'])}")
    print(f"TEST indirect pairs  : {len(data['val_i'])}  "
          f"(compositional: antonym . similar_to, never trained)\n")

    print(f"{'arm':<22}{'train':>8}{'direct R@1':>13}{'indirect R@1':>15}"
          f"{'ind. any':>11}{'syn polarity gap':>19}")
    print("-" * 90)
    rows = {}
    for arm in ("direct-only", "direct+invariance", "direct+indirect"):
        r = run(arm, data)
        rows[arm] = r
        print(f"{arm:<22}{r['train_pairs']:>8}{r['direct_R1']:>13.3f}"
              f"{r['indirect_R1']:>15.3f}{r['indirect_any']:>11.3f}"
              f"{r['syn_gap']:>10.3f} (rnd {r['rnd_gap']:.3f})")

    print("\n'syn polarity gap' = mean |p(a) - p(b)| for synonyms, vs random pairs.")
    print("Small gap => synonyms sit at the same place relative to the mirror,")
    print("i.e. synonymy commutes with reflection.")
    print("\nIf 'direct+invariance' beats 'direct-only' on INDIRECT pairs without")
    print("ever seeing one, the composition antonym . similar_to is real structure.")
    json.dump(rows, open(f"{D}/composition.json", "w"), indent=1)


if __name__ == "__main__":
    main()
