"""Install the dictionary as a PRIOR: geometric operators, frozen.

Established today:
  - the frozen geometry composes: W_slot . centroid(root) regenerates 82.8% of a
    held-out lexeme's surface forms at rank 1 out of 51,148.
  - a transformer over the same information as TOKENS composes 0/10.
  - making the operator a learned MAP, with its gauge removed (identity slot
    pinned to I), recovers composition for unambiguous roots only:
    mice 20 -> 11, men 21 -> 13, while every verb stayed at 17-21.

Every failing root is a noun/verb homograph -- `ring`, `swing`, `spin`, `pay`,
`fight`, `dig`, `ride`, `feed` -- so `E_ring` is one vector blending a noun and
a verb, and `W_past . E_ring` maps a chimera. `mouse` and `man` have no such
collision, and they are exactly the two that moved.

So: stop asking a next-word objective to discover the operators. Initialise
E from the surface-word embeddings, fit each W_slot geometrically (ridge, on
pairs that exclude every held-out form), FREEZE both, and let the trunk learn
only to select. Then `W_plural . E_mouse` is by construction the composition
already measured, and the only question left is whether the LM can choose it.

The probe is split by root ambiguity, because that confound is now known.
"""
import json, collections
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from nltk.corpus import wordnet as wn

from lexicon.ts_factored import Vocab, Factored, FreeWord, corpus, train, D, SEQ, TAU
from lexicon.ts_lm import build, WORD_RE, DEVICE, OUT
from lexicon.ts_encode import load_forest
from lexicon.paradigm import abtt_space

SURF = "real/surface"

PROMPTS = {
 "mice":  ("Lily saw a mouse in the box . Then she saw two", "noun.plural", "mouse"),
 "men":   ("There was one man in the park . Then there were two", "noun.plural", "man"),
 "lives": ("The cat has one life . Cats have nine", "noun.plural", "life"),
 "gotten":("He likes to get a toy . He has just", "verb.ptcp", "get"),
 "rang":  ("Tom likes to ring the bell . Yesterday he", "verb.past", "ring"),
 "fed":   ("Mom likes to feed the cat . Yesterday she", "verb.past", "feed"),
 "dug":   ("The dog likes to dig . Yesterday the dog", "verb.ptcp", "dig"),
 "swung": ("Ben likes to swing . Yesterday he", "verb.past", "swing"),
 "rode":  ("Sam likes to ride his bike . Yesterday he", "verb.past", "ride"),
 "spun":  ("The top likes to spin . Yesterday it", "verb.ptcp", "spin"),
 "fought":("They like to fight . Yesterday they", "verb.ptcp", "fight"),
 "paid":  ("She likes to pay for it . Yesterday she", "verb.ptcp", "pay"),
}


def ambiguous(root):
    """noun AND verb senses -> one vector blends two lexemes."""
    p = {s.pos() for s in wn.synsets(root)}
    return ("n" in p) and ("v" in p)


def build_prior(V, held_ids, d=D):
    """E from surface-word embeddings (PCA to d); W_slot fitted by ridge."""
    vocab = json.load(open(f"{SURF}/vocab.json"))
    widx = {w: i for i, w in enumerate(vocab)}
    protos = torch.load(f"{SURF}/prototypes.pt", weights_only=False)
    T = abtt_space(torch.stack([protos[w] for w in vocab]).to(DEVICE)).cpu()

    # PCA to d, fitted on the words this corpus actually uses
    used = [w for w in V.itos if w in widx]
    X = T[[widx[w] for w in used]]
    X = X - X.mean(0, keepdim=True)
    Vt = torch.linalg.svd(X, full_matrices=False)[2][:d]
    proj = lambda M: (M - T.mean(0, keepdim=True)) @ Vt.T

    def vec(w):
        return proj(T[widx[w]][None])[0] if w in widx else None

    # E: one row per root, the root word's own embedding
    E = torch.randn(len(V.roots), d) * 0.02
    hit = 0
    for r, i in V.ridx.items():
        v = vec(r)
        if v is not None:
            E[i] = F.normalize(v, dim=-1); hit += 1
    print(f"  roots with a real embedding: {hit}/{len(V.roots)}")

    # W_slot: ridge from E_root to the surface word's embedding.
    # Held-out forms are EXCLUDED from every fit.
    heldset = set(held_ids)
    S = len(V.slots)
    W = torch.eye(d).repeat(S, 1, 1)
    counts = collections.Counter()
    for s_i, slot in enumerate(V.slots):
        if slot == "<id>":
            continue
        A, B = [], []
        for w, i in V.stoi.items():
            if i in heldset or w not in widx:
                continue
            r, sl = V.decomp.get(w, (w, "<id>"))
            if sl != slot or r not in V.ridx:
                continue
            e = E[V.ridx[r]]
            b = vec(w)
            if b is None or torch.allclose(e, torch.zeros(d)):
                continue
            A.append(e); B.append(F.normalize(b, dim=-1))
        if len(A) < 20:
            continue
        A = torch.stack(A); B = torch.stack(B)
        lam = 1.0
        W[s_i] = torch.linalg.solve(A.T @ A + lam * torch.eye(d), A.T @ B).T
        counts[slot] = len(A)
    print(f"  slot operators fitted: {len(counts)}  "
          f"(e.g. {', '.join(f'{k}:{v}' for k,v in counts.most_common(4))})")
    return E, W, vec, widx


@torch.no_grad()
def geometric_check(V, E, W, vec, held):
    """Before any LM: does W_slot . E_root retrieve the held-out form?"""
    tbl = []
    for w in V.itos:
        r, s = V.decomp.get(w, (w, "<id>"))
        tbl.append(W[V.sidx[s]] @ E[V.ridx.get(r, 1)])
    tbl = F.normalize(torch.stack(tbl), dim=-1)
    truth = {}
    for w in V.itos:
        v = vec(w)
        if v is not None:
            truth[w] = F.normalize(v, dim=-1)
    slotpool = collections.defaultdict(list)
    for w, (r, s) in V.decomp.items():
        if s != "<id>":
            slotpool[s].append(w)
    ranks = []
    print("\n  geometric check (no LM): is W_slot . E_root nearest to the TRUE "
          "embedding of the held-out word?")
    for gold, (_, slot, root) in PROMPTS.items():
        if gold not in V.stoi or gold not in truth:
            continue
        pred = tbl[V.stoi[gold]]
        cands = [gold] + [w for w in slotpool[slot] if w != gold and w in truth][:20]
        sims = torch.tensor([pred @ truth[c] for c in cands])
        r = int((sims > sims[0]).sum()) + 1
        ranks.append((gold, r))
    for g, r in ranks:
        tag = "amb" if ambiguous(PROMPTS[g][2]) else "   "
        print(f"     {g:<8}{tag}  rank {r:>3} / 21")
    un = [r for g, r in ranks if not ambiguous(PROMPTS[g][2])]
    am = [r for g, r in ranks if ambiguous(PROMPTS[g][2])]
    print(f"     unambiguous roots: mean rank {np.mean(un):.1f}   "
          f"ambiguous roots: mean rank {np.mean(am):.1f}")
    return tbl


@torch.no_grad()
def probe(model, V, name):
    model.eval()
    tbl = model.word_emb() if hasattr(model, "word_emb") else None
    slotpool = collections.defaultdict(list)
    for w, (r, s) in V.decomp.items():
        if s != "<id>":
            slotpool[s].append(w)
    rows = []
    print(f"\n[{name}]")
    print(f"   {'gold':<9}{'root':<8}{'amb?':<6}{'rank/21':>9}   top-3")
    for gold, (p, slot, root) in PROMPTS.items():
        if gold not in V.stoi:
            continue
        ids = V.enc(p.split())
        x = torch.tensor(ids[-SEQ:], device=DEVICE)[None]
        lg = F.log_softmax(model(x, tbl)[0, -1], -1)
        cands = [gold] + [w for w in slotpool[slot] if w != gold][:20]
        cid = torch.tensor([V.stoi[c] for c in cands], device=DEVICE)
        sc = lg[cid]
        r = int((sc > sc[0]).sum().item()) + 1
        top3 = [cands[j] for j in torch.topk(sc, 3).indices.tolist()]
        amb = ambiguous(root)
        rows.append((gold, r, amb))
        print(f"   {gold:<9}{root:<8}{'yes' if amb else '-':<6}{r:>9}   {', '.join(top3)}")
    un = [r for _, r, a in rows if not a]
    am = [r for _, r, a in rows if a]
    top1 = sum(1 for _, r, _ in rows if r == 1)
    print(f"   unambiguous roots: mean rank {np.mean(un):.1f} (n={len(un)})   "
          f"ambiguous: {np.mean(am):.1f} (n={len(am)})   top-1 {top1}/{len(rows)}")
    return dict(mean_un=float(np.mean(un)), mean_am=float(np.mean(am)),
                top1=top1, n=len(rows))


def main():
    train_texts, clean_eval, test_texts, held = build()
    parent, _ = load_forest()
    hset = {w for w, _, _ in held} | {g for g in PROMPTS}
    V = Vocab(train_texts, parent)
    for w in hset:
        if w in parent:
            V.add_word(w, parent)
    held_ids = [V.stoi[w] for w in hset if w in V.stoi]
    print(f"\nvocab {len(V.itos)}, roots {len(V.roots)}, slots {len(V.slots)}")
    print(f"held-out (masked from the softmax): {len(held_ids)}")

    E, W, vec, widx = build_prior(V, held_ids)
    geometric_check(V, E, W, vec, held)

    data = corpus(V, train_texts)
    hid = torch.tensor(held_ids, device=DEVICE)
    res = {}

    print("\n--- factored, operators FROZEN from geometry ---")
    m = Factored(V).to(DEVICE)
    with torch.no_grad():
        m.E.weight.copy_(E.to(DEVICE))
        m.Wop.copy_(W[1:].to(DEVICE))
    m.E.weight.requires_grad_(False)
    m.Wop.requires_grad_(False)
    print(f"    trainable params: "
          f"{sum(p.numel() for p in m.parameters() if p.requires_grad)/1e6:.1f}M "
          f"(E and W frozen)")
    train(m, data, mask_ids=hid)
    res["frozen prior"] = probe(m, V, "factored, frozen geometric operators")

    print("\n--- free-word baseline (same trunk) ---")
    b = FreeWord(V).to(DEVICE)
    train(b, data, mask_ids=hid)
    res["free-word"] = probe(b, V, "free-word baseline")

    print("\n" + "=" * 74)
    print(f"{'model':<34}{'unamb mean rank':>18}{'amb':>8}{'top-1':>8}")
    print("-" * 74)
    for k, v in res.items():
        print(f"{k:<34}{v['mean_un']:>18.1f}{v['mean_am']:>8.1f}{v['top1']:>8}")
    print("\nchance mean rank 11.  Learned-operator run: unamb 12.0, amb 19.5, top-1 0.")
    json.dump(res, open(f"{OUT}/prior.json", "w"), indent=1)


if __name__ == "__main__":
    main()
