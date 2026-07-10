"""Full-English training with derived analogies. Ablatable, and leak-free.

LEAKAGE. The held-out split is taken on the ORIGINAL pairs first. Only then
are closures and quadruples derived, and only from the training half. Deriving
first would let symmetric closure hand (b,a) to training while (a,b) is the
test item -- the model would look like it generalised when it had been told
the answer.

LOSSES
  pair       multi-positive InfoNCE, sampled negatives. "given a and r, rank b"
  quad       a:b::c:d  =>  A(b)-A(a) parallel to A(d)-A(c).
             This does not ask the model to recall b; it imposes the
             'relation is a direction' property as a constraint on the SPACE.
  opposing   (w, syn, ant) => A(syn)-A(w) must point AWAY from A(ant)-A(w).
             The corpus cannot supply this: synonyms and antonyms are equally
             close (0.809 vs 0.801), because both are substitutable in context.

Run:  python -m lexicon.scale_train2            # pairs only (baseline)
      python -m lexicon.scale_train2 --quads --opposing
"""
import json, os, sys, collections, random
import torch
import torch.nn.functional as F

from lexicon.model import Adapter
from lexicon.scale_train import Ops, sampled_loss, full_rank_eval
from lexicon.derive import (transitive_closure, symmetric_closure,
                            inverse_closure, build_quadruples, opposing_triples,
                            TRANSITIVE, SYMMETRIC, INVERSE_OF)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
D = "real/english"
USE_QUADS = "--quads" in sys.argv
USE_OPPOSING = "--opposing" in sys.argv
TAG = ("quads" if USE_QUADS else "") + ("_opp" if USE_OPPOSING else "") or "baseline"


def split_then_derive(val_frac=0.10, seed=0):
    rels = {k: [tuple(p) for p in v]
            for k, v in json.load(open(f"{D}/relations.json")).items()}
    rng = random.Random(seed)
    train_rels, val_pairs = {}, []
    for r, pl in rels.items():
        pl = list(pl); rng.shuffle(pl)
        k = max(1, int(len(pl) * val_frac))
        val_pairs += [(r, a, b) for a, b in pl[:k]]
        train_rels[r] = pl[k:]

    # --- derive ONLY from the training half ---
    derived = collections.defaultdict(list)
    for r in TRANSITIVE:
        if r in train_rels:
            derived[r] += transitive_closure(train_rels[r])
    for r in SYMMETRIC:
        if r in train_rels:
            derived[r] += symmetric_closure(train_rels[r])
    for r, inv in INVERSE_OF.items():
        if r in train_rels and inv in train_rels:
            derived[inv] += inverse_closure(train_rels[r], train_rels[inv])
    aug = {r: sorted(set(train_rels[r]) | set(derived.get(r, [])))
           for r in train_rels}

    # scrub anything that collides with a held-out pair (a derived pair may
    # coincide with a test pair by chance; that would still be leakage)
    val_set = {(r, a, b) for r, a, b in val_pairs}
    aug = {r: [(a, b) for a, b in pl if (r, a, b) not in val_set]
           for r, pl in aug.items()}

    train_pairs = [(r, a, b) for r, pl in aug.items() for a, b in pl]
    train_pairs += [(r + "_inv", b, a) for r, a, b in train_pairs]
    val_pairs += [(r + "_inv", b, a) for r, a, b in val_pairs]
    return aug, train_pairs, val_pairs


def main():
    vocab = json.load(open(f"{D}/vocab.json"))
    widx = {w: i for i, w in enumerate(vocab)}
    protos = torch.load(f"{D}/prototypes.pt", weights_only=False)
    P = torch.stack([protos[w] for w in vocab]).to(DEVICE)
    V = len(vocab)

    aug, train, val = split_then_derive()
    rel_names = sorted({p[0] for p in train})
    pos = collections.defaultdict(set)
    for r, s, t in train:
        pos[(r, s)].add(widx[t])
    for r, s, t in val:                     # all sanctioned answers count
        pos[(r, s)].add(widx[t])
    print(f"mode {TAG}: vocab {V}, relations {len(rel_names)}, "
          f"train {len(train)}, held-out {len(val)}")

    quads = {}
    if USE_QUADS:
        for r, pl in aug.items():
            if len(pl) >= 8:
                q = build_quadruples(pl, max_per_rel=60000)
                if q:
                    quads[r] = q
        flat_q = [(r, *q) for r, ql in quads.items() for q in ql]
        print(f"  quadruples: {len(flat_q):,}")
    trips = []
    if USE_OPPOSING:
        trips = [t for t in opposing_triples(aug)
                 if all(w in widx for w in t)]
        print(f"  opposing triples: {len(trips):,}")

    adapter = Adapter().to(DEVICE)
    ops = Ops(rel_names).to(DEVICE)
    gen = torch.Generator(device=DEVICE).manual_seed(0)
    opt = torch.optim.AdamW(list(adapter.parameters()) + list(ops.parameters()),
                            lr=6e-4, weight_decay=1e-2)
    EPOCHS, BS = int(os.environ.get("EPOCHS", 10)), 512
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS)
    P0n = F.normalize(P, dim=-1)

    for ep in range(EPOCHS):
        random.Random(ep).shuffle(train)
        if USE_QUADS:
            random.Random(ep).shuffle(flat_q)
        tot = 0.0
        nb = 0
        for i in range(0, len(train), BS):
            batch = train[i:i + BS]
            s = torch.tensor([widx[b[1]] for b in batch], device=DEVICE)
            t = torch.tensor([widx[b[2]] for b in batch], device=DEVICE)
            r = torch.tensor([ops.rel_index[b[0]] for b in batch], device=DEVICE)
            zs = F.normalize(adapter(P[s]), dim=-1)
            out = F.normalize(ops(zs, r), dim=-1)
            loss = sampled_loss(out, adapter, P, t, batch, pos, V, gen)

            if USE_QUADS and flat_q:
                j = (nb * 256) % max(1, len(flat_q) - 256)
                qb = flat_q[j:j + 256]
                qa = torch.tensor([widx[q[1]] for q in qb], device=DEVICE)
                qbb = torch.tensor([widx[q[2]] for q in qb], device=DEVICE)
                qc = torch.tensor([widx[q[3]] for q in qb], device=DEVICE)
                qd = torch.tensor([widx[q[4]] for q in qb], device=DEVICE)
                A = adapter(P[torch.cat([qa, qbb, qc, qd])])
                A = F.normalize(A, dim=-1)
                n = len(qb)
                d1 = A[n:2*n] - A[:n]
                d2 = A[3*n:] - A[2*n:3*n]
                loss = loss + 0.5 * (1 - F.cosine_similarity(d1, d2, dim=-1)).mean()

            if USE_OPPOSING and trips:
                j = (nb * 128) % max(1, len(trips) - 128)
                tb = trips[j:j + 128]
                w = torch.tensor([widx[x[0]] for x in tb], device=DEVICE)
                sy = torch.tensor([widx[x[1]] for x in tb], device=DEVICE)
                an = torch.tensor([widx[x[2]] for x in tb], device=DEVICE)
                A = F.normalize(adapter(P[torch.cat([w, sy, an])]), dim=-1)
                n = len(tb)
                dsyn = A[n:2*n] - A[:n]
                dant = A[2*n:] - A[:n]
                c = F.cosine_similarity(dsyn, dant, dim=-1)
                loss = loss + 0.5 * F.relu(c + 0.2).mean()   # push apart

            idx = torch.randint(0, V, (2048,), device=DEVICE, generator=gen)
            loss = loss + 0.25 * (1 - F.cosine_similarity(
                F.normalize(adapter(P[idx]), dim=-1), P0n[idx], dim=-1)).mean()

            opt.zero_grad(); loss.backward(); opt.step()
            tot += loss.item(); nb += 1
        sched.step()
        va_s, va_a = full_rank_eval(adapter, ops, P, val, widx, pos, vocab, 3000)
        print(f"ep {ep:2d} loss {tot/nb:.3f} | held-out R@1 {va_s:.3f} "
              f"(any {va_a:.3f})")

    torch.save({"adapter": adapter.state_dict(), "ops": ops.state_dict(),
                "relation_names": rel_names, "vocab": vocab},
               f"{D}/model_{TAG}.pt")
    json.dump({"train": [list(x) for x in train], "val": [list(x) for x in val]},
              open(f"{D}/split_{TAG}.json", "w"))
    print(f"saved {D}/model_{TAG}.pt")


if __name__ == "__main__":
    main()
