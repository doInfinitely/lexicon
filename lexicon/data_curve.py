"""Does expanding the data fix antonymy? The only fair way to ask.

Everything is held fixed -- 38k vocabulary, the same leak-free held-out pairs,
the same architecture and schedule -- and ONLY the number of training pairs
varies, from BATS's 35 up to WordNet's ~2900.

Comparing the old BATS result (R@1 0.267, retrieval over 2768 words) with the
new one (0.269, retrieval over 38142 words) would be meaningless: the second
task is 14x harder. This curve removes that confound.

Pairs are canonicalised before splitting, because WordNet stores antonymy in
both directions and 88% of a naive held-out set has its reverse in training.
"""
import json, random, sys
import numpy as np
import torch
import torch.nn.functional as F

from lexicon.model import Adapter
from lexicon.involution import (MLPOp, InvolutionOp, load_antonyms, infonce,
                                evaluate, DEVICE, D)


def run_n(n_pairs, kind="mlp+rt", epochs=60, bs=256, lr=4e-4, seed=0, k=8):
    vocab, widx, P, train_all, val, pos, pos_eval = load_antonyms()
    V = len(vocab)
    # train_all holds both directions of each canonical pair; subsample by
    # canonical pair so the two directions stay together
    canon = sorted({tuple(sorted(p)) for p in train_all})
    rng = random.Random(seed)
    rng.shuffle(canon)
    sub = canon[:n_pairs]
    train = [(a, b) for a, b in sub] + [(b, a) for a, b in sub]

    adapter = Adapter().to(DEVICE)
    op = (InvolutionOp(k=k) if kind.startswith("involution") else MLPOp()).to(DEVICE)
    use_rt = kind == "mlp+rt"
    gen = torch.Generator(device=DEVICE).manual_seed(seed)
    opt = torch.optim.AdamW(list(adapter.parameters()) + list(op.parameters()),
                            lr=lr, weight_decay=1e-2)
    # equalise optimisation steps across data sizes: small n gets more epochs
    steps_per_epoch = max(1, len(train) // bs)
    eff_epochs = max(epochs, int(epochs * 60 / max(steps_per_epoch, 1)))
    eff_epochs = min(eff_epochs, 600)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=eff_epochs)
    P0n = F.normalize(P, dim=-1)

    for ep in range(eff_epochs):
        random.Random(ep).shuffle(train)
        for i in range(0, len(train), bs):
            b = train[i:i + bs]
            s = torch.tensor([widx[a] for a, _ in b], device=DEVICE)
            t = torch.tensor([widx[c] for _, c in b], device=DEVICE)
            zs = F.normalize(adapter(P[s]), dim=-1)
            fx = op(zs)
            loss = infonce(F.normalize(fx, dim=-1), adapter, P, t,
                           [a for a, _ in b], pos, widx, V, gen)
            if use_rt:
                loss = loss + (1 - F.cosine_similarity(op(fx), zs, dim=-1)).mean()
            idx = torch.randint(0, V, (2048,), device=DEVICE, generator=gen)
            loss = loss + 0.25 * (1 - F.cosine_similarity(
                F.normalize(adapter(P[idx]), dim=-1), P0n[idx], dim=-1)).mean()
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(
                list(adapter.parameters()) + list(op.parameters()), 1.0)
            opt.step()
        sched.step()
    return evaluate(adapter, op, P, val, widx, pos_eval, vocab), len(train) // 2, eff_epochs


def main():
    vocab, widx, P, train_all, val, pos, pos_eval = load_antonyms()
    canon = len({tuple(sorted(p)) for p in train_all})
    print(f"held-out (canonical, leak-free): {len(val)} pairs; "
          f"retrieval over {len(vocab)} words")
    print(f"available training pairs: {canon}\n")

    with torch.no_grad():
        tbl = F.normalize(P, dim=-1)
    s = torch.tensor([widx[a] for a, _ in val], device=DEVICE)
    t = torch.tensor([widx[b] for _, b in val], device=DEVICE)
    sims = tbl[s] @ tbl.T
    sims.scatter_(1, s.unsqueeze(1), -2)
    print(f"frozen space, identity operator: R@1 "
          f"{(sims.argmax(1) == t).float().mean().item():.3f}\n")

    print(f"{'train pairs':>12}{'epochs':>9}{'held-out R@1':>15}{'R@1 any':>10}"
          f"{'direction':>12}{'round-trip':>13}")
    print("-" * 71)
    out = {}
    for n in (35, 100, 300, 1000, canon):
        m, used, eps = run_n(n)
        out[used] = {kk: float(vv) for kk, vv in m.items()}
        print(f"{used:>12}{eps:>9}{m['R1']:>15.3f}{m['R1_any']:>10.3f}"
              f"{m['align']:>12.3f}{m['roundtrip']:>13.3f}")

    print("\nAll rows share the same 38k vocabulary and the same held-out pairs,")
    print("so these R@1 values are directly comparable. 'direction' is the")
    print("alignment of held-out displacements: whether antonymy became a vector.")
    json.dump(out, open(f"{D}/antonym_data_curve.json", "w"), indent=1)


if __name__ == "__main__":
    main()
