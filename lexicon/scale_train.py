"""Train the operator algebra on the full English lexicon.

What changes at 38k words / 196k pairs:

  sampled softmax   Recomputing the adapter over the whole table every step
                    (as the 2.7k-word version did) is 38k x 768 x 1024 of
                    wasted work per batch. Instead each batch scores against
                    its own targets plus a large random negative sample. The
                    ranking metric is still computed against the FULL vocab,
                    so the number reported is not inflated by easy negatives.

  multi-positive    WordNet relations are massively one-to-many: 'dog' has
                    many hypernyms, all correct. Every sanctioned answer for a
                    (source, relation) is a positive and is excluded from the
                    negatives. Without this the model is punished for being
                    right, which is what wrecked the lexicographic operators
                    at small scale.

  held-out split    10% of pairs per relation, so generalization is measured
                    on word pairs never seen.
"""
import json, os, sys, collections, random
import torch
import torch.nn as nn
import torch.nn.functional as F

from lexicon.model import Adapter, EMB_DIM, REL_DIM, HIDDEN

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
D = "real/english"
N_NEG = 8192
TAU = 0.05


class Ops(nn.Module):
    def __init__(self, relation_names):
        super().__init__()
        self.relation_names = list(relation_names)
        self.rel_index = {r: i for i, r in enumerate(self.relation_names)}
        self.rel_emb = nn.Embedding(len(self.relation_names), REL_DIM)
        self.trunk = nn.Sequential(
            nn.Linear(EMB_DIM + REL_DIM, HIDDEN), nn.GELU(),
            nn.Linear(HIDDEN, HIDDEN), nn.GELU(),
            nn.Linear(HIDDEN, EMB_DIM))
        nn.init.normal_(self.trunk[-1].weight, std=1e-3)
        nn.init.zeros_(self.trunk[-1].bias)

    def forward(self, z, rid):
        return z + self.trunk(torch.cat([z, self.rel_emb(rid)], dim=-1))

    def apply_named(self, rel, z):
        single = z.dim() == 1
        if single:
            z = z.unsqueeze(0)
        rid = torch.full((z.shape[0],), self.rel_index[rel],
                         dtype=torch.long, device=z.device)
        y = self.forward(z, rid)
        return y.squeeze(0) if single else y


def load(with_inverses=True, val_frac=0.10, seed=0):
    vocab = json.load(open(f"{D}/vocab.json"))
    rels = json.load(open(f"{D}/relations.json"))
    protos = torch.load(f"{D}/prototypes.pt", weights_only=False)
    widx = {w: i for i, w in enumerate(vocab)}

    pairs = []
    for r, pl in rels.items():
        for s, t in pl:
            pairs.append((r, s, t))
    if with_inverses:
        pairs += [(r + "_inv", t, s) for r, s, t in pairs]

    rng = random.Random(seed)
    byrel = collections.defaultdict(list)
    for p in pairs:
        byrel[p[0]].append(p)
    train, val = [], []
    for r, pl in byrel.items():
        rng.shuffle(pl)
        k = max(1, int(len(pl) * val_frac))
        val += pl[:k]
        train += pl[k:]

    # every sanctioned answer for a (relation, source): all of them are correct
    pos = collections.defaultdict(set)
    for r, s, t in pairs:
        pos[(r, s)].add(widx[t])

    rel_names = sorted(byrel)
    P = torch.stack([protos[w] for w in vocab]).to(DEVICE)
    return vocab, widx, rel_names, train, val, pos, P


def batch_tensors(batch, widx, rel_index, pos, V):
    s = torch.tensor([widx[b[1]] for b in batch], device=DEVICE)
    t = torch.tensor([widx[b[2]] for b in batch], device=DEVICE)
    r = torch.tensor([rel_index[b[0]] for b in batch], device=DEVICE)
    return s, t, r


def sampled_loss(out, adapter, P, t, batch, pos, V, gen):
    """InfoNCE over batch targets + random negatives, with every sanctioned
    answer masked out of the negative set."""
    negs = torch.randint(0, V, (N_NEG,), device=DEVICE, generator=gen)
    cand = torch.cat([t, negs]).unique()
    cand_tbl = F.normalize(adapter(P[cand]), dim=-1)
    logits = out @ cand_tbl.T / TAU                        # [B, C]

    cpos = torch.zeros_like(logits, dtype=torch.bool)
    inv = {int(c): j for j, c in enumerate(cand.tolist())}
    for i, b in enumerate(batch):
        for w in pos[(b[0], b[1])]:
            j = inv.get(w)
            if j is not None:
                cpos[i, j] = True
    best = logits.masked_fill(~cpos, float("-inf")).max(1).values
    neg = logits.masked_fill(cpos, float("-inf"))
    return (torch.logsumexp(torch.cat([best.unsqueeze(1), neg], 1), 1) - best).mean()


@torch.no_grad()
def full_rank_eval(adapter, ops, P, data, widx, pos, vocab, n=4000, chunk=4096):
    """Rank against the FULL 38k vocabulary. Reports strict R@1 (the listed
    target) and 'any' R@1 (any WordNet-sanctioned answer)."""
    table = torch.cat([F.normalize(adapter(P[i:i + chunk]), dim=-1)
                       for i in range(0, len(P), chunk)])
    # data is grouped by relation; data[:n] would score only the alphabetically
    # first relations. Sample uniformly instead.
    if len(data) > n:
        idx = random.Random(1234).sample(range(len(data)), n)
        sub = [data[i] for i in idx]
    else:
        sub = data
    s = torch.tensor([widx[b[1]] for b in sub], device=DEVICE)
    r = torch.tensor([ops.rel_index[b[0]] for b in sub], device=DEVICE)
    t = torch.tensor([widx[b[2]] for b in sub], device=DEVICE)
    strict = anyv = 0
    for i in range(0, len(sub), 512):
        out = F.normalize(ops(table[s[i:i+512]], r[i:i+512]), dim=-1)
        sims = out @ table.T
        sims.scatter_(1, s[i:i+512].unsqueeze(1), -2)      # exclude the source
        top1 = sims.argmax(1)
        strict += (top1 == t[i:i+512]).sum().item()
        for j, b in enumerate(sub[i:i+512]):
            if top1[j].item() in pos[(b[0], b[1])]:
                anyv += 1
    return strict / len(sub), anyv / len(sub)


def main():
    vocab, widx, rel_names, train, val, pos, P = load()
    V = len(vocab)
    print(f"vocab {V}, relations {len(rel_names)}, "
          f"train pairs {len(train)}, held-out {len(val)}")

    adapter = Adapter().to(DEVICE)
    ops = Ops(rel_names).to(DEVICE)
    n_par = sum(p.numel() for p in list(adapter.parameters()) + list(ops.parameters()))
    print(f"model {n_par:,} params on {DEVICE}")

    gen = torch.Generator(device=DEVICE).manual_seed(0)
    opt = torch.optim.AdamW(list(adapter.parameters()) + list(ops.parameters()),
                            lr=6e-4, weight_decay=1e-2)
    EPOCHS, BS = int(os.environ.get("EPOCHS", 12)), 512
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS)
    P0n = F.normalize(P, dim=-1)

    for ep in range(EPOCHS):
        random.Random(ep).shuffle(train)
        tot = 0.0
        for i in range(0, len(train), BS):
            batch = train[i:i + BS]
            s, t, r = batch_tensors(batch, widx, ops.rel_index, pos, V)
            zs = F.normalize(adapter(P[s]), dim=-1)
            out = F.normalize(ops(zs, r), dim=-1)
            l = sampled_loss(out, adapter, P, t, batch, pos, V, gen)
            # keep the adapted space anchored to distilbert's geometry
            idx = torch.randint(0, V, (2048,), device=DEVICE, generator=gen)
            l_anc = (1 - F.cosine_similarity(
                F.normalize(adapter(P[idx]), dim=-1), P0n[idx], dim=-1)).mean()
            loss = l + 0.25 * l_anc
            opt.zero_grad(); loss.backward(); opt.step()
            tot += loss.item()
        sched.step()
        tr_s, tr_a = full_rank_eval(adapter, ops, P, train, widx, pos, vocab, 2000)
        va_s, va_a = full_rank_eval(adapter, ops, P, val, widx, pos, vocab, 2000)
        print(f"ep {ep:2d} loss {tot/(len(train)/BS):.3f} | "
              f"train R@1 {tr_s:.3f} (any {tr_a:.3f}) | "
              f"held-out R@1 {va_s:.3f} (any {va_a:.3f})")

    os.makedirs(D, exist_ok=True)
    torch.save({"adapter": adapter.state_dict(), "ops": ops.state_dict(),
                "relation_names": rel_names, "vocab": vocab},
               f"{D}/model.pt")
    json.dump({"train": [list(x) for x in train], "val": [list(x) for x in val]},
              open(f"{D}/split.json", "w"))
    print(f"saved {D}/model.pt")


if __name__ == "__main__":
    main()
