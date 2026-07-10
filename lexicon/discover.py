"""Virtual words and branched inverses: what structure does language leave implicit?

Two ideas, one mechanism.

BRANCHED INVERSES. The fan-in probe shows many relations are many-to-one
(50 animals -> 10 hypernyms). Their inverses are therefore NOT functions: no
single operator can send 'animal' to both 'sparrow' and 'beaver'. So each
inverse gets K branches, trained by hard EM -- every pair updates only the
branch that already predicts it best. The branches are not a capacity hack;
if the idea is right, each branch should converge on a coherent SUBTREE of
the taxonomy, and reading them off tells you how the lexicon fans out.

VIRTUAL WORDS. The base lexicon is forced to consist of real words. But the
best hub for a cluster of words is often a concept English never lexicalized
-- a point in meaning-space between the words. We therefore learn M free
vectors ("virtual words") that serve as derivation roots. Operators stay
FROZEN and semantically grounded on BATS, so a virtual word is only ever
interpretable through them: anchor_7 is whatever thing whose plural is
'wolves' and whose young is 'cub'. We then name each anchor by its nearest
real words -- and the interesting ones have no good name.

Because every word is one operator application from an anchor, this also
dissolves the depth problem: chains never compound.
"""
import json, collections, sys
import torch
import torch.nn as nn
import torch.nn.functional as F

from lexicon.model import Adapter, EMB_DIM, REL_DIM, HIDDEN

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
K_BRANCHES = int(sys.argv[1]) if len(sys.argv) > 1 else 4


class BranchedOps(nn.Module):
    """One trunk; each relation has K branch embeddings. Branch k of relation
    r is a distinct operator, written r#k."""

    def __init__(self, relation_names, k=K_BRANCHES):
        super().__init__()
        self.relation_names = list(relation_names)
        self.k = k
        self.rel_index = {r: i for i, r in enumerate(self.relation_names)}
        self.rel_emb = nn.Embedding(len(self.relation_names) * k, REL_DIM)
        nn.init.normal_(self.rel_emb.weight, std=0.5)  # spread branches apart
        self.trunk = nn.Sequential(
            nn.Linear(EMB_DIM + REL_DIM, HIDDEN), nn.GELU(),
            nn.Linear(HIDDEN, HIDDEN), nn.GELU(),
            nn.Linear(HIDDEN, EMB_DIM))
        nn.init.normal_(self.trunk[-1].weight, std=1e-3)
        nn.init.zeros_(self.trunk[-1].bias)

    def forward(self, z, bid):
        return z + self.trunk(torch.cat([z, self.rel_emb(bid)], dim=-1))

    def branch_ids(self, rel_ids, k):
        return rel_ids * self.k + k

    def apply_branch(self, rel, k, z):
        single = z.dim() == 1
        if single:
            z = z.unsqueeze(0)
        bid = torch.full((z.shape[0],), self.rel_index[rel] * self.k + k,
                         dtype=torch.long, device=z.device)
        y = self.forward(z, bid)
        return y.squeeze(0) if single else y


def positive_mask(pairs, widx, V):
    M = torch.zeros(len(pairs), V, dtype=torch.bool)
    for i, p in enumerate(pairs):
        for a in set(p.get("alternates") or []) | {p["target"]}:
            if a in widx:
                M[i, widx[a]] = True
    return M


def multi_pos_loss(out, table, pos, tau=0.05, reduce=True):
    logits = out @ table.T / tau
    neg = logits.masked_fill(pos, float("-inf"))
    best = logits.masked_fill(~pos, float("-inf")).max(1).values
    l = torch.logsumexp(torch.cat([best.unsqueeze(1), neg], 1), 1) - best
    return l.mean() if reduce else l


def train_branched(adapter, ops, pairs, vocab, P, epochs=500, bs=256, tau=0.05):
    """Hard-EM over branches: each pair trains only its best-fitting branch."""
    widx = {w: i for i, w in enumerate(vocab)}
    V = len(vocab)
    POS = positive_mask(pairs, widx, V).to(DEVICE)
    st = torch.tensor([widx[p["source"]] for p in pairs], device=DEVICE)
    rt = torch.tensor([ops.rel_index[p["relation"]] for p in pairs], device=DEVICE)
    P0n = F.normalize(P, dim=-1)

    params = list(adapter.parameters()) + list(ops.parameters())
    opt = torch.optim.AdamW(params, lr=8e-4, weight_decay=1e-2)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)

    for ep in range(epochs):
        order = torch.randperm(len(pairs), device=DEVICE)
        for i in range(0, len(order), bs):
            b = order[i:i + bs]
            table = F.normalize(adapter(P), dim=-1)
            zs, pos = table[st[b]], POS[b]
            # E-step: which branch already explains this pair best?
            with torch.no_grad():
                losses = torch.stack([
                    multi_pos_loss(F.normalize(ops(zs, ops.branch_ids(rt[b], k)), -1),
                                   table, pos, tau, reduce=False)
                    for k in range(ops.k)])                # [K, B]
                assign = losses.argmin(0)                  # [B]
            # M-step: update only that branch
            out = F.normalize(ops(zs, ops.branch_ids(rt[b], assign)), dim=-1)
            l_ret = multi_pos_loss(out, table, pos, tau)
            l_cos = (1 - (out @ table.T).masked_fill(~pos, -2).max(1).values).mean()
            l_anc = (1 - F.cosine_similarity(table, P0n, dim=-1)).mean()
            loss = l_ret + l_cos + 0.25 * l_anc
            opt.zero_grad()
            loss.backward()
            opt.step()
        sched.step()
        if ep % 50 == 0:
            with torch.no_grad():
                table = F.normalize(adapter(P), dim=-1)
                best = torch.stack([
                    (F.normalize(ops(table[st], ops.branch_ids(rt, k)), -1)
                     @ table.T).argmax(1) for k in range(ops.k)])
                hit = POS[torch.arange(len(st), device=DEVICE).unsqueeze(0), best]
                acc = hit.any(0).float().mean().item()
            print(f"  ep {ep:3d}  best-branch R@1(any) {acc:.3f}")
    return adapter, ops


@torch.no_grad()
def branch_report(adapter, ops, pairs, vocab, P, out_path):
    """Which branch claims which pairs? A branch should be a coherent subtree."""
    widx = {w: i for i, w in enumerate(vocab)}
    table = F.normalize(adapter(P), dim=-1)
    POS = positive_mask(pairs, widx, len(vocab)).to(DEVICE)
    st = torch.tensor([widx[p["source"]] for p in pairs], device=DEVICE)
    rt = torch.tensor([ops.rel_index[p["relation"]] for p in pairs], device=DEVICE)
    losses = torch.stack([
        multi_pos_loss(F.normalize(ops(table[st], ops.branch_ids(rt, k)), -1),
                       table, POS, reduce=False) for k in range(ops.k)])
    assign = losses.argmin(0).tolist()
    byrel = collections.defaultdict(lambda: collections.defaultdict(list))
    for p, k in zip(pairs, assign):
        byrel[p["relation"]][k].append((p["source"], p["target"]))
    json.dump({r: {str(k): v for k, v in d.items()} for r, d in byrel.items()},
              open(out_path, "w"), indent=1)
    return byrel


class VirtualWords(nn.Module):
    """M free vectors that act as derivation roots. Operators stay frozen, so
    a virtual word means exactly what the operators say it means."""

    def __init__(self, table, m):
        super().__init__()
        # init as k-means centroids of the real vocabulary
        idx = torch.randperm(len(table))[:m]
        C = table[idx].clone()
        for _ in range(25):
            a = (table @ C.T).argmax(1)
            for j in range(m):
                sel = table[a == j]
                if len(sel):
                    C[j] = F.normalize(sel.mean(0), dim=-1)
        self.anchors = nn.Parameter(C)

    def normalized(self):
        return F.normalize(self.anchors, dim=-1)


def train_virtual(vw, ops, table, epochs=300, tau=0.05, lr=3e-3, margin=0.02):
    """EM: assign every word to its best (anchor, relation, branch); push that
    anchor's image onto the word. Only anchors move -- operators are frozen,
    which is what keeps the virtual words interpretable."""
    for p in ops.parameters():
        p.requires_grad_(False)
    opt = torch.optim.Adam(vw.parameters(), lr=lr)
    R, K = len(ops.relation_names), ops.k
    bids = torch.arange(R * K, device=DEVICE)
    tgt = torch.arange(len(table), device=DEVICE)

    for ep in range(epochs):
        A = vw.normalized()                                   # [M,768]
        M = len(A)
        z = A.repeat_interleave(R * K, 0)                     # [M*R*K, 768]
        b = bids.repeat(M)
        img = F.normalize(ops(z, b), dim=-1)                  # images of anchors
        sims = img @ table.T                                  # [M*R*K, V]
        # E-step: each word picks the (anchor,rel,branch) that best hits it
        best_slot = sims.argmax(0)                            # [V]
        # M-step: InfoNCE pulling that slot's image onto the word
        chosen = img[best_slot]                               # [V,768]
        logits = chosen @ table.T / tau
        loss = F.cross_entropy(logits, tgt)
        opt.zero_grad()
        loss.backward()
        opt.step()
        if ep % 50 == 0:
            with torch.no_grad():
                cov = coverage(vw, ops, table, margin)
            print(f"  ep {ep:3d}  loss {loss.item():.3f}  "
                  f"covered {cov['n_covered']}/{len(table)} "
                  f"({cov['n_covered']/len(table):.1%})")
    for p in ops.parameters():
        p.requires_grad_(True)
    return vw


@torch.no_grad()
def coverage(vw, ops, table, margin=0.02):
    """A word is covered when some anchor's image under some operator decodes
    to it as nearest neighbour, with a margin over the runner-up word."""
    A = vw.normalized()
    R, K, M = len(ops.relation_names), ops.k, len(A)
    bids = torch.arange(R * K, device=DEVICE)
    z, b = A.repeat_interleave(R * K, 0), bids.repeat(M)
    img = F.normalize(ops(z, b), dim=-1)
    sims = img @ table.T                                       # [slots, V]
    top2 = sims.topk(2, dim=1)
    decoded, marg = top2.indices[:, 0], top2.values[:, 0] - top2.values[:, 1]
    ok = marg >= margin
    covered, prov = {}, {}
    for slot in ok.nonzero(as_tuple=True)[0].tolist():
        w = decoded[slot].item()
        if w not in covered or marg[slot] > covered[w]:
            covered[w] = marg[slot].item()
            j, rest = divmod(slot, R * K)
            r, k = divmod(rest, K)
            prov[w] = (j, ops.relation_names[r], k)
    return {"n_covered": len(covered), "provenance": prov}


def main():
    vocab = json.load(open("harbor/workspace/data/vocab.json"))
    relations = json.load(open("harbor/workspace/data/relations.json"))
    train_pairs = json.load(open("harbor/workspace/data/train_pairs.json"))
    test_pairs = json.load(open("harbor/tests/test_pairs.json"))
    protos = torch.load("real/embeddings/prototypes.pt", weights_only=False)
    P = torch.stack([protos[w] for w in vocab]).to(DEVICE)

    pairs = train_pairs + test_pairs
    pairs = pairs + [{"relation": p["relation"] + "_inv", "source": p["target"],
                      "target": p["source"], "alternates": [p["source"]]}
                     for p in pairs]
    rel_names = sorted(relations) + [r + "_inv" for r in sorted(relations)]

    adapter = Adapter().to(DEVICE)
    ops = BranchedOps(rel_names, K_BRANCHES).to(DEVICE)
    print(f"branched operators: {len(rel_names)} relations x {K_BRANCHES} "
          f"branches = {len(rel_names)*K_BRANCHES} operators")
    train_branched(adapter, ops, pairs, vocab, P)

    with torch.no_grad():
        table = F.normalize(adapter(P), dim=-1)
    print("\nbranch assignment report -> real/branches.json")
    branch_report(adapter, ops, pairs, vocab, P, "real/branches.json")

    print("\nlearning virtual words (operators frozen)...")
    results = {}
    for m in (16, 32, 64, 128, 256):
        vw = VirtualWords(table, m).to(DEVICE)
        train_virtual(vw, ops, table)
        cov = coverage(vw, ops, table)
        n_cov = cov["n_covered"]
        gen = len(vocab) - n_cov + m       # virtual words + words still needing storage
        results[m] = {"anchors": m, "covered": n_cov,
                      "generating_set": gen,
                      "ratio": round(len(vocab) / gen, 3)}
        print(f"  M={m:>3}: covers {n_cov}/{len(vocab)} at depth 1 -> "
              f"generating set {gen}  ({len(vocab)/gen:.2f}x)")
        torch.save({"anchors": vw.anchors.detach().cpu(),
                    "provenance": cov["provenance"]}, f"real/virtual_{m}.pt")

    torch.save({"adapter": adapter.state_dict(), "ops": ops.state_dict(),
                "relation_names": rel_names, "k": K_BRANCHES, "vocab": vocab},
               "real/branched.pt")
    json.dump(results, open("real/virtual_sweep.json", "w"), indent=1)
    print("\nsaved real/branched.pt, real/virtual_*.pt")


if __name__ == "__main__":
    main()
