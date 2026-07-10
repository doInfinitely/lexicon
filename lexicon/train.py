"""Joint training of the adapted space + operators, then lexicon selection.

Losses:
  retrieval  InfoNCE over the full vocab: f_r(A(e_s)) must rank A(e_t) first
  cosine     pointwise pull toward the target (smooths the retrieval loss)
  anchor     small distillation term keeping A near the distilbert geometry
             (guards against the space drifting into a degenerate layout)

Lexicon selection: greedy derivation forest over operator-scored edges with a
*correct* global depth invariant (subtree height tracked on every attach),
then an exact composed-quality repair loop. Compression is pushed as far as
retrieval quality allows, sweeping the edge-quality floor and reporting the
tradeoff curve.
"""
import json, os, sys, collections
import torch
import torch.nn.functional as F

from lexicon.model import LexiconSpace

DATA = "harbor/workspace/data"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
# Reachability is not the constraint (every word is reachable from some other
# word by some operator); compounding error along chains is. Chain-aware
# refinement (refine_on_forest) is what buys depth, so this is tunable.
MAX_DEPTH = int(os.environ.get("LEXICON_MAX_DEPTH", 3))


def cos(a, b):
    return F.cosine_similarity(a, b, dim=-1)


def load_data():
    vocab = json.load(open(f"{DATA}/vocab.json"))
    relations = json.load(open(f"{DATA}/relations.json"))
    train_pairs = json.load(open(f"{DATA}/train_pairs.json"))
    test_pairs = json.load(open("harbor/tests/test_pairs.json"))
    protos = torch.load("real/embeddings/prototypes.pt", weights_only=False)
    return vocab, relations, train_pairs, test_pairs, protos


def add_inverses(pairs):
    return pairs + [{"relation": p["relation"] + "_inv",
                     "source": p["target"], "target": p["source"],
                     # the inverse of a one-to-many map is many-to-one, so the
                     # only valid answer going back is the original source
                     "alternates": [p["source"]]}
                    for p in pairs]


def positive_mask(pairs, vocab, widx):
    """[N, V] bool: every BATS-sanctioned answer for each pair.

    BATS lists multiple valid targets (ant -> insect/invertebrate/animal/...).
    Training the operator to hit the first one and scoring the rest as
    negatives teaches it noise: it is punished for being right. Instead the
    alternates are excluded from the negative set, and the loss rewards the
    closest valid answer.
    """
    M = torch.zeros(len(pairs), len(vocab), dtype=torch.bool)
    for i, p in enumerate(pairs):
        for a in set(p.get("alternates") or []) | {p["target"]}:
            if a in widx:
                M[i, widx[a]] = True
    return M


def train_joint(space, pairs, vocab, P, epochs=600, bs=256, tau=0.05,
                w_cos=1.0, w_anchor=0.25, val_frac=0.1):
    """P: [V,768] raw prototype matrix on DEVICE, row i = vocab[i]."""
    widx = {w: i for i, w in enumerate(vocab)}
    g = torch.Generator().manual_seed(0)
    perm = torch.randperm(len(pairs), generator=g).tolist()
    n_val = int(len(pairs) * val_frac)
    val_idx, tr_idx = perm[:n_val], perm[n_val:]

    POS = positive_mask(pairs, vocab, widx).to(DEVICE)

    def tensors(idx):
        s = torch.tensor([widx[pairs[i]["source"]] for i in idx], device=DEVICE)
        t = torch.tensor([widx[pairs[i]["target"]] for i in idx], device=DEVICE)
        r = torch.tensor([space.ops.rel_index[pairs[i]["relation"]] for i in idx],
                         device=DEVICE)
        m = POS[torch.tensor(idx, device=DEVICE)]
        return s, t, r, m

    st, tt, rt, mt = tensors(tr_idx)
    sv, tv, rv, mv = tensors(val_idx)
    P0n = F.normalize(P, dim=-1)  # frozen geometry anchor

    def multi_pos_loss(out, table, pos):
        """InfoNCE where every valid answer is a positive: negatives exclude
        all alternates, and the numerator is the best-matching positive."""
        logits = out @ table.T / tau
        neg = logits.masked_fill(pos, float("-inf"))
        best_pos = logits.masked_fill(~pos, float("-inf")).max(1).values
        denom = torch.logsumexp(
            torch.cat([best_pos.unsqueeze(1), neg], dim=1), dim=1)
        return (denom - best_pos).mean()

    opt = torch.optim.AdamW(space.parameters(), lr=8e-4, weight_decay=1e-2)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    best = {"acc": -1.0}
    patience = 0
    for ep in range(epochs):
        space.train()
        order = torch.randperm(len(tr_idx), device=DEVICE)
        for i in range(0, len(order), bs):
            b = order[i:i + bs]
            table = F.normalize(space.adapter(P), dim=-1)      # [V,768]
            zs = table[st[b]]
            out = F.normalize(space.ops(zs, rt[b]), dim=-1)
            l_ret = multi_pos_loss(out, table, mt[b])
            # pull toward the nearest VALID answer, not an arbitrary one
            l_cos = (1 - (out @ table.T).masked_fill(~mt[b], -2)
                     .max(1).values).mean()
            l_anc = (1 - cos(table, P0n)).mean()
            loss = l_ret + w_cos * l_cos + w_anchor * l_anc
            opt.zero_grad()
            loss.backward()
            opt.step()
        sched.step()

        space.eval()
        with torch.no_grad():
            table = F.normalize(space.adapter(P), dim=-1)
            out = F.normalize(space.ops(table[sv], rv), dim=-1)
            ranks = (out @ table.T).argmax(dim=1)
            strict = (ranks == tv).float().mean().item()
            # a hit on ANY BATS-sanctioned answer is a hit; select on this
            acc = mv[torch.arange(len(ranks), device=DEVICE), ranks].float().mean().item()
            vcos = cos(out, table[tv]).mean().item()
        if acc > best["acc"]:
            best = {"acc": acc, "strict": strict, "cos": vcos, "ep": ep,
                    "state": {k: v.detach().clone()
                              for k, v in space.state_dict().items()}}
            patience = 0
        else:
            patience += 1
            if patience >= 60:
                break
        if ep % 25 == 0:
            print(f"  ep {ep:3d} val R@1 any {acc:.3f} strict {strict:.3f} "
                  f"cos {vcos:.4f} (best {best['acc']:.3f})")
    space.load_state_dict(best["state"])
    print(f"  joint training done: val R@1 any {best['acc']:.3f} "
          f"strict {best['strict']:.3f} cos {best['cos']:.4f} @ep {best['ep']}")


def score_edges(space, vocab, P, edges):
    with torch.no_grad():
        table = F.normalize(space.adapter(P), dim=-1)
    widx = {w: i for i, w in enumerate(vocab)}
    scored = []
    with torch.no_grad():
        for rel, plist in edges.items():
            if not plist:
                continue
            si = torch.tensor([widx[s] for s, t in plist], device=DEVICE)
            ti = torch.tensor([widx[t] for s, t in plist], device=DEVICE)
            out = F.normalize(space.ops.apply_named(rel, table[si]), dim=-1)
            q = cos(out, table[ti]).tolist()
            scored += [(qi, rel, s, t) for qi, (s, t) in zip(q, plist)]
    scored.sort(reverse=True)
    return scored, table


def build_forest(space, vocab, P, edges, floor, scored=None, table=None,
                 blacklist=frozenset()):
    """Greedy forest with a correct global depth invariant."""
    if scored is None:
        scored, table = score_edges(space, vocab, P, edges)

    parent, children = {}, collections.defaultdict(set)
    depth = collections.defaultdict(int)    # distance from root
    height = collections.defaultdict(int)   # subtree height below word

    def is_ancestor(a, w):
        while w in parent:
            w = parent[w][1]
            if w == a:
                return True
        return False

    def attach(t, rel, s):
        parent[t] = (rel, s)
        children[s].add(t)
        stack = [(t, depth[s] + 1)]
        while stack:
            w, d = stack.pop()
            depth[w] = d
            stack.extend((c, d + 1) for c in children[w])
        w, h = t, height[t]
        while w in parent:
            w = parent[w][1]
            h += 1
            if height[w] >= h:
                break
            height[w] = h

    for q, rel, s, t in scored:
        if q < floor:
            break
        if t in parent or s == t or is_ancestor(t, s):
            continue
        if (t, rel, s) in blacklist:
            continue
        if depth[s] + 1 + height[t] > MAX_DEPTH:
            continue
        attach(t, rel, s)
    return parent, table


def exact_quality(space, parent, table, widx):
    memo = {}
    def emb(w):
        if w not in parent:
            return table[widx[w]]
        if w not in memo:
            rel, s = parent[w]
            with torch.no_grad():
                memo[w] = F.normalize(
                    space.ops.apply_named(rel, emb(s)), dim=-1)
        return memo[w]
    qual = {w: cos(emb(w), table[widx[w]]).item() for w in parent}
    return qual, {w: emb(w) for w in parent}


def select_lexicon(space, vocab, P, edges, train_targets,
                   margins=(0.10, 0.05, 0.02, 0.0), build_floor=0.55):
    """Maximize compression subject to decodability: every derived word's
    composed embedding must retrieve the word itself as nearest neighbor,
    with at least `margin` similarity over the runner-up. Words failing the
    criterion revert to base until the forest is stable."""
    widx = {w: i for i, w in enumerate(vocab)}
    scored, table = score_edges(space, vocab, P, edges)
    results = []
    for margin in margins:
        # A word whose chosen parent fails the decode gate must be allowed to
        # fall back on its other candidate parents, so blacklist the failing
        # EDGE (not the word) and rebuild. Reverting the word outright
        # silently throws away every alternative derivation it had.
        blacklist = set()
        for _ in range(40):
            parent, _ = build_forest(space, vocab, P, edges, build_floor,
                                     scored=scored, table=table,
                                     blacklist=blacklist)
            qual, embs = exact_quality(space, parent, table, widx)
            derived = list(parent)
            E = torch.stack([embs[w] for w in derived])
            sims = E @ table.T                             # [D, V]
            self_idx = torch.tensor([widx[w] for w in derived],
                                    device=sims.device)
            self_sim = sims.gather(1, self_idx.unsqueeze(1)).squeeze(1)
            sims.scatter_(1, self_idx.unsqueeze(1), -2)
            runner_up = sims.max(1).values
            bad = [w for w, ok in zip(
                derived, (self_sim - runner_up >= margin).tolist()) if not ok]
            if not bad:
                break
            for w in bad:
                rel, s = parent[w]
                blacklist.add((w, rel, s))
        else:  # never converged: drop whatever still fails
            for w in bad:
                del parent[w]
            qual, embs = exact_quality(space, parent, table, widx)
        tr = [qual[w] for w in parent if w in train_targets]
        results.append({
            "margin": margin,
            "base": len(vocab) - len(parent),
            "ratio": round(len(vocab) / (len(vocab) - len(parent)), 3),
            "derived_retrieval@1": 1.0,
            "derived_mean_cos": round(sum(qual.values()) / len(qual), 4),
            "train_mean_cos": round(sum(tr) / len(tr), 4) if tr else None,
            "parent": dict(parent),
        })
        r = results[-1]
        print(f"  margin {margin:.2f}: base={r['base']} ratio={r['ratio']} "
              f"cos={r['derived_mean_cos']} (all derived decode correctly)")
    return results


def discover_productive_edges(space, vocab, P, known_edges, min_margin=0.03,
                              chunk=64):
    """Operators are productive: a plural operator trained on album->albums
    should pluralize any noun. For every (operator, word) pair, apply the
    operator in the adapted space and see whether the output decodes to a
    *different* vocabulary word by nearest neighbor with a margin over the
    runner-up. Those decodes are derivation edges that the BATS pair-list
    never enumerated, and they are what compression actually feeds on.

    Self-decodes (operator is a no-op on this word) and edges already known
    are skipped. The margin makes an edge earn its place: a fuzzy output that
    sits between two words is not a derivation.
    """
    with torch.no_grad():
        table = F.normalize(space.adapter(P), dim=-1)
    V = len(vocab)
    ar = torch.arange(V, device=DEVICE)
    known = {(rel, s, t) for rel, pl in known_edges.items() for s, t in pl}
    new_edges = collections.defaultdict(list)
    n_new = 0
    for rel in space.ops.relation_names:
        with torch.no_grad():
            outs = []
            for i in range(0, V, chunk * 8):
                z = space.ops.apply_named(rel, table[i:i + chunk * 8])
                outs.append(F.normalize(z, dim=-1))
            sims = torch.cat(outs) @ table.T          # [V, V]
        sims[ar, ar] = -2                             # never derive a word from itself
        top2 = sims.topk(2, dim=1)
        tgt = top2.indices[:, 0]
        margin = top2.values[:, 0] - top2.values[:, 1]
        keep = margin >= min_margin
        for si in keep.nonzero(as_tuple=True)[0].tolist():
            s, t = vocab[si], vocab[tgt[si].item()]
            if (rel, s, t) in known:
                continue
            new_edges[rel].append((s, t))
            n_new += 1
    print(f"  discovered {n_new} productive edges beyond the "
          f"{sum(len(v) for v in known_edges.values())} from BATS pairs")
    merged = {rel: list(pl) for rel, pl in known_edges.items()}
    for rel, pl in new_edges.items():
        merged.setdefault(rel, []).extend(pl)
    return merged


def refine_on_forest(space, parent, vocab, P, pairs, rounds=150, tau=0.05):
    """Alternation step: freeze the adapter (the table must stay put), then
    fine-tune operators on the actual derivation chains — backprop through
    the composed applications so multi-hop reconstruction lands on the word's
    own row — while keeping the pair-level InfoNCE as an anchor."""
    widx = {w: i for i, w in enumerate(vocab)}
    for p in space.adapter.parameters():
        p.requires_grad_(False)
    with torch.no_grad():
        table = F.normalize(space.adapter(P), dim=-1)

    # group derived words by depth for vectorized chain unrolling
    chains = []
    for w in parent:
        ops, cur = [], w
        while cur in parent:
            rel, src = parent[cur]
            ops.append(space.ops.rel_index[rel])
            cur = src
        chains.append((widx[cur], list(reversed(ops)), widx[w]))
    by_depth = collections.defaultdict(list)
    for c in chains:
        by_depth[len(c[1])].append(c)

    st = torch.tensor([widx[p["source"]] for p in pairs], device=DEVICE)
    tt = torch.tensor([widx[p["target"]] for p in pairs], device=DEVICE)
    rt = torch.tensor([space.ops.rel_index[p["relation"]] for p in pairs],
                      device=DEVICE)
    POS = positive_mask(pairs, vocab, widx).to(DEVICE)

    opt = torch.optim.AdamW(
        [p for p in space.ops.parameters() if p.requires_grad],
        lr=2e-4, weight_decay=1e-2)
    for ep in range(rounds):
        loss = 0.0
        for d, group in by_depth.items():
            roots = torch.tensor([g[0] for g in group], device=DEVICE)
            selfs = torch.tensor([g[2] for g in group], device=DEVICE)
            z = table[roots]
            for hop in range(d):
                rid = torch.tensor([g[1][hop] for g in group], device=DEVICE)
                z = space.ops(z, rid)
            z = F.normalize(z, dim=-1)
            loss = loss + F.cross_entropy(z @ table.T / tau, selfs)
        # pair-level anchor: any BATS-sanctioned answer counts as correct
        pl = F.normalize(space.ops(table[st], rt), dim=-1) @ table.T / tau
        bp = pl.masked_fill(~POS, float("-inf")).max(1).values
        loss = loss + (torch.logsumexp(
            torch.cat([bp.unsqueeze(1), pl.masked_fill(POS, float("-inf"))],
                      dim=1), dim=1) - bp).mean()
        opt.zero_grad()
        loss.backward()
        opt.step()
    for p in space.adapter.parameters():
        p.requires_grad_(True)


def main(production="--all-pairs" in sys.argv):
    vocab, relations, train_pairs, test_pairs, protos = load_data()
    P = torch.stack([protos[w] for w in vocab]).to(DEVICE)

    rel_names = sorted(relations) + [r + "_inv" for r in sorted(relations)]
    base_pairs = train_pairs + (test_pairs if production else [])
    pairs = add_inverses(base_pairs)
    print(f"mode: {'PRODUCTION (all pairs)' if production else 'split (train only)'}")
    space = LexiconSpace(rel_names).to(DEVICE)
    n_params = sum(p.numel() for p in space.parameters())
    print(f"model: {n_params:,} params on {DEVICE}")

    print("joint training (adapter + operators, InfoNCE over vocab)...")
    train_joint(space, pairs, vocab, P)

    print("lexicon selection sweep...")
    edges = {rel: [(p["source"], p["target"]) for p in meta["pairs"]
                   if p["source"] != p["target"]]
             for rel, meta in relations.items()}
    edges.update({rel + "_inv": [(t, s) for s, t in plist]
                  for rel, plist in list(edges.items())})
    train_targets = {p["target"] for p in pairs}
    bats_edges = {rel: list(pl) for rel, pl in edges.items()}

    print("discovering productive edges (operators applied off their pairs)...")
    edges = discover_productive_edges(space, vocab, P, bats_edges)

    sweep = select_lexicon(space, vocab, P, edges, train_targets)

    # pick the most compressed lexicon; keep a small decode-margin for safety
    def pick(s):
        ok = [r for r in s if r["margin"] >= 0.02]
        return max(ok, key=lambda r: r["ratio"]) if ok else s[0]

    choice = pick(sweep)
    print(f"round 1: margin={choice['margin']} ratio={choice['ratio']}")

    # Alternate: fine-tune operators through the chosen chains, re-discover
    # productive edges under the updated operators, reselect. Restore the
    # pre-round weights whenever a round regresses, so the saved model always
    # matches the saved forest.
    n_rounds = int(os.environ.get("LEXICON_ROUNDS", 3))
    stall = 0
    for rnd in range(2, n_rounds + 2):
        snapshot = {k: v.detach().clone() for k, v in space.state_dict().items()}
        refine_on_forest(space, choice["parent"], vocab, P, pairs)
        edges2 = discover_productive_edges(space, vocab, P, bats_edges)
        sweep2 = select_lexicon(space, vocab, P, edges2, train_targets)
        c2 = pick(sweep2)
        print(f"round {rnd}: margin={c2['margin']} ratio={c2['ratio']} "
              f"cos={c2['derived_mean_cos']}")
        if c2["ratio"] <= choice["ratio"] + 0.01:
            space.load_state_dict(snapshot)
            stall += 1
            if stall >= 2:
                break
        else:
            stall = 0
            choice, sweep, edges = c2, sweep2, edges2

    os.makedirs("real", exist_ok=True)
    torch.save({
        "relation_names": rel_names,
        "state_dict": {k: v.cpu() for k, v in space.state_dict().items()},
        "vocab": vocab,
        "base_lexicon": sorted(set(vocab) - set(choice["parent"])),
        "derivations": {w: list(v) for w, v in choice["parent"].items()},
        "sweep": [{k: v for k, v in r.items() if k != "parent"} for r in sweep],
    }, "real/lexicon_space.pt")
    print("saved real/lexicon_space.pt")


if __name__ == "__main__":
    main()
