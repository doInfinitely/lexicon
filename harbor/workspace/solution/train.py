"""Train operators, select the base lexicon, save the checkpoint.

Strategy (the alternating joint optimization the task demands):
  1. Compute frozen distilbert reference embeddings for all vocab words.
  2. Train the conditional operator network on the training pairs
     (early-stopped on a carved-out validation subset).
  3. Iteratively select the base lexicon: start with all words base, build a
     derivation forest by greedily removing words whose embedding the
     current operators reconstruct well from another word, avoiding cycles
     and capping derivation depth. Then re-measure exact composed
     reconstruction quality in topological order and revert the worst words
     back to base until the training-split mean cosine is safely above
     threshold. Repeat the prune/repair alternation until stable.
"""
import json, os, sys, collections
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(__file__))
from common import compute_reference_embeddings
from operators import OperatorNet

DATA = os.path.join(os.path.dirname(__file__), "..", "data")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

MAX_DEPTH = 3            # derivation chains at most this long
PRUNE_COS = 0.85         # hint-probe pruning tolerance
TRAIN_MEAN_TARGET = 0.84 # safety margin over the 0.80 gate
FLOOR_COS = 0.60         # revert any derived word reconstructed worse than this
COMPRESS_TARGET = 2.5


def cos(a, b):
    return F.cosine_similarity(a, b, dim=-1)


def train_operators(net, pairs, ref, epochs=400, batch_size=128, val_frac=0.1):
    g = torch.Generator().manual_seed(0)
    perm = torch.randperm(len(pairs), generator=g).tolist()
    n_val = int(len(pairs) * val_frac)
    val_idx, tr_idx = perm[:n_val], perm[n_val:]

    def tensors(idx):
        x = torch.stack([ref[pairs[i]["source"]] for i in idx]).to(DEVICE)
        y = torch.stack([ref[pairs[i]["target"]] for i in idx]).to(DEVICE)
        r = torch.tensor([net.rel_index[pairs[i]["relation"]] for i in idx],
                         device=DEVICE)
        return x, y, r

    xt, yt, rt = tensors(tr_idx)
    xv, yv, rv = tensors(val_idx)

    opt = torch.optim.AdamW(net.parameters(), lr=1e-3, weight_decay=1e-2)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    best_val, best_state, patience = -1.0, None, 0
    for ep in range(epochs):
        net.train()
        order = torch.randperm(len(tr_idx), device=DEVICE)
        for i in range(0, len(order), batch_size):
            b = order[i:i + batch_size]
            loss = (1 - cos(net(xt[b], rt[b]), yt[b])).mean()
            opt.zero_grad()
            loss.backward()
            opt.step()
        sched.step()
        net.eval()
        with torch.no_grad():
            v = cos(net(xv, rv), yv).mean().item()
        if v > best_val:
            best_val, patience = v, 0
            best_state = {k: t.detach().clone() for k, t in net.state_dict().items()}
        else:
            patience += 1
            if patience >= 40:
                break
        if ep % 25 == 0:
            print(f"  epoch {ep:3d}  val cosine {v:.4f}  (best {best_val:.4f})")
    net.load_state_dict(best_state)
    print(f"  operator training done: best val cosine {best_val:.4f}")
    return best_val


def build_forest(net, ref, vocab, edges, train_targets):
    """Greedy derivation-forest selection with exact-quality repair."""
    words = set(vocab)
    R = {w: ref[w].to(DEVICE) for w in words}

    # score every candidate edge with the trained operators
    scored = []
    with torch.no_grad():
        for rel, plist in edges.items():
            if not plist:
                continue
            x = torch.stack([R[s] for s, t in plist])
            y = torch.stack([R[t] for s, t in plist])
            q = cos(net.apply_named(rel, x), y).tolist()
            scored += [(qi, rel, s, t) for qi, (s, t) in zip(q, plist)]
    scored.sort(reverse=True)

    parent = {}   # derived word -> (relation, source)
    depth = collections.defaultdict(int)

    def creates_cycle(t, s):
        while s in parent:
            s = parent[s][1]
            if s == t:
                return True
        return False

    def chain_depth(s):
        d = 0
        while s in parent:
            s = parent[s][1]
            d += 1
        return d

    for q, rel, s, t in scored:
        if q < PRUNE_COS:
            break
        if t in parent or s == t or creates_cycle(t, s):
            continue
        if chain_depth(s) + 1 > MAX_DEPTH:
            continue
        parent[t] = (rel, s)

    # second pass: fill toward the compression target with lower-quality edges
    n_total = len(words)
    for q, rel, s, t in scored:
        if n_total / (n_total - len(parent)) >= COMPRESS_TARGET * 1.15:
            break
        if q < FLOOR_COS:
            break
        if t in parent or s == t or creates_cycle(t, s):
            continue
        if chain_depth(s) + 1 > MAX_DEPTH:
            continue
        parent[t] = (rel, s)

    # exact composed-quality measurement + repair loop
    def exact_quality():
        memo = {}
        def emb(w):
            if w not in parent:
                return R[w]
            if w not in memo:
                rel, s = parent[w]
                with torch.no_grad():
                    memo[w] = net.apply_named(rel, emb(s))
            return memo[w]
        return {w: cos(emb(w), R[w]).item() for w in parent}

    while True:
        qual = exact_quality()
        derived_train = [w for w in qual if w in train_targets]
        mean_train = (sum(qual[w] for w in derived_train) / len(derived_train)
                      if derived_train else 1.0)
        ratio = n_total / (n_total - len(parent))
        bad = [w for w, v in qual.items() if v < FLOOR_COS]
        print(f"  lexicon pass: base={n_total - len(parent)} ratio={ratio:.2f} "
              f"train-mean={mean_train:.4f} below-floor={len(bad)}")
        if not bad and mean_train >= TRAIN_MEAN_TARGET:
            break
        # revert offenders (and worst words if the mean is low) to base
        revert = set(bad)
        if mean_train < TRAIN_MEAN_TARGET:
            worst = sorted(derived_train, key=qual.get)
            revert |= set(worst[:max(1, len(worst) // 20)])
        if not revert:
            break
        for w in revert:
            del parent[w]
        # a reverted word's children now chain from base again - fine, but
        # if reverting pushed us below the compression gate, stop reverting
        if n_total / (n_total - len(parent)) < COMPRESS_TARGET:
            print("  warning: reverting would break compression gate; stopping")
            break
    return parent


def train(checkpoint_path):
    vocab = json.load(open(os.path.join(DATA, "vocab.json")))
    relations = json.load(open(os.path.join(DATA, "relations.json")))
    train_pairs = json.load(open(os.path.join(DATA, "train_pairs.json")))

    print("computing reference embeddings...")
    ref = compute_reference_embeddings(vocab, device=DEVICE)

    # each relation also gets a learned inverse operator (e.g. plural_inv =
    # singularize): any word on either side of a pair becomes derivable,
    # which is what lets compression climb past the target-only ceiling
    rel_names = sorted(relations.keys()) + [r + "_inv" for r in sorted(relations)]
    train_pairs = train_pairs + [
        {"relation": p["relation"] + "_inv",
         "source": p["target"], "target": p["source"]}
        for p in train_pairs]

    net = OperatorNet(rel_names).to(DEVICE)
    n_params = sum(p.numel() for p in net.parameters())
    print(f"operator net: {n_params:,} params, device={DEVICE}")

    print("training operators on train split...")
    train_operators(net, train_pairs, ref)

    print("selecting base lexicon...")
    edges = {rel: [(p["source"], p["target"]) for p in meta["pairs"]
                   if p["source"] != p["target"]]
             for rel, meta in relations.items()}
    edges.update({rel + "_inv": [(t, s) for s, t in plist]
                  for rel, plist in list(edges.items())})
    train_targets = {p["target"] for p in train_pairs}
    parent = build_forest(net, ref, vocab, edges, train_targets)

    base = sorted(set(vocab) - set(parent))
    print(f"final: base={len(base)} derived={len(parent)} "
          f"ratio={len(vocab)/len(base):.2f}")

    torch.save({
        "relation_names": net.relation_names,
        "state_dict": {k: v.cpu() for k, v in net.state_dict().items()},
        "base_lexicon": base,
        "derivations": {w: list(parent[w]) for w in parent},
        "vocab": vocab,
    }, checkpoint_path)
    print(f"saved {checkpoint_path}")


if __name__ == "__main__":
    train(sys.argv[1] if len(sys.argv) > 1 else
          os.path.join(os.path.dirname(__file__), "checkpoint.pt"))
