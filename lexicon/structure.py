"""What actually sets the size of the base lexicon?

A derivation forest is a *branching*: every derived word has exactly one
parent, and there are no cycles. So

    |base| = |V| - |edges in the branching|

and minimizing the base lexicon means maximizing the branching. Ignoring
depth, the minimum possible number of roots is exactly the number of SOURCE
COMPONENTS in the condensation of the reliable-edge digraph (each source
component must contain at least one root, and one per source component is
achievable). This is computable in linear time -- no search required.

The point of this module is to show that reachability is *not* the binding
constraint. If the reliable-edge graph is strongly connected, the
unconstrained optimum is a single base word, which would obviously
reconstruct nothing: the error compounds along the chain. What actually sets
|base| is the DEPTH BUDGET. Under a depth cap the problem becomes
bounded-depth minimum branching, which is NP-hard, so we approximate.

Run: python -m lexicon.structure [checkpoint]
"""
import sys, collections
import numpy as np
import torch
import torch.nn.functional as F
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components

from lexicon.model import LexiconSpace

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def reliable_edges(space, vocab, P, margin, chunk=2048):
    """u -> v when some operator applied to u decodes to v with margin."""
    with torch.no_grad():
        table = F.normalize(space.adapter(P), dim=-1)
    V = len(vocab)
    ar = torch.arange(V, device=table.device)
    src, dst = [], []
    for rel in space.ops.relation_names:
        with torch.no_grad():
            outs = [F.normalize(space.ops.apply_named(rel, table[i:i + chunk]),
                                dim=-1) for i in range(0, V, chunk)]
            sims = torch.cat(outs) @ table.T
        sims[ar, ar] = -2
        top2 = sims.topk(2, dim=1)
        keep = (top2.values[:, 0] - top2.values[:, 1]) >= margin
        idx = keep.nonzero(as_tuple=True)[0]
        src.append(idx)
        dst.append(top2.indices[idx, 0])
    return torch.cat(src).cpu().numpy(), torch.cat(dst).cpu().numpy(), table


def min_roots_unconstrained(src, dst, n):
    """Number of source components in the condensation = exact minimum roots
    for a branching with no depth cap."""
    g = csr_matrix((np.ones(len(src)), (src, dst)), shape=(n, n))
    ncomp, labels = connected_components(g, directed=True, connection="strong")
    has_incoming = np.zeros(ncomp, dtype=bool)
    for u, v in zip(src, dst):
        if labels[u] != labels[v]:
            has_incoming[labels[v]] = True
    sources = int((~has_incoming).sum())
    sizes = collections.Counter(labels.tolist())
    return sources, ncomp, max(sizes.values())


def greedy_bounded_depth(src, dst, qual, n, max_depth):
    """Greedy maximum branching subject to a depth cap (the NP-hard part).
    Edges are considered best-first; the depth invariant is maintained
    exactly by tracking each node's subtree height."""
    order = np.argsort(-qual)
    parent = {}
    children = collections.defaultdict(set)
    depth = collections.defaultdict(int)
    height = collections.defaultdict(int)

    def is_ancestor(a, w):
        while w in parent:
            w = parent[w]
            if w == a:
                return True
        return False

    for e in order:
        u, v = int(src[e]), int(dst[e])
        if v in parent or u == v or is_ancestor(v, u):
            continue
        if depth[u] + 1 + height[v] > max_depth:
            continue
        parent[v] = u
        children[u].add(v)
        stack = [(v, depth[u] + 1)]
        while stack:
            w, d = stack.pop()
            depth[w] = d
            stack.extend((c, d + 1) for c in children[w])
        w, h = v, height[v]
        while w in parent:
            w = parent[w]
            h += 1
            if height[w] >= h:
                break
            height[w] = h
    return n - len(parent)


def main(ckpt_path="real/lexicon_space_production.pt"):
    ckpt = torch.load(ckpt_path, weights_only=False)
    vocab = ckpt["vocab"]
    space = LexiconSpace(ckpt["relation_names"]).to(DEVICE)
    space.load_state_dict(ckpt["state_dict"])
    space.eval()
    protos = torch.load("real/embeddings/prototypes.pt", weights_only=False)
    P = torch.stack([protos[w] for w in vocab]).to(DEVICE)
    n = len(vocab)

    print(f"{'margin':>7} {'edges':>8} {'SCCs':>6} {'largest':>8} "
          f"{'min roots':>10} {'ceiling':>9} | greedy base @ depth 2/3/6/inf")
    for margin in (0.0, 0.02, 0.05, 0.10, 0.20):
        src, dst, table = reliable_edges(space, vocab, P, margin)
        sources, ncomp, largest = min_roots_unconstrained(src, dst, n)
        # edge quality = decode similarity, used to order the greedy branching
        with torch.no_grad():
            q = (table[src] * table[dst]).sum(-1).cpu().numpy()
        greedy = [greedy_bounded_depth(src, dst, q, n, d) for d in (2, 3, 6, n)]
        ceil = f"{n/sources:.1f}x" if sources else "inf"
        print(f"{margin:>7.2f} {len(src):>8} {ncomp:>6} {largest:>8} "
              f"{sources:>10} {ceil:>9} | " +
              "  ".join(f"{b}({n/b:.2f}x)" for b in greedy))

    print("\nReading: 'min roots' is the exact optimum with NO depth cap "
          "(source components of the condensation).\n"
          "The gap between that and the greedy columns is the price of the "
          "depth budget -- i.e. of compounding error, not of coverage.")


if __name__ == "__main__":
    main(*sys.argv[1:])
