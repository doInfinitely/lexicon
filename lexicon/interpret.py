"""Read the discovered structure back out in words.

Two readings:

BRANCHES. For each one-to-many relation, print what each branch of its
inverse claimed. If the branches are real, each is a coherent region of the
taxonomy -- 'the birds', 'the insects' -- rather than an arbitrary split.

VIRTUAL WORDS. An anchor has no name. It is defined only by what the frozen,
semantically-grounded operators make of it. So we describe each anchor two
ways: (1) its nearest real words, and (2) the words it generates and by which
operator. An anchor whose nearest real word is far away is a concept the
language never lexicalized -- the interesting case.
"""
import json, collections
import torch
import torch.nn.functional as F

from lexicon.model import Adapter
from lexicon.discover import BranchedOps, coverage

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def load():
    ck = torch.load("real/branched.pt", weights_only=False)
    vocab = ck["vocab"]
    adapter = Adapter().to(DEVICE)
    adapter.load_state_dict(ck["adapter"])
    ops = BranchedOps(ck["relation_names"], ck["k"]).to(DEVICE)
    ops.load_state_dict(ck["ops"])
    adapter.eval(); ops.eval()
    protos = torch.load("real/embeddings/prototypes.pt", weights_only=False)
    P = torch.stack([protos[w] for w in vocab]).to(DEVICE)
    with torch.no_grad():
        table = F.normalize(adapter(P), dim=-1)
    return ck, adapter, ops, vocab, table


def show_branches(top_relations=6, max_show=9):
    br = json.load(open("real/branches.json"))
    det = json.load(open("real/algebra.json"))["determinism"]
    # the relations whose INVERSE is most one-to-many are the interesting ones
    order = sorted((r for r in det), key=lambda r: -det[r]["fan_in"])[:top_relations]
    print("=" * 96)
    print("BRANCHED INVERSES: how does each fan-out split?")
    print("(relation r has fan-in f, so r_inv is one-to-many; each branch of "
          "r_inv claims a set of pairs)")
    print("=" * 96)
    for r in order:
        inv = r + "_inv"
        if inv not in br:
            continue
        print(f"\n{inv}   (forward fan-in {det[r]['fan_in']:.2f}: "
              f"{det[r]['n_sources']} sources -> {det[r]['gold_distinct']} targets)")
        for k, plist in sorted(br[inv].items()):
            outs = [t for _, t in plist]
            srcs = collections.Counter(s for s, _ in plist)
            hub = ", ".join(w for w, _ in srcs.most_common(2))
            sample = ", ".join(outs[:max_show])
            print(f"   branch #{k}  ({len(plist):>3} pairs)  from [{hub}]")
            print(f"       -> {sample}")


@torch.no_grad()
def show_virtual(m=64, n_show=20):
    ck, adapter, ops, vocab, table = load()
    v = torch.load(f"real/virtual_{m}.pt", weights_only=False)
    A = F.normalize(v["anchors"].to(DEVICE), dim=-1)
    prov = v["provenance"]

    # what each anchor generates
    gen = collections.defaultdict(list)
    for w_idx, (j, rel, k) in prov.items():
        gen[j].append((vocab[w_idx], rel, k))

    sims = A @ table.T
    top = sims.topk(5, dim=1)

    print("\n" + "=" * 96)
    print(f"VIRTUAL WORDS (M={m}): concepts that root the lexicon")
    print("An anchor's 'distance to language' is 1 - cos(anchor, nearest real word).")
    print("High distance = the language has no word for this.")
    print("=" * 96)

    rows = []
    for j in range(len(A)):
        if not gen[j]:
            continue
        near = [vocab[i] for i in top.indices[j].tolist()]
        dist = 1 - top.values[j, 0].item()
        rows.append((dist, j, near, gen[j]))
    rows.sort(reverse=True)

    print(f"\n--- the {min(n_show, len(rows))} most UNLEXICALIZED anchors "
          f"(no good English word) ---")
    for dist, j, near, g in rows[:n_show]:
        gens = ", ".join(f"{w}" for w, _, _ in g[:6])
        rels = collections.Counter(r for _, r, _ in g).most_common(2)
        rel_s = ", ".join(f"{r}" for r, _ in rels)
        print(f"\n  anchor_{j:<3} distance-to-language {dist:.3f}   "
              f"generates {len(g)} words")
        print(f"     nearest real words : {', '.join(near)}")
        print(f"     chief operators    : {rel_s}")
        print(f"     generates          : {gens}")

    print(f"\n--- the {min(6, len(rows))} most LEXICALIZED anchors "
          f"(language has a word for this) ---")
    for dist, j, near, g in rows[-6:]:
        gens = ", ".join(f"{w}" for w, _, _ in g[:6])
        print(f"  anchor_{j:<3} d={dist:.3f}  ~= '{near[0]}'   generates: {gens}")

    covered = len(prov)
    print(f"\ncoverage: {covered}/{len(vocab)} words are ONE operator "
          f"application from a virtual word (depth 1, no chains).")
    print(f"generating set: {m} virtual words + {len(vocab)-covered} "
          f"irreducible real words = {m + len(vocab) - covered}")


def show_irreducible(m=64, n=40):
    """Which real words can no operator produce? These are the primitives."""
    ck, adapter, ops, vocab, table = load()
    v = torch.load(f"real/virtual_{m}.pt", weights_only=False)
    prov = v["provenance"]
    left = [vocab[i] for i in range(len(vocab)) if i not in prov]
    print("\n" + "=" * 96)
    print(f"IRREDUCIBLE WORDS: {len(left)} words no operator reaches from any anchor")
    print("=" * 96)
    print("  " + ", ".join(left[:n]) + (" ..." if len(left) > n else ""))


if __name__ == "__main__":
    import sys
    m = int(sys.argv[1]) if len(sys.argv) > 1 else 64
    show_branches()
    show_virtual(m)
    show_irreducible(m)
