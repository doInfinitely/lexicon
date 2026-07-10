"""Controls. Are the branches and the virtual words structure, or artifacts?

Two claims are on trial.

CLAIM 1: "branches of a one-to-many inverse discover coherent subtrees."
  Null: a random partition of the same pairs into the same branch sizes.
  Statistic: how predictable is the branch from the SOURCE word (a color, a
  hypernym)? If branch #0 were 'the birds', knowing the source is 'fowl'
  would tell you the branch. Measured as normalized mutual information
  between branch label and source word, and as target-cluster purity.

CLAIM 2: "an anchor far from every real word is an unlexicalized concept."
  Null: how far is a REAL word from its nearest OTHER real word? If real
  words are themselves ~0.5 away from their neighbours, then an anchor at
  0.5 is unremarkable and the phrase 'no word for this' is empty.
"""
import json, collections, math
import numpy as np
import torch
import torch.nn.functional as F
from sklearn.cluster import KMeans
from sklearn.metrics import normalized_mutual_info_score

from lexicon.model import Adapter
from lexicon.discover import BranchedOps

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
RNG = np.random.default_rng(0)


def load():
    ck = torch.load("real/branched.pt", weights_only=False)
    vocab = ck["vocab"]
    adapter = Adapter().to(DEVICE); adapter.load_state_dict(ck["adapter"]); adapter.eval()
    ops = BranchedOps(ck["relation_names"], ck["k"]).to(DEVICE)
    ops.load_state_dict(ck["ops"]); ops.eval()
    protos = torch.load("real/embeddings/prototypes.pt", weights_only=False)
    P = torch.stack([protos[w] for w in vocab]).to(DEVICE)
    with torch.no_grad():
        table = F.normalize(adapter(P), dim=-1)
    return ck, adapter, ops, vocab, table


def branch_coherence(table, widx):
    br = json.load(open("real/branches.json"))
    det = json.load(open("real/algebra.json"))["determinism"]
    inv_rels = [r + "_inv" for r in sorted(det, key=lambda r: -det[r]["fan_in"])[:8]]

    print("=" * 100)
    print("CONTROL 1: do inverse branches carve coherent subtrees, or an "
          "arbitrary enumeration?")
    print("=" * 100)
    print(f"{'relation':<32}{'NMI(branch;source)':>20}{'null':>10}"
          f"{'target purity':>16}{'null':>10}")

    for r in inv_rels:
        if r not in br:
            continue
        pairs, labels = [], []
        for k, pl in br[r].items():
            for s, t in pl:
                pairs.append((s, t)); labels.append(int(k))
        labels = np.array(labels)
        srcs = np.array([s for s, _ in pairs])

        # --- NMI between branch and source word ---
        _, src_id = np.unique(srcs, return_inverse=True)
        nmi = normalized_mutual_info_score(src_id, labels)
        # null: shuffle labels, keeping branch sizes
        nulls = [normalized_mutual_info_score(src_id, RNG.permutation(labels))
                 for _ in range(200)]
        nmi_null = float(np.mean(nulls))

        # --- purity of branches w.r.t. semantic clusters of the TARGETS ---
        T = torch.stack([table[widx[t]] for _, t in pairs]).cpu().numpy()
        k = len(set(labels))
        if len(T) > k * 2:
            cl = KMeans(n_clusters=k, n_init=5, random_state=0).fit_predict(T)
            purity = sum(collections.Counter(
                cl[labels == b]).most_common(1)[0][1] for b in set(labels)) / len(labels)
            pn = []
            for _ in range(200):
                perm = RNG.permutation(labels)
                pn.append(sum(collections.Counter(
                    cl[perm == b]).most_common(1)[0][1] for b in set(labels)) / len(labels))
            purity_null = float(np.mean(pn))
        else:
            purity = purity_null = float("nan")
        print(f"{r:<32}{nmi:>20.3f}{nmi_null:>10.3f}{purity:>16.3f}{purity_null:>10.3f}")
    print("\nIf NMI ~= null, the branch tells you nothing about which source it "
          "came from:\nthe branches are an INDEX into a set, not a taxonomy.")


@torch.no_grad()
def anchor_distance_null(table, vocab, ms=(16, 32, 64, 128, 256)):
    print("\n" + "=" * 100)
    print("CONTROL 2: is an anchor's 'distance to language' actually unusual?")
    print("=" * 100)
    # null: each real word's distance to its nearest OTHER real word
    sims = table @ table.T
    sims.fill_diagonal_(-2)
    nn_sim = sims.max(1).values
    nn_dist = (1 - nn_sim).cpu().numpy()
    print(f"real words: distance to nearest OTHER real word  "
          f"mean {nn_dist.mean():.3f}  median {np.median(nn_dist):.3f}  "
          f"p90 {np.percentile(nn_dist, 90):.3f}  max {nn_dist.max():.3f}")

    for m in ms:
        try:
            v = torch.load(f"real/virtual_{m}.pt", weights_only=False)
        except FileNotFoundError:
            continue
        A = F.normalize(v["anchors"].to(DEVICE), dim=-1)
        ad = (1 - (A @ table.T).max(1).values).cpu().numpy()
        # what fraction of anchors are further from language than the median
        # real word is from its neighbour?
        frac = float((ad > np.median(nn_dist)).mean())
        pct = float((ad.mean() > nn_dist).mean())
        print(f"anchors M={m:>3}: mean {ad.mean():.3f}  median {np.median(ad):.3f} "
              f" | {frac:.0%} of anchors are further from language than the "
              f"median real word is from its neighbour "
              f"(mean anchor sits at the {pct:.0%}th pct of real words)")
    print("\nIf anchors are no further from language than real words are from "
          "each other,\nthen 'a concept with no word' is not established by "
          "distance alone.")


@torch.no_grad()
def anchor_generation_coherence(ops, table, vocab, m=64):
    """A meaningful anchor should generate a semantically coherent family.
    Compare the mean pairwise similarity of the words an anchor generates
    against random word sets of the same size."""
    print("\n" + "=" * 100)
    print(f"CONTROL 3: are the words an anchor generates a coherent family? (M={m})")
    print("=" * 100)
    v = torch.load(f"real/virtual_{m}.pt", weights_only=False)
    prov = v["provenance"]
    gen = collections.defaultdict(list)
    for w_idx, (j, rel, k) in prov.items():
        gen[j].append(w_idx)

    obs, nul = [], []
    for j, idxs in gen.items():
        if len(idxs) < 3:
            continue
        E = table[idxs]
        s = (E @ E.T)
        n = len(idxs)
        obs.append(((s.sum() - n) / (n * (n - 1))).item())
        r = torch.tensor(RNG.choice(len(table), size=n, replace=False), device=DEVICE)
        Er = table[r]
        sr = Er @ Er.T
        nul.append(((sr.sum() - n) / (n * (n - 1))).item())
    obs, nul = np.array(obs), np.array(nul)
    print(f"mean pairwise cosine within an anchor's generated set : {obs.mean():.3f}")
    print(f"same, for random word sets of equal size              : {nul.mean():.3f}")
    print(f"anchors more coherent than their random null          : "
          f"{(obs > nul).mean():.0%} of {len(obs)} anchors")
    # a strong baseline: coherence of a real word's k nearest neighbours
    k = int(np.mean([len(v) for v in gen.values() if len(v) >= 3]))
    idx = torch.tensor(RNG.choice(len(table), size=64, replace=False), device=DEVICE)
    knn = (table[idx] @ table.T).topk(k, dim=1).indices
    kn = []
    for row in knn:
        E = table[row]; s = E @ E.T
        kn.append(((s.sum() - k) / (k * (k - 1))).item())
    print(f"for reference, a real word's {k} nearest neighbours    : "
          f"{np.mean(kn):.3f}  <- what genuine semantic coherence looks like")


def main():
    ck, adapter, ops, vocab, table = load()
    widx = {w: i for i, w in enumerate(vocab)}
    branch_coherence(table, widx)
    anchor_distance_null(table, vocab)
    anchor_generation_coherence(ops, table, vocab, 64)


if __name__ == "__main__":
    main()
