"""Does a word have several positions, or is it just a cloud you may pick from?

The previous null was wrong. Choosing the best of K x K sense pairs can only
raise a score, so the comparison must hold the FREEDOM OF CHOICE fixed and
vary only whether the K points are meaningful:

  single      one mean vector per word
  senses      K sense centroids (k-means on the word's contextual occurrences)
  cloud null  K occurrences drawn at random from the same word's cloud
  jitter null K samples of the mean + gaussian noise matched to cloud spread

If `senses` does not beat `cloud null` at equal K, then "a word is several
points" buys nothing over "a word is a fuzzy blob and we let you pick."

Second test, the one that matters for antonymy: after choosing the best sense
pair, does the DISPLACEMENT DIRECTION become consistent across pairs? A
relation is "a direction" only if its displacements align. Global antonymy
has alignment ~0.68. If sense selection raises that sharply, antonymy is a
LOCAL direction that global mean-pooling destroys.
"""
import json, collections
import numpy as np
import torch
import torch.nn.functional as F

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
RNG = np.random.default_rng(0)
FOCUS = ["L10_antonyms_binary", "L09_antonyms_gradable", "L08_synonyms_exact",
         "L07_synonyms_intensity", "L01_hypernyms_animals", "L02_hypernyms_misc",
         "I01_noun_plural_reg", "I06_verb_inf_Ving", "E01_country_capital",
         "E10_male_female"]


def load():
    rels = json.load(open("harbor/workspace/data/relations.json"))
    protos = torch.load("real/embeddings/prototypes.pt", weights_only=False)
    senses = torch.load("real/embeddings/senses.pt", weights_only=False)
    occ = torch.load("real/embeddings/occurrences.pt", weights_only=False)
    P = {w: F.normalize(v, dim=-1).to(DEVICE) for w, v in protos.items()}
    S = {w: F.normalize(torch.stack(v), dim=-1).to(DEVICE) for w, v in senses.items()}
    O = {w: F.normalize(v, dim=-1).to(DEVICE) for w, v in occ.items()}
    pairs = {r: [(p["source"], p["target"]) for p in m["pairs"]
                 if p["source"] != p["target"]] for r, m in rels.items()}
    return pairs, P, S, O


def split(pl, frac=0.3, seed=0):
    idx = np.random.default_rng(seed).permutation(len(pl))
    k = int(len(pl) * frac)
    return [pl[i] for i in idx[k:]], [pl[i] for i in idx[:k]]


def best_pair_score(reps_a, reps_b, d):
    """max over sense pairs of cos(a + d, b); also return the argmax pair."""
    pred = F.normalize(reps_a + d, dim=-1)               # [Ka,768]
    sims = pred @ F.normalize(reps_b, dim=-1).T          # [Ka,Kb]
    i = int(sims.argmax())
    ia, ib = divmod(i, sims.shape[1])
    return sims.max().item(), ia, ib


def cloud_sample(O, w, k, P):
    """k random occurrences of w (its own cloud), or the mean if too few."""
    c = O.get(w)
    if c is None or len(c) < k:
        return P[w].unsqueeze(0).repeat(k, 1)
    idx = RNG.choice(len(c), size=k, replace=False)
    return c[torch.tensor(idx, device=DEVICE)]


def jitter_sample(O, w, k, P):
    c = O.get(w)
    sd = c.std(0).mean().item() if c is not None and len(c) > 2 else 0.05
    return F.normalize(P[w].unsqueeze(0).repeat(k, 1)
                       + sd * torch.randn(k, P[w].shape[0], device=DEVICE), dim=-1)


def random_partition_centroids(O, w, k, P):
    """THE fair null for sense structure. Sense centroids average many
    occurrences, so they are denoised; a single random occurrence is not.
    This null splits the SAME cloud into k random groups and averages each,
    giving identical denoising and identical freedom of choice, but with no
    semantic clustering. If real senses do not beat this, then k-means found
    convenient variance, not distinct meanings."""
    c = O.get(w)
    if c is None or len(c) < 2 * k:
        return P[w].unsqueeze(0).repeat(k, 1)
    idx = RNG.permutation(len(c))
    groups = np.array_split(idx, k)
    return F.normalize(torch.stack(
        [c[torch.tensor(g, device=DEVICE)].mean(0) for g in groups]), dim=-1)


def main():
    pairs, P, S, O = load()
    print("=" * 108)
    print("DOES A WORD HAVE SEVERAL POSITIONS?  (held-out cosine; all methods "
          "get the SAME K x K freedom of choice)")
    print("=" * 108)
    print(f"{'relation':<26}{'K':>5}{'single':>9}{'senses':>9}{'randpart':>10}"
          f"{'cloud':>9}{'senses-randpart':>17}")

    summary = {}
    for r in FOCUS:
        pl = pairs[r]
        tr, te = split(pl)
        d = torch.stack([P[b] - P[a] for a, b in tr]).mean(0)
        K = int(round(np.mean([len(S[a]) for a, _ in te] +
                              [len(S[b]) for _, b in te])))
        K = max(K, 2)

        sing = np.mean([F.cosine_similarity(
            F.normalize(P[a] + d, dim=-1), P[b], dim=0).item() for a, b in te])
        sens = np.mean([best_pair_score(S[a], S[b], d)[0] for a, b in te])
        # nulls: repeat draws to average out sampling luck
        cl, rp = [], []
        for _ in range(5):
            cl.append(np.mean([best_pair_score(cloud_sample(O, a, K, P),
                                               cloud_sample(O, b, K, P), d)[0]
                               for a, b in te]))
            rp.append(np.mean([best_pair_score(random_partition_centroids(O, a, K, P),
                                               random_partition_centroids(O, b, K, P), d)[0]
                               for a, b in te]))
        cloud, randpart = float(np.mean(cl)), float(np.mean(rp))
        summary[r] = dict(single=sing, senses=sens, cloud=cloud, randpart=randpart)
        print(f"{r:<26}{K:>5}{sing:>9.3f}{sens:>9.3f}{randpart:>10.3f}"
              f"{cloud:>9.3f}{sens-randpart:>17.3f}")

    print("\n'randpart' = centroids of a RANDOM split of the same cloud: identical")
    print("denoising, identical freedom of choice, no semantic clustering.")
    print("'senses - randpart' is the honest value of a word having several")
    print("MEANINGS as opposed to merely having variance.\n")

    # ---- the antonymy question: is it a LOCAL direction? ----
    print("=" * 108)
    print("IS THE RELATION A DIRECTION, ONCE YOU PICK THE RIGHT SENSES?")
    print("displacement alignment = mean cos(individual displacement, mean "
          "displacement). 1.0 = a true direction.")
    print("=" * 108)
    print(f"{'relation':<26}{'align (mean pts)':>19}{'align (best senses)':>22}"
          f"{'align (best cloud)':>21}")
    for r in FOCUS:
        pl = pairs[r]
        tr, _ = split(pl)
        d = torch.stack([P[b] - P[a] for a, b in tr]).mean(0)
        K = max(2, int(round(np.mean([len(S[a]) for a, _ in tr]))))

        def align(disp):
            D = torch.stack(disp)
            mu = D.mean(0, keepdim=True)
            return F.cosine_similarity(D, mu, dim=-1).mean().item()

        a_mean = align([P[b] - P[a] for a, b in tr])
        ds = []
        for a, b in tr:
            _, ia, ib = best_pair_score(S[a], S[b], d)
            ds.append(S[b][ib] - S[a][ia])
        a_sense = align(ds)
        dc = []
        for a, b in tr:
            ca, cb = cloud_sample(O, a, K, P), cloud_sample(O, b, K, P)
            _, ia, ib = best_pair_score(ca, cb, d)
            dc.append(cb[ib] - ca[ia])
        a_cloud = align(dc)
        print(f"{r:<26}{a_mean:>19.3f}{a_sense:>22.3f}{a_cloud:>21.3f}")

    print("\nIf 'best senses' alignment >> 'mean pts' AND >> 'best cloud',")
    print("the relation is a direction between SENSES that mean-pooling destroys.")
    json.dump(summary, open("real/multipoint.json", "w"), indent=1)


if __name__ == "__main__":
    main()
