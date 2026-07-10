"""Held-out evaluation of the real system.

Headline metric: retrieval — apply the correct operator to the source
embedding and decode by nearest neighbor over the full vocab; did we get the
target word? Reported for the adapted space vs. two baselines in the frozen
space (identity operator, per-relation mean offset). Also runs the
temperature decoder demo and a homograph sense report.
"""
import json, torch
import torch.nn.functional as F

from lexicon.model import LexiconSpace
from lexicon.decode import Decoder

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def retrieval(out, table, target_idx, source_idx=None, pos_mask=None):
    """R@1/R@5 against the single listed target ('strict'), and — when a
    positive mask of all BATS-sanctioned answers is given — against any valid
    answer ('any'), which is the standard BATS scoring convention."""
    sims = out @ table.T
    if source_idx is not None:  # standard analogy convention: exclude source
        sims[torch.arange(len(sims)), source_idx] = -2
    top5 = sims.topk(5, dim=1).indices
    r1 = (top5[:, 0] == target_idx).float().mean().item()
    r5 = (top5 == target_idx.unsqueeze(1)).any(1).float().mean().item()
    out_d = {"R@1": round(r1, 4), "R@5": round(r5, 4)}
    if pos_mask is not None:
        rows = torch.arange(len(top5), device=top5.device)
        a1 = pos_mask[rows, top5[:, 0]].float().mean().item()
        a5 = torch.stack([pos_mask[rows, top5[:, k]] for k in range(5)]) \
            .any(0).float().mean().item()
        out_d["R@1_any"] = round(a1, 4)
        out_d["R@5_any"] = round(a5, 4)
    return out_d


def main():
    ckpt = torch.load("real/lexicon_space.pt", weights_only=False)
    vocab = ckpt["vocab"]
    widx = {w: i for i, w in enumerate(vocab)}
    space = LexiconSpace(ckpt["relation_names"]).to(DEVICE)
    space.load_state_dict(ckpt["state_dict"])
    space.eval()

    protos = torch.load("real/embeddings/prototypes.pt", weights_only=False)
    P = torch.stack([protos[w] for w in vocab]).to(DEVICE)
    test_pairs = json.load(open("harbor/tests/test_pairs.json"))
    train_pairs = json.load(open("harbor/workspace/data/train_pairs.json"))

    si = torch.tensor([widx[p["source"]] for p in test_pairs], device=DEVICE)
    ti = torch.tensor([widx[p["target"]] for p in test_pairs], device=DEVICE)
    rels = [p["relation"] for p in test_pairs]

    results = {}

    # mask of every BATS-sanctioned answer per held-out pair
    POS = torch.zeros(len(test_pairs), len(vocab), dtype=torch.bool, device=DEVICE)
    for i, p in enumerate(test_pairs):
        for a in set(p.get("alternates") or []) | {p["target"]}:
            if a in widx:
                POS[i, widx[a]] = True

    # ---- ours: adapted space + learned operators ----
    with torch.no_grad():
        table = F.normalize(space.adapter(P), dim=-1)
        rid = torch.tensor([space.ops.rel_index[r] for r in rels], device=DEVICE)
        out = F.normalize(space.ops(table[si], rid), dim=-1)
    results["ours_adapted"] = {
        **retrieval(out, table, ti, si, POS),
        "cos": round(F.cosine_similarity(out, table[ti]).mean().item(), 4)}

    # per-category breakdown (strict and any-valid-answer)
    top1 = (out @ table.T).scatter(1, si.unsqueeze(1), -2).argmax(1)
    rows = torch.arange(len(top1), device=DEVICE)
    strict_hit, any_hit = (top1 == ti).tolist(), POS[rows, top1].tolist()
    bycat = {}
    for p, sh, ah in zip(test_pairs, strict_hit, any_hit):
        bycat.setdefault(p["relation"][0], []).append((sh, ah))
    results["ours_by_category"] = {
        {"I": "inflectional", "D": "derivational", "E": "encyclopedic",
         "L": "lexicographic"}[k]: {
            "R@1": round(sum(s for s, _ in v) / len(v), 4),
            "R@1_any": round(sum(a for _, a in v) / len(v), 4)}
        for k, v in sorted(bycat.items())}

    # ---- baseline 1: frozen space, identity operator ----
    Pn = F.normalize(P, dim=-1)
    results["frozen_identity"] = retrieval(Pn[si], Pn, ti, si, POS)

    # ---- baseline 2: frozen space, per-relation mean offset (word2vec-style) ----
    offs = {}
    for p in train_pairs:
        offs.setdefault(p["relation"], []).append(
            P[widx[p["target"]]] - P[widx[p["source"]]])
    out_off = F.normalize(torch.stack(
        [P[widx[p["source"]]] + torch.stack(offs[p["relation"]]).mean(0)
         for p in test_pairs]), dim=-1)
    results["frozen_offset"] = retrieval(out_off, Pn, ti, si, POS)

    # ---- lexicon stats ----
    results["lexicon"] = {"vocab": len(vocab),
                          "base": len(ckpt["base_lexicon"]),
                          "ratio": round(len(vocab) / len(ckpt["base_lexicon"]), 3)}
    results["sweep"] = ckpt["sweep"]

    print(json.dumps(results, indent=2))

    # ---- decoder demo ----
    senses = torch.load("real/embeddings/senses.pt", weights_only=False)
    dec = Decoder(space, vocab, protos, senses, device=DEVICE)
    print("\ndecoder demo (operator output -> temperature-sampled words):")
    for p in test_pairs[:3] + test_pairs[300:302]:
        with torch.no_grad():
            v = space.ops.apply_named(p["relation"], table[widx[p["source"]]])
        greedy = dec.decode(v, temperature=0.0, top_k=3)
        sampled = dec.decode(v, temperature=0.05, top_k=3,
                             generator=torch.Generator(device=DEVICE).manual_seed(0))
        print(f"  {p['relation']}({p['source']}) -> want {p['target']}")
        print(f"    T=0 : {[(w, round(s, 3)) for w, k, s in greedy]}")
        print(f"    T=.05:{[(w, round(s, 3)) for w, k, s in sampled]}")


if __name__ == "__main__":
    main()
