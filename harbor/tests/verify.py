"""Independent verifier for the Lexicon Compression task.

Computes reference embeddings with its OWN frozen distilbert copy, loads the
agent's model through the required interface, and checks every gate:

  structural: base subset of vocab, valid acyclic derivations, >=3 named
              operators, >=1000 params
  compression: N_total / N_base >= 2.5
  train:      mean cos(encode(w), ref(w)) over training-split derived words >= 0.80
  held-out:   mean cos(op_r(encode(s)), ref(t)) over hidden test pairs >= 0.65
"""
import json, os, sys
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
WS = os.path.join(ROOT, "workspace")
T_COMPRESS, T_TRAIN, T_TEST = 2.5, 0.80, 0.65


@torch.no_grad()
def reference_embeddings(words, device):
    tok = AutoTokenizer.from_pretrained("distilbert-base-uncased")
    model = AutoModel.from_pretrained("distilbert-base-uncased").to(device).eval()
    out = {}
    for i in range(0, len(words), 256):
        batch = words[i:i + 256]
        enc = tok(batch, return_tensors="pt", padding=True).to(device)
        hidden = model(**enc).last_hidden_state
        ids, mask = enc["input_ids"], enc["attention_mask"].bool()
        keep = mask & ~((ids == tok.cls_token_id) | (ids == tok.sep_token_id))
        for j, w in enumerate(batch):
            out[w] = hidden[j][keep[j]].mean(dim=0).cpu()
    return out


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    vocab = json.load(open(os.path.join(WS, "data", "vocab.json")))
    train_pairs = json.load(open(os.path.join(WS, "data", "train_pairs.json")))
    test_pairs = json.load(open(os.path.join(ROOT, "tests", "test_pairs.json")))

    sys.path.insert(0, os.path.join(WS, "solution"))
    from model import LexiconCompressor
    lc = LexiconCompressor()

    results, ok = {}, True

    # ---- structural gates ----
    base = lc.get_base_lexicon()
    base_set, vocab_set = set(base), set(vocab)
    assert base_set <= vocab_set, "base lexicon not a subset of vocab"
    assert len(base_set) < len(vocab_set), "no compression at all"

    ops = lc.get_operators()
    results["n_operators"] = len(ops)
    assert len(ops) >= 3, "fewer than 3 named operators"

    n_params = sum(p.numel() for p in lc.net.parameters())
    results["n_params"] = n_params
    assert n_params >= 1000, "model too small"

    # every non-base word must have a valid acyclic derivation
    derived = [w for w in vocab if w not in base_set]
    for w in derived:
        expr, _ = lc.reconstruct(w)
        seen, cur = set(), w
        while cur not in base_set:
            assert cur not in seen, f"circular derivation at {w}"
            seen.add(cur)
            rel, src = lc.derivations[cur]
            assert rel in ops, f"unknown operator {rel}"
            cur = src
        assert cur in base_set

    # ---- compression gate ----
    ratio = len(vocab) / len(base)
    results["compression_ratio"] = round(ratio, 3)
    ok &= ratio >= T_COMPRESS

    # ---- reference embeddings (independent, frozen) ----
    ref = reference_embeddings(vocab, device)

    # ---- train reconstruction gate ----
    train_targets = {p["target"] for p in train_pairs}
    train_derived = [w for w in derived if w in train_targets]
    sims = [F.cosine_similarity(lc.encode(w), ref[w], dim=0).item()
            for w in train_derived]
    results["n_train_derived"] = len(train_derived)
    results["train_recon_cos"] = round(sum(sims) / len(sims), 4)
    ok &= results["train_recon_cos"] >= T_TRAIN

    # ---- held-out gate: correct operator applied to source embedding ----
    hsims = []
    with torch.no_grad():
        for p in test_pairs:
            src_emb = lc.encode(p["source"])
            out = ops[p["relation"]](src_emb)
            hsims.append(F.cosine_similarity(out, ref[p["target"]], dim=0).item())
    results["n_test_pairs"] = len(hsims)
    results["test_recon_cos"] = round(sum(hsims) / len(hsims), 4)
    ok &= results["test_recon_cos"] >= T_TEST

    # per-category breakdown for the held-out metric
    bycat = {}
    for p, s in zip(test_pairs, hsims):
        bycat.setdefault(p["relation"][:1], []).append(s)
    results["test_by_category"] = {
        {"I": "inflectional", "D": "derivational",
         "E": "encyclopedic", "L": "lexicographic"}[k]: round(sum(v)/len(v), 4)
        for k, v in sorted(bycat.items())}

    results["gates"] = {
        "compression>=2.5": ratio >= T_COMPRESS,
        "train>=0.80": results["train_recon_cos"] >= T_TRAIN,
        "test>=0.65": results["test_recon_cos"] >= T_TEST,
        "operators>=3": len(ops) >= 3,
        "params>=1000": n_params >= 1000,
        "derivations_valid": True,
    }
    results["PASS"] = bool(ok)
    print(json.dumps(results, indent=2))
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
