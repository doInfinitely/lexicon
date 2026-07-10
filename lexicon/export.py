"""Export the trained lexicon as a self-contained artifact for LLM training.

real/lexicon_export/
  tokens.json      base-word tokens + operator tokens (<op:NAME> / </op>),
                   ready to seed a tokenizer vocabulary
  embeddings.npz   adapted-space embedding table: one row per (word, sense),
                   plus operator-net weights for on-the-fly derivation
  derivations.json word -> operator expression over base words
  senses.json      multi-sense (homograph) metadata with sample contexts
"""
import json, os
import numpy as np
import torch
import torch.nn.functional as F

from lexicon.model import LexiconSpace

OUT = "real/lexicon_export"


def main():
    ckpt = torch.load("real/lexicon_space.pt", weights_only=False)
    vocab = ckpt["vocab"]
    space = LexiconSpace(ckpt["relation_names"])
    space.load_state_dict(ckpt["state_dict"])
    space.eval()
    protos = torch.load("real/embeddings/prototypes.pt", weights_only=False)
    senses = torch.load("real/embeddings/senses.pt", weights_only=False)

    os.makedirs(OUT, exist_ok=True)

    # token inventory: base words are first-class tokens; every operator gets
    # start/end tokens so derived words serialize as <op:X> base </op>
    tokens = {
        "base_words": ckpt["base_lexicon"],
        "operator_tokens": [f"<op:{r}>" for r in ckpt["relation_names"]],
        "operator_end_token": "</op>",
    }
    json.dump(tokens, open(f"{OUT}/tokens.json", "w"), indent=1)

    # derivation expressions in token form
    derivs = {}
    for w in vocab:
        chain, cur = [], w
        while cur in ckpt["derivations"]:
            rel, src = ckpt["derivations"][cur]
            chain.append(rel)
            cur = src
        if chain:
            expr = cur
            for rel in reversed(chain):
                expr = f"<op:{rel}> {expr} </op>"
            derivs[w] = expr
    json.dump(derivs, open(f"{OUT}/derivations.json", "w"), indent=1)

    # adapted embedding table, one row per (word, sense)
    rows, meta = [], []
    with torch.no_grad():
        for w in vocab:
            for k, e in enumerate(senses.get(w) or [protos[w]]):
                rows.append(F.normalize(space.adapter(e.unsqueeze(0)), dim=-1)[0])
                meta.append([w, k])
    np.savez_compressed(
        f"{OUT}/embeddings.npz",
        table=torch.stack(rows).numpy().astype(np.float32),
        entries=np.array(meta, dtype=object),
        operator_state={k: v.numpy() for k, v in
                        space.ops.state_dict().items()} if False else 0)
    torch.save({"adapter": space.adapter.state_dict(),
                "operators": space.ops.state_dict(),
                "relation_names": ckpt["relation_names"]},
               f"{OUT}/networks.pt")

    sense_meta = json.load(open("real/embeddings/sense_meta.json"))
    samples = json.load(open("real/embeddings/sense_samples.json"))
    json.dump({w: {**m, "samples": samples.get(w, {})}
               for w, m in sense_meta.items()},
              open(f"{OUT}/senses.json", "w"), indent=1)

    n_rows = len(rows)
    print(f"exported: {len(tokens['base_words'])} base tokens, "
          f"{len(tokens['operator_tokens'])} operator tokens, "
          f"{len(derivs)} derivations, {n_rows} (word,sense) embedding rows")


if __name__ == "__main__":
    main()
