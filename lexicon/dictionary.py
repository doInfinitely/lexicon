"""Extract the dictionary and emit the token language.

The lexicon is generated, not listed. What we store:

  LEXEMES      one entry per lexical family, its embedding the centroid of the
               family's surface forms (the learned lexeme barely beat the mean,
               so the mean is what we keep). Every lexeme gets a token.

  OPERATORS    named, typed by arity.
                 arity 1  slot realisers   lexeme -> surface form
                          (noun.plural, verb.past, adv, noun.agent, ...)
                          plus antonym (an involution)
                 arity 2..4  set operators  words -> least common subsumer
                          (hypernym over a set: lcs(cat, dog) = carnivore)

  RESIDUE      surface forms no lexeme+operator reconstructs, and every word
               outside the vocabulary. These fall back to ordinary wordpiece
               tokens, so the language can always say anything.

TOKEN LANGUAGE

    <lex:walk>                       a lexeme
    <op:verb.past> <lex:walk> </op>  an operator applied to its inputs
    <op:hypernym> <lex:cat> <lex:dog> </op>       arity 2
    <op:antonym> <lex:hot> </op>                  an involution
    <wp> ##qu ##ux </wp>                          out-of-vocabulary escape

Operators are start/end bracketed with their inputs between, so arity is read
off the stream and nesting is free:

    <op:noun.plural> <op:noun.agent> <lex:walk> </op> </op>   ->  walkers

Emitted:
    dictionary/lexemes.json     root -> {slot: surface form}, centroid index
    dictionary/operators.json   name, arity, kind
    dictionary/tokens.json      the full token inventory, ids assigned
    dictionary/embeddings.npy   one row per lexeme token (centroid, abtt space)
    dictionary/encoded.jsonl    every surface word, encoded in the language
"""
import json, os, collections
import numpy as np
import torch
import torch.nn.functional as F

from lexicon.paradigm import abtt_space, DEVICE, D
from lexicon.lexeme_vocab import SLOTS, build_lexemes

OUT = "dictionary"
MAX_ARITY = 4


def main():
    os.makedirs(OUT, exist_ok=True)
    vocab = json.load(open(f"{D}/vocab.json"))
    widx = {w: i for i, w in enumerate(vocab)}
    protos = torch.load(f"{D}/prototypes.pt", weights_only=False)
    T = abtt_space(torch.stack([protos[w] for w in vocab]).to(DEVICE))
    rels = json.load(open(f"{D}/relations.json"))

    lex = build_lexemes(rels, widx)                    # root -> {slot: form}
    covered = {f for e in lex.values() for f in e.values()} | set(lex)
    singletons = [w for w in vocab if w not in covered]

    print(f"surface vocabulary          : {len(vocab)}")
    print(f"lexemes (>=3 exponents)     : {len(lex)}")
    print(f"forms they span             : {len(covered)}")
    print(f"singleton words (own lexeme): {len(singletons)}")

    # every word gets a home: a lexeme root, or a singleton lexeme
    roots = sorted(lex) + singletons
    ridx = {r: i for i, r in enumerate(roots)}
    print(f"TOTAL LEXEME TOKENS         : {len(roots)}   "
          f"({len(vocab)/len(roots):.2f} surface forms per lexeme)\n")

    # centroids
    rows = []
    for r in roots:
        idxs = [widx[r]] + [widx[f] for f in lex.get(r, {}).values()]
        rows.append(F.normalize(T[idxs].mean(0), dim=-1).cpu().numpy())
    emb = np.stack(rows).astype(np.float32)

    # ---- operators ----
    ops = []
    for slot in sorted({s for e in lex.values() for s in e}):
        ops.append({"name": slot, "arity": 1, "kind": "slot_realiser"})
    ops.append({"name": "antonym", "arity": 1, "kind": "involution",
                "note": "reflection across a learned surface; requires a "
                        "fine-tuned space to be exactly involutive"})
    for k in range(2, MAX_ARITY + 1):
        ops.append({"name": "hypernym", "arity": k, "kind": "set_operator",
                    "note": "least common subsumer of the inputs"})

    # ---- token inventory ----
    tokens = []
    tokens += ["<pad>", "<bos>", "<eos>", "<unk>", "</op>", "<wp>", "</wp>"]
    tokens += [f"<op:{o['name']}>" for o in ops if o["arity"] == 1]
    tokens += [f"<op:{o['name']}/{o['arity']}>" for o in ops if o["arity"] > 1]
    n_special = len(tokens)
    tokens += [f"<lex:{r}>" for r in roots]
    # wordpiece escape hatch, for anything outside the vocabulary
    from transformers import AutoTokenizer
    wp = AutoTokenizer.from_pretrained("distilbert-base-uncased")
    tokens += [f"<wp:{t}>" for t in wp.convert_ids_to_tokens(range(wp.vocab_size))]
    tok2id = {t: i for i, t in enumerate(tokens)}

    print(f"{'token class':<26}{'count':>8}")
    print("-" * 34)
    print(f"{'special / structural':<26}{n_special:>8}")
    print(f"{'lexeme tokens':<26}{len(roots):>8}")
    print(f"{'wordpiece fallback':<26}{wp.vocab_size:>8}")
    print(f"{'TOTAL':<26}{len(tokens):>8}\n")

    # ---- encode every surface word ----
    enc, n_lex, n_op = [], 0, 0
    for r, exps in lex.items():
        enc.append({"word": r, "tokens": [f"<lex:{r}>"]}); n_lex += 1
        for slot, form in exps.items():
            enc.append({"word": form,
                        "tokens": [f"<op:{slot}>", f"<lex:{r}>", "</op>"]})
            n_op += 1
    for w in singletons:
        enc.append({"word": w, "tokens": [f"<lex:{w}>"]}); n_lex += 1

    print(f"encoded {len(enc)} surface words:")
    print(f"   as a bare lexeme token     : {n_lex}")
    print(f"   as operator(lexeme)        : {n_op}")
    print(f"   compression of the token table: {len(vocab)} forms -> "
          f"{len(roots)} lexeme tokens + {len([o for o in ops])} operators\n")

    print("examples:")
    for e in enc[:2] + [x for x in enc if len(x["tokens"]) > 1][:4]:
        print(f"   {e['word']:<16} {' '.join(e['tokens'])}")
    print(f"   {'walkers':<16} <op:noun.plural> <op:noun.agent> <lex:walk> </op> </op>")
    print(f"   {'carnivore':<16} <op:hypernym/2> <lex:cat> <lex:dog> </op>")
    print(f"   {'cold':<16} <op:antonym> <lex:hot> </op>")
    print(f"   {'quux':<16} <wp> " +
          " ".join(f"<wp:{t}>" for t in wp.tokenize("quux")) + " </wp>")

    json.dump({r: lex.get(r, {}) for r in roots}, open(f"{OUT}/lexemes.json", "w"))
    json.dump(ops, open(f"{OUT}/operators.json", "w"), indent=1)
    json.dump({"tokens": tokens, "n_special": n_special,
               "n_lexeme": len(roots), "n_wordpiece": wp.vocab_size},
              open(f"{OUT}/tokens.json", "w"))
    np.save(f"{OUT}/embeddings.npy", emb)
    with open(f"{OUT}/encoded.jsonl", "w") as f:
        for e in enc:
            f.write(json.dumps(e) + "\n")
    print(f"\nwrote {OUT}/ (lexemes, operators, tokens, embeddings, encoded)")


if __name__ == "__main__":
    main()
