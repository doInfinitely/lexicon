"""Run the three-tokenizer TinyStories experiment."""
import json, collections, numpy as np, torch
from lexicon.ts_lm import (WordTok, BPETok, LexTok, build, train_one,
                           bits_per_char, heldout_eval, WORD_RE, OUT)

def main():
    train_texts, clean_eval, test_texts, held = build()
    hset = {w for w, _, _ in held}

    # integrity: no held-out FORM may appear in training under any scheme
    leaked = [w for w in hset for t in train_texts[:2000]
              if w in {m.lower() for m in WORD_RE.findall(t)}]
    print(f"leak check: held-out forms found in a training sample: {len(leaked)}")

    lex = LexTok(train_texts)
    # a lexeme-model-specific leak: is `mice` itself a <lex:> token in the stream?
    bad = [w for w in hset if f"<lex:{w}>" in lex.stoi]
    print(f"  held-out forms present as BARE LEXEME tokens (would be a leak): {bad}")
    # they must instead be reachable as operator+root
    ok = [w for w, s, r in held if f"<op:{s}>" in lex.stoi and f"<lex:{r}>" in lex.stoi]
    print(f"  held-out forms composable from seen tokens: {len(ok)}/{len(held)}")
    print(f"  e.g. mice -> {lex.lex.encode_word('mice')}\n")

    # distractors: other inflections of common lemmas + frequent words
    c = collections.Counter()
    for t in train_texts[:4000]:
        c.update(m.lower() for m in WORD_RE.findall(t) if m.isalpha())
    distract = [w for w, _ in c.most_common(60) if w not in hset][:40]

    results = {}
    for Tok in (WordTok, BPETok, LexTok):
        tok = Tok(train_texts) if Tok is not LexTok else lex
        m = train_one(tok, train_texts)
        bpc = bits_per_char(m, tok, clean_eval)
        hr = heldout_eval(m, tok, test_texts, held, distract)
        results[tok.name] = dict(bpc=bpc, **hr)
        print(f"  [{tok.name}] bits/char {bpc:.3f} | held-out form acc {hr['acc']:.3f} "
              f"mrr {hr['mrr']:.3f} nll {hr['nll']:.2f} (n={hr['n']})\n", flush=True)
        del m; torch.cuda.empty_cache()

    print("=" * 74)
    print(f"{'tokenizer':<14}{'bits/char':>11}{'held-out form acc':>20}{'MRR':>8}{'NLL':>9}")
    print("-" * 74)
    for k, v in results.items():
        print(f"{k:<14}{v['bpc']:>11.3f}{v['acc']:>20.3f}{v['mrr']:>8.3f}{v['nll']:>9.2f}")
    print("\nchance acc = 1/41 = 0.024.  The held-out forms never appeared in training;")
    print("only the lexeme scheme can compose them from tokens it has seen.")
    json.dump(results, open(f"{OUT}/results.json", "w"), indent=1)

if __name__ == "__main__":
    main()
