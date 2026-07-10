import json, collections, torch, numpy as np
from lexicon.ts_lm import WordTok, BPETok, LexTok, build, train_one, WORD_RE, OUT
from lexicon.ts_eval2 import bits_per_char, evaluate, has_unk

def main():
    train_texts, clean_eval, test_texts, held = build()
    lex = LexTok(train_texts)

    # slot-matched distractors: same operator, different root, all SEEN in training
    seen = collections.Counter()
    for t in train_texts[:6000]:
        seen.update(m.lower() for m in WORD_RE.findall(t) if m.isalpha())
    hset = {w for w,_,_ in held}
    slot_pool = collections.defaultdict(list)
    for form,(slot,root) in lex.lex.parent.items():
        if form in hset or seen[form] < 40: continue
        slot_pool[slot].append(form)
    for s in slot_pool: slot_pool[s] = sorted(slot_pool[s], key=lambda w:-seen[w])[:20]
    print("\nslot-matched distractor pools (all seen in training):")
    for s in sorted({sl for _,sl,_ in held}):
        print(f"   {s:<14} {', '.join(slot_pool[s][:8])}")
    print()

    res = {}
    for Tok in (WordTok, BPETok, LexTok):
        tok = lex if Tok is LexTok else Tok(train_texts)
        m = train_one(tok, train_texts)
        bpc = bits_per_char(m, tok, clean_eval)
        ev = evaluate(m, tok, test_texts, held, slot_pool,
                      lextok=(lex if Tok is LexTok else None))
        res[tok.name] = dict(bpc=bpc, **ev)
        print(f"  [{tok.name}] bits/char {bpc:.3f}  expressible {ev['expressible']:.3f} "
              f"acc {ev['acc']:.3f}  mrr {ev['mrr']:.3f}", flush=True)
        if "root_top1" in ev:
            print(f"           root|operator: top1 {ev['root_top1']:.3f} "
                  f"top10 {ev['root_top10']:.3f} median rank {ev['root_rank_median']:.0f}")
        torch.save(m.state_dict(), f"{OUT}/{tok.name}.pt")
        del m; torch.cuda.empty_cache()

    print("\n" + "="*80)
    print(f"{'tokenizer':<12}{'bits/char':>11}{'can express':>13}{'acc':>8}{'MRR':>8}")
    print("-"*80)
    for k,v in res.items():
        print(f"{k:<12}{v['bpc']:>11.3f}{v['expressible']:>13.3f}{v['acc']:>8.3f}{v['mrr']:>8.3f}")
    n = res['lexeme']['n']
    print(f"\nchance acc ~= 1/21 = 0.048   (n={n} held-out-form contexts)")
    print("'can express' = fraction of gold forms the scheme can even emit.")
    if "root_top1" in res["lexeme"]:
        r = res["lexeme"]
        print(f"\nlexeme diagnostic -- operator teacher-forced, which root does it pick?")
        print(f"   correct root ranked #1 : {r['root_top1']:.3f}")
        print(f"   correct root in top-10 : {r['root_top10']:.3f}")
        print(f"   median rank of the correct root among all lexemes: {r['root_rank_median']:.0f}")
    json.dump(res, open(f"{OUT}/results2.json","w"), indent=1)

if __name__ == "__main__":
    main()
