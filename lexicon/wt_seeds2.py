"""Does the corrected dictionary change the wikitext result? 3 seeds, L6.

Arms:
  bpe            GPT-2 BPE control
  lex-v1         forest.json, hand-rule derivations, BUGGY outermost-first postfix
                 (= every lexeme run before today; reproduces the -0.072 at L6)
  lex-v2         forest_v2.json (MorphyNet derivations + cosine opacity filter +
                 rare-operator pruning), CORRECT innermost-first postfix
  lex-v2-lean    as v2, without the <wp>/</wp> escape brackets

v1 -> v2 isolates the dictionary+order fixes; v2 -> v2-lean isolates the brackets.
Three seeds everywhere, because no wikitext cell has ever had a variance estimate.
"""
import json, collections
import numpy as np, torch

from lexicon.ts_lm import GPT, DEVICE, OUT, WORD_RE
from lexicon.ts_postfix import PostfixTok
from lexicon.ts_eval2 import bits_per_char
from lexicon.ts_scale import trunk_params
from lexicon.wt_scale import BPETok, corpus, train_and_loss, N_PARA

SEEDS = [0, 1, 2]
L, D, H = 6, 384, 6


class Lex(PostfixTok):
    def __init__(self, texts, forest, correct_order, brackets, name, max_vocab=16000):
        from lexicon.ts_encode import LexemeTokenizer
        self.name, self.correct_order, self.brackets = name, correct_order, brackets
        self.lex = LexemeTokenizer()
        f = json.load(open(forest))
        self.lex.parent = {k: tuple(v) for k, v in f["parent"].items()}
        self.lex.roots = set(f["roots"]); self.lex._cache = {}
        self._pc = {}
        c = collections.Counter()
        for t in texts[:20000]:
            for m in WORD_RE.findall(t):
                c.update(self.enc_word_str(m) if (m.isalpha() or "'" in m) else [f"<p:{m}>"])
        self.itos = ["<pad>", "<unk>"] + [w for w, _ in c.most_common(max_vocab)]
        self.stoi = {w: i for i, w in enumerate(self.itos)}

    def enc_word_str(self, w):
        if w in self._pc: return self._pc[w]
        lw = w.lower()
        if lw in self.lex.roots:
            out = [f"<lex:{lw}>"]
        elif lw in self.lex.parent:
            chain, cur = [], lw                       # outermost -> innermost
            while cur in self.lex.parent:
                s, cur = self.lex.parent[cur]; chain.append(s)
            ops = list(reversed(chain)) if self.correct_order else chain
            out = [f"<lex:{cur}>"] + [f"<op:{s}>" for s in ops]
        else:
            wp = [f"<wp:{t}>" for t in self.lex.wp.tokenize(lw)] or ["<unk>"]
            out = (["<wp>"] + wp + ["</wp>"]) if self.brackets else wp
        self._pc[w] = out
        return out


def main():
    from datasets import load_dataset
    ds = load_dataset("Salesforce/wikitext", "wikitext-103-raw-v1", split="train")
    texts = []
    for r in ds:
        t = r["text"]
        if len(t) > 200 and not t.lstrip().startswith("="): texts.append(t)
        if len(texts) >= N_PARA: break
    tr, ev = texts[:-2000], texts[-2000:]

    toks = {"bpe": BPETok(tr)}
    for nm, fo, co, br in [("lex-v1", "dictionary/forest.json", False, True),
                           ("lex-v2", "dictionary/forest_v2.json", True, True),
                           ("lex-v2-lean", "dictionary/forest_v2.json", True, False)]:
        toks[nm] = Lex(tr, fo, co, br, nm)

    data = {n: corpus(t, tr) for n, t in toks.items()}
    for n in toks:
        print(f"{n:<13} vocab {len(toks[n].itos):>6}  tokens {len(data[n]):>12,}  "
              f"{len(data[n])/len(data['bpe']):.3f}x bpe", flush=True)
    print()

    res = collections.defaultdict(list)
    for seed in SEEDS:
        for n, tok in toks.items():
            torch.manual_seed(seed); np.random.seed(seed)
            m = GPT(len(tok.itos), d=D, layers=L, heads=H).to(DEVICE)
            train_and_loss(m, data[n], len(tok.itos))
            b = bits_per_char(m, tok, ev)
            res[n].append(b)
            print(f"  seed {seed}  {n:<13} bits/char {b:.4f}", flush=True)
            del m; torch.cuda.empty_cache()

    base = np.array(res["bpe"])
    print("\n" + "=" * 76)
    print(f"{'arm':<14}{'mean':>8}{'sd':>8}{'vs bpe':>10}{'sd(d)':>9}{'tokens':>9}{'compute-eq':>12}")
    print("-" * 76)
    # bpe L6->L8 curve: 0.075 bits per 2.37x params => bits per doubling
    BPD = 0.075 / np.log2(2.37)
    for n in toks:
        a = np.array(res[n]); d = a - base
        r = len(data[n]) / len(data["bpe"])
        ce = (2 ** (-d.mean() / BPD)) / r if n != "bpe" else 1.0
        print(f"{n:<14}{a.mean():>8.4f}{a.std(ddof=1):>8.4f}{d.mean():>+10.4f}"
              f"{d.std(ddof=1):>9.4f}{r:>8.2f}x{ce:>11.2f}x")
    print(f"\nbpe seed sd: {base.std(ddof=1):.4f}   (noise band assumed so far: +/-0.05)")
    print("compute-eq = params-equivalent gain / token cost; >1 means cheaper than "
          "buying the same bits with a bigger trunk")
    json.dump({k: v for k, v in res.items()}, open(f"{OUT}/wt_seeds2.json", "w"), indent=1)


if __name__ == "__main__":
    main()
