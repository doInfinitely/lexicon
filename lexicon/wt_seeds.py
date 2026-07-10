"""Disentangle brackets from dictionary expansion, and get a noise band.

wt_scale2 changed TWO things at once (bracket removal + 9,938 decomposition
entries) and the L6 delta moved -0.072 -> -0.051. Unattributable. Also every
wikitext cell so far is a single seed, while the only noise estimate I have
(+/-0.05) came from TinyStories.

2x2 over {brackets, expanded-dict} plus a BPE control, 3 seeds, L6 only.
"""
import json, collections, itertools
import numpy as np, torch

from lexicon.ts_lm import GPT, DEVICE, OUT, WORD_RE
from lexicon.ts_postfix import PostfixTok
from lexicon.ts_eval2 import bits_per_char
from lexicon.ts_scale import trunk_params
from lexicon.wt_scale import BPETok, corpus, train_and_loss, N_PARA

SEEDS = [0, 1, 2]
L, D, H = 6, 384, 6


class Lex(PostfixTok):
    def __init__(self, texts, brackets, expanded, max_vocab=16000):
        from lexicon.ts_encode import LexemeTokenizer, load_forest
        self.name = f"lex[br={int(brackets)},exp={int(expanded)}]"
        self.brackets = brackets
        self.lex = LexemeTokenizer()
        bp, br = load_forest()
        self.lex.parent, self.lex.roots = dict(bp), set(br)
        if expanded:
            e = json.load(open("dictionary/forest_expanded.json"))
            for w, v in e["parent"].items():
                if w not in bp and w not in br:
                    self.lex.parent[w] = tuple(v)      # decomposition only
        self.lex._cache = {}
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
            ops, cur = [], lw
            while cur in self.lex.parent:
                s, cur = self.lex.parent[cur]; ops.append(s)
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
    for br, ex in itertools.product([True, False], [False, True]):
        t = Lex(tr, br, ex); toks[t.name] = t
    data = {n: corpus(t, tr) for n, t in toks.items()}
    for n in toks:
        print(f"{n:<22} vocab {len(toks[n].itos):>6}  tokens {len(data[n]):>12,}  "
              f"{len(data[n])/len(data['bpe']):.3f}x", flush=True)

    res = collections.defaultdict(list)
    for seed in SEEDS:
        for n, tok in toks.items():
            torch.manual_seed(seed); np.random.seed(seed)
            m = GPT(len(tok.itos), d=D, layers=L, heads=H).to(DEVICE)
            train_and_loss(m, data[n], len(tok.itos))
            b = bits_per_char(m, tok, ev)
            res[n].append(b)
            print(f"  seed {seed}  {n:<22} bits/char {b:.4f}", flush=True)
            del m; torch.cuda.empty_cache()

    print("\n" + "=" * 78)
    base = np.array(res["bpe"])
    print(f"{'arm':<22}{'mean':>8}{'sd':>8}{'vs bpe':>9}{'sd(delta)':>11}{'tokens':>9}")
    print("-" * 78)
    for n in toks:
        a = np.array(res[n]); d = a - base
        print(f"{n:<22}{a.mean():>8.4f}{a.std(ddof=1):>8.4f}"
              f"{d.mean():>+9.4f}{d.std(ddof=1):>11.4f}"
              f"{len(data[n])/len(data['bpe']):>8.2f}x")
    print(f"\nseed noise on bpe (sd): {base.std(ddof=1):.4f}")
    json.dump({k: v for k, v in res.items()}, open(f"{OUT}/wt_seeds.json", "w"), indent=1)


if __name__ == "__main__":
    main()
