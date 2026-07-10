"""Does the lexeme advantage grow with trunk size, on a corpus with real morphology?

TinyStories has the vocabulary of a four-year-old: thin morphology, short tail.
The worst corpus for testing a morphological prior. wikitext-103 has 21.6% of
tokens as derived forms and 70.5% of types occurring <=5 times.

It is also harder for us: 7.1% of words fall through to the wordpiece escape,
so the lexeme stream is 1.23x BPE tokens (vs 1.28x word tokens on TinyStories),
and the honest baseline is BPE, not word-level (72k types).

  CORRECTION: this docstring said 1.655x. That was the PREFIX encoder, which
  emits a `</op>` per operator; this sweep uses PostfixTok, which does not.
  Measured here: 40,249,846 lexeme vs 32,674,764 BPE = 1.23x.

Sweep trunks across the TinyStories crossover (~5M). Report train loss beside
eval bits/char: if the delta tracks train loss, it is overfitting, not
representation.
"""
import collections, json
import numpy as np
import torch
import torch.nn.functional as F

from lexicon.ts_lm import GPT, DEVICE, OUT, CTX, WORD_RE
from lexicon.ts_postfix import PostfixTok
from lexicon.ts_eval2 import bits_per_char
from lexicon.ts_scale import trunk_params, train

N_PARA = 200000
CONFIGS = [(4, 256, 4), (6, 384, 6), (8, 512, 8)]
STEPS = 4000


class BPETok:
    name = "bpe"
    def __init__(self, texts, max_vocab=16000):
        from transformers import GPT2TokenizerFast
        self.t = GPT2TokenizerFast.from_pretrained("gpt2")
        c = collections.Counter()
        for x in texts[:20000]:
            c.update(self.t.encode(x))
        self.itos = [-1, -2] + [i for i, _ in c.most_common(max_vocab)]
        self.map = {g: i for i, g in enumerate(self.itos)}
    def enc_word(self, w):
        return [self.map.get(g, 1) for g in self.t.encode(" " + w)]
    def enc_words(self, ws):
        out = []
        for w in ws: out += self.enc_word(w)
        return out


def corpus(tok, texts):
    ids = []
    for t in texts:
        ids += tok.enc_words(WORD_RE.findall(t)) + [0]
    return np.array(ids, dtype=np.int32)


def main():
    from datasets import load_dataset
    ds = load_dataset("Salesforce/wikitext", "wikitext-103-raw-v1", split="train")
    texts = []
    for r in ds:
        t = r["text"]
        if len(t) > 200 and not t.lstrip().startswith("="):
            texts.append(t)
        if len(texts) >= N_PARA: break
    train_texts, eval_texts = texts[:-2000], texts[-2000:]
    chars = sum(len(t) for t in train_texts)
    print(f"wikitext: {len(train_texts):,} paragraphs, {chars/1e6:.1f}M chars\n")

    toks = {"bpe": BPETok(train_texts), "lexeme": PostfixTok(train_texts, max_vocab=16000)}
    data = {}
    for n, t in toks.items():
        data[n] = corpus(t, train_texts)
        print(f"{n:<8} vocab {len(t.itos):>6}   tokens {len(data[n]):>12,}   "
              f"{len(data[n])/len(data['bpe']):.2f}x bpe")
    print()

    rows = {}
    for layers, d, heads in CONFIGS:
        for name, tok in toks.items():
            torch.manual_seed(0)
            m = GPT(len(tok.itos), d=d, layers=layers, heads=heads).to(DEVICE)
            tp = trunk_params(m)
            opt_loss = train_and_loss(m, data[name], len(tok.itos))
            bpc = bits_per_char(m, tok, eval_texts)
            rows.setdefault((layers, d), {})[name] = (bpc, tp, opt_loss)
            print(f"  L{layers} d{d} {name:<7} trunk {tp/1e6:6.2f}M  "
                  f"train loss {opt_loss:.3f}  eval bits/char {bpc:.3f}", flush=True)
            del m; torch.cuda.empty_cache()

    print("\n" + "=" * 82)
    print(f"{'trunk':<13}{'params':>10}{'bpe bpc':>10}{'lexeme bpc':>13}{'delta':>9}"
          f"{'bpe trainL':>13}{'lex trainL':>12}")
    print("-" * 82)
    for (l, d), v in rows.items():
        b, x = v["bpe"], v["lexeme"]
        print(f"L{l} d{d:<8}{b[1]/1e6:>9.2f}M{b[0]:>10.3f}{x[0]:>13.3f}"
              f"{x[0]-b[0]:>+9.3f}{b[2]:>13.3f}{x[2]:>12.3f}")
    print("\nnegative delta = lexeme better. |delta| growing with trunk => it scales.")
    print("TinyStories reference (vs word): L4 +0.052, L6 -0.098")
    json.dump({f"L{l}_d{d}": {k: v[0] for k, v in r.items()} for (l, d), r in rows.items()},
              open(f"{OUT}/wt_scale.json", "w"), indent=1)


def train_and_loss(m, data, vocab_size, steps=STEPS, bs=24, lr=6e-4, seed=0):
    torch.manual_seed(seed)
    opt = torch.optim.AdamW(m.parameters(), lr=lr, weight_decay=0.1)
    sch = torch.optim.lr_scheduler.OneCycleLR(opt, lr, total_steps=steps, pct_start=0.05)
    d = torch.from_numpy(data).long()
    g = torch.Generator().manual_seed(seed)
    m.train(); last = []
    for s in range(steps):
        ix = torch.randint(0, len(d) - CTX - 1, (bs,), generator=g)
        x = torch.stack([d[i:i+CTX] for i in ix]).to(DEVICE)
        y = torch.stack([d[i+1:i+1+CTX] for i in ix]).to(DEVICE)
        loss = F.cross_entropy(m(x).reshape(-1, vocab_size), y.reshape(-1))
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0)
        opt.step(); sch.step()
        if s >= steps - 50: last.append(loss.item())
    return float(np.mean(last))


if __name__ == "__main__":
    main()
