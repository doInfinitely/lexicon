"""Train small LMs on TinyStories in three tokenizations, and test whether the
lexeme language can say words it has never seen.

The question is not perplexity. It is whether moving morphology out of the
weights and into the LANGUAGE buys compositional generalisation.

Sixteen opaque irregular forms are removed from training entirely -- every
story containing one becomes test data. Each form's ROOT and OPERATOR remain
abundant:

    mice  = noun.plural(mouse)      `mouse` 10,459x, `<op:noun.plural>` everywhere
    rang  = verb.past(ring)         `ring` 2,482x
    fought = verb.ptcp(fight)       `fight` frequent

They are OPAQUE: the root is not an orthographic prefix, and GPT-2 gives each
one atomic token. So:

    word model   the token `mice` exists but was never trained. Dead.
    bpe model    `Ġmice` is one token, never trained. Dead.
    lexeme model `<op:noun.plural> <lex:mouse> </op>` -- every token seen
                 thousands of times. It can compose the word it never saw.

Scoring is comparable across tokenizers: a candidate word's score is the summed
log-probability of its own token sequence given the same word-level context.

  bits/char        on clean held-out stories (general quality)
  held-out form    rank of the true form among distractors, and its NLL
"""
import json, math, os, random, collections
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from lexicon.ts_encode import LexemeTokenizer, WORD_RE

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
OUT = "real/ts"
N_STORIES = 60000
CTX = 512


# ------------------------------------------------------------------ tokenizers
class WordTok:
    name = "word"

    def __init__(self, texts, max_vocab=12000):
        c = collections.Counter()
        for t in texts:
            c.update(m.lower() for m in WORD_RE.findall(t))
        self.itos = ["<pad>", "<unk>"] + [w for w, _ in c.most_common(max_vocab)]
        self.stoi = {w: i for i, w in enumerate(self.itos)}

    def enc_words(self, ws):
        return [self.stoi.get(w.lower(), 1) for w in ws]

    def enc_word(self, w):
        return [self.stoi.get(w.lower(), 1)]


class BPETok:
    name = "bpe"

    def __init__(self, texts, max_vocab=12000):
        from transformers import GPT2TokenizerFast
        self.t = GPT2TokenizerFast.from_pretrained("gpt2")
        c = collections.Counter()
        for t in texts[:8000]:
            c.update(self.t.encode(t))
        keep = [i for i, _ in c.most_common(max_vocab)]
        self.itos = [-1, -2] + keep                 # pad, unk
        self.map = {g: i for i, g in enumerate(self.itos)}

    def enc_words(self, ws):
        out = []
        for w in ws:
            out += self.enc_word(w)
        return out

    def enc_word(self, w):
        return [self.map.get(g, 1) for g in self.t.encode(" " + w)]


class LexTok:
    name = "lexeme"

    def __init__(self, texts, max_vocab=12000):
        self.lex = LexemeTokenizer()
        c = collections.Counter()
        for t in texts[:8000]:
            c.update(self.lex.encode(t))
        self.itos = ["<pad>", "<unk>"] + [w for w, _ in c.most_common(max_vocab)]
        self.stoi = {w: i for i, w in enumerate(self.itos)}

    def enc_words(self, ws):
        out = []
        for w in ws:
            out += self.enc_word(w)
        return out

    def enc_word(self, w):
        return [self.stoi.get(t, 1) for t in self.lex.encode_word(w)]


# ------------------------------------------------------------------ model
class Block(nn.Module):
    def __init__(self, d, h):
        super().__init__()
        self.ln1 = nn.LayerNorm(d); self.ln2 = nn.LayerNorm(d)
        self.att = nn.MultiheadAttention(d, h, batch_first=True)
        self.mlp = nn.Sequential(nn.Linear(d, 4 * d), nn.GELU(), nn.Linear(4 * d, d))

    def forward(self, x, mask):
        h = self.ln1(x)
        a, _ = self.att(h, h, h, attn_mask=mask, need_weights=False)
        x = x + a
        return x + self.mlp(self.ln2(x))


class GPT(nn.Module):
    def __init__(self, vocab, d=384, layers=6, heads=6, ctx=CTX):
        super().__init__()
        self.tok = nn.Embedding(vocab, d)
        self.pos = nn.Embedding(ctx, d)
        self.blocks = nn.ModuleList([Block(d, heads) for _ in range(layers)])
        self.ln = nn.LayerNorm(d)
        self.head = nn.Linear(d, vocab, bias=False)
        self.head.weight = self.tok.weight
        self.ctx = ctx

    def forward(self, idx):
        B, T = idx.shape
        m = torch.triu(torch.full((T, T), float("-inf"), device=idx.device), 1)
        x = self.tok(idx) + self.pos(torch.arange(T, device=idx.device))[None]
        for b in self.blocks:
            x = b(x, m)
        return self.head(self.ln(x))


def build():
    from datasets import load_dataset
    os.makedirs(OUT, exist_ok=True)
    ds = load_dataset("roneneldan/TinyStories", split="train")
    held = [tuple(x) for x in json.load(open("dictionary/heldout_forms.json"))
            if x[0] != "ups"]
    hset = {w for w, _, _ in held}
    print(f"held-out forms ({len(held)}): {', '.join(sorted(hset))}\n")

    train_texts, test_texts = [], []
    i = 0
    while len(train_texts) < N_STORIES and i < 400000:
        t = ds[i]["text"]; i += 1
        ws = {m.lower() for m in WORD_RE.findall(t) if m.isalpha()}
        (test_texts if (ws & hset) else train_texts).append(t)
    test_texts = test_texts[:3000]
    clean_eval = train_texts[-2000:]
    train_texts = train_texts[:-2000]
    print(f"train {len(train_texts)}, clean-eval {len(clean_eval)}, "
          f"held-out-form test {len(test_texts)}")
    json.dump({"held": held}, open(f"{OUT}/spec.json", "w"))
    return train_texts, clean_eval, test_texts, held


def tokenize_corpus(tok, texts):
    ids = []
    for t in texts:
        ids += tok.enc_words(WORD_RE.findall(t)) + [0]
    return np.array(ids, dtype=np.int32)


def train_one(tok, train_texts, steps=4000, bs=24, lr=3e-4, seed=0):
    torch.manual_seed(seed)
    data = tokenize_corpus(tok, train_texts)
    print(f"  [{tok.name}] {len(data):,} training tokens, vocab {len(tok.itos)}")
    m = GPT(len(tok.itos)).to(DEVICE)
    n = sum(p.numel() for p in m.parameters())
    opt = torch.optim.AdamW(m.parameters(), lr=lr, weight_decay=0.1)
    sch = torch.optim.lr_scheduler.OneCycleLR(opt, lr, total_steps=steps, pct_start=0.05)
    d = torch.from_numpy(data).long()
    g = torch.Generator().manual_seed(seed)
    m.train()
    for s in range(steps):
        ix = torch.randint(0, len(d) - CTX - 1, (bs,), generator=g)
        x = torch.stack([d[i:i + CTX] for i in ix]).to(DEVICE)
        y = torch.stack([d[i + 1:i + 1 + CTX] for i in ix]).to(DEVICE)
        loss = F.cross_entropy(m(x).reshape(-1, len(tok.itos)), y.reshape(-1))
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0)
        opt.step(); sch.step()
        if s % 500 == 0:
            print(f"    step {s:5d}  loss {loss.item():.3f}", flush=True)
    print(f"  [{tok.name}] done, {n/1e6:.1f}M params")
    return m


# ------------------------------------------------------------------ evaluation
@torch.no_grad()
def bits_per_char(m, tok, texts):
    m.eval()
    tot_nll, tot_chars = 0.0, 0
    for t in texts[:400]:
        ids = tok.enc_words(WORD_RE.findall(t))[:CTX]
        if len(ids) < 8:
            continue
        x = torch.tensor(ids, device=DEVICE)[None]
        lg = F.log_softmax(m(x)[0, :-1], -1)
        tot_nll += -lg[torch.arange(len(ids) - 1), x[0, 1:]].sum().item()
        tot_chars += len(t)
    return tot_nll / tot_chars / math.log(2)


@torch.no_grad()
def score_word(m, tok, ctx_ids, word):
    """log P(word | context), summed over the word's own tokens."""
    wt = tok.enc_word(word)
    ids = ctx_ids + wt
    ids = ids[-CTX:]
    x = torch.tensor(ids, device=DEVICE)[None]
    lg = F.log_softmax(m(x)[0], -1)
    n = len(wt)
    s = 0.0
    for k in range(n):
        pos = len(ids) - n + k - 1
        s += lg[pos, ids[len(ids) - n + k]].item()
    return s


@torch.no_grad()
def heldout_eval(m, tok, test_texts, held, distractors, max_cases=300):
    m.eval()
    hmap = {w: (slot, root) for w, slot, root in held}
    ranks, nlls, hits = [], [], 0
    cases = 0
    for t in test_texts:
        ws = WORD_RE.findall(t)
        for i, w in enumerate(ws):
            if w.lower() not in hmap or i < 6:
                continue
            ctx = tok.enc_words(ws[max(0, i - 80):i])
            gold = w.lower()
            cands = [gold] + [d for d in distractors if d != gold]
            sc = np.array([score_word(m, tok, ctx, c) for c in cands])
            r = int((sc > sc[0]).sum()) + 1
            ranks.append(r); hits += (r == 1); nlls.append(-sc[0])
            cases += 1
            if cases >= max_cases:
                break
        if cases >= max_cases:
            break
    return dict(acc=hits / len(ranks), mrr=float(np.mean(1 / np.array(ranks))),
                nll=float(np.mean(nlls)), n=len(ranks))
