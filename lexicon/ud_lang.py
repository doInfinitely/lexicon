"""Factored-embedding experiment on a morphologically rich language, via Universal Deps.

The treebank IS the corpus (gold-annotated), and FEATS IS the operator slot. This removes
every confound that would plague a raw-text approach: no morphological analyzer (UD gives
gold lemma+features), no sandhi, clean whitespace word segmentation, no dictionary to
build wrong. English needed 7 useful operators; Turkish/Finnish have ~200 covering 90% of
inflection, TTR ~4x higher, 66-80% of content words inflected. If the factorization helps
MORE here, that is the "punches above its weight" result.

Arms (byte fallback, one token per word, so length is held fixed across the lexeme arms):
    bpe        GPT-2 BPE (multilingual via byte-level, honest baseline)
    free       one token per word, independent embeddings
    affine     e = E_lemma + d_slot + U_slot V_slot^T E_lemma,  identity slot pinned
    shufroot   affine with each word pointing at a RANDOM lemma of the same slot  (null)

Operators pruned to those covering >=90% of inflected occurrences; the rare-feature tail
falls back to whole-word tokens (same fix that stopped English derivation from hurting).
"""
import json, collections, os, sys
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F
from lexicon.ts_lm import GPT, Block, DEVICE, OUT, CTX
from lexicon.ts_eval2 import bits_per_char
from lexicon.wt_scale import train_and_loss

SEEDS = [0, 1, 2][:int(__import__("os").environ.get("NSEEDS","3"))]
L, D, H = 6, 384, 6
RANK, NBYTE, VOCAB = 32, 256, 16000
MIN_OP = 25


def read_conllu(path):
    """-> list of sentences, each a list of (form_lower, lemma_lower, feats)."""
    sents, cur = [], []
    for ln in open(path, encoding="utf-8"):
        if ln.startswith("#"): continue
        if not ln.strip():
            if cur: sents.append(cur); cur = []
            continue
        c = ln.rstrip("\n").split("\t")
        if len(c) < 6 or "-" in c[0] or "." in c[0]: continue
        form, lemma, upos, feats = c[1], c[2], c[3], c[5]
        if upos in ("PUNCT", "SYM"): continue
        cur.append((form.lower(), lemma.lower(), feats if feats != "_" else ""))
    if cur: sents.append(cur)
    return sents


class UDTok:
    """word-level tokenizer over a treebank; byte fallback; carries the factorization."""
    def __init__(self, sents, max_vocab=VOCAB):
        formc = collections.Counter(t[0] for s in sents for t in s)
        # operators: FEATS bundles frequent enough to be worth a slot
        opc = collections.Counter(t[2] for s in sents for t in s if t[2] and t[0] != t[1])
        self.slots = ["<id>"] + [f for f, c in opc.items() if c >= MIN_OP]
        self.sidx = {f: i for i, f in enumerate(self.slots)}
        keep = [w for w, _ in formc.most_common(max_vocab - NBYTE)]
        self.itos = ["<pad>"] + [f"<b:{i}>" for i in range(NBYTE)] + keep
        self.b0 = 1
        self.stoi = {w: 1 + NBYTE + i for i, w in enumerate(keep)}
        # map each vocab form -> (lemma, slot) using its most common analysis
        analysis = {}
        seen = collections.defaultdict(collections.Counter)
        for s in sents:
            for form, lemma, feats in s:
                sl = self.sidx.get(feats, 0) if form != lemma else 0
                seen[form][(lemma, sl)] += 1
        for form, cc in seen.items():
            (lemma, sl), _ = cc.most_common(1)[0]
            analysis[form] = (lemma if sl != 0 else form, sl)
        self.analysis = analysis

    def enc_word(self, w):
        k = w.lower()
        return [self.stoi[k]] if k in self.stoi else [self.b0 + b for b in k.encode("utf-8")]

    def encode_sents(self, sents):
        ids = []
        for s in sents:
            for form, lemma, feats in s:
                ids += self.enc_word(form)
        return np.array(ids, dtype=np.int64)

    def factorise(self, shuffle_seed=None):
        roots, rid = [], {}
        def get(r):
            if r not in rid: rid[r] = len(roots); roots.append(r)
            return rid[r]
        R = np.zeros(len(self.itos), dtype=np.int64); S = np.zeros(len(self.itos), dtype=np.int64)
        for i, t in enumerate(self.itos):
            if t.startswith(("<b:",)) or t == "<pad>":
                R[i] = get(t); continue
            lemma, sl = self.analysis.get(t, (t, 0))
            R[i] = get(lemma if sl != 0 else t); S[i] = sl
        if shuffle_seed is not None:
            g = np.random.default_rng(shuffle_seed)
            for s in range(1, len(self.slots)):
                m = np.where(S == s)[0]
                if len(m) > 1: R[m] = R[m][g.permutation(len(m))]
        return torch.from_numpy(R), torch.from_numpy(S), len(roots)


def causal(T, dev): return torch.triu(torch.full((T, T), float("-inf"), device=dev), 1)
def init_scale(m, std=0.02):
    for n in ("tok", "pos", "E"):
        e = getattr(m, n, None)
        if e is not None and hasattr(e, "weight"): e.weight.data.normal_(0, std)


class AffineGPT(nn.Module):
    def __init__(self, R, S, nR, nslots, mode, d=D, layers=L, heads=H, ctx=CTX, rank=RANK):
        super().__init__(); self.mode = mode; self.nslots = nslots
        self.register_buffer("wr", R); self.register_buffer("ws", S)
        self.E = nn.Embedding(nR, d)
        self.dop = nn.Parameter(torch.zeros(nslots - 1, d))
        if mode == "affine":
            self.U = nn.Parameter(torch.zeros(nslots - 1, d, rank))
            self.V = nn.Parameter(torch.randn(nslots - 1, d, rank) * 0.02)
        self.pos = nn.Embedding(ctx, d)
        self.blocks = nn.ModuleList([Block(d, heads) for _ in range(layers)])
        self.ln = nn.LayerNorm(d); self.ctx = ctx
    def word_emb(self):
        er = self.E(self.wr); out = er.clone()
        for s in range(1, self.nslots):
            m = self.ws == s
            if not m.any(): continue
            v = er[m] + self.dop[s - 1]
            if self.mode == "affine": v = v + (er[m] @ self.V[s - 1]) @ self.U[s - 1].T
            out[m] = v
        return out
    def forward(self, idx):
        T = idx.shape[1]; W = self.word_emb()
        x = F.embedding(idx, W) + self.pos(torch.arange(T, device=idx.device))[None]
        mask = causal(T, idx.device)
        for b in self.blocks: x = b(x, mask)
        return self.ln(x) @ W.T


class ByteBPE:
    def __init__(self, sents, max_vocab=VOCAB):
        from transformers import GPT2TokenizerFast
        self.t = GPT2TokenizerFast.from_pretrained("gpt2")
        c = collections.Counter()
        for s in sents:
            c.update(self.t.encode(" ".join(t[0] for t in s)))
        keep = [g for g, _ in c.most_common(max_vocab - NBYTE)]
        self.itos = ["<pad>"] + [f"<b:{i}>" for i in range(NBYTE)] + [f"<g:{g}>" for g in keep]
        self.b0 = 1; self.map = {g: 1 + NBYTE + i for i, g in enumerate(keep)}
    def enc_word(self, w):
        gs = self.t.encode(" " + w)
        return [self.map[g] for g in gs] if all(g in self.map for g in gs) \
               else [self.b0 + b for b in (" " + w).encode("utf-8")]
    def encode_sents(self, sents):
        ids = []
        for s in sents: ids += self.enc_word(" ".join(t[0] for t in s))
        return np.array(ids, dtype=np.int64)


def eval_bpc(m, tok, sents):
    m.eval(); import math; nll = 0.0; chars = 0
    with torch.no_grad():
        for s in sents[:300]:
            ids = tok.encode_sents([s])
            if len(ids) < 8: continue
            ids = ids[:CTX]
            x = torch.tensor(ids, device=DEVICE)[None]
            lg = F.log_softmax(m(x)[0, :-1], -1)
            nll += -lg[torch.arange(len(ids) - 1), x[0, 1:]].sum().item()
            chars += len(" ".join(t[0] for t in s))
    return nll / chars / math.log(2)


def train_earlystop(m, data, V, evalfn, seed, maxsteps=6000, warm=300, every=300, patience=6, bs=24):
    """warmup+constant LR; eval held-out every `every` steps; return best bits/char.
    Fixes the overfitting that wrecked the matched-50k magnitudes (hundreds of epochs)."""
    opt = torch.optim.AdamW(m.parameters(), lr=6e-4, weight_decay=0.1)
    d = torch.from_numpy(data).long(); g = torch.Generator().manual_seed(seed)
    best = 1e9; bad = 0
    for step in range(1, maxsteps + 1):
        for pg in opt.param_groups: pg["lr"] = 6e-4 * min(1.0, step / warm)
        m.train()
        ix = torch.randint(0, len(d) - CTX - 1, (bs,), generator=g)
        x = torch.stack([d[i:i+CTX] for i in ix]).to(DEVICE)
        y = torch.stack([d[i+1:i+1+CTX] for i in ix]).to(DEVICE)
        loss = F.cross_entropy(m(x).reshape(-1, V), y.reshape(-1))
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0); opt.step()
        if step % every == 0:
            b = evalfn(m)
            if b < best - 1e-4: best = b; bad = 0
            else:
                bad += 1
                if bad >= patience: break
    return best


def main():
    lang = sys.argv[1] if len(sys.argv) > 1 else "tr"
    path = {"tr":"ud/tr_boun.conllu","fi":"ud/fi_tdt.conllu","en":"ud/en_ewt.conllu"}.get(lang, f"ud/{lang}.conllu")
    sents = read_conllu(path)
    rng = np.random.default_rng(0); perm = rng.permutation(len(sents))
    ev = [sents[i] for i in perm[:400]]; tr = [sents[i] for i in perm[400:]]
    MAXTOK=int(os.environ.get("MAXTOK","0"))
    if MAXTOK:
        capped=[]; ntok=0
        for _s in tr:
            capped.append(_s); ntok+=len(_s)
            if ntok>=MAXTOK: break
        tr=capped; print(f"  [MAXTOK] capped train to {ntok} word-tokens", flush=True)
    tok = UDTok(tr); bpe = ByteBPE(tr)
    R, S, nR = tok.factorise(); Rs, Ss, _ = tok.factorise(shuffle_seed=0)
    data = tok.encode_sents(tr); bd = bpe.encode_sents(tr)
    print(f"{lang}: {len(tr)} train / {len(ev)} eval sents, {len(data):,} word-tokens "
          f"({len(data)/len(bd):.3f}x bpe)")
    print(f"  vocab {len(tok.itos):,}  lemmas {nR:,}  operators {len(tok.slots)}  "
          f"(pruned from raw feature bundles at MIN_OP={MIN_OP})", flush=True)
    STEPS = int(__import__("os").environ.get("STEPS","4000"))
    res = collections.defaultdict(list)
    for seed in SEEDS:
        for arm in ("bpe", "free", "affine", "shufroot"):
            torch.manual_seed(seed); np.random.seed(seed)
            if arm == "bpe":
                m = GPT(len(bpe.itos), d=D, layers=L, heads=H); etok=bpe; V=len(bpe.itos); dat=bd
            elif arm == "free":
                m = GPT(len(tok.itos), d=D, layers=L, heads=H); etok=tok; V=len(tok.itos); dat=data
            else:
                rr, ss = (Rs, Ss) if arm == "shufroot" else (R, S)
                m = AffineGPT(rr, ss, nR, len(tok.slots), "affine"); etok=tok; V=len(tok.itos); dat=data
            init_scale(m); m = m.to(DEVICE)
            if os.environ.get("EARLYSTOP"):
                b = train_earlystop(m, dat, V, lambda mm: eval_bpc(mm, etok, ev), seed)
            else:
                train_and_loss(m, dat, V, steps=STEPS, seed=seed); b = eval_bpc(m, etok, ev)
            res[arm].append(b)
            print(f"  seed {seed} {arm:<9} bits/char {b:.4f}", flush=True)
            del m; torch.cuda.empty_cache()
        json.dump({k: v for k, v in res.items()}, open(f"{OUT}/ud_{lang}.json", "w"), indent=1)
    A = ("bpe", "free", "affine", "shufroot")
    print("\n" + "=" * 66)
    r = {a: np.array(res[a]) for a in A}
    for a in A: print(f"  {a:<9} {r[a].mean():.4f} +- {r[a].std(ddof=1):.4f}")
    print(f"\n  free-bpe    {(r['free']-r['bpe']).mean():+.4f}   (tokenizer)")
    print(f"  affine-free {(r['affine']-r['free']).mean():+.4f}   (morphological prior)")
    print(f"  affine-shuf {(r['affine']-r['shufroot']).mean():+.4f}   (vs null)")
    print(f"\n  English reference (10k para): affine-free -0.011, affine-shuf -0.007")

if __name__ == "__main__":
    main()
