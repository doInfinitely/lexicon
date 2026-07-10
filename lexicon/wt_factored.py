"""Operator as an AFFINE map applied before the model: e(w) = E_root + d_slot + low-rank.

Remy: "make the operator a matrix not a token? apply it to the lexeme before the model
gets it?" then "do we also need an additive c vector?"

YES, and the bias is the MAIN TERM. A linear map W.x fixes the origin and cannot express
x -> x + d. The atlas says inflection IS a translation (identity->translation gains
+0.179 on inflectional relations; a single mean displacement vector carries child->children
at rank 1). So d_slot is the operator and W should be a small perturbation of I.

The first version of this file had `Wop = I + 0.01*randn` and NO bias -- it could not
represent a displacement at all. It would not have failed loudly: BERT embeddings are
strongly anisotropic, so a linear map can use the shared mean direction as a pseudo-bias
and FAKE a translation. It would have limped along and produced a mediocre number.

PARAMETERISATION
    e(word) = E_root + d_slot + U_slot (V_slot^T E_root)          rank r correction
    slot 0 (identity): d_0 = 0, U_0 = V_0 = 0, PINNED.
Pinning slot 0 ties E_root to the real embedding of the root and kills the gauge freedom
that has bitten this project three times (mirror V, ts_factored identity slot, here).

Full d x d matrices would be 147k params per slot fit from a few hundred words each --
overparameterised, and free enough to UNDO the sharing. Rank 32 is 24.6k. d_slot is 384,
initialised to the measured mean displacement, so the model STARTS at the atlas's
prediction and can only improve on it.

ARMS (identical token stream in every lexeme arm -- length exactly controlled):
    bpe         reference (byte fallback, no <unk>)
    free        16k independent word vectors                    <- honest baseline
    translate   e = E_root + d_slot          (W = I; the atlas's pure claim)
    affine      e = E_root + d_slot + U V^T E_root   (rank 32)
    shufroot    affine, but each word points at a RANDOM root of the same slot   <- NULL

translate vs free : does enforced morphological sharing help?
affine vs translate : does inflection need more than a translation?   (atlas says no)
affine vs shufroot : is it MORPHOLOGY or just a low-rank reparameterisation?  (the null)

An INITIALISATION is a suggestion; a FACTORISATION is a constraint. wt_prior showed the
LM walks away from a prototype init immediately (frozen prototypes score 3.10 vs 2.31
trainable). Here `walked` has no parameter of its own to drift with.
"""
import json, collections
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F
from lexicon.ts_lm import GPT, Block, DEVICE, OUT, CTX, WORD_RE
from lexicon.ts_eval2 import bits_per_char
from lexicon.wt_scale import corpus, N_PARA
from lexicon.bytetok import ByteBPETok, NBYTE, VOCAB
from lexicon.ts_encode import LexemeTokenizer

SEEDS = [0, 1, 2]
L, D, H = 6, 384, 6
RUNGS = [10000, 40000]
EPOCHS, BS, CTX_ = 2, 24, 512
RANK = 32

def causal(T, device):
    return torch.triu(torch.full((T, T), float("-inf"), device=device), 1)


def init_scale(m, std=0.02):
    """GPT-2 style: token AND positional embeddings at the same small scale.

    ts_lm.GPT leaves both at PyTorch's default N(0,1). wt_prior normalised only the TOKEN
    rows to 0.02, making the positional signal 50x larger than word identity -- which is
    why rand02/proto/shuf came out identical to 3 dp at the 10k rung. Any arm that touches
    embedding scale must touch both, or it is comparing optimisation regimes.
    """
    for name in ("tok", "pos", "E"):
        e = getattr(m, name, None)
        if e is not None and hasattr(e, "weight"):
            e.weight.data.normal_(0, std)

SLOTS = ["<id>", "noun.plural", "verb.ger", "verb.3sg", "verb.ptcp",
         "verb.past", "adj.comp", "adj.sup"]


class ByteWordTok:
    """one token per word; byte fallback (no <unk>). shared by free/translate/affine/shuf."""
    name = "word"
    def __init__(self, texts, max_vocab=VOCAB):
        self.lex = LexemeTokenizer()
        f = json.load(open("dictionary/forest_v6.json"))
        self.parent = {k: tuple(v) for k, v in f["parent"].items()}
        c = collections.Counter()
        for t in texts[:20000]:
            for m in WORD_RE.findall(t):
                c[m.lower() if (m.isalpha() or "'" in m) else f"<p:{m}>"] += 1
        keep = [w for w, _ in c.most_common(max_vocab - NBYTE)]
        self.itos = ["<pad>"] + [f"<b:{i}>" for i in range(NBYTE)] + keep
        self.b0 = 1
        self.stoi = {w: 1 + NBYTE + i for i, w in enumerate(keep)}
        self._pc = {}

    def enc_word(self, w):
        if w in self._pc: return self._pc[w]
        key = w.lower() if (w.isalpha() or "'" in w) else f"<p:{w}>"
        out = [self.stoi[key]] if key in self.stoi else [self.b0 + b for b in w.encode("utf-8")]
        self._pc[w] = out
        return out

    def enc_words(self, ws):
        o = []
        for w in ws: o += self.enc_word(w)
        return o

    def factorise(self, shuffle_seed=None):
        roots, rid = [], {}
        def get(r):
            if r not in rid: rid[r] = len(roots); roots.append(r)
            return rid[r]
        R = np.zeros(len(self.itos), dtype=np.int64)
        S = np.zeros(len(self.itos), dtype=np.int64)
        for i, t in enumerate(self.itos):
            if t.startswith(("<p:", "<b:")) or t == "<pad>":
                R[i] = get(t); S[i] = 0; continue
            if t in self.parent:
                slot, base = self.parent[t]
                R[i] = get(base); S[i] = SLOTS.index(slot) if slot in SLOTS else 0
            else:
                R[i] = get(t); S[i] = 0
        if shuffle_seed is not None:
            g = np.random.default_rng(shuffle_seed)
            for s in range(1, len(SLOTS)):
                m = np.where(S == s)[0]
                if len(m) > 1: R[m] = R[m][g.permutation(len(m))]
        return torch.from_numpy(R), torch.from_numpy(S), len(roots), roots


def init_displacements(roots, itos, R, S, d):
    """d_slot = mean(e[child] - e[base]) in the projected prototype space; else zeros."""
    P = torch.load("real/surface/prototypes.pt", map_location="cpu", weights_only=False)
    have = [w for w in roots if w in P]
    if len(have) < 100: return torch.zeros(len(SLOTS), d)
    X = np.stack([np.asarray(P[w], dtype=np.float32) for w in have])
    mu = X.mean(0)
    _, _, Vt = np.linalg.svd(X - mu, full_matrices=False)
    W = Vt[:d].T
    acc = collections.defaultdict(list)
    for i, t in enumerate(itos):
        s = int(S[i])
        if s == 0: continue
        base = roots[int(R[i])]
        if t in P and base in P:
            acc[s].append((np.asarray(P[t], np.float32) - np.asarray(P[base], np.float32)) @ W)
    dd = torch.zeros(len(SLOTS), d)
    for s, v in acc.items():
        if len(v) >= 20:
            m = np.mean(v, 0)
            dd[s] = torch.from_numpy(m / (np.linalg.norm(m) + 1e-9) * 0.02 * np.sqrt(d))
    return dd


class AffineGPT(nn.Module):
    def __init__(self, R, S, n_roots, d0, mode, d=D, layers=L, heads=H, ctx=CTX, rank=RANK):
        super().__init__()
        self.mode = mode
        self.register_buffer("wr", R); self.register_buffer("ws", S)
        self.E = nn.Embedding(n_roots, d)
        S_ = len(SLOTS)
        self.register_buffer("d0", torch.zeros(1, d))                 # slot 0 bias pinned to 0
        self.dop = nn.Parameter(d0[1:].clone())                       # [S-1, d]
        if mode == "affine":
            self.U = nn.Parameter(torch.zeros(S_ - 1, d, rank))       # correction starts at 0
            self.V = nn.Parameter(torch.randn(S_ - 1, d, rank) * 0.02)
        self.pos = nn.Embedding(ctx, d)
        self.blocks = nn.ModuleList([Block(d, heads) for _ in range(layers)])
        self.ln = nn.LayerNorm(d)
        self.ctx = ctx

    def word_emb(self):
        er = self.E(self.wr)
        out = er.clone()
        for s in range(1, len(SLOTS)):
            m = self.ws == s
            if not m.any(): continue
            v = er[m] + self.dop[s - 1]
            if self.mode == "affine":
                v = v + (er[m] @ self.V[s - 1]) @ self.U[s - 1].T
            out[m] = v
        return out

    def forward(self, idx):
        B, T = idx.shape
        W = self.word_emb()
        mask = causal(T, idx.device)
        x = F.embedding(idx, W) + self.pos(torch.arange(T, device=idx.device))[None]
        for b in self.blocks: x = b(x, mask)
        return self.ln(x) @ W.T


def train(m, data, V, steps, seed):
    from lexicon.wt_scale import train_and_loss
    return train_and_loss(m, data, V, steps=steps, seed=seed)


def main():
    from datasets import load_dataset
    ds = load_dataset("Salesforce/wikitext","wikitext-103-raw-v1",split="train")
    texts=[]
    for r in ds:
        t=r["text"]
        if len(t)>200 and not t.lstrip().startswith("="): texts.append(t)
        if len(texts)>=N_PARA: break
    full_tr, ev = texts[:-2000], texts[-2000:]

    res = collections.defaultdict(list)
    for rung in RUNGS:
        tr = full_tr[:rung]
        bpe = ByteBPETok(tr); bd = corpus(bpe, tr)
        tok = ByteWordTok(tr); data = corpus(tok, tr)
        R, S, nR, roots = tok.factorise()
        Rs, Ss, _, _ = tok.factorise(shuffle_seed=0)
        # d_slot from PROTOTYPE-space displacements is meaningless when E_root is a random
        # Gaussian -- different bases. Start at zero and let the model learn the operator.
        # (wt_prior showed a prototype init is washed out anyway: proto == shuf.)
        d0 = torch.zeros(len(SLOTS), D)
        infl = int((S > 0).sum())
        print(f"\n--- {rung:,} paragraphs ---")
        print(f"  bpe tokens {len(bd):,}   word tokens {len(data):,}  {len(data)/len(bd):.3f}x")
        print(f"  vocab {len(tok.itos):,}  roots {nR:,}  inflected {infl:,}  "
              f"rank {RANK}  |d_slot| {[round(float(x.norm()),3) for x in d0[1:]]}", flush=True)
        steps  = max(60, int(EPOCHS * len(data) / (BS * CTX_)))
        bsteps = max(60, int(EPOCHS * len(bd) / (BS * CTX_)))

        for seed in SEEDS:
            for arm in ("bpe","free","translate","affine","shufroot"):
                torch.manual_seed(seed); np.random.seed(seed)
                if arm == "bpe":
                    m = GPT(len(bpe.itos), d=D, layers=L, heads=H); init_scale(m); m=m.to(DEVICE)
                    train(m, bd, len(bpe.itos), bsteps, seed); b = bits_per_char(m, bpe, ev)
                elif arm == "free":
                    m = GPT(len(tok.itos), d=D, layers=L, heads=H); init_scale(m); m=m.to(DEVICE)
                    train(m, data, len(tok.itos), steps, seed); b = bits_per_char(m, tok, ev)
                else:
                    rr, ss = (Rs, Ss) if arm == "shufroot" else (R, S)
                    mode = "translate" if arm == "translate" else "affine"
                    m = AffineGPT(rr, ss, nR, d0, mode); init_scale(m); m = m.to(DEVICE)
                    train(m, data, len(tok.itos), steps, seed); b = bits_per_char(m, tok, ev)
                res[(rung, arm)].append(b)
                print(f"  seed {seed} {arm:<10} bits/char {b:.4f}", flush=True)
                del m; torch.cuda.empty_cache()
        json.dump({f"{k[0]}_{k[1]}": v for k,v in res.items()},
                  open(f"{OUT}/wt_factored.json","w"), indent=1)

    print("\n" + "="*92)
    A = ("bpe","free","translate","affine","shufroot")
    print(f"{'paragraphs':<12}" + "".join(f"{a:>11}" for a in A) +
          f"{'tr-free':>10}{'aff-tr':>9}{'aff-shuf':>11}")
    print("-"*92)
    for rung in RUNGS:
        r = {a: np.array(res[(rung,a)]).mean() for a in A}
        print(f"{rung:<12,}" + "".join(f"{r[a]:>11.4f}" for a in A) +
              f"{r['translate']-r['free']:>+10.4f}{r['affine']-r['translate']:>+9.4f}"
              f"{r['affine']-r['shufroot']:>+11.4f}")
    print("\ntr-free  : does ENFORCED morphological sharing help? (negative = yes)")
    print("aff-tr   : does inflection need more than a translation? (atlas says no => ~0)")
    print("aff-shuf : morphology, or just a low-rank reparameterisation? (the null)")

if __name__=="__main__":
    main()
