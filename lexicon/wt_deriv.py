"""Does DERIVATION come back when the operator is a matrix instead of a token?

The deflation (v6 == v4, derivation worth 0 bits/char) was measured under TWO handicaps
at once, and neither was noticed:

  1. Derivational operators were TOKENS, so every derived word cost extra sequence
     length -- and length was the largest single effect in the whole study
     (v4 -> v3: +0.073 bits/char for +4% tokens).
  2. A token embedding enters the residual stream ADDITIVELY, so it can only ever be a
     TRANSLATION. The atlas says morphology translates but MEANING ROTATES
     (identity->translation: +0.179 inflection, +0.009 lexicographic; orthogonal maps
     gain +0.201 on lexicographic and win 10/14). Derivation was forbidden the one
     operation it needs.

As a pre-model affine map both handicaps vanish: `worker` stays ONE token, and
A_agent . E_work + d_agent can rotate.

    e(w) = A_s ( e(parent) ) + d_s ,    A_s = I + U_s V_s^T   (rank r)
    composed along the chain: settlements = A_plural(A_ment(E_settle) + d_ment) + d_plural
    slot 0 (identity) pinned: A_0 = I, d_0 = 0     <- gauge

Low-rank rather than exactly orthogonal, because the atlas's rotations live in LOW-DIM
subspaces (the antonymy polarity subspace is 1-8 dims). I + UV^T rotates inside a small
subspace and is the identity elsewhere. An orthogonal A would also force ||Ax|| = ||x||,
and there is no evidence meaning-change preserves norm.

ARMS (identical token stream everywhere -- one token per word, byte fallback, no <unk>):
    free        16k independent word vectors
    infl        affine ops on INFLECTION only (forest_v6); derived words are own roots
    deriv-tr    all edges (forest_v3), TRANSLATION only (A = I)
    deriv-aff   all edges, affine (A = I + UV^T)
    shufroot    deriv-aff with each word pointing at a RANDOM root  <- the null

deriv-aff vs infl      : does derivation pay once it is free of token cost?
deriv-aff vs deriv-tr  : does derivation need the MATRIX? (atlas predicts yes)
infl-aff  vs infl-tr   : inflection should NOT need it (atlas predicts ~0)  [via deriv-tr]
deriv-aff vs shufroot  : morphology, or just a low-rank reparameterisation?
"""
import json, collections
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F
from lexicon.ts_lm import GPT, Block, DEVICE, OUT, CTX, WORD_RE
from lexicon.ts_eval2 import bits_per_char
from lexicon.wt_scale import corpus, N_PARA, train_and_loss
from lexicon.bytetok import ByteBPETok, NBYTE, VOCAB
from lexicon.ts_encode import LexemeTokenizer

SEEDS = [0, 1, 2]
L, D, H = 6, 384, 6
RUNGS = [10000, 40000]
EPOCHS, BS, CTX_ = 2, 24, 512
RANK, MIN_SLOT = 24, 30

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



class ByteWordTok:
    def __init__(self, texts, max_vocab=VOCAB):
        self.lex = LexemeTokenizer()
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
        k = w.lower() if (w.isalpha() or "'" in w) else f"<p:{w}>"
        out = [self.stoi[k]] if k in self.stoi else [self.b0 + b for b in w.encode("utf-8")]
        self._pc[w] = out
        return out
    def enc_words(self, ws):
        o=[]
        for w in ws: o += self.enc_word(w)
        return o


def build_graph(itos, forest_path, shuffle_seed=None):
    """nodes = vocab tokens + any intermediate words on their chains.
    returns node parent/slot arrays, root ids, vocab->node map, slot names."""
    f = json.load(open(forest_path))
    par = {k: tuple(v) for k, v in f["parent"].items()}
    use = collections.Counter(s for s, _ in par.values())
    par = {w: v for w, v in par.items() if use[v[0]] >= MIN_SLOT}

    nodes, nid = [], {}
    def get(x):
        if x not in nid: nid[x] = len(nodes); nodes.append(x)
        return nid[x]
    slots = ["<id>"] + sorted({s for s, _ in par.values()})
    sidx = {s: i for i, s in enumerate(slots)}

    for t in itos: get(t)
    i = 0
    while i < len(nodes):                 # expand chains
        w = nodes[i]; i += 1
        if w in par: get(par[w][1])
    P = np.arange(len(nodes)); S = np.zeros(len(nodes), dtype=np.int64)
    for w, k in nid.items():
        if w in par:
            s, b = par[w]; P[k] = nid[b]; S[k] = sidx[s]
    if shuffle_seed is not None:
        g = np.random.default_rng(shuffle_seed)
        for s in range(1, len(slots)):
            m = np.where(S == s)[0]
            if len(m) > 1: P[m] = P[m][g.permutation(len(m))]
    depth = np.zeros(len(nodes), dtype=np.int64)
    for _ in range(6):
        depth = np.where(S > 0, depth[P] + 1, 0)
    roots = np.where(S == 0)[0]
    vocab_nodes = np.array([nid[t] for t in itos])
    return (torch.from_numpy(P), torch.from_numpy(S), torch.from_numpy(depth),
            torch.from_numpy(roots), torch.from_numpy(vocab_nodes), slots)


class DerivGPT(nn.Module):
    def __init__(self, P, S, depth, roots, vnodes, nslots, mode,
                 d=D, layers=L, heads=H, ctx=CTX, rank=RANK):
        super().__init__()
        self.mode = mode
        for n, t in (("P",P),("S",S),("depth",depth),("roots",roots),("vnodes",vnodes)):
            self.register_buffer(n, t)
        self.maxd = int(depth.max())
        self.E = nn.Embedding(len(roots), d)
        self.rmap = torch.full((len(P),), -1, dtype=torch.long)
        self.rmap[roots] = torch.arange(len(roots))
        self.register_buffer("rmap_b", self.rmap)
        self.dop = nn.Parameter(torch.zeros(nslots - 1, d))
        if mode == "affine":
            self.U = nn.Parameter(torch.zeros(nslots - 1, d, rank))
            self.V = nn.Parameter(torch.randn(nslots - 1, d, rank) * 0.02)
        self.pos = nn.Embedding(ctx, d)
        self.blocks = nn.ModuleList([Block(d, heads) for _ in range(layers)])
        self.ln = nn.LayerNorm(d)
        self.ctx = ctx
        self.nslots = nslots

    def node_emb(self):
        """Group by SLOT inside each depth: one [d,rank] matmul per slot.

        The per-node gather V[sl-1] materialised [n, d, rank] -- for n~12k nodes that is
        442 MB per tensor and it blew up inside triton. Slots are few (<=88); nodes are
        many. Same fix as the [V, d, d] blowup in ts_factored.
        """
        N = len(self.P)
        X = torch.zeros(N, self.E.weight.shape[1], device=self.E.weight.device)
        X = X.index_copy(0, self.roots,
                         self.E(torch.arange(len(self.roots), device=X.device)))
        for dpt in range(1, self.maxd + 1):
            at_depth = (self.depth == dpt)
            if not at_depth.any(): continue
            newX = X.clone()
            for s in range(1, self.nslots):
                m = at_depth & (self.S == s)
                if not m.any(): continue
                idx = m.nonzero(as_tuple=True)[0]
                src = X[self.P[idx]]
                out = src + self.dop[s - 1]
                if self.mode == "affine":
                    out = out + (src @ self.V[s - 1]) @ self.U[s - 1].T
                newX = newX.index_copy(0, idx, out)
            X = newX
        return X[self.vnodes]

    def forward(self, idx):
        B, T = idx.shape
        W = self.node_emb()
        mask = causal(T, idx.device)
        x = F.embedding(idx, W) + self.pos(torch.arange(T, device=idx.device))[None]
        for b in self.blocks: x = b(x, mask)
        return self.ln(x) @ W.T


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
        tok = ByteWordTok(tr); data = corpus(tok, tr)
        G6 = build_graph(tok.itos, "dictionary/forest_v6.json")
        G3 = build_graph(tok.itos, "dictionary/forest_v3.json")
        G3s = build_graph(tok.itos, "dictionary/forest_v3.json", shuffle_seed=0)
        print(f"\n--- {rung:,} paragraphs --- word tokens {len(data):,}")
        print(f"  v6 graph: {len(G6[5])} slots, roots {len(G6[3]):,}, maxdepth {int(G6[2].max())}")
        print(f"  v3 graph: {len(G3[5])} slots, roots {len(G3[3]):,}, maxdepth {int(G3[2].max())}",
              flush=True)
        steps = max(60, int(EPOCHS * len(data) / (BS * CTX_)))

        for seed in SEEDS:
            for arm in ("free","infl","deriv-tr","deriv-aff","shufroot"):
                torch.manual_seed(seed); np.random.seed(seed)
                if arm == "free":
                    m = GPT(len(tok.itos), d=D, layers=L, heads=H)
                else:
                    G = G6 if arm=="infl" else (G3s if arm=="shufroot" else G3)
                    mode = "translate" if arm=="deriv-tr" else "affine"
                    m = DerivGPT(*G[:5], len(G[5]), mode)
                init_scale(m); m = m.to(DEVICE)
                train_and_loss(m, data, len(tok.itos), steps=steps, seed=seed)
                b = bits_per_char(m, tok, ev)
                res[(rung,arm)].append(b)
                print(f"  seed {seed} {arm:<10} bits/char {b:.4f}", flush=True)
                del m; torch.cuda.empty_cache()
        json.dump({f"{k[0]}_{k[1]}": v for k,v in res.items()},
                  open(f"{OUT}/wt_deriv.json","w"), indent=1)

    A = ("free","infl","deriv-tr","deriv-aff","shufroot")
    print("\n" + "="*94)
    print(f"{'paragraphs':<12}" + "".join(f"{a:>12}" for a in A) +
          f"{'aff-infl':>11}{'aff-tr':>9}{'aff-shuf':>11}")
    print("-"*94)
    for rung in RUNGS:
        r = {a: np.array(res[(rung,a)]).mean() for a in A}
        print(f"{rung:<12,}" + "".join(f"{r[a]:>12.4f}" for a in A) +
              f"{r['deriv-aff']-r['infl']:>+11.4f}{r['deriv-aff']-r['deriv-tr']:>+9.4f}"
              f"{r['deriv-aff']-r['shufroot']:>+11.4f}")
    print("\naff-infl : does derivation pay once free of token cost? (negative = yes)")
    print("aff-tr   : does derivation need the MATRIX? (atlas predicts yes => negative)")
    print("aff-shuf : morphology, or a low-rank reparameterisation? (the null)")

if __name__=="__main__":
    main()
