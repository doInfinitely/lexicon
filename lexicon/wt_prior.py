"""Initialise the token embeddings from the surface prototype space.

Remy: "we are absolutely tossing away information with our tokenization ... a stream of
lexeme embeddings and operator embeddings would exploit the knowledge in the word
embedding space."

Right. `<lex:walk>` is an arbitrary integer; the model must learn from 1.4M words that
walk is near run. We have that geometry: real/surface/prototypes.pt, 51,148 x 768.
Under the data-scarce frame a pretrained embedding IS the sample-efficiency device.

GPT already ties head.weight = tok.weight, so the "sample by temperature over embedding
distance" decoder is already what it does: P(w) ∝ exp(-||h-e_w||^2/T) expands to a
softmax with tied embeddings and a norm bias. Nothing new to build on the output side;
the missing piece is that the space is random.

OPERATOR INIT FROM THE ATLAS. The atlas says morphology TRANSLATES (identity→translation
gains +0.179 on inflection, +0.009 on lexicographic; orthogonal wins on meaning). A token
embedding enters the residual stream ADDITIVELY, so it can natively express a translation
and cannot natively express a rotation. That is very likely WHY derivation deflated to
zero while inflection carries everything. So: init <op:s> to the mean displacement
mean(e[child] - e[base]) over that slot's pairs.

THE NULL (mandatory): SHUFFLED prototypes -- same vectors, permuted across lexemes. Same
norms, same anisotropy, same init scale, no semantic alignment. If proto beats random but
shuffled matches proto, the win is initialisation statistics, not geometry. Cf. the
random-untrained-V mirror result (0.300 vs 0.301).

FIRST RUN CAUGHT A STRAWMAN BASELINE. `rand` used nn.Embedding's default N(0,1) while the
prototype rows are normalised to std 0.02. proto 2.309 vs rand 2.691 was an INIT-SCALE
effect, and shuf scored 2.309 too -- identical, because the projected prototypes are near
isotropic (mean cos between rows +0.0042; top singular value carries 1.2% of the spectrum).
So `rand02` (scale-matched Gaussian) is the honest baseline, and proto-vs-shuf is the only
contrast that can speak about geometry.

LEAKAGE, stated up front: distilbert was pretrained on Wikipedia+BookCorpus, and
wikitext-103 IS Wikipedia. So a prototype-initialised model is doing TRANSFER from a
large corpus, not extracting more from the small one. This CANNOT support a claim that
"our language exploits a small dataset better". It can support "transfer helps
low-resource", which is a different and weaker claim. The shuffled null does not fix this;
only a prototype space trained on disjoint text would.
"""
import json, collections
import numpy as np, torch, torch.nn as nn
from lexicon.ts_lm import GPT, DEVICE, OUT
from lexicon.ts_eval2 import bits_per_char
from lexicon.wt_scale import BPETok, corpus, train_and_loss, N_PARA
from lexicon.wt_seeds2 import Lex

SEEDS = [0, 1, 2]
L, D, H = 6, 384, 6
RUNGS = [10000, 40000]
EPOCHS, BS, CTX_ = 2, 24, 512
INFL = {"noun.plural","verb.ger","verb.3sg","verb.ptcp","verb.past","adj.comp","adj.sup"}


def build_prior(itos, d):
    """-> [V, d] init matrix, and a mask of which rows were actually filled."""
    P = torch.load("real/surface/prototypes.pt", map_location="cpu", weights_only=False)
    forest = json.load(open("dictionary/forest_v6.json"))
    parent = {k: tuple(v) for k, v in forest["parent"].items()}

    lex = {t[5:-1] for t in itos if t.startswith("<lex:")}
    words = [w for w in lex if w in P]
    X = np.stack([np.asarray(P[w], dtype=np.float32) for w in words])
    mu = X.mean(0)
    U, S, Vt = np.linalg.svd(X - mu, full_matrices=False)
    W = Vt[:d].T                                     # 768 x d
    proj = {w: (np.asarray(P[w], dtype=np.float32) - mu) @ W for w in words}

    # operator = mean displacement over its pairs, projected WITHOUT the mean (linear)
    disp = collections.defaultdict(list)
    for child, (slot, base) in parent.items():
        if child in P and base in P:
            disp[slot].append((np.asarray(P[child], dtype=np.float32) -
                               np.asarray(P[base], dtype=np.float32)) @ W)
    ops = {s: np.mean(v, 0) for s, v in disp.items() if len(v) >= 20}

    E = torch.randn(len(itos), d) * 0.02
    filled = torch.zeros(len(itos), dtype=torch.bool)
    for i, t in enumerate(itos):
        if t.startswith("<lex:") and t[5:-1] in proj:
            E[i] = torch.from_numpy(proj[t[5:-1]]); filled[i] = True
        elif t.startswith("<op:") and t[4:-1] in ops:
            E[i] = torch.from_numpy(ops[t[4:-1]]); filled[i] = True
    # match the scale of default init on the filled rows
    E[filled] = E[filled] / E[filled].std() * 0.02
    return E, filled, len(ops)


def shuffle_prior(E, filled, seed=0):
    E2 = E.clone()
    idx = torch.where(filled)[0]
    g = torch.Generator().manual_seed(seed)
    E2[idx] = E[idx[torch.randperm(len(idx), generator=g)]]
    return E2


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
        bpe = BPETok(tr); nb = len(corpus(bpe, tr))
        tok = Lex(tr, "dictionary/forest_v6.json", True, True, "lex-v6")
        data = corpus(tok, tr)
        E0, filled, nops = build_prior(tok.itos, D)
        print(f"\n--- {rung:,} paragraphs --- lex-v6 tokens {len(data):,} ({len(data)/nb:.3f}x bpe)")
        print(f"    prior filled {filled.sum().item():,}/{len(tok.itos):,} rows "
              f"({nops} operators from atlas displacements)", flush=True)
        steps = max(60, int(EPOCHS * len(data) / (BS * CTX_)))
        bsteps = max(60, int(EPOCHS * nb / (BS * CTX_)))

        for seed in SEEDS:
            for arm in ("bpe", "rand", "rand02", "proto", "shuf", "frozen"):
                torch.manual_seed(seed); np.random.seed(seed)
                if arm == "bpe":
                    m = GPT(len(bpe.itos), d=D, layers=L, heads=H).to(DEVICE)
                    train_and_loss(m, corpus(bpe, tr), len(bpe.itos), steps=bsteps)
                    b = bits_per_char(m, bpe, ev)
                else:
                    m = GPT(len(tok.itos), d=D, layers=L, heads=H).to(DEVICE)
                    if arm == "rand02":
                        m.tok.weight.data.normal_(0, 0.02)      # scale-matched baseline
                    if arm == "proto" or arm == "frozen":
                        m.tok.weight.data.copy_(E0.to(DEVICE))
                    elif arm == "shuf":
                        m.tok.weight.data.copy_(shuffle_prior(E0, filled, seed).to(DEVICE))
                    if arm == "frozen":
                        m.tok.weight.requires_grad_(False)
                    train_and_loss(m, data, len(tok.itos), steps=steps)
                    b = bits_per_char(m, tok, ev)
                res[(rung, arm)].append(b)
                print(f"  seed {seed} {arm:<7} bits/char {b:.4f}", flush=True)
                del m; torch.cuda.empty_cache()
        json.dump({f"{k[0]}_{k[1]}": v for k,v in res.items()},
                  open(f"{OUT}/wt_prior.json","w"), indent=1)

    print("\n" + "="*80)
    print(f"{'paragraphs':<12}{'bpe':>9}{'rand':>9}{'rand02':>9}{'proto':>9}{'shuf':>9}"
          f"{'frozen':>9}{'proto-rand02':>13}{'proto-shuf':>12}")
    print("-"*90)
    for rung in RUNGS:
        A = ("bpe","rand","rand02","proto","shuf","frozen")
        r = {a: np.array(res[(rung,a)]).mean() for a in A}
        print(f"{rung:<12,}" + "".join(f"{r[a]:>9.3f}" for a in A) +
              f"{r['proto']-r['rand02']:>13.4f}{r['proto']-r['shuf']:>12.4f}")
    print("\nrand   : nn.Embedding default N(0,1) -- a STRAWMAN, kept only to show the trap")
    print("rand02 : scale-matched Gaussian, the honest baseline")
    print("proto-rand02 : does the pretrained geometry help over a well-scaled random init?")
    print("proto-shuf   : is it the GEOMETRY, or just the init statistics? (the null)")
    print("LEAKAGE: distilbert saw Wikipedia; wikitext IS Wikipedia. This is transfer,")
    print("         not 'extracting more from a small corpus'. See module docstring.")

if __name__=="__main__":
    main()
