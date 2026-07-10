"""Does factoring buy a SMALLER COMPETENT MODEL?

The goal is TinyStories-style: shrink what must be learned. The embedding table
is where the parameters live -- 12,018 words x 384 = 4.6M. Factoring stores
7,671 roots (2.9M), but full-rank operators cost 24 x 384^2 = 3.5M and make the
model BIGGER. Low-rank operators (W_s = I + U_s V_s^T, rank r) cost
24 x 2 x 384 x r; at r=32 that is 0.6M, and morphology becomes nearly free.

Three arms, identical trunk, identical steps, identical word-level vocabulary
(so per-word NLL is directly comparable -- no tokenizer confound):

    free-word        one embedding row per word          4.6M embedding params
    factored (full)  W_slot . E_root, full-rank W        6.4M
    factored (r=32)  W_s = I + U_s V_s^T                 3.5M

Measured: per-word negative log-likelihood on held-out text, BINNED BY TRAINING
FREQUENCY. The claim worth making is not zero-shot on `mice`; it is that a rare
word with three occurrences is modelled better when its embedding is computed
from a frequent root than when it is a nearly-untrained row of its own. If the
curves separate in the 1-50 bin, the language pays for itself on any corpus.

Bin 0 (never seen) is the 15 held-out forms, masked out of the training softmax
so they are never negatives either.
"""
import json, collections, math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from lexicon.ts_factored import Vocab, FreeWord, corpus, train, D, SEQ, TAU
from lexicon.ts_lm import Block, build, WORD_RE, DEVICE, OUT
from lexicon.ts_encode import load_forest


class FactoredLR(nn.Module):
    """W_slot = I + U_s V_s^T. rank=None -> full-rank matrices."""

    def __init__(self, vocab, d=D, layers=6, heads=6, rank=32):
        super().__init__()
        self.E = nn.Embedding(len(vocab.roots), d)
        nn.init.normal_(self.E.weight, std=0.02)
        S = len(vocab.slots)
        self.rank = rank
        if rank is None:
            self.Wop = nn.Parameter(torch.eye(d).repeat(S - 1, 1, 1)
                                    + 0.01 * torch.randn(S - 1, d, d))
        else:
            self.U = nn.Parameter(0.01 * torch.randn(S - 1, d, rank))
            self.V = nn.Parameter(0.01 * torch.randn(S - 1, d, rank))
        self.register_buffer("W0", torch.eye(d))
        self.pos = nn.Embedding(SEQ, d)
        self.blocks = nn.ModuleList([Block(d, heads) for _ in range(layers)])
        self.ln = nn.LayerNorm(d)
        self.register_buffer("wr", vocab.wr.clone())
        self.register_buffer("ws", vocab.ws.clone())
        self.d = d

    def slot_W(self, s):
        if s == 0:
            return self.W0
        if self.rank is None:
            return self.Wop[s - 1]
        return torch.eye(self.d, device=self.U.device) + self.U[s-1] @ self.V[s-1].T

    def word_emb(self):
        er = self.E(self.wr)
        out = torch.empty_like(er)
        n_slots = 1 + (self.Wop.shape[0] if self.rank is None else self.U.shape[0])
        for s in range(n_slots):
            m = self.ws == s
            if m.any():
                out[m] = er[m] @ self.slot_W(s).T
        return out

    def forward(self, idx, table=None):
        T = idx.shape[1]
        tbl = self.word_emb() if table is None else table
        m = torch.triu(torch.full((T, T), float("-inf"), device=idx.device), 1)
        x = tbl[idx] + self.pos(torch.arange(T, device=idx.device))[None]
        for b in self.blocks:
            x = b(x, m)
        h = F.normalize(self.ln(x), dim=-1)
        return h @ F.normalize(tbl, dim=-1).T / TAU


@torch.no_grad()
def per_word_nll(model, V, texts, want=None, max_tok=200000):
    """mean -log P(word | context) for each word type."""
    model.eval()
    tbl = model.word_emb() if hasattr(model, "word_emb") else None
    acc = collections.defaultdict(list)
    seen = 0
    for t in texts:
        ids = V.enc(WORD_RE.findall(t))
        if len(ids) < 8:
            continue
        ids = ids[:SEQ]
        x = torch.tensor(ids, device=DEVICE)[None]
        lg = F.log_softmax(model(x, tbl)[0, :-1], -1)
        tgt = x[0, 1:]
        nll = -lg[torch.arange(len(ids) - 1), tgt]
        for j, wid in enumerate(tgt.tolist()):
            w = V.itos[wid]
            if want is None or w in want:
                acc[w].append(nll[j].item())
        seen += len(ids)
        if seen > max_tok:
            break
    return {w: float(np.mean(v)) for w, v in acc.items() if v}


def main():
    train_texts, clean_eval, test_texts, held = build()
    parent, _ = load_forest()
    hset = {w for w, _, _ in held}
    V = Vocab(train_texts, parent)
    for w in hset:
        if w in parent:
            V.add_word(w, parent)
    held_ids = torch.tensor([V.stoi[w] for w in hset if w in V.stoi], device=DEVICE)

    freq = collections.Counter()
    for t in train_texts:
        freq.update(m.lower() for m in WORD_RE.findall(t) if m.isalpha())
    for w in hset:
        freq[w] = 0

    data = corpus(V, train_texts)
    print(f"\nvocab {len(V.itos)}, roots {len(V.roots)}, slots {len(V.slots)}")
    print(f"masked (never seen, never a negative): {len(held_ids)}\n")

    arms = [("free-word", lambda: FreeWord(V)),
            ("factored (full-rank W)", lambda: FactoredLR(V, rank=None)),
            ("factored (rank 32)", lambda: FactoredLR(V, rank=32))]

    def emb_params(m):
        n = 0
        for nm, p in m.named_parameters():
            if nm.startswith(("tok", "E.", "Wop", "U", "V")):
                n += p.numel()
        return n

    res, nlls = {}, {}
    for name, ctor in arms:
        print(f"--- {name} ---")
        m = ctor().to(DEVICE)
        tot = sum(p.numel() for p in m.parameters())
        ep = emb_params(m)
        print(f"    total {tot/1e6:.2f}M   embedding/operator params {ep/1e6:.2f}M")
        train(m, data, mask_ids=held_ids)
        # clean held-out text for the seen bins; test text for bin 0
        d1 = per_word_nll(m, V, clean_eval)
        d0 = per_word_nll(m, V, test_texts, want=hset)
        d1.update({k: v for k, v in d0.items()})
        nlls[name] = d1
        res[name] = dict(total=tot, emb=ep)
        del m; torch.cuda.empty_cache()

    BINS = [(0, 0, "0 (never seen)"), (1, 5, "1-5"), (6, 50, "6-50"),
            (51, 500, "51-500"), (501, 10**9, "501+")]
    print("\n" + "=" * 84)
    print("per-word NLL on held-out text, binned by TRAINING frequency (lower is better)")
    print("=" * 84)
    hdr = f"{'freq bin':<16}{'n words':>9}"
    for name, _ in arms:
        hdr += f"{name[:16]:>18}"
    print(hdr); print("-" * len(hdr))
    for lo, hi, lab in BINS:
        ws = [w for w in nlls[arms[0][0]] if lo <= freq[w] <= hi]
        ws = [w for w in ws if all(w in nlls[n] for n, _ in arms)]
        if len(ws) < 3:
            continue
        row = f"{lab:<16}{len(ws):>9}"
        for name, _ in arms:
            row += f"{np.mean([nlls[name][w] for w in ws]):>18.3f}"
        print(row)
    print("\n" + f"{'model':<26}{'total params':>14}{'emb+op params':>16}")
    print("-" * 58)
    for name, _ in arms:
        print(f"{name:<26}{res[name]['total']/1e6:>13.2f}M{res[name]['emb']/1e6:>15.2f}M")
    json.dump({k: v for k, v in res.items()}, open(f"{OUT}/freq.json", "w"), indent=1)


if __name__ == "__main__":
    main()
