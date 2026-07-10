"""The vector route: operators as FUNCTIONS, decoded by DISTANCE.

Update. A dot-product head ranks unseen compositions last regardless of
direction, because their magnitude was never trained: an untrained row scores
~0 while trained rows have grown large positive logits. Measured: free-word
baseline mean rank 19.9, factored 18.1, chance 11 -- BOTH worse than chance,
and the ordering told us the factored direction was right and its scale wrong.

So decode by cosine with a temperature, as originally specified. Then an unseen
product `W_plural . E_mouse` competes on direction alone.

Original docstring follows.

The vector route: operators as FUNCTIONS, not tokens.

Established: a transformer over the lexeme token language cannot compose. It
learns `P(root | operator)` as a co-occurrence table, and `mouse` never
co-occurs with `noun.plural`, so `mice` is unreachable (root median rank 646
prefix / 738 postfix, 0/10 composed). Meanwhile the SAME operators applied
geometrically -- `W_slot . centroid(root)` -> nearest word -- regenerate 82.8%
of a held-out lexeme's forms. The composition lives in the geometry; tokenising
it threw the geometry away.

So put it back. A word is a (root, slot) pair, and its embedding is *computed*:

    e(root, slot) = W_slot . E_root          (bare root: slot = identity)

Both the input embedding and the output logit use this. Nothing about `mice` is
stored anywhere: `E_mouse` is trained only from `mouse`, `W_plural` only from
other plurals, and

    logit(mice | h) = h . (W_plural . E_mouse)

is defined even though the pair never occurred. That is the whole claim, and it
is exactly what a token softmax cannot do.

Baseline: the identical trunk with a free per-word embedding table (`word`
model). There, `mice` has its own untrained row and is unreachable by
construction.

Held-out forms never appear as a training target. Everything else is equal:
same trunk, same steps, same word-level sequences, same vocabulary.
"""
import json, math, collections, random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from lexicon.ts_lm import Block, build, WORD_RE, DEVICE, OUT, CTX

D = 384
SEQ = 256
TAU = 0.07          # temperature for the cosine decoder


class Vocab:
    """Every word is (root_id, slot_id). slot 0 = identity (a bare lexeme)."""

    def __init__(self, texts, lexparent, max_words=12000):
        c = collections.Counter()
        for t in texts:
            c.update(m.lower() for m in WORD_RE.findall(t) if m.isalpha())
        words = [w for w, _ in c.most_common(max_words)]
        self.slots = ["<id>"] + sorted({lexparent[w][0] for w in lexparent})
        self.sidx = {s: i for i, s in enumerate(self.slots)}
        roots = set()
        self.decomp = {}
        for w in words:
            if w in lexparent:
                slot, r = lexparent[w]
                if r in c or True:
                    self.decomp[w] = (r, slot); roots.add(r)
                else:
                    self.decomp[w] = (w, "<id>"); roots.add(w)
            else:
                self.decomp[w] = (w, "<id>"); roots.add(w)
        self.roots = ["<pad>", "<unk>"] + sorted(roots)
        self.ridx = {r: i for i, r in enumerate(self.roots)}
        self.itos = ["<pad>", "<unk>", "<p>"] + words
        self.stoi = {w: i for i, w in enumerate(self.itos)}
        # per-word (root, slot) index arrays
        self.wr = torch.zeros(len(self.itos), dtype=torch.long)
        self.ws = torch.zeros(len(self.itos), dtype=torch.long)
        for w, i in self.stoi.items():
            r, s = self.decomp.get(w, (w, "<id>"))
            self.wr[i] = self.ridx.get(r, 1)
            self.ws[i] = self.sidx.get(s, 0)

    def add_word(self, w, lexparent):
        """Make a never-seen word ADDRESSABLE without giving it parameters."""
        if w in self.stoi:
            return self.stoi[w]
        slot, r = lexparent[w]
        i = len(self.itos)
        self.itos.append(w); self.stoi[w] = i
        # decomp must be updated too: anything reading it (the geometric sanity
        # check did) would otherwise treat `mice` as a BARE lexeme with an <unk>
        # root, and score the composition at chance while the model -- which
        # reads wr/ws -- was using the right factorisation all along.
        self.decomp[w] = (r, slot)
        self.wr = torch.cat([self.wr, torch.tensor([self.ridx.get(r, 1)])])
        self.ws = torch.cat([self.ws, torch.tensor([self.sidx.get(slot, 0)])])
        return i

    def enc(self, ws):
        return [self.stoi.get(w.lower(), 1) for w in ws if w.isalpha()]


class Factored(nn.Module):
    """e(word) = W_slot . E_root, used for BOTH input and output."""

    def __init__(self, vocab, d=D, layers=6, heads=6, rank=None):
        super().__init__()
        self.V = vocab
        self.E = nn.Embedding(len(vocab.roots), d)
        nn.init.normal_(self.E.weight, std=0.02)
        S = len(vocab.slots)
        # GAUGE. Slot 0 is the identity slot (a bare lexeme). If it is learnable,
        # a bare word only requires W_id . E_root ~= emb(root), leaving E_root free
        # to be ANY pre-image -- and then W_plural . E_mouse is that operator applied
        # to an arbitrary vector, tied to nothing. A free map in front of an operator
        # makes the operator meaningless. (The same gauge symmetry made a random
        # reflection plane match a trained one, once an adapter could rotate.)
        # Pin slot 0 to the identity, so E_root IS the word's embedding.
        self.register_buffer("W0", torch.eye(d))
        self.Wop = nn.Parameter(torch.eye(d).repeat(S - 1, 1, 1)
                                + 0.01 * torch.randn(S - 1, d, d))

        self.pos = nn.Embedding(SEQ, d)
        self.blocks = nn.ModuleList([Block(d, heads) for _ in range(layers)])
        self.ln = nn.LayerNorm(d)
        self.register_buffer("wr", vocab.wr.clone())
        self.register_buffer("ws", vocab.ws.clone())

    @property
    def W(self):
        return torch.cat([self.W0[None], self.Wop], 0)

    def word_emb(self):
        """[V, d]: every word's embedding, computed from root and slot.

        Grouped by slot. `self.W[self.ws]` would materialise a [V, d, d] tensor
        (~7 GB at V=12k, d=384) on every forward pass, and again in backward.
        Twenty-four matmuls of [n_s, d] @ [d, d] do the same work in megabytes."""
        er = self.E(self.wr)                       # [V, d]
        out = torch.empty_like(er)
        for s in range(self.W.shape[0]):
            m = self.ws == s
            if m.any():
                out[m] = er[m] @ self.W[s].T
        return out

    def forward(self, idx, table=None):
        T = idx.shape[1]
        tbl = self.word_emb() if table is None else table
        m = torch.triu(torch.full((T, T), float("-inf"), device=idx.device), 1)
        x = tbl[idx] + self.pos(torch.arange(T, device=idx.device))[None]
        for b in self.blocks:
            x = b(x, m)
        h = F.normalize(self.ln(x), dim=-1)
        return h @ F.normalize(tbl, dim=-1).T / TAU     # distance decoder


class FreeWord(nn.Module):
    """Baseline: a free embedding row per word. `mice` has its own untrained row."""

    def __init__(self, vocab, d=D, layers=6, heads=6):
        super().__init__()
        self.tok = nn.Embedding(len(vocab.itos), d)
        nn.init.normal_(self.tok.weight, std=0.02)
        self.pos = nn.Embedding(SEQ, d)
        self.blocks = nn.ModuleList([Block(d, heads) for _ in range(layers)])
        self.ln = nn.LayerNorm(d)

    def forward(self, idx, table=None):
        T = idx.shape[1]
        m = torch.triu(torch.full((T, T), float("-inf"), device=idx.device), 1)
        x = self.tok(idx) + self.pos(torch.arange(T, device=idx.device))[None]
        for b in self.blocks:
            x = b(x, m)
        h = F.normalize(self.ln(x), dim=-1)
        return h @ F.normalize(self.tok.weight, dim=-1).T / TAU


def corpus(V, texts):
    ids = []
    for t in texts:
        ids += V.enc(WORD_RE.findall(t)) + [0]
    return torch.tensor(ids, dtype=torch.long)


def train(model, data, steps=4000, bs=32, lr=3e-4, seed=0, mask_ids=None):
    """`mask_ids` are removed from the training softmax DENOMINATOR.

    Without this, a held-out word is not "never seen" -- it is seen ten million
    times as a negative. Cross-entropy normalises over the whole vocabulary, so
    every step where `mice` is not the target pushes logit(mice) down, and in
    the factored model that gradient flows straight into W_plural and E_mouse.
    We were training the composition machinery not to compose. Measured: gold
    ranked 21st of 21 (chance is 11th)."""
    torch.manual_seed(seed)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.1)
    sch = torch.optim.lr_scheduler.OneCycleLR(opt, lr, total_steps=steps, pct_start=0.05)
    g = torch.Generator().manual_seed(seed)
    model.train()
    for s in range(steps):
        ix = torch.randint(0, len(data) - SEQ - 1, (bs,), generator=g)
        x = torch.stack([data[i:i+SEQ] for i in ix]).to(DEVICE)
        y = torch.stack([data[i+1:i+1+SEQ] for i in ix]).to(DEVICE)
        lg = model(x)
        if mask_ids is not None:
            lg[..., mask_ids] = -1e9          # never a positive, never a negative
        loss = F.cross_entropy(lg.reshape(-1, lg.shape[-1]), y.reshape(-1))
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step(); sch.step()
        if s % 500 == 0:
            print(f"    step {s:5d} loss {loss.item():.3f}", flush=True)
    return model


PROMPTS = {
 "mice":("Lily saw a mouse in the box . Then she saw two","noun.plural","mouse"),
 "men":("There was one man in the park . Then there were two","noun.plural","man"),
 "rang":("Tom likes to ring the bell . Yesterday he","verb.past","ring"),
 "fed":("Mom likes to feed the cat . Yesterday she","verb.past","feed"),
 "dug":("The dog likes to dig . Yesterday the dog","verb.ptcp","dig"),
 "swung":("Ben likes to swing . Yesterday he","verb.past","swing"),
 "rode":("Sam likes to ride his bike . Yesterday he","verb.past","ride"),
 "spun":("The top likes to spin . Yesterday it","verb.ptcp","spin"),
 "fought":("They like to fight . Yesterday they","verb.ptcp","fight"),
 "paid":("She likes to pay for it . Yesterday she","verb.ptcp","pay"),
}


@torch.no_grad()
def probe(model, V, name):
    model.eval()
    tbl = model.word_emb() if hasattr(model, "word_emb") else None
    slotpool = collections.defaultdict(list)
    for w, (r, s) in V.decomp.items():
        if s != "<id>":
            slotpool[s].append(w)
    ranks, hits = [], 0
    print(f"\n[{name}] minimal-pair probe (root is in the prompt)")
    print(f"   {'gold':<8}{'rank among slot-matched':>26}{'top1':>7}   top-3 predictions")
    for gold, (p, slot, root) in PROMPTS.items():
        if gold not in V.stoi:
            print(f"   {gold:<8}{'NOT IN VOCAB':>26}"); continue
        ids = V.enc(p.split())
        x = torch.tensor(ids[-SEQ:], device=DEVICE)[None]
        lg = F.log_softmax(model(x, tbl)[0, -1], -1)
        cands = [gold] + [w for w in slotpool[slot] if w != gold][:20]
        cid = torch.tensor([V.stoi[c] for c in cands], device=DEVICE)
        sc = lg[cid]
        r = int((sc > sc[0]).sum().item()) + 1
        ranks.append(r); hits += (r == 1)
        top3 = [cands[j] for j in torch.topk(sc, 3).indices.tolist()]
        print(f"   {gold:<8}{r:>26}{'yes' if r==1 else '':>7}   {', '.join(top3)}")
    print(f"   {'MEAN':<8}{np.mean(ranks):>26.1f}   top1 {hits}/{len(ranks)}  "
          f"MRR {np.mean(1/np.array(ranks)):.3f}")
    return dict(acc=hits/len(ranks), mrr=float(np.mean(1/np.array(ranks))),
                mean_rank=float(np.mean(ranks)))


def main():
    train_texts, clean_eval, test_texts, held = build()
    from lexicon.ts_encode import load_forest
    parent, _ = load_forest()
    hset = {w for w, _, _ in held}

    V = Vocab(train_texts, parent)
    # held-out words are ADDRESSABLE but were never a training target
    for w in hset:
        if w in parent:
            V.add_word(w, parent)
    print(f"\nvocab: {len(V.itos)} words, {len(V.roots)} roots, {len(V.slots)} slots")
    print(f"held-out words addressable: {sum(w in V.stoi for w in hset)}/{len(hset)}")
    print(f"   mice = W[{V.decomp.get('mice', ('?','?'))[1]}] . E[{V.decomp.get('mice',('?','?'))[0]}]"
          if "mice" in V.decomp else "")
    data = corpus(V, train_texts)
    print(f"training words: {len(data):,}   (held-out forms appear 0 times)\n")
    assert not any(V.stoi[w] in set(data.tolist()) for w in hset if w in V.stoi), "LEAK"
    held_ids = torch.tensor([V.stoi[w] for w in hset if w in V.stoi], device=DEVICE)
    print(f"masking {len(held_ids)} held-out words out of the training softmax "
          f"(so they are never negatives either)\n")

    res = {}
    for name, M in (("free-word baseline", FreeWord), ("factored (operators as maps)", Factored)):
        print(f"--- {name} ---")
        m = M(V).to(DEVICE)
        print(f"    {sum(p.numel() for p in m.parameters())/1e6:.1f}M params")
        train(m, data, mask_ids=held_ids)
        res[name] = probe(m, V, name)
    print("\n" + "="*72)
    print(f"{'model':<32}{'top-1':>9}{'MRR':>9}{'mean rank':>12}")
    print("-"*72)
    for k, v in res.items():
        print(f"{k:<32}{v['acc']:>9.2f}{v['mrr']:>9.3f}{v['mean_rank']:>12.1f}")
    print("\nchance: top-1 = 1/21 = 0.048, mean rank 11")
    json.dump(res, open(f"{OUT}/factored.json","w"), indent=1)


if __name__ == "__main__":
    main()
