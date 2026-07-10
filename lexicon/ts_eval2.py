"""Corrected evaluation. The first version had three bugs, each biasing a
different way:

  bits/char   truncated the token stream to 512 but divided by the FULL story's
              characters. The lexeme stream is 1.46x longer, so it was credited
              with characters it never scored. Fix: count only the characters
              of the words actually inside the window.

  <unk> cheat `mice` is absent from the word/BPE vocabularies (built on training
              text), so "scoring mice" scored the <unk> token -- a frequent,
              high-probability token. Those models never produce the word. Fix:
              a candidate whose encoding contains <unk> scores -inf. That is
              what "cannot say it" means.

  length bias the lexeme gold costs 3 tokens; the distractors were 1-token
              frequent words. Summed log-probs punish longer candidates. Fix:
              distractors are the SAME SLOT applied to other roots, so every
              candidate has identical token length within a scheme.

Plus the diagnostic that isolates the real question: teacher-force the operator
and ask whether the model puts mass on the right root.
"""
import json, math, collections
import numpy as np
import torch
import torch.nn.functional as F
from lexicon.ts_lm import WORD_RE, CTX, DEVICE


@torch.no_grad()
def bits_per_char(m, tok, texts, n=300):
    m.eval(); nll = 0.0; chars = 0
    for t in texts[:n]:
        ws = WORD_RE.findall(t)
        ids, kept = [], 0
        for w in ws:
            e = tok.enc_word(w) if (w.isalpha() or "'" in w) else tok.enc_words([w])
            if len(ids) + len(e) > CTX: break
            ids += e; kept += 1
        if len(ids) < 8: continue
        x = torch.tensor(ids, device=DEVICE)[None]
        lg = F.log_softmax(m(x)[0, :-1], -1)
        nll += -lg[torch.arange(len(ids) - 1), x[0, 1:]].sum().item()
        chars += len(" ".join(ws[:kept]))          # only what was scored
    return nll / chars / math.log(2)


def has_unk(tok, w):
    return 1 in tok.enc_word(w)


@torch.no_grad()
def score(m, tok, ctx_ids, word):
    wt = tok.enc_word(word)
    if 1 in wt:                       # cannot express it -> cannot say it
        return -float("inf")
    ids = (ctx_ids + wt)[-CTX:]
    x = torch.tensor(ids, device=DEVICE)[None]
    lg = F.log_softmax(m(x)[0], -1)
    n = len(wt); s = 0.0
    for k in range(n):
        p = len(ids) - n + k
        s += lg[p - 1, ids[p]].item()
    return s


@torch.no_grad()
def root_prob(m, lextok, ctx_ids, slot, root):
    """Teacher-force the operator; how much mass on the right root?"""
    op = lextok.stoi.get(f"<op:{slot}>"); lx = lextok.stoi.get(f"<lex:{root}>")
    if op is None or lx is None: return None
    ids = (ctx_ids + [op])[-CTX:]
    x = torch.tensor(ids, device=DEVICE)[None]
    lg = F.log_softmax(m(x)[0, -1], -1)
    lex_ids = [i for t, i in lextok.stoi.items() if t.startswith("<lex:")]
    sub = lg[lex_ids]
    rank = int((sub > lg[lx]).sum().item()) + 1
    return rank, len(lex_ids), lg[lx].item()


def evaluate(m, tok, test_texts, held, slot_pool, max_cases=250, lextok=None):
    hmap = {w: (s, r) for w, s, r in held}
    hset = set(hmap)
    ranks, hits, expressible, cases = [], 0, 0, 0
    rootranks = []
    for t in test_texts:
        ws = WORD_RE.findall(t)
        for i, w in enumerate(ws):
            lw = w.lower()
            if lw not in hmap or i < 6: continue
            slot, root = hmap[lw]
            cands = [lw] + [c for c in slot_pool.get(slot, []) if c != lw][:20]
            if len(cands) < 6: continue
            ctx = tok.enc_words(ws[max(0, i - 60):i])
            sc = np.array([score(m, tok, ctx, c) for c in cands])
            expressible += int(np.isfinite(sc[0]))
            r = int((sc > sc[0]).sum()) + 1 if np.isfinite(sc[0]) else len(cands)
            ranks.append(r); hits += (r == 1)
            if lextok is not None:
                rr = root_prob(m, lextok, ctx, slot, root)
                if rr: rootranks.append(rr[0])
            cases += 1
            if cases >= max_cases: break
        if cases >= max_cases: break
    out = dict(n=cases, acc=hits / max(cases, 1),
               mrr=float(np.mean(1 / np.array(ranks))) if ranks else 0.0,
               expressible=expressible / max(cases, 1))
    if rootranks:
        out["root_rank_median"] = float(np.median(rootranks))
        out["root_top1"] = float(np.mean(np.array(rootranks) == 1))
        out["root_top10"] = float(np.mean(np.array(rootranks) <= 10))
    return out
