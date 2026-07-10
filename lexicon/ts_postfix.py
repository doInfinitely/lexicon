"""Postfix (argument-first) encoding, and whether it buys composition.

Prefix `<op:noun.plural> <lex:mouse> </op>` requires the model to commit to an
operator and THEN produce a root it has never seen in that slot. Measured: the
operator is chosen correctly (mean rank 2.5) and the root lands at median rank
646 of 4,656 -- the model emits whichever lexeme that operator was most often
seen with (`thing`, `friend`, `be`). It learned a bigram, not a function.

Postfix `<lex:mouse> <op:noun.plural>` puts the argument first, where it can be
copied out of the context by an induction head, and leaves the operator as a
syntactic decision conditioned on it. Same tokens, same information, reversed
order. Reverse Polish is still unambiguous.
"""
import json, collections, torch, numpy as np, torch.nn.functional as F
from lexicon.ts_lm import LexTok, GPT, build, train_one, WORD_RE, DEVICE, OUT, CTX
from lexicon.ts_eval2 import bits_per_char

class PostfixTok(LexTok):
    name = "lexeme_postfix"
    def __init__(self, texts, max_vocab=12000):
        self.lex = __import__("lexicon.ts_encode", fromlist=["LexemeTokenizer"]).LexemeTokenizer()
        self._pc = {}
        c = collections.Counter()
        for t in texts[:8000]:
            for m in WORD_RE.findall(t):
                c.update(self.enc_word_str(m) if (m.isalpha() or "'" in m) else [f"<p:{m}>"])
        self.itos = ["<pad>", "<unk>"] + [w for w,_ in c.most_common(max_vocab)]
        self.stoi = {w:i for i,w in enumerate(self.itos)}
    def enc_word_str(self, w):
        if w in self._pc: return self._pc[w]
        pre = self.lex.encode_word(w)
        if pre[0] == "<wp>":
            out = pre
        else:
            ops = [t for t in pre if t.startswith("<op:")]
            root = [t for t in pre if t.startswith("<lex:")][0]
            out = [root] + list(reversed(ops))          # innermost operator first
        self._pc[w] = out
        return out
    def enc_word(self, w):
        return [self.stoi.get(t,1) for t in self.enc_word_str(w)]
    def enc_words(self, ws):
        o=[]
        for w in ws:
            o += self.enc_word(w) if (w.isalpha() or "'" in w) else [self.stoi.get(f"<p:{w}>",1)]
        return o

def main():
    train_texts, clean_eval, test_texts, held = build()
    tok = PostfixTok(train_texts)
    print(f"\npostfix examples:")
    for w in ["walkers","mice","happiness","cat"]:
        print(f"   {w:<12} {' '.join(tok.enc_word_str(w))}")
    m = train_one(tok, train_texts)
    torch.save(m.state_dict(), f"{OUT}/lexeme_postfix.pt")
    bpc = bits_per_char(m, tok, clean_eval)
    print(f"\n[postfix] bits/char {bpc:.3f}")

    PROMPTS={"mice":("Lily saw a mouse in the box. Then she saw two","noun.plural","mouse"),
     "men":("There was one man in the park. Then there were two","noun.plural","man"),
     "rang":("Tom likes to ring the bell. Yesterday he","verb.past","ring"),
     "fed":("Mom likes to feed the cat. Yesterday she","verb.past","feed"),
     "dug":("The dog likes to dig. Yesterday the dog","verb.ptcp","dig"),
     "swung":("Ben likes to swing. Yesterday he","verb.past","swing"),
     "rode":("Sam likes to ride his bike. Yesterday he","verb.past","ride"),
     "spun":("The top likes to spin. Yesterday it","verb.ptcp","spin"),
     "fought":("They like to fight. Yesterday they","verb.ptcp","fight"),
     "paid":("She likes to pay for it. Yesterday she","verb.ptcp","pay")}
    lex_ids=[i for t,i in tok.stoi.items() if t.startswith("<lex:")]
    op_ids ={t:i for t,i in tok.stoi.items() if t.startswith("<op:")}
    m.eval()
    print(f"\n{'gold':<8}{'root rank':>11}{'root top1':>11}{'op rank | root':>16}{'op top1':>9}   composed?")
    print("-"*76)
    rr, opr, both = [], [], 0
    for gold,(prompt,slot,root) in PROMPTS.items():
        ctx = tok.enc_words(WORD_RE.findall(prompt))
        x=torch.tensor(ctx[-CTX:],device=DEVICE)[None]
        with torch.no_grad(): lg=F.log_softmax(m(x)[0,-1],-1)
        ri=tok.stoi[f"<lex:{root}>"]
        root_rank=int((lg[lex_ids]>lg[ri]).sum().item())+1
        x2=torch.tensor((ctx+[ri])[-CTX:],device=DEVICE)[None]
        with torch.no_grad(): lg2=F.log_softmax(m(x2)[0,-1],-1)
        oi=op_ids[f"<op:{slot}>"]
        op_rank=int((lg2[list(op_ids.values())]>lg2[oi]).sum().item())+1
        ok = (root_rank==1 and op_rank==1); both += ok
        rr.append(root_rank); opr.append(op_rank)
        print(f"{gold:<8}{root_rank:>11}{'yes' if root_rank==1 else '':>11}"
              f"{op_rank:>16}{'yes' if op_rank==1 else '':>9}   {'YES' if ok else ''}")
    print("-"*76)
    print(f"{'MEAN':<8}{np.mean(rr):>11.1f}{'':>11}{np.mean(opr):>16.1f}")
    print(f"\nroot in top-1: {np.mean(np.array(rr)==1):.0%}   root in top-10: {np.mean(np.array(rr)<=10):.0%}")
    print(f"fully composed (root #1 AND operator #1): {both}/{len(PROMPTS)}")
    print(f"\nPREFIX baseline: root median rank 646, top-10 0%, fully composed 0/10")

if __name__=="__main__":
    main()
