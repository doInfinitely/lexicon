"""Contextual embeddings for the full English lexicon (38k words).

Same definition as before -- a word's embedding is the mean of distilbert's
last-layer states over its subword tokens, averaged over real occurrences in
wikitext-103 -- but at 14x the vocabulary. Words too rare to appear in the
corpus fall back to the isolated-word embedding, and we record which, since
their geometry is less trustworthy.
"""
import json, os, re, collections
import torch
from transformers import AutoTokenizer, AutoModel

MAX_CTX = 16
MIN_SENT_CHARS, MAX_SENT_CHARS = 24, 400
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
OUT = os.environ.get("LEXICON_DIR", "real/english")
WORD_RE = re.compile(r"[a-z]+")


def mine(vocab):
    from datasets import load_dataset
    ds = load_dataset("Salesforce/wikitext", "wikitext-103-raw-v1", split="train")
    want = set(vocab)
    hits = collections.defaultdict(list)
    unfilled = set(want)
    for n, row in enumerate(ds):
        text = row["text"]
        if len(text) < 24 or text.lstrip().startswith("="):
            continue
        for sent in re.split(r"(?<=[.!?]) ", text):
            if not (MIN_SENT_CHARS <= len(sent) <= MAX_SENT_CHARS):
                continue
            for w in set(WORD_RE.findall(sent.lower())) & unfilled:
                hits[w].append(sent.strip())
                if len(hits[w]) >= MAX_CTX:
                    unfilled.discard(w)
        if not unfilled:
            break
        if n % 400000 == 0:
            print(f"  row {n}: {len(want)-len(unfilled)}/{len(want)} words filled")
    return hits


@torch.no_grad()
def embed(vocab, hits):
    tok = AutoTokenizer.from_pretrained("distilbert-base-uncased")
    model = AutoModel.from_pretrained("distilbert-base-uncased").to(DEVICE).eval()

    # 1) isolated embeddings for everything (fallback + for rare words)
    iso = {}
    for i in range(0, len(vocab), 512):
        batch = vocab[i:i + 512]
        enc = tok(batch, return_tensors="pt", padding=True).to(DEVICE)
        h = model(**enc).last_hidden_state
        ids, mask = enc["input_ids"], enc["attention_mask"].bool()
        keep = mask & ~((ids == tok.cls_token_id) | (ids == tok.sep_token_id))
        for j, w in enumerate(batch):
            iso[w] = h[j][keep[j]].mean(0).cpu()
    print(f"  isolated embeddings: {len(iso)}")

    # 2) contextual embeddings where the corpus provides them
    ctx = {}
    words = [w for w in vocab if len(hits.get(w, [])) >= 3]
    flat = [(w, s) for w in words for s in hits[w][:MAX_CTX]]
    print(f"  contextual occurrences to embed: {len(flat)} over {len(words)} words")
    acc = collections.defaultdict(list)
    B = 256
    for i in range(0, len(flat), B):
        chunk = flat[i:i + B]
        sents = [s for _, s in chunk]
        enc = tok(sents, return_tensors="pt", padding=True, truncation=True,
                  max_length=96, return_offsets_mapping=True).to(DEVICE)
        offs = enc.pop("offset_mapping")
        h = model(**enc).last_hidden_state
        for j, (w, s) in enumerate(chunk):
            low = s.lower()
            spans = [(m.start(), m.end())
                     for m in re.finditer(rf"\b{re.escape(w)}\b", low)]
            if not spans:
                continue
            m = torch.zeros(h.shape[1], dtype=torch.bool)
            for a, b in spans:
                for k, (oa, ob) in enumerate(offs[j].tolist()):
                    if oa != ob and oa < b and ob > a:
                        m[k] = True
            if m.any():
                acc[w].append(h[j][m.to(DEVICE)].mean(0).cpu())
        if i % (B * 400) == 0:
            print(f"    {i}/{len(flat)}")
    for w, v in acc.items():
        ctx[w] = torch.stack(v).mean(0)
    print(f"  contextual embeddings: {len(ctx)}")

    protos = {w: ctx.get(w, iso[w]) for w in vocab}
    from_corpus = {w: (w in ctx) for w in vocab}
    return protos, from_corpus


def main():
    vocab = json.load(open(f"{OUT}/vocab.json"))
    print(f"mining contexts for {len(vocab)} words...")
    hits = mine(vocab)
    covered = sum(1 for w in vocab if len(hits.get(w, [])) >= 3)
    print(f"corpus coverage: {covered}/{len(vocab)} words with >=3 occurrences")
    protos, from_corpus = embed(vocab, hits)
    torch.save(protos, f"{OUT}/prototypes.pt")
    json.dump(from_corpus, open(f"{OUT}/from_corpus.json", "w"))
    print(f"wrote {OUT}/prototypes.pt")


if __name__ == "__main__":
    main()
