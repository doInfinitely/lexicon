"""Mine real-sentence contexts for every vocab word from wikitext-103,
embed each occurrence in context with distilbert, and build:

  prototypes.pt   word -> mean contextual embedding (the word's anchor)
  senses.pt       word -> [sense centroids] (>=2 only for detected homographs)
  contexts.json   word -> sample sentences per sense (for inspection/decoding)

An occurrence embedding is the mean of last-layer hidden states over the
word's subword tokens inside the sentence. Words with too few corpus hits
fall back to the isolated-word embedding.
"""
import json, os, re, collections
import torch
import numpy as np
from transformers import AutoTokenizer, AutoModel
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

MAX_CTX_PER_WORD = 64
MIN_SENT_WORDS = 6
MAX_SENT_CHARS = 400
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

WORD_RE = re.compile(r"[a-z]+")


def mine_sentences(vocab, dataset):
    """One streaming pass over the corpus collecting sentences per word."""
    want = set(vocab)
    hits = collections.defaultdict(list)
    unfilled = set(want)
    for row in dataset:
        text = row["text"]
        if len(text) < 20 or text.lstrip().startswith("="):
            continue
        for sent in re.split(r"(?<=[.!?]) ", text):
            if not (MIN_SENT_WORDS * 3 <= len(sent) <= MAX_SENT_CHARS):
                continue
            toks = set(WORD_RE.findall(sent.lower()))
            for w in toks & unfilled:
                hits[w].append(sent.strip())
                if len(hits[w]) >= MAX_CTX_PER_WORD:
                    unfilled.discard(w)
        if not unfilled:
            break
    return hits


@torch.no_grad()
def embed_occurrences(word, sentences, tok, model, batch_size=64):
    """Embed each occurrence of `word` inside its sentence (subword mean)."""
    embs = []
    for i in range(0, len(sentences), batch_size):
        batch = sentences[i:i + batch_size]
        enc = tok(batch, return_tensors="pt", padding=True, truncation=True,
                  max_length=128, return_offsets_mapping=True).to(DEVICE)
        offsets = enc.pop("offset_mapping")
        hidden = model(**enc).last_hidden_state
        for j, sent in enumerate(batch):
            low = sent.lower()
            spans = [(m.start(), m.end()) for m in
                     re.finditer(rf"\b{re.escape(word)}\b", low)]
            if not spans:
                continue
            tok_mask = torch.zeros(hidden.shape[1], dtype=torch.bool)
            for (a, b) in spans:
                for k, (oa, ob) in enumerate(offsets[j].tolist()):
                    if oa == ob == 0:
                        continue
                    if oa < b and ob > a:
                        tok_mask[k] = True
            if tok_mask.any():
                embs.append(hidden[j][tok_mask.to(DEVICE)].mean(0).cpu())
    return embs


def detect_senses(occ_embs, max_k=3, min_cluster=5, sil_threshold=0.30):
    """Cluster occurrence embeddings; >1 cluster => homograph/polysemy."""
    X = torch.stack(occ_embs)
    Xn = (X / X.norm(dim=1, keepdim=True)).numpy()
    best = (1, None, None)
    for k in range(2, max_k + 1):
        if len(occ_embs) < k * min_cluster * 2:
            break
        km = KMeans(n_clusters=k, n_init=5, random_state=0).fit(Xn)
        sizes = np.bincount(km.labels_)
        if sizes.min() < min_cluster:
            continue
        sil = silhouette_score(Xn, km.labels_, metric="cosine")
        if sil > sil_threshold and sil > (best[2] or 0):
            best = (k, km.labels_, sil)
    k, labels, sil = best
    if k == 1:
        return [X.mean(0)], None, 0.0
    centroids = [X[labels == c].mean(0) for c in range(k)]
    return centroids, labels, sil


def build(vocab, out_dir):
    from datasets import load_dataset
    os.makedirs(out_dir, exist_ok=True)
    ds = load_dataset("Salesforce/wikitext", "wikitext-103-raw-v1", split="train")
    print("mining sentences...")
    hits = mine_sentences(vocab, ds)
    covered = sum(1 for w in vocab if hits.get(w))
    print(f"corpus coverage: {covered}/{len(vocab)} words")

    tok = AutoTokenizer.from_pretrained("distilbert-base-uncased")
    model = AutoModel.from_pretrained("distilbert-base-uncased").to(DEVICE).eval()

    prototypes, senses, sense_meta, samples = {}, {}, {}, {}
    occurrences, sentences = {}, {}
    for n, w in enumerate(vocab):
        sents = hits.get(w, [])
        occ = embed_occurrences(w, sents, tok, model) if sents else []
        if len(occ) < 3:  # corpus-poor: isolated-word fallback
            enc = tok(w, return_tensors="pt").to(DEVICE)
            with torch.no_grad():
                h = model(**enc).last_hidden_state[0, 1:-1]
            prototypes[w] = h.mean(0).cpu()
            senses[w] = [prototypes[w]]
            continue
        cents, labels, sil = detect_senses(occ)
        prototypes[w] = torch.stack(occ).mean(0)
        senses[w] = cents
        occurrences[w] = torch.stack(occ)
        sentences[w] = sents
        if labels is not None:
            sense_meta[w] = {"n_senses": len(cents), "silhouette": round(sil, 3)}
            samples[w] = {int(c): [s for s, l in zip(sents, labels) if l == c][:3]
                          for c in range(len(cents))}
        if n % 250 == 0:
            print(f"  embedded {n}/{len(vocab)}")

    torch.save(prototypes, os.path.join(out_dir, "prototypes.pt"))
    torch.save(senses, os.path.join(out_dir, "senses.pt"))
    torch.save(occurrences, os.path.join(out_dir, "occurrences.pt"))
    json.dump(sentences, open(os.path.join(out_dir, "sentences.json"), "w"))
    json.dump(sense_meta, open(os.path.join(out_dir, "sense_meta.json"), "w"),
              indent=1)
    json.dump(samples, open(os.path.join(out_dir, "sense_samples.json"), "w"),
              indent=1)
    print(f"multi-sense words: {len(sense_meta)}")
    return prototypes, senses, sense_meta


if __name__ == "__main__":
    vocab = json.load(open("harbor/workspace/data/vocab.json"))
    build(vocab, "real/embeddings")
