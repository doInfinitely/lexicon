"""Vector -> word decoding: the adaptor for the embeddings-out LLM variant.

Given an arbitrary vector in the adapted space, decode by softmax over
cosine similarity to the (sense-aware) vocabulary table with a temperature:
T -> 0 recovers nearest-neighbor argmax; higher T samples among close words.
Homographs contribute one entry per sense, so context vectors land on the
right sense cluster (apple-fruit vs apple-company).
"""
import torch
import torch.nn.functional as F


class Decoder:
    def __init__(self, space, vocab, protos, senses=None, device="cpu"):
        self.space = space.to(device).eval()
        self.device = device
        entries, mat = [], []
        for w in vocab:
            sense_list = (senses or {}).get(w) or [protos[w]]
            for k, e in enumerate(sense_list):
                entries.append((w, k))
                mat.append(e)
        with torch.no_grad():
            self.table = F.normalize(
                self.space.adapter(torch.stack(mat).to(device)), dim=-1)
        self.entries = entries

    @torch.no_grad()
    def decode(self, vec, temperature=0.0, top_k=5, generator=None):
        v = F.normalize(vec.to(self.device), dim=-1)
        sims = self.table @ v
        if temperature <= 0:
            idx = torch.topk(sims, top_k).indices.tolist()
            return [(self.entries[i][0], self.entries[i][1], sims[i].item())
                    for i in idx]
        probs = F.softmax(sims / temperature, dim=0)
        picks = torch.multinomial(probs, top_k, replacement=False,
                                  generator=generator).tolist()
        return [(self.entries[i][0], self.entries[i][1], probs[i].item())
                for i in picks]
