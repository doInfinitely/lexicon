"""LexiconCompressor: the task's required interface."""
import os, sys
import torch

sys.path.insert(0, os.path.dirname(__file__))
from common import compute_reference_embeddings
from operators import OperatorNet

CKPT = os.path.join(os.path.dirname(__file__), "checkpoint.pt")


class LexiconCompressor:
    def __init__(self, checkpoint_path=CKPT):
        ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        self.net = OperatorNet(ckpt["relation_names"])
        self.net.load_state_dict(ckpt["state_dict"])
        self.net.eval()
        self.base = list(ckpt["base_lexicon"])
        self.base_set = set(self.base)
        self.derivations = {w: tuple(v) for w, v in ckpt["derivations"].items()}
        self.vocab = ckpt["vocab"]
        self._emb_cache = {}
        self._base_emb = None

    def _base_embedding(self, word):
        if self._base_emb is None:
            self._base_emb = compute_reference_embeddings(self.vocab)
        return self._base_emb[word]

    def get_base_lexicon(self):
        return list(self.base)

    def get_operators(self):
        return {rel: (lambda x, r=rel: self.net.apply_named(r, x))
                for rel in self.net.relation_names}

    def encode(self, word):
        if word in self._emb_cache:
            return self._emb_cache[word]
        if word in self.derivations:
            rel, src = self.derivations[word]
            with torch.no_grad():
                emb = self.net.apply_named(rel, self.encode(src))
        else:
            emb = self._base_embedding(word)
        self._emb_cache[word] = emb
        return emb

    def reconstruct(self, word):
        if word not in self.derivations:
            return word, self.encode(word)
        rel, src = self.derivations[word]
        src_expr, _ = self.reconstruct(src)
        return f"{rel}({src_expr})", self.encode(word)
