"""Joint model: a space adapter + relation operators, trained together.

  adapter  A(x) = LN(x + MLP(x))      — fine-tunes the embedding space itself
  operator f_r(z) = z + MLP([z; e_r]) — acts inside the adapted space

The adapted space is the real product: it is the embedding table the future
LLM will live in, so it must stay discriminative. Training therefore uses
InfoNCE retrieval loss over the whole vocabulary (operator output must rank
the correct target word first), not just pointwise cosine.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

EMB_DIM = 768
REL_DIM = 128
HIDDEN = 1024


class Adapter(nn.Module):
    def __init__(self):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(EMB_DIM, HIDDEN), nn.GELU(), nn.Linear(HIDDEN, EMB_DIM))
        self.norm = nn.LayerNorm(EMB_DIM)
        nn.init.normal_(self.mlp[-1].weight, std=1e-3)
        nn.init.zeros_(self.mlp[-1].bias)

    def forward(self, x):
        return self.norm(x + self.mlp(x))


class OperatorNet(nn.Module):
    def __init__(self, relation_names):
        super().__init__()
        self.relation_names = list(relation_names)
        self.rel_index = {r: i for i, r in enumerate(self.relation_names)}
        self.rel_emb = nn.Embedding(len(self.relation_names), REL_DIM)
        self.trunk = nn.Sequential(
            nn.Linear(EMB_DIM + REL_DIM, HIDDEN), nn.GELU(),
            nn.Linear(HIDDEN, HIDDEN), nn.GELU(),
            nn.Linear(HIDDEN, EMB_DIM))
        nn.init.normal_(self.trunk[-1].weight, std=1e-3)
        nn.init.zeros_(self.trunk[-1].bias)

    def forward(self, z, rel_ids):
        return z + self.trunk(torch.cat([z, self.rel_emb(rel_ids)], dim=-1))

    def apply_named(self, relation, z):
        single = z.dim() == 1
        if single:
            z = z.unsqueeze(0)
        rid = torch.full((z.shape[0],), self.rel_index[relation],
                         dtype=torch.long, device=z.device)
        y = self.forward(z, rid)
        return y.squeeze(0) if single else y


class LexiconSpace(nn.Module):
    """Adapter + operators + the adapted vocabulary table."""

    def __init__(self, relation_names):
        super().__init__()
        self.adapter = Adapter()
        self.ops = OperatorNet(relation_names)

    def adapt_table(self, proto_matrix):
        """Adapted, L2-normalized embedding table for the whole vocab."""
        with torch.no_grad():
            return F.normalize(self.adapter(proto_matrix), dim=-1)
