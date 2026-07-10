"""Operator network: one named operator per BATS relation, implemented as a
shared residual MLP conditioned on a learned relation embedding.

f_r(x) = x + MLP([x ; e_r])

Sharing the trunk across all 40 relations lets morphologically similar
relations (e.g. the ten verb inflections) pool statistical strength while the
relation embedding keeps them distinct — critical with only ~35 training
pairs per relation.
"""
import torch
import torch.nn as nn

EMB_DIM = 768
REL_DIM = 128
HIDDEN = 1024


class OperatorNet(nn.Module):
    def __init__(self, relation_names):
        super().__init__()
        self.relation_names = list(relation_names)
        self.rel_index = {r: i for i, r in enumerate(self.relation_names)}
        self.rel_emb = nn.Embedding(len(self.relation_names), REL_DIM)
        self.trunk = nn.Sequential(
            nn.Linear(EMB_DIM + REL_DIM, HIDDEN),
            nn.GELU(),
            nn.Linear(HIDDEN, HIDDEN),
            nn.GELU(),
            nn.Linear(HIDDEN, EMB_DIM),
        )
        # start near the identity map: related words are already close in
        # distilbert space, so the residual should begin small
        nn.init.normal_(self.trunk[-1].weight, std=1e-3)
        nn.init.zeros_(self.trunk[-1].bias)

    def forward(self, x, rel_ids):
        e = self.rel_emb(rel_ids)
        return x + self.trunk(torch.cat([x, e], dim=-1))

    def apply_named(self, relation, x):
        """Apply one named operator to a batch (or single) embedding."""
        single = x.dim() == 1
        if single:
            x = x.unsqueeze(0)
        rid = torch.full((x.shape[0],), self.rel_index[relation],
                         dtype=torch.long, device=x.device)
        y = self.forward(x, rid)
        return y.squeeze(0) if single else y
