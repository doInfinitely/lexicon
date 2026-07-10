"""Shared utilities: reference embedding definition used across the task.

A word's reference embedding is the mean of distilbert-base-uncased's
last-layer hidden states over the word's subword tokens (special tokens
excluded), for the word encoded alone.
"""
import torch
from transformers import AutoTokenizer, AutoModel

MODEL_NAME = "distilbert-base-uncased"


@torch.no_grad()
def compute_reference_embeddings(words, device="cpu", batch_size=256):
    tok = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModel.from_pretrained(MODEL_NAME).to(device).eval()
    out = {}
    for i in range(0, len(words), batch_size):
        batch = words[i:i + batch_size]
        enc = tok(batch, return_tensors="pt", padding=True).to(device)
        hidden = model(**enc).last_hidden_state  # [B, T, 768]
        # mask out CLS/SEP/PAD: special_tokens_mask via attention + offsets
        ids = enc["input_ids"]
        mask = enc["attention_mask"].bool()
        special = (ids == tok.cls_token_id) | (ids == tok.sep_token_id)
        keep = mask & ~special
        for j, w in enumerate(batch):
            out[w] = hidden[j][keep[j]].mean(dim=0).cpu()
    return out
