from __future__ import annotations
from typing import List, Tuple
import torch
import torch.nn as nn
from app import config
from app.utils import tokens as T

# Minimal hierarchical-style encoder: encode grids via color embedding average,
# concat across I/O pairs, then attend with program token embeddings.


class GridEncoder(nn.Module):
    def __init__(self, d_model: int):
        super().__init__()
        self.emb = nn.Embedding(int(config.DSL_NUM_COLORS), d_model)
        self.proj = nn.Linear(d_model, d_model)

    def forward(self, grid: torch.Tensor) -> torch.Tensor:
        # grid: (H, W) int64
        x = self.emb(grid.long())  # (H, W, D)
        x = x.mean(dim=(0, 1))  # (D)
        return self.proj(x)


class ProgramPolicy(nn.Module):
    def __init__(self):
        super().__init__()
        D = int(config.MODEL_CFG["d_model"])
        H = int(config.MODEL_CFG["nhead"])
        L = int(config.MODEL_CFG["nlayers"])
        drop = float(config.MODEL_CFG["dropout"])
        self.grid_enc = GridEncoder(D)
        encoder_layer = nn.TransformerEncoderLayer(d_model=D, nhead=H, dropout=drop, batch_first=True)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=L)
        self.tok_emb = nn.Embedding(int(T.VOCAB_SIZE), D)
        self.pos = nn.Parameter(torch.zeros(1, int(config.MODEL_CFG["max_tokens"]), D))
        self.policy = nn.Linear(D, int(T.VOCAB_SIZE))
        self.value = nn.Linear(D, 1)

    def encode_task(self, pairs: List[Tuple[torch.Tensor, torch.Tensor]]) -> torch.Tensor:
        # returns (D)
        embs = []
        for inp, out in pairs:
            ei = self.grid_enc(inp)
            eo = self.grid_enc(out)
            embs.append((ei + eo) / 2)
        return torch.stack(embs, dim=0).mean(dim=0)

    def forward(self, pairs: List[Tuple[torch.Tensor, torch.Tensor]], prog_tokens: torch.Tensor):
        # prog_tokens: (T,) int64
        ctx = self.encode_task(pairs)  # (D)
        tok = self.tok_emb(prog_tokens.long())  # (T, D)
        seq = torch.cat([ctx.unsqueeze(0), tok], dim=0).unsqueeze(0)  # (1, 1+T, D)
        seq = seq + self.pos[:, : seq.size(1), :]
        h = self.encoder(seq)  # (1, 1+T, D)
        h_last = h[:, -1, :]  # (1, D)
        logits = self.policy(h_last)
        val = self.value(h_last).squeeze(-1)
        return logits.squeeze(0), val.squeeze(0)
