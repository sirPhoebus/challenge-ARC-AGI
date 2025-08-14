from __future__ import annotations
from typing import List, Tuple
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from app import config
from app.utils import tokens as T
from app.dsl.ops import connected_components_nonzero


# ----------------------------
# CNN backbone blocks
# ----------------------------
class ConvBNReLU(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, k: int, norm: str = "batch"):
        super().__init__()
        p = k // 2
        self.conv = nn.Conv2d(in_ch, out_ch, kernel_size=k, padding=p, bias=(norm == "none"))
        if norm == "batch":
            self.norm = nn.BatchNorm2d(out_ch)
        elif norm == "layer":
            self.norm = nn.GroupNorm(1, out_ch)
        else:
            self.norm = nn.Identity()
        self.act = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        x = self.norm(x)
        x = self.act(x)
        return x


class BasicBlock(nn.Module):
    def __init__(self, ch: int, k: int, norm: str = "batch"):
        super().__init__()
        self.f1 = ConvBNReLU(ch, ch, k, norm)
        self.f2 = ConvBNReLU(ch, ch, k, norm)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.f2(self.f1(x)) + x


class CNNBackbone(nn.Module):
    def __init__(self):
        super().__init__()
        cfg = config.MODEL_CFG
        self.use_onehot = bool(cfg.get("input_onehot", True))
        self.norm = str(cfg.get("cnn_norm", "batch"))
        k = int(cfg.get("cnn_kernel_size", 3))
        chans = list(cfg.get("cnn_channels", [64, 128]))
        blocks_per = list(cfg.get("cnn_blocks_per_stage", [1, 1]))
        in_ch = int(config.DSL_NUM_COLORS) if self.use_onehot else int(chans[0])
        layers: List[nn.Module] = []
        # Optional embedding if not onehot
        if not self.use_onehot:
            self.pix_emb = nn.Embedding(int(config.DSL_NUM_COLORS), in_ch)
        else:
            self.pix_emb = None
        # Stem
        layers.append(ConvBNReLU(in_ch, chans[0], k, self.norm))
        # Stages
        cur = chans[0]
        for si, ch in enumerate(chans):
            if si > 0:
                layers.append(ConvBNReLU(cur, ch, k, self.norm))
                cur = ch
            for _ in range(int(blocks_per[si]) if si < len(blocks_per) else 1):
                layers.append(BasicBlock(cur, k, self.norm))
        self.net = nn.Sequential(*layers)
        self.out_ch = cur
        self.pool_type = str(cfg.get("cnn_pool", "avg"))

    def _to_chw(self, grid: torch.Tensor) -> torch.Tensor:
        # grid: (H, W) long -> (C,H,W) float
        H, W = grid.shape
        if self.use_onehot:
            C = int(config.DSL_NUM_COLORS)
            x = F.one_hot(grid.long(), num_classes=C).permute(2, 0, 1).to(dtype=torch.float32)
            return x
        else:
            emb = self.pix_emb(grid.long())  # (H,W,in_ch)
            return emb.permute(2, 0, 1).contiguous()

    def forward(self, grid: torch.Tensor) -> torch.Tensor:
        # returns (Cout)
        x = self._to_chw(grid)
        x = self.net(x.unsqueeze(0))  # (1,C,H,W)
        if self.pool_type == "max":
            x = F.adaptive_max_pool2d(x, (1, 1))
        else:
            x = F.adaptive_avg_pool2d(x, (1, 1))
        x = x.view(x.size(0), -1).squeeze(0)
        return x  # (Cout)


# ----------------------------
# Object-centric branch (connected components + attention pooling)
# ----------------------------
class ObjectBranch(nn.Module):
    def __init__(self):
        super().__init__()
        cfg = config.MODEL_CFG
        self.hidden = int(cfg.get("obj_hidden", 128))
        self.max_comp = int(cfg.get("obj_max_components", 64))
        self.heads = int(cfg.get("obj_heads", 4))
        self.layers = int(cfg.get("obj_layers", 2))
        self.pool = str(cfg.get("obj_pool", "attn"))
        self.color_emb = nn.Embedding(int(config.DSL_NUM_COLORS), self.hidden)
        # geometric projection for [h, w, area_frac]
        self.geo = nn.Sequential(
            nn.Linear(3, self.hidden), nn.ReLU(inplace=True), nn.Linear(self.hidden, self.hidden)
        )
        # attention pooling
        self.query = nn.Parameter(torch.zeros(1, 1, self.hidden))
        self.atts = nn.ModuleList([
            nn.MultiheadAttention(self.hidden, self.heads, batch_first=True)
            for _ in range(self.layers)
        ])
        self.norms = nn.ModuleList([nn.LayerNorm(self.hidden) for _ in range(self.layers)])

    def _objects_from_grid(self, grid: torch.Tensor) -> torch.Tensor:
        # grid: (H,W) long on device -> features: (Nobj, hidden) on same device
        dev = grid.device
        a = grid.detach().cpu().numpy().astype(np.int32)
        labels, n = connected_components_nonzero(a)
        if n <= 0:
            return torch.zeros(0, self.hidden, device=dev, dtype=torch.float32)
        # sizes for top-k
        sizes = [(labels == i).sum() for i in range(n)]
        order = np.argsort(sizes)[::-1][: self.max_comp]
        feats = []
        H, W = a.shape
        total = float(H * W) if H * W > 0 else 1.0
        for i in order:
            mask = (labels == int(i))
            ys, xs = np.where(mask)
            if ys.size == 0:
                continue
            h = float(ys.max() - ys.min() + 1)
            w = float(xs.max() - xs.min() + 1)
            area_frac = float(mask.sum()) / total
            cols = a[mask]
            # mean color embedding (by averaging embeddings of each pixel's color)
            col_emb = self.color_emb(torch.tensor(cols, device=dev, dtype=torch.long)).mean(dim=0)
            geo = self.geo(torch.tensor([h, w, area_frac], device=dev, dtype=torch.float32))
            feats.append((col_emb + geo) / 2.0)
        if not feats:
            return torch.zeros(0, self.hidden, device=dev, dtype=torch.float32)
        return torch.stack(feats, dim=0)

    def forward(self, grid: torch.Tensor) -> torch.Tensor:
        # returns (hidden)
        objs = self._objects_from_grid(grid)
        if objs.size(0) == 0:
            # no objects -> learnable or zero representation
            return torch.zeros(self.hidden, device=grid.device, dtype=torch.float32)
        if self.pool == "mean":
            return objs.mean(dim=0)
        # attention pooling with learnable query
        q = self.query.expand(objs.size(0) // objs.size(0) + 1, -1, -1)[:1].to(objs.device)
        x = objs.unsqueeze(0)  # (1, N, H)
        for att, ln in zip(self.atts, self.norms):
            out, _ = att(q, x, x)
            q = ln(q + out)
        return q.squeeze(0).squeeze(0)


class ProgramPolicy(nn.Module):
    def __init__(self):
        super().__init__()
        D = int(config.MODEL_CFG["d_model"])
        Hh = int(config.MODEL_CFG["nhead"])
        L = int(config.MODEL_CFG["nlayers"])
        drop = float(config.MODEL_CFG["dropout"])

        # Encoders
        self.cnn = CNNBackbone()
        self.use_obj = bool(config.MODEL_CFG.get("use_object_branch", True))
        if self.use_obj:
            self.obj = ObjectBranch()
            obj_dim = int(config.MODEL_CFG.get("obj_hidden", 128))
        else:
            self.obj = None
            obj_dim = 0

        # Projections to D
        self.proj_grid = nn.Linear(self.cnn.out_ch, D)
        fuse = str(config.MODEL_CFG.get("context_fusion", "concat_proj"))
        self.fuse = fuse
        if fuse == "concat_proj" and self.use_obj:
            self.proj_ctx = nn.Linear(D + obj_dim, D)
        else:
            self.proj_ctx = nn.Identity()

        # Sequence model over program tokens
        encoder_layer = nn.TransformerEncoderLayer(d_model=D, nhead=Hh, dropout=drop, batch_first=True)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=L)
        self.tok_emb = nn.Embedding(int(T.VOCAB_SIZE), D)
        self.pos = nn.Parameter(torch.zeros(1, int(config.MODEL_CFG["max_tokens"]), D))
        self.policy = nn.Linear(D, int(T.VOCAB_SIZE))
        self.value = nn.Linear(D, 1)

    def _encode_grid(self, grid: torch.Tensor) -> torch.Tensor:
        # CNN feature + (optional) object feature -> fused D-dim
        # Accept numpy arrays too
        if isinstance(grid, np.ndarray):
            grid = torch.from_numpy(grid)
        grid = grid.to(dtype=torch.long)
        g = self.cnn(grid)  # (Cout)
        g = self.proj_grid(g)
        if self.use_obj and self.obj is not None:
            o = self.obj(grid)  # (obj_dim)
            if self.fuse == "concat_proj":
                x = torch.cat([g, o], dim=-1)
                ctx = self.proj_ctx(x)
            else:
                # default to grid-only if unsupported
                ctx = g
        else:
            ctx = g
        return ctx

    def encode_task(self, pairs: List[Tuple[torch.Tensor, torch.Tensor]]) -> torch.Tensor:
        # returns (D)
        embs = []
        for inp, out in pairs:
            ei = self._encode_grid(inp)
            eo = self._encode_grid(out)
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
