"""Ours (Claude). A starting point for an executive-controlled bicameral
model: a TriNet side, a Transformer side, and an executive between them.

- The TriNet side is a stack of `trinet.modeling_trinet.TriNetLayer`: a
  memory of fixed size that tracks values as they change.
- The Transformer side is a small causal Transformer with rotary positions
  (no learned position table, so longer texts do not meet unseen positions),
  optionally attending only to the last `window` tokens.
- Each side has its own output head and predicts the next token on its own.
- The executive is a separate network with its own parameters. For each
  token it reads a compressed summary of each side and how unsure each side
  is (the entropy of its prediction), and gives a weight `alpha`: how far to
  trust the TriNet side. The model's prediction is the mixture
  `alpha * p_trinet + (1 - alpha) * p_transformer`.
- By default the executive's inputs are detached: it learns where to send
  trust without changing what the sides learn. Each side is also trained
  on its own prediction (`side_loss_weight`), so neither withers when the
  executive favours the other.

`sides` gives the single-side models used as controls: "trinet" or
"transformer" alone, with the same embedding and head. `mix="fixed"` gives
the two sides with a constant alpha of 0.5 instead of the executive.

This is the first, weighing level of the executive. It does not yet decide
which side runs at all (routing), or steer either side's settings.
"""
from dataclasses import asdict, dataclass
from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from trinet.configuration_trinet import TriNetConfig
from trinet.modeling_trinet import TriNetLayer


@dataclass
class HybridConfig:
    vocab_size: int = 256
    d_embed: int = 32
    sides: str = "both"  # "both", "trinet" or "transformer"
    mix: str = "executive"  # with both sides: "executive" or "fixed"
    # TriNet side
    tri_d: int = 24
    tri_key: int = 16
    tri_layers: int = 2
    tri_delay: int = 4
    tri_receptors: int = 8
    tri_residual_error: bool = False
    # Transformer side
    tf_d: int = 24
    tf_heads: int = 4
    tf_ff: int = 64
    tf_layers: int = 2
    tf_window: Optional[int] = None  # None: attend to every earlier token
    # Executive
    exec_d: int = 8
    detach_executive: bool = True
    side_loss_weight: float = 0.5

    def __post_init__(self):
        assert self.sides in ("both", "trinet", "transformer"), self.sides
        assert self.mix in ("executive", "fixed"), self.mix
        assert self.tf_d % self.tf_heads == 0 and (self.tf_d // self.tf_heads) % 2 == 0


def rotary(x: torch.Tensor, base: float = 10000.0) -> torch.Tensor:
    """Rotary position encoding over the last dimension of (B, H, T, D)."""
    t, d = x.shape[-2], x.shape[-1]
    half = d // 2
    freq = base ** (-torch.arange(half, device=x.device, dtype=x.dtype) / half)
    angle = torch.arange(t, device=x.device, dtype=x.dtype)[:, None] * freq[None, :]
    cos, sin = angle.cos(), angle.sin()
    x1, x2 = x[..., :half], x[..., half:]
    return torch.cat([x1 * cos - x2 * sin, x1 * sin + x2 * cos], dim=-1)


class AttentionBlock(nn.Module):
    def __init__(self, d: int, heads: int, ff: int, window: Optional[int]):
        super().__init__()
        self.heads, self.window = heads, window
        self.norm1, self.norm2 = nn.LayerNorm(d), nn.LayerNorm(d)
        self.qkv, self.out = nn.Linear(d, 3 * d), nn.Linear(d, d)
        self.ff = nn.Sequential(nn.Linear(d, ff), nn.GELU(), nn.Linear(ff, d))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, t, d = x.shape
        q, k, v = self.qkv(self.norm1(x)).view(b, t, 3, self.heads, d // self.heads).permute(2, 0, 3, 1, 4)
        q, k = rotary(q), rotary(k)
        i = torch.arange(t, device=x.device)
        allowed = i[:, None] >= i[None, :]
        if self.window is not None:
            allowed = allowed & (i[:, None] - i[None, :] < self.window)
        att = F.scaled_dot_product_attention(q, k, v, attn_mask=allowed)
        x = x + self.out(att.transpose(1, 2).reshape(b, t, d))
        return x + self.ff(self.norm2(x))


class TransformerSide(nn.Module):
    def __init__(self, cfg: HybridConfig):
        super().__init__()
        self.inp = nn.Linear(cfg.d_embed, cfg.tf_d)
        self.blocks = nn.ModuleList(
            [AttentionBlock(cfg.tf_d, cfg.tf_heads, cfg.tf_ff, cfg.tf_window) for _ in range(cfg.tf_layers)])
        self.norm = nn.LayerNorm(cfg.tf_d)

    def forward(self, e: torch.Tensor) -> torch.Tensor:
        h = self.inp(e)
        for block in self.blocks:
            h = block(h)
        return self.norm(h)


class TriNetSide(nn.Module):
    def __init__(self, cfg: HybridConfig):
        super().__init__()
        tcfg = TriNetConfig(vocab_size=cfg.vocab_size, d_model=cfg.tri_d, d_key=cfg.tri_key, d_val=cfg.tri_key,
                            n_layers=cfg.tri_layers, delay_lags=cfg.tri_delay, n_receptors=cfg.tri_receptors,
                            residual_error=cfg.tri_residual_error)
        self.inp = nn.Linear(cfg.d_embed, cfg.tri_d)
        self.layers = nn.ModuleList([TriNetLayer(tcfg, layer_idx=i) for i in range(cfg.tri_layers)])
        for layer in self.layers:  # the gate biases TriNet's own initial values set (change 5)
            nn.init.constant_(layer.yellow_to_write.bias, 0.25)
            nn.init.constant_(layer.yellow_to_retain.bias, 0.90)
        self.norm = nn.LayerNorm(cfg.tri_d)

    def forward(self, e: torch.Tensor) -> torch.Tensor:
        h = self.inp(e)
        for layer in self.layers:
            h = layer(h)[0]
        return self.norm(h)


class Executive(nn.Module):
    """For each token: how far to trust the TriNet side (alpha, 0 to 1)."""

    def __init__(self, cfg: HybridConfig):
        super().__init__()
        self.from_tri = nn.Linear(cfg.tri_d, cfg.exec_d)
        self.from_tf = nn.Linear(cfg.tf_d, cfg.exec_d)
        self.gate = nn.Sequential(nn.Linear(2 * cfg.exec_d + 2, cfg.exec_d), nn.GELU(), nn.Linear(cfg.exec_d, 1))
        nn.init.zeros_(self.gate[-1].bias)  # starts even-handed

    def forward(self, h_tri, h_tf, logp_tri, logp_tf) -> torch.Tensor:
        ent = lambda lp: -(lp.exp() * lp).sum(-1, keepdim=True)  # how unsure each side is
        x = torch.cat([F.silu(self.from_tri(h_tri)), F.silu(self.from_tf(h_tf)), ent(logp_tri), ent(logp_tf)], -1)
        return torch.sigmoid(self.gate(x))


class BicameralHybrid(nn.Module):
    def __init__(self, cfg: HybridConfig):
        super().__init__()
        self.cfg = cfg
        self.embed = nn.Embedding(cfg.vocab_size, cfg.d_embed)
        if cfg.sides in ("both", "trinet"):
            self.tri = TriNetSide(cfg)
            self.head_tri = nn.Linear(cfg.tri_d, cfg.vocab_size)
        if cfg.sides in ("both", "transformer"):
            self.tf = TransformerSide(cfg)
            self.head_tf = nn.Linear(cfg.tf_d, cfg.vocab_size)
        if cfg.sides == "both" and cfg.mix == "executive":
            self.executive = Executive(cfg)

    def forward(self, ids: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Log-probabilities of the next token: the model's ("logp"), each
        side's ("logp_tri", "logp_tf"), and the executive's alpha."""
        e = self.embed(ids)
        out: Dict[str, torch.Tensor] = {}
        if self.cfg.sides in ("both", "trinet"):
            h_tri = self.tri(e)
            out["logp_tri"] = F.log_softmax(self.head_tri(h_tri), -1)
        if self.cfg.sides in ("both", "transformer"):
            h_tf = self.tf(e)
            out["logp_tf"] = F.log_softmax(self.head_tf(h_tf), -1)
        if self.cfg.sides == "trinet":
            out["logp"] = out["logp_tri"]
            return out
        if self.cfg.sides == "transformer":
            out["logp"] = out["logp_tf"]
            return out
        if self.cfg.mix == "fixed":
            alpha = torch.full_like(out["logp_tri"][..., :1], 0.5)
        else:
            inputs = (h_tri, h_tf, out["logp_tri"], out["logp_tf"])
            if self.cfg.detach_executive:
                inputs = tuple(x.detach() for x in inputs)
            alpha = self.executive(*inputs)
        out["alpha"] = alpha
        out["logp"] = torch.logsumexp(
            torch.stack([torch.log(alpha + 1e-9) + out["logp_tri"], torch.log1p(-alpha + 1e-9) + out["logp_tf"]]), 0)
        return out

    def loss(self, out: Dict[str, torch.Tensor], select, targets: torch.Tensor) -> torch.Tensor:
        """Loss at the positions `select` picks out of (B, T, V) tensors: the
        model's prediction, plus each side's own, weighted by side_loss_weight."""
        loss = F.nll_loss(select(out["logp"]), targets)
        if self.cfg.sides == "both":
            for key in ("logp_tri", "logp_tf"):
                loss = loss + self.cfg.side_loss_weight * F.nll_loss(select(out[key]), targets)
        return loss

    def describe(self) -> str:
        n = sum(p.numel() for p in self.parameters())
        return f"{n:,} parameters; " + ", ".join(f"{k}={v}" for k, v in asdict(self.cfg).items())
