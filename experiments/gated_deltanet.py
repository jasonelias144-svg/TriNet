"""Ours. A small gated delta-rule memory (after Gated DeltaNet, Yang, Kautz
& Hatamizadeh, 2024, arXiv:2412.06464), for the fair comparison of
STATUS.md, Next 1: a plain memory of the same kind as TriNet's fast
memory, without TriNet's own parts.

Each layer: a short causal convolution over the input; a key, a query and
a value; a decay alpha and a write strength beta, each one number per
step, from the input alone; then, with the memory S (d_key x d_val, one
per sequence):

    S <- alpha * S + beta * k^T (v - alpha * k S)
    o  = q S

(the delta rule with a gate, written as rows, as TriNet's memory is). The
read passes a normalization and an output gate, then a projection; a small
MLP follows, as in Gated DeltaNet's blocks. Unlike TriNet, the gates read
only the input, never the memory, which is what lets the published model
train in parallel chunks; here it runs one step at a time, like TriNet,
so the two take comparable time on a CPU.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class GatedDeltaLayer(nn.Module):
    def __init__(self, d_model: int, d_key: int, d_val: int, conv: int = 4, mlp: int = 2):
        super().__init__()
        self.conv = nn.Conv1d(d_model, d_model, conv, groups=d_model)  # depthwise, causal by padding
        self.pad = conv - 1
        self.q = nn.Linear(d_model, d_key, bias=False)
        self.k = nn.Linear(d_model, d_key, bias=False)
        self.v = nn.Linear(d_model, d_val, bias=False)
        self.a = nn.Linear(d_model, 1)
        self.b = nn.Linear(d_model, 1)
        self.g = nn.Linear(d_model, d_val, bias=False)
        self.o = nn.Linear(d_val, d_model, bias=False)
        self.norm1 = nn.LayerNorm(d_model)
        self.mlp = nn.Sequential(nn.Linear(d_model, mlp * d_model), nn.SiLU(), nn.Linear(mlp * d_model, d_model))
        self.norm2 = nn.LayerNorm(d_model)
        self.d_key, self.d_val = d_key, d_val

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        bsz, seq_len, _ = x.shape
        h = F.silu(self.conv(F.pad(x.transpose(1, 2), (self.pad, 0))).transpose(1, 2))
        q = F.normalize(F.silu(self.q(h)), dim=-1)
        k = F.normalize(F.silu(self.k(h)), dim=-1)
        v = self.v(h)
        alpha = torch.sigmoid(self.a(h))  # (B, T, 1)
        beta = torch.sigmoid(self.b(h))
        s = x.new_zeros(bsz, self.d_key, self.d_val)
        outs = []
        for t in range(seq_len):
            kt, at, bt = k[:, t], alpha[:, t].unsqueeze(-1), beta[:, t].unsqueeze(-1)
            err = v[:, t] - at.squeeze(-1) * torch.bmm(kt.unsqueeze(1), s).squeeze(1)
            s = at * s + bt * torch.bmm(kt.unsqueeze(2), err.unsqueeze(1))
            outs.append(torch.bmm(q[:, t].unsqueeze(1), s).squeeze(1))
        read = torch.stack(outs, dim=1)
        read = F.layer_norm(read, (self.d_val,)) * F.silu(self.g(h))
        x = self.norm1(x + self.o(read))
        return self.norm2(x + self.mlp(x))


class GatedDeltaLM(nn.Module):
    """Token ids to logits, with the same embedding and head as a TriNet
    of answers_only.SMALL (256 byte tokens, width 32), so the memories are
    what differs."""

    def __init__(self, vocab: int = 256, d_model: int = 32, d_key: int = 24, d_val: int = 24, layers: int = 2):
        super().__init__()
        self.emb = nn.Embedding(vocab, d_model)
        self.layers = nn.ModuleList(GatedDeltaLayer(d_model, d_key, d_val) for _ in range(layers))
        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab)

    def forward(self, ids: torch.Tensor) -> torch.Tensor:
        x = self.emb(ids)
        for layer in self.layers:
            x = layer(x)
        return self.head(self.norm(x))
