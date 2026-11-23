"""Ours, not Gemini's. Complex retention: a fast memory that fades and
rotates.

In TriNetLayer the fast memory F is multiplied each step by a real
retention lam (from the yellow gate), so old traces only fade. Here F is
complex, held as two real matrices (F_re, F_im side by side), and each
step multiplies row j (key dimension j) by lam * exp(i * theta_j):

    F <- lam * exp(i*theta) . F + eta * k^T (v - k F)

so a trace's phase advances with its age, like a clock. The angles theta
(one per key dimension) are learned. The read adds the imaginary part,
scaled per value dimension by w_im (learned, starting at zero):

    r = q W_base + Re(q F) + w_im * Im(q F)

The error is complex too: its real part is v - k F_re (with W_base added
when residual_error is on, change 7), its imaginary part - k F_im, and the
surprise the yellow gate reads is the size of both.

With theta = 0, F_im stays zero and this is TriNetLayer exactly
(tests/test_complex_retention.py checks it, bit for bit). The founder
agreed to this test on 2026-11-23 (STATUS.md, Decided); the reasons are in
docs/NEURON-CATALOG.md.

    make_complex(model)   # turns a TriNetForCausalLM's layers complex,
                          # keeping every weight it has
    angles(model)         # the angles and imaginary weights, summarized
"""
import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from trinet.modeling_trinet import TriNetLayer

# Starting angles, drawn from [0, theta_max). The first version started
# them in [0, pi/10) (THETA_MAX_WIDE): a trace then turns half a circle in
# 10 characters or more, so an old trace read through Re(q F) is scaled by
# cos(theta * age), which swings between 1 and -1 within one episode of
# the harder task. It scored 37 to 81% in six seeds, well below TriNet
# (experiments/README.md); the angles are the likely cause, and this
# version tests that. Now they start in [0, 0.003) (THETA_MAX): a
# trace turns at most 0.38 radians over the training length (about 125
# characters), so the model starts close to TriNet, and the angles it ends
# with are learned rather than given. At exactly zero they would not move:
# with theta = 0 and w_im = 0 neither gets a gradient.
THETA_MAX = 0.003
THETA_MAX_WIDE = math.pi / 10


class ComplexRetentionLayer(TriNetLayer):
    def __init__(self, config, layer_idx: int):
        super().__init__(config, layer_idx)
        self.theta = nn.Parameter(torch.zeros(config.d_key))
        self.w_im = nn.Parameter(torch.zeros(config.d_val))

    def forward(
        self,
        hidden_states: torch.Tensor,
        fast_weights: Optional[torch.Tensor] = None,
        yellow_tonic: Optional[torch.Tensor] = None,
        delay_buffer: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        # As TriNetLayer.forward, but for the memory's real and imaginary
        # parts; the lines that differ are marked "complex".
        bsz, seq_len, _ = hidden_states.shape
        device = hidden_states.device
        dtype = hidden_states.dtype

        h_t = hidden_states.transpose(1, 2)
        buf_len = self.delay_lags - 1
        if buf_len > 0:
            if delay_buffer is None:
                delay_buffer = torch.zeros(bsz, self.d_model, buf_len, device=device, dtype=dtype)
            h_cat = torch.cat([delay_buffer, h_t], dim=-1)
            next_delay_buffer = h_cat[:, :, -buf_len:].contiguous()
        else:
            h_cat = h_t
            next_delay_buffer = torch.zeros(bsz, self.d_model, 0, device=device, dtype=dtype)

        d_ctx = F.silu(self.delay_conv(h_cat).transpose(1, 2))
        rec_feat = self.receptor_bus(d_ctx)

        k_write = F.silu(self.W_key(d_ctx))
        k_write = k_write / torch.sqrt((k_write * k_write).sum(dim=-1, keepdim=True) + 1e-4)
        q_read = F.silu(self.W_query(d_ctx))
        q_read = q_read / torch.sqrt((q_read * q_read).sum(dim=-1, keepdim=True) + 1e-4)
        v_basal_all = self.W_val(hidden_states)

        # complex: the memory is (B, d_key, 2 * d_val), real part first. A
        # cache made by TriNetCache holds only d_val columns, all zero at
        # the start; the imaginary part then starts at zero too.
        if fast_weights is None:
            fast_weights = torch.zeros(bsz, self.d_key, self.d_val, device=device, dtype=dtype)
        if fast_weights.shape[-1] == self.d_val:
            fast_weights = torch.cat([fast_weights, torch.zeros_like(fast_weights)], dim=-1)
        f_re, f_im = fast_weights[..., : self.d_val], fast_weights[..., self.d_val :]

        if yellow_tonic is None:
            y_tonic = torch.zeros(bsz, 1, device=device, dtype=dtype)
        else:
            y_tonic = yellow_tonic

        tonic_decay = torch.sigmoid(self.tonic_decay_logit)
        g0 = torch.exp(self.log_g0.clamp(-4.0, 1.0))
        shunt_w = F.softplus(self.shunt_w)
        retain_span = 1.0 - self.min_retention
        cos = torch.cos(self.theta).view(1, -1, 1)  # complex: rotation of row j
        sin = torch.sin(self.theta).view(1, -1, 1)

        mem_outs = []
        eta_sum = hidden_states.new_zeros(())

        for t in range(seq_len):
            qt = q_read[:, t, :]
            kt = k_write[:, t, :]
            vt = v_basal_all[:, t, :]
            rt = rec_feat[:, t, :]

            # complex: Re(q F) and, scaled by w_im, Im(q F)
            r_read = (
                torch.matmul(qt, self.W_base)
                + torch.bmm(qt.unsqueeze(1), f_re).squeeze(1)
                + self.w_im * torch.bmm(qt.unsqueeze(1), f_im).squeeze(1)
            )
            b_shunt = (F.relu(r_read) * shunt_w).sum(dim=-1, keepdim=True)
            mem_outs.append(r_read / (g0 + b_shunt))

            v_apical = torch.bmm(kt.unsqueeze(1), f_re).squeeze(1)
            if self.residual_error:
                v_apical = v_apical + torch.matmul(kt, self.W_base)
            delta_err = vt - v_apical
            delta_im = -torch.bmm(kt.unsqueeze(1), f_im).squeeze(1)  # complex: v is real
            surprise = torch.sqrt(
                ((delta_err * delta_err) + (delta_im * delta_im)).mean(dim=-1, keepdim=True) + 1e-6
            )

            mod_drive = F.relu(self.mod_in(torch.cat([rt, surprise], dim=-1)))
            y_tonic = torch.clamp(tonic_decay * y_tonic + mod_drive, 0.0, 2.0)

            gate_in = torch.cat([rt, surprise, y_tonic], dim=-1)
            eta = self.max_write_rate * torch.clamp(self.yellow_to_write(gate_in), 0.0, 1.0)
            lam = self.min_retention + retain_span * torch.clamp(
                self.yellow_to_retain(gate_in), 0.0, 1.0
            )
            eta_sum = eta_sum + eta.mean()

            lam3, eta3 = lam.unsqueeze(-1), eta.unsqueeze(-1)
            outer_re = torch.bmm(kt.unsqueeze(2), delta_err.unsqueeze(1))
            outer_im = torch.bmm(kt.unsqueeze(2), delta_im.unsqueeze(1))
            # complex: fade and rotate, then write; each part clipped as F_fast is
            new_re = lam3 * (cos * f_re - sin * f_im) + eta3 * outer_re
            new_im = lam3 * (sin * f_re + cos * f_im) + eta3 * outer_im
            f_re = torch.clamp(new_re, -self.max_fast_weight, self.max_fast_weight)
            f_im = torch.clamp(new_im, -self.max_fast_weight, self.max_fast_weight)

        mem_seq = torch.stack(mem_outs, dim=1)
        fused = torch.cat([mem_seq, d_ctx, rec_feat], dim=-1)
        next_hidden = self.norm(hidden_states + self.out_proj(fused))
        f_fast = torch.cat([f_re, f_im], dim=-1)
        return next_hidden, f_fast, y_tonic, next_delay_buffer, eta_sum / max(1, seq_len)


def make_complex(model, theta_max: float = THETA_MAX):
    """Replace each layer of a TriNetForCausalLM by a ComplexRetentionLayer
    with the same weights, angles drawn from [0, theta_max) and w_im zero.
    The random state is restored afterwards, so a run with the same seed
    sees the same training as the plain model."""
    layers = model.model.layers
    with torch.random.fork_rng(devices=[]):
        for idx, layer in enumerate(layers):
            new = ComplexRetentionLayer(model.config, idx)
            missing, unexpected = new.load_state_dict(layer.state_dict(), strict=False)
            assert set(missing) == {"theta", "w_im"} and not unexpected, (missing, unexpected)
            with torch.no_grad():
                new.theta.uniform_(0.0, theta_max)
            new.train(layer.training)
            layers[idx] = new.to(next(layer.parameters()).device)
    return model


def angles(model) -> str:
    """One line per layer: the angles (mean and largest, in radians per
    character, and the largest turn over 125 characters) and the mean size
    of w_im."""
    out = []
    for i, layer in enumerate(model.model.layers):
        th = layer.theta.detach().abs()
        out.append(f"layer {i}: angle mean {th.mean():.4f}, max {th.max():.4f} "
                   f"(turn over 125 characters up to {125 * th.max():.2f}); "
                   f"|w_im| mean {layer.w_im.detach().abs().mean():.3f}")
    return "; ".join(out)
