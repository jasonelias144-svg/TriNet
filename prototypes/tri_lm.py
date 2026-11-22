import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class TriLanguageLayer(nn.Module):
    """
    One stackable heterogeneous language layer:
      - DelayLineUnit:      K-tap causal conv parsing local character syntax
      - TriChannel memory:  W_base + dissolving fast weights F_fast (O(1) state)
      - ShuntingInterneuron: Blue divisive normalization on memory readout
      - PyramidalComparator: basal value vs apical F_fast prediction error
      - ModulatorNucleus:   slow tonic Yellow + sparse dead-zone write/retain gates
    """

    def __init__(
        self,
        d_model: int = 32,
        d_key: int = 20,
        d_val: int = 20,
        delay_lags: int = 4,
    ):
        super().__init__()
        self.d_model = d_model
        self.d_key = d_key
        self.d_val = d_val
        self.delay_lags = delay_lags

        # Archetype 6: Causal DelayLineUnit over token representations
        self.delay_conv = nn.Conv1d(d_model, d_model, kernel_size=delay_lags, padding=0)

        # Separate write-key, read-query, and basal value projections
        self.W_key = nn.Linear(d_model, d_key, bias=False)
        self.W_query = nn.Linear(d_model, d_key, bias=False)
        self.W_val = nn.Linear(d_model, d_val, bias=False)

        # Archetype 1: Slow baseline structure
        self.W_base = nn.Parameter(torch.randn(d_key, d_val) * 0.05)

        # Archetype 3: Blue ShuntingInterneuron pool
        self.shunt_w = nn.Parameter(torch.ones(d_val) * 0.15)
        self.log_g0 = nn.Parameter(torch.tensor(float(np.log(0.25))))

        # Archetypes 2 & 4: Pyramidal surprise -> ModulatorNucleus (sparse Yellow gates)
        self.tonic_decay_logit = nn.Parameter(torch.tensor(1.0))
        self.mod_in = nn.Linear(d_model + 1, 1)
        self.yellow_to_write = nn.Linear(d_model + 2, 1)
        self.yellow_to_retain = nn.Linear(d_model + 2, 1)
        nn.init.constant_(self.yellow_to_write.bias, 0.2)
        nn.init.constant_(self.yellow_to_retain.bias, 1.0)

        # Residual fusion back into d_model
        self.out_proj = nn.Linear(d_val + d_model, d_model)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, h: torch.Tensor, return_gates: bool = False):
        B, T, _ = h.shape

        # 1. Causal DelayLineUnit syntax extraction
        h_pad = F.pad(h.transpose(1, 2), (self.delay_lags - 1, 0))
        d_ctx = F.silu(self.delay_conv(h_pad).transpose(1, 2))

        k_write = F.silu(self.W_key(d_ctx))
        k_write = k_write / torch.sqrt((k_write ** 2).sum(dim=-1, keepdim=True) + 1e-4)

        q_read = F.silu(self.W_query(d_ctx))
        q_read = q_read / torch.sqrt((q_read ** 2).sum(dim=-1, keepdim=True) + 1e-4)

        v_basal_all = self.W_val(h)

        # 2. Recurrent dissolving fast-weight & ModulatorNucleus loop
        F_fast = torch.zeros(B, self.d_key, self.d_val, device=h.device)
        yellow_tonic = torch.zeros(B, 1, device=h.device)
        tonic_decay = torch.sigmoid(self.tonic_decay_logit)
        g0 = torch.exp(self.log_g0.clamp(-4.0, 1.0))
        shunt_w = F.softplus(self.shunt_w)

        mem_outs = []
        eta_sum = 0.0
        for t in range(T):
            qt = q_read[:, t, :]
            kt = k_write[:, t, :]
            vt = v_basal_all[:, t, :]
            dt = d_ctx[:, t, :]

            # Readout through W_base + F_fast under Blue divisive shunting
            r_read = qt @ self.W_base + torch.bmm(qt.unsqueeze(1), F_fast).squeeze(1)
            b_shunt = (F.relu(r_read) * shunt_w).sum(dim=-1, keepdim=True)
            mem_outs.append(r_read / (g0 + b_shunt))

            # Pyramidal mismatch (basal value vs apical fast-memory prediction)
            v_apical = torch.bmm(kt.unsqueeze(1), F_fast).squeeze(1)
            delta_err = vt - v_apical
            surprise = torch.sqrt((delta_err ** 2).mean(dim=-1, keepdim=True) + 1e-6)

            # ModulatorNucleus tonic Yellow + sparse dead-zone write/retain gates
            mod_drive = F.relu(self.mod_in(torch.cat([dt, surprise], dim=-1)))
            yellow_tonic = torch.clamp(tonic_decay * yellow_tonic + mod_drive, 0.0, 2.0)

            gate_in = torch.cat([dt, surprise, yellow_tonic], dim=-1)
            eta = 1.5 * torch.clamp(self.yellow_to_write(gate_in), 0.0, 1.0)
            lam = 0.50 + 0.50 * torch.clamp(self.yellow_to_retain(gate_in), 0.0, 1.0)
            eta_sum = eta_sum + eta.mean()

            outer = torch.bmm(kt.unsqueeze(2), delta_err.unsqueeze(1))
            F_fast = torch.clamp(
                lam.unsqueeze(-1) * F_fast + eta.unsqueeze(-1) * outer, -6.0, 6.0
            )

        mem_seq = torch.stack(mem_outs, dim=1)
        h_next = self.norm(h + self.out_proj(torch.cat([mem_seq, d_ctx], dim=-1)))
        if return_gates:
            return h_next, eta_sum / T
        return h_next


class TriLanguageModel(nn.Module):
    """Multi-layer heterogeneous character-level language model."""

    def __init__(
        self,
        vocab_size: int = 22,
        d_model: int = 32,
        d_key: int = 20,
        d_val: int = 20,
        n_layers: int = 2,
        delay_lags: int = 4,
    ):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, d_model)
        self.layers = nn.ModuleList(
            [
                TriLanguageLayer(
                    d_model=d_model,
                    d_key=d_key,
                    d_val=d_val,
                    delay_lags=delay_lags,
                )
                for _ in range(n_layers)
            ]
        )
        self.head = nn.Linear(d_model, vocab_size)

    def forward(self, idx: torch.Tensor, return_meta: bool = False):
        h = self.emb(idx)
        eta_total = 0.0
        for layer in self.layers:
            h, m_eta = layer(h, return_gates=True)
            eta_total = eta_total + m_eta
        logits = self.head(h)
        if return_meta:
            return logits, eta_total / len(self.layers)
        return logits
