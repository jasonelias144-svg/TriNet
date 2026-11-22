import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class MetaMicroBrain(nn.Module):
    """
    Fully differentiable, 100% randomly initialized 6-archetype network:
      1. TriChannelUnit        -> Associative working-memory column (W_base + F_fast)
      2. PyramidalUnit         -> Basal-vs-apical mismatch (v_t - k_t @ F_fast)
      3. ShuntingInterneuron   -> Divisive Blue normalization in sensory & memory pools
      4. ModulatorNucleus      -> Slow tonic Yellow gating plasticity (eta) & decay (lam)
      5. ReceptorBroadcastUnit -> Learnable Michaelis-Menten 3-color hue receptors
      6. DelayLineUnit         -> Learnable K-tap causal shift register
    """

    def __init__(
        self,
        n_receptors: int = 16,
        d_key: int = 16,
        d_val: int = 16,
        delay_lags: int = 4,
        n_vals: int = 6,
    ):
        super().__init__()
        self.n_receptors = n_receptors
        self.d_key = d_key
        self.d_val = d_val
        self.delay_lags = delay_lags
        self.n_vals = n_vals

        # Archetype 5: Learnable ReceptorBroadcastUnits (random affinities & widths)
        self.affinity_logits = nn.Parameter(torch.randn(n_receptors, 3) * 0.5)
        self.log_sigma = nn.Parameter(torch.full((n_receptors,), float(np.log(0.35))))
        self.log_g0 = nn.Parameter(torch.tensor(float(np.log(0.03))))

        # Archetype 3: Learnable Blue Shunting Interneuron pools
        self.shunt_w_sensory = nn.Parameter(torch.ones(n_receptors) * 0.25)
        self.log_g0_sensory = nn.Parameter(torch.tensor(float(np.log(0.15))))
        self.shunt_w_mem = nn.Parameter(torch.ones(d_val) * 0.20)
        self.log_g0_mem = nn.Parameter(torch.tensor(float(np.log(0.20))))

        # Key & Value encoders
        self.W_key = nn.Linear(n_receptors, d_key, bias=False)
        self.W_val = nn.Linear(n_vals, d_val, bias=False)

        # Archetype 6: Learnable DelayLineUnit (K-tap causal shift register)
        self.delay_proj = nn.Linear(n_receptors, 4, bias=False)
        self.delay_taps = nn.Parameter(torch.randn(delay_lags, 4) * 0.1)

        # Archetype 1: Slow baseline weights W_base (outer-loop learned)
        self.W_base = nn.Parameter(torch.randn(d_key, d_val) * 0.05)

        # Archetypes 2 & 4: Pyramidal surprise -> ModulatorNucleus tonic Yellow
        self.tonic_decay_logit = nn.Parameter(torch.tensor(1.5))
        self.mod_in = nn.Linear(3, 1)
        self.yellow_to_write = nn.Linear(4, 1)
        self.yellow_to_retain = nn.Linear(4, 1)

        # Task readout
        self.readout = nn.Linear(d_val + 4, n_vals)

    def receptor_bus_seq(self, field_rby: torch.Tensor) -> torch.Tensor:
        c = F.relu(field_rby)                                        # (B, T, 3)
        intensity = c.sum(dim=-1, keepdim=True)                      # (B, T, 1)
        hue = c / (intensity + 1e-6)                                 # (B, T, 3)
        A = F.softmax(self.affinity_logits, dim=-1)                  # (M, 3)
        s2 = torch.exp(2.0 * self.log_sigma.clamp(-3.0, 1.0))        # (M,)
        g0 = torch.exp(self.log_g0.clamp(-5.0, 0.0))

        dist_sq = ((hue.unsqueeze(2) - A.view(1, 1, -1, 3)) ** 2).sum(dim=-1)
        raw_rec = (s2 / (s2 + dist_sq)) * (intensity / (g0 + intensity))

        b_pool = (raw_rec * F.softplus(self.shunt_w_sensory)).sum(dim=-1, keepdim=True)
        g0_s = torch.exp(self.log_g0_sensory.clamp(-4.0, 1.0))
        return raw_rec / (g0_s + b_pool)

    def forward(self, x: torch.Tensor, return_gates: bool = False):
        """
        x: (B, T, 3 + n_vals + 2) -> [R, B, Y, val_onehot, is_bind, is_query]
        """
        B, T, _ = x.shape

        # 1. Sensory Receptor Bus + Blue shunting
        s_red = self.receptor_bus_seq(x[:, :, :3])                   # (B, T, M)
        k = F.silu(self.W_key(s_red))                                # (B, T, d_key)
        k_unit = k / torch.sqrt((k ** 2).sum(dim=-1, keepdim=True) + 1e-4)
        v_basal_all = self.W_val(x[:, :, 3 : 3 + self.n_vals])       # (B, T, d_val)
        ctrl_all = x[:, :, 3 + self.n_vals :]                        # (B, T, 2)

        # 2. Causal DelayLineUnit over sensory stream
        d_in = self.delay_proj(s_red)                                # (B, T, 4)
        d_pad = F.pad(d_in, (0, 0, self.delay_lags - 1, 0))
        delay_ctx_all = torch.zeros(B, T, 4, device=x.device)
        for lag in range(self.delay_lags):
            start = (self.delay_lags - 1) - lag
            delay_ctx_all = delay_ctx_all + d_pad[:, start : start + T, :] * self.delay_taps[lag]

        # 3. Episodic dissolving fast-weight & ModulatorNucleus loop
        F_fast = torch.zeros(B, self.d_key, self.d_val, device=x.device)
        yellow_tonic = torch.zeros(B, 1, device=x.device)
        tonic_decay = torch.sigmoid(self.tonic_decay_logit)
        g0_m = torch.exp(self.log_g0_mem.clamp(-4.0, 1.0))
        shunt_w_m = F.softplus(self.shunt_w_mem)

        out_mem_list, eta_list, lam_list = [], [], []
        for t in range(T):
            kt = k_unit[:, t, :]
            vt = v_basal_all[:, t, :]
            ct = ctrl_all[:, t, :]

            # AssociativeTriColumn Red readout + Blue shunting
            v_apical = torch.bmm(kt.unsqueeze(1), F_fast).squeeze(1)
            r_mem = kt @ self.W_base + v_apical
            b_mem = (F.relu(r_mem) * shunt_w_m).sum(dim=-1, keepdim=True)
            out_mem_list.append(r_mem / (g0_m + b_mem))

            # Pyramidal mismatch & surprise
            delta_err = vt - v_apical
            surprise = torch.sqrt((delta_err ** 2).mean(dim=-1, keepdim=True) + 1e-6)

            # ModulatorNucleus slow Yellow state -> plasticity & retention gates
            mod_drive = F.relu(self.mod_in(torch.cat([ct, surprise], dim=-1)))
            yellow_tonic = torch.clamp(tonic_decay * yellow_tonic + mod_drive, 0.0, 2.0)

            gate_in = torch.cat([ct, surprise, yellow_tonic], dim=-1)
            eta = 1.5 * torch.sigmoid(self.yellow_to_write(gate_in))
            lam = 0.50 + 0.50 * torch.sigmoid(self.yellow_to_retain(gate_in))
            if return_gates:
                eta_list.append(eta.mean().item())
                lam_list.append(lam.mean().item())

            # Yellow-gated dissolving fast-weight Delta update
            outer = torch.bmm(kt.unsqueeze(2), delta_err.unsqueeze(1))
            F_fast = torch.clamp(
                lam.unsqueeze(-1) * F_fast + eta.unsqueeze(-1) * outer, -6.0, 6.0
            )

        out_mem_all = torch.stack(out_mem_list, dim=1)
        logits = self.readout(torch.cat([out_mem_all, delay_ctx_all], dim=-1))
        if return_gates:
            return logits, eta_list, lam_list
        return logits
