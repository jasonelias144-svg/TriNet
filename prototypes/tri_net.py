import numpy as np


class TriColumn:
    """
    One heterogeneous 3-channel cortical column:
      - Red:    excitatory drive through W_base + dissolving F(t)
      - Blue:   local shunting interneuron pool (divisive normalization)
      - Yellow: modulator nucleus driven by pyramidal prediction mismatch
    """

    def __init__(
        self,
        d_in: int,
        d_out: int,
        *,
        g0: float = 0.2,
        shunt_strength: float = 0.25,
        decay_quiet: float = 0.75,
        decay_attend: float = 0.98,
        tonic_decay: float = 0.94,
        lr_fast: float = 1.0,
        consolidation_rate: float = 0.0,
        consolidation_thresh: float = 1.2,
        seed: int = 0,
    ):
        rng = np.random.default_rng(seed)
        self.d_in = d_in
        self.d_out = d_out

        # Slow baseline structure (learned across tasks / episodes)
        self.W_base = rng.normal(0.0, 1.0 / np.sqrt(d_in), size=(d_in, d_out))
        self.b_base = np.zeros(d_out)

        # Fast dissolving synaptic state F(t) (lives within an episode)
        self.F = np.zeros((d_in, d_out))

        # Modulator nucleus tonic Yellow state
        self.yellow_tonic = 0.0

        # Hyperparameters
        self.g0 = g0
        self.shunt_strength = shunt_strength
        self.decay_quiet = decay_quiet
        self.decay_attend = decay_attend
        self.tonic_decay = tonic_decay
        self.lr_fast = lr_fast
        self.consolidation_rate = consolidation_rate
        self.consolidation_thresh = consolidation_thresh

    def reset_fast_state(self):
        self.F.fill(0.0)
        self.yellow_tonic = 0.0

    def forward(self, x_red: np.ndarray, apical_yellow: np.ndarray | None = None):
        """
        x_red:         (d_in,) bottom-up Red input
        apical_yellow: (d_out,) optional top-down expectation from a higher column
        Returns: (out_red, blue_pool, surprise_yellow)
        """
        # 1. Excitatory drive through baseline + temporary dissolving weights
        r_raw = x_red @ (self.W_base + self.F) + self.b_base

        # 2. Local Blue shunting interneuron pool (divisive gain control)
        blue_pool = self.shunt_strength * float(np.sum(np.maximum(0.0, r_raw)))
        r_shunted = r_raw / (self.g0 + blue_pool)

        # 3. Pyramidal coincidence / surprise check against top-down apical Yellow
        if apical_yellow is not None:
            mismatch = np.maximum(0.0, r_shunted - apical_yellow)
            surprise_yellow = float(np.mean(mismatch))
            # Coincidence boost where bottom-up and top-down align
            coincidence = np.minimum(np.maximum(0.0, r_shunted), np.maximum(0.0, apical_yellow))
            out_red = np.maximum(0.0, r_shunted + 0.25 * coincidence)
        else:
            surprise_yellow = 0.0
            out_red = np.maximum(0.0, r_shunted)

        return out_red, blue_pool, surprise_yellow

    def plastic_update(
        self,
        x_red: np.ndarray,
        target_red: np.ndarray | None = None,
        external_yellow: float = 0.0,
        surprise_yellow: float = 0.0,
    ):
        """
        Runs one local step of ModulatorNucleus + attention-gated dissolution
        + 3-factor Delta fast-weight update + thresholded consolidation.
        """
        # Update slow ModulatorNucleus Yellow field
        self.yellow_tonic = min(
            1.5,
            self.tonic_decay * self.yellow_tonic + external_yellow + surprise_yellow,
        )
        att = min(1.0, max(0.0, self.yellow_tonic))

        # Attention-gated dissolution: high Yellow holds F(t); low Yellow dissolves it
        lam = self.decay_quiet + (self.decay_attend - self.decay_quiet) * att
        self.F *= lam

        # 3-factor local fast-weight update when attended and a teaching/binding target exists
        if target_red is not None and att > 0.05:
            pred = x_red @ (self.W_base + self.F) + self.b_base
            error = target_red - pred
            norm_sq = float(np.dot(x_red, x_red)) + 1e-6
            self.F += (self.lr_fast * att / norm_sq) * np.outer(x_red, error)
            np.clip(self.F, -5.0, 5.0, out=self.F)

        # Thresholded consolidation: only persistently strong traces bleed into W_base
        if self.consolidation_rate > 0.0:
            mask = np.abs(self.F) >= self.consolidation_thresh
            self.W_base[mask] += self.consolidation_rate * self.F[mask]
