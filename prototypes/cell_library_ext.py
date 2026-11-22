import numpy as np


class ReceptorBroadcastUnit:
    """
    Archetype 5: Composite-color volume-transmission receiver.
    Reads a shared 3-color field C = (R, B, Y) via a Michaelis-Menten / Cauchy
    chromaticity binding curve and shunted intensity gate. Supports Yellow-gated
    temporary receptor tuning that dissolves back to `base_affinity`.
    """

    def __init__(
        self,
        affinity: tuple[float, float, float],
        *,
        sigma: float = 0.18,
        g0: float = 0.05,
        threshold: float = 0.45,
        emit_channel: str = "red",
        decay_quiet: float = 0.80,
        decay_attend: float = 0.96,
        lr_tune: float = 0.30,
    ):
        a = np.maximum(0.0, np.asarray(affinity, dtype=np.float64))
        self.base_affinity = a / (np.sum(a) + 1e-8)
        self.fast_shift = np.zeros(3, dtype=np.float64)
        self.sigma = float(sigma)
        self.g0 = float(g0)
        self.threshold = float(threshold)
        self.emit_channel = emit_channel
        self.decay_quiet = decay_quiet
        self.decay_attend = decay_attend
        self.lr_tune = lr_tune

    def effective_affinity(self) -> np.ndarray:
        eff = np.maximum(0.0, self.base_affinity + self.fast_shift)
        return eff / (np.sum(eff) + 1e-8)

    def reset_fast_state(self):
        self.fast_shift.fill(0.0)

    def step(self, field_rby: np.ndarray, attend_yellow: float = 0.0) -> np.ndarray:
        c = np.maximum(0.0, np.asarray(field_rby, dtype=np.float64))
        intensity = float(np.sum(c))
        if intensity <= 1e-9:
            hue = np.zeros(3)
            act = 0.0
        else:
            hue = c / intensity
            a = self.effective_affinity()
            dist_sq = float(np.sum((hue - a) ** 2))
            s2 = self.sigma ** 2
            act = (s2 / (s2 + dist_sq)) * (intensity / (self.g0 + intensity))

        out_val = (
            max(0.0, (act - self.threshold) / (1.0 - self.threshold))
            if act >= self.threshold
            else 0.0
        )

        # Attention-gated receptor drift (dissolves back to base_affinity when unattended)
        att = float(np.clip(attend_yellow, 0.0, 1.0))
        lam = self.decay_quiet + (self.decay_attend - self.decay_quiet) * att
        self.fast_shift *= lam
        if att > 0.0 and intensity > 1e-9:
            self.fast_shift += self.lr_tune * att * (hue - self.effective_affinity())
            np.clip(self.fast_shift, -0.5, 0.5, out=self.fast_shift)

        out = np.zeros(3, dtype=np.float64)
        idx = {"red": 0, "blue": 1, "yellow": 2}[self.emit_channel]
        out[idx] = out_val
        return out


class DelayLineUnit:
    """
    Archetype 6: Cerebellar / Jeffress tapped delay-line unit.
    Maintains a K-tick history of Red cue inputs:
      - `predict_cancel`: emits timed Blue cancellation using `w_base + f_fast`
        and adapts online from a Yellow climbing-fiber error signal.
      - `check_coincidence`: detects exact ordered intervals between two streams.
    """

    def __init__(
        self,
        max_lag: int = 8,
        *,
        lr_fast: float = 0.80,
        decay_quiet: float = 0.995,
        decay_error: float = 0.90,
        consolidation_rate: float = 0.0,
        consolidation_thresh: float = 0.85,
    ):
        self.max_lag = int(max_lag)
        self.lr_fast = lr_fast
        self.decay_quiet = decay_quiet
        self.decay_error = decay_error
        self.consolidation_rate = consolidation_rate
        self.consolidation_thresh = consolidation_thresh

        self.buf_a = np.zeros(self.max_lag + 1, dtype=np.float64)
        self.buf_b = np.zeros(self.max_lag + 1, dtype=np.float64)
        self.w_base = np.zeros(self.max_lag, dtype=np.float64)
        self.f_fast = np.zeros(self.max_lag, dtype=np.float64)

    def reset_fast_state(self):
        self.buf_a.fill(0.0)
        self.buf_b.fill(0.0)
        self.f_fast.fill(0.0)

    def push(self, a_red: float, b_red: float = 0.0):
        self.buf_a = np.roll(self.buf_a, 1)
        self.buf_a[0] = float(a_red)
        self.buf_b = np.roll(self.buf_b, 1)
        self.buf_b[0] = float(b_red)

    def check_coincidence(self, lag_a: int, lag_b: int = 0, threshold: float = 0.7) -> float:
        va = max(0.0, self.buf_a[lag_a])
        vb = max(0.0, self.buf_b[lag_b])
        score = float(np.sqrt(va * vb))
        return score if score >= threshold else 0.0

    def predict_cancel(self) -> float:
        """Emits Blue predictive cancellation from taps at lags 1..max_lag."""
        taps = self.buf_a[1:]
        return float(np.dot(self.w_base + self.f_fast, taps))

    def climbing_fiber_update(self, error_yellow: float):
        """Updates dissolving fast tap weights from Yellow prediction error."""
        taps = self.buf_a[1:]
        lam = self.decay_error if abs(error_yellow) > 0.25 else self.decay_quiet
        self.f_fast = np.clip(
            lam * self.f_fast + self.lr_fast * error_yellow * taps, -2.0, 2.0
        )
        if self.consolidation_rate > 0.0:
            mask = np.abs(self.f_fast) >= self.consolidation_thresh
            self.w_base[mask] += self.consolidation_rate * self.f_fast[mask]
