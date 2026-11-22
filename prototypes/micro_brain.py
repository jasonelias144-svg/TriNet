import numpy as np

# Canonical 5-hue sensory palette (Red, Blue, Purple, Orange, Green)
DEFAULT_HUES = np.array(
    [
        [1.0, 0.0, 0.0],  # 0: Red
        [0.0, 1.0, 0.0],  # 1: Blue
        [0.5, 0.5, 0.0],  # 2: Purple
        [0.5, 0.0, 0.5],  # 3: Orange
        [0.0, 0.5, 0.5],  # 4: Green (disturbance warning)
    ],
    dtype=np.float64,
)


class ReceptorBroadcastUnit:
    """Archetype 5: Scale-invariant Michaelis-Menten composite-hue receiver."""

    def __init__(self, affinity, sigma=0.16, g0=0.015, threshold=0.45):
        a = np.maximum(0.0, np.asarray(affinity, dtype=np.float64))
        self.affinity = a / (np.sum(a) + 1e-8)
        self.sigma = float(sigma)
        self.g0 = float(g0)
        self.threshold = float(threshold)

    def read(self, field_rby: np.ndarray) -> float:
        c = np.maximum(0.0, field_rby)
        intensity = float(np.sum(c))
        if intensity <= 1e-9:
            return 0.0
        hue = c / intensity
        dist_sq = float(np.sum((hue - self.affinity) ** 2))
        s2 = self.sigma ** 2
        act = (s2 / (s2 + dist_sq)) * (intensity / (self.g0 + intensity))
        if act < self.threshold:
            return 0.0
        return (act - self.threshold) / (1.0 - self.threshold)


class ShuntingInterneuronPool:
    """Archetype 3: Fast divisive Blue gain-control pool."""

    def __init__(self, g0=0.15, blue_gain=0.30):
        self.g0 = float(g0)
        self.blue_gain = float(blue_gain)

    def normalize(self, red_vec: np.ndarray):
        blue = self.blue_gain * float(np.sum(np.maximum(0.0, red_vec)))
        return red_vec / (self.g0 + blue), blue


class DelayLineUnit:
    """Archetype 6: Cerebellar Purkinje cancellation & Jeffress order detector."""

    def __init__(self, max_lag=8, lr_fast=0.85, decay_quiet=0.992, decay_error=0.88):
        self.max_lag = int(max_lag)
        self.lr_fast = float(lr_fast)
        self.decay_quiet = float(decay_quiet)
        self.decay_error = float(decay_error)
        self.buf_a = np.zeros(self.max_lag + 1, dtype=np.float64)
        self.buf_b = np.zeros(self.max_lag + 1, dtype=np.float64)
        self.w_base = np.zeros(self.max_lag, dtype=np.float64)
        self.f_fast = np.zeros(self.max_lag, dtype=np.float64)

    def push(self, a_red: float, b_red: float = 0.0):
        self.buf_a = np.roll(self.buf_a, 1)
        self.buf_a[0] = float(a_red)
        self.buf_b = np.roll(self.buf_b, 1)
        self.buf_b[0] = float(b_red)

    def check_coincidence(self, lag_a: int, lag_b: int = 0, thresh: float = 0.55) -> int:
        va = max(0.0, self.buf_a[lag_a])
        vb = max(0.0, self.buf_b[lag_b])
        return 1 if np.sqrt(va * vb) >= thresh else 0

    def predict_cancel(self) -> float:
        return float(np.dot(self.w_base + self.f_fast, self.buf_a[1:]))

    def climbing_fiber_update(self, error_yellow: float):
        taps = self.buf_a[1:]
        lam = self.decay_error if abs(error_yellow) > 0.25 else self.decay_quiet
        self.f_fast = np.clip(
            lam * self.f_fast + self.lr_fast * error_yellow * taps, -2.0, 2.0
        )


class PyramidalComparatorBank:
    """Archetype 2: Compares basal feedback against apical prediction to emit Yellow surprise."""

    def compare(self, basal_true_vec: np.ndarray, apical_pred_vec: np.ndarray) -> float:
        return float(
            np.clip(np.sum(np.maximum(0.0, basal_true_vec - apical_pred_vec)), 0.0, 1.0)
        )


class ModulatorNucleus:
    """Archetype 4: Slow tonic Yellow broadcaster gating dissolution & plasticity."""

    def __init__(self, tonic_decay=0.84, surprise_gain=1.2, override_gain=0.45):
        self.tonic_decay = float(tonic_decay)
        self.surprise_gain = float(surprise_gain)
        self.override_gain = float(override_gain)
        self.yellow = 0.0

    def step(self, surprise_yellow: float, override_used: float) -> float:
        self.yellow = min(
            1.5,
            self.tonic_decay * self.yellow
            + self.surprise_gain * surprise_yellow
            + self.override_gain * override_used,
        )
        return min(1.0, self.yellow)


class MicroBrain:
    """
    Full 4-module, 6-archetype heterogeneous network:
      1. SensoryReceptorBus       (ReceptorBroadcastUnit x5 + ShuntingInterneuron)
      2. CerebellarTimingModule   (Purkinje DelayLineUnit + Jeffress DelayLineUnit)
      3. AssociativeMemoryColumn  (TriChannelUnit x4 + ShuntingInterneuron + dissolving F)
      4. ExecutiveModulatorModule (PyramidalComparatorBank + ModulatorNucleus)
    """

    def __init__(self, hues=DEFAULT_HUES, w_base=None):
        self.receptors = [ReceptorBroadcastUnit(hues[i]) for i in range(5)]
        self.sensory_shunt = ShuntingInterneuronPool(g0=0.15, blue_gain=0.30)

        self.purkinje = DelayLineUnit(max_lag=8, lr_fast=0.85, decay_quiet=0.992, decay_error=0.88)
        self.jeffress = DelayLineUnit(max_lag=4)

        self.W_base = (
            np.eye(4, dtype=np.float64) * 2.0
            if w_base is None
            else np.array(w_base, dtype=np.float64)
        )
        self.F = np.zeros((4, 4), dtype=np.float64)
        self.mem_shunt = ShuntingInterneuronPool(g0=0.20, blue_gain=0.25)
        self.decay_quiet = 0.72
        self.decay_attend = 0.992
        self.lr_fast = 2.5

        self.pyramidal = PyramidalComparatorBank()
        self.modulator = ModulatorNucleus(tonic_decay=0.84, surprise_gain=1.2, override_gain=0.45)

    def step(self, field_rby: np.ndarray, true_action: int | None = None, shock: float = 0.0):
        # 1. Sensory decoding & divisive normalization
        raw_hues = np.array([r.read(field_rby) for r in self.receptors], dtype=np.float64)
        cue_red, _ = self.sensory_shunt.normalize(raw_hues[:4])
        cue_max = float(np.max(cue_red))
        cue_vec = (cue_red / cue_max) if cue_max > 0.25 else np.zeros(4)
        warn_red = 1.0 if raw_hues[4] > 0.35 else 0.0

        # 2. Cerebellar predictive cancellation & temporal order detection
        self.purkinje.push(warn_red)
        brace_out = self.purkinje.predict_cancel()
        self.jeffress.push(cue_vec[0], cue_vec[1])
        seq_alert = self.jeffress.check_coincidence(lag_a=2, lag_b=0, thresh=0.55)
        self.purkinje.climbing_fiber_update(shock - brace_out)

        # 3. Associative working-memory column (TriChannel units + Blue shunting)
        r_drive = cue_vec @ (self.W_base + self.F)
        out_red, _ = self.mem_shunt.normalize(r_drive)
        pred_action = int(np.argmax(out_red))

        # 4. Pyramidal surprise comparison -> Modulator Yellow -> fast-weight plasticity
        surprise_y = 0.0
        override_used = 0.0
        if true_action is not None:
            basal_true = np.zeros(4)
            basal_true[true_action] = 1.0
            apical_pred = np.zeros(4)
            apical_pred[pred_action] = 1.0
            surprise_y = self.pyramidal.compare(basal_true, apical_pred)
            base_pred = int(np.argmax(cue_vec @ self.W_base))
            override_used = 1.0 if (pred_action == true_action and base_pred != true_action) else 0.0

        att = self.modulator.step(surprise_y, override_used)
        lam = self.decay_quiet + (self.decay_attend - self.decay_quiet) * att
        self.F *= lam

        if true_action is not None and surprise_y > 0.0 and float(np.dot(cue_vec, cue_vec)) > 1e-4:
            basal_true = np.zeros(4)
            basal_true[true_action] = 1.0
            current = cue_vec @ (self.W_base + self.F)
            norm_sq = float(np.dot(cue_vec, cue_vec)) + 1e-6
            self.F += (self.lr_fast * att / norm_sq) * np.outer(
                cue_vec, (2.0 * basal_true - current) / 2.0
            )
            np.clip(self.F, -5.0, 5.0, out=self.F)

        return pred_action, brace_out, seq_alert
