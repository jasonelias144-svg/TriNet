"""Ours. What Gemini said of their prototypes that can be checked in a
moment (prototypes/README.md)."""
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "prototypes"))

import cell_library_ext  # noqa: E402
import meta_micro_brain  # noqa: E402
import tri_lm  # noqa: E402

NONE = (0.0, 0.0)


def count(model):
    return sum(p.numel() for p in model.parameters())


def test_the_two_pytorch_prototypes_have_the_sizes_gemini_gives():
    assert count(meta_micro_brain.MetaMicroBrain()) == 928
    assert count(tri_lm.TriLanguageModel()) == 18_098


@pytest.mark.parametrize("ticks, fires", [
    ([(1, 0), NONE, NONE, (0, 1), NONE], True),           # A, then B three ticks later
    ([(0, 1), NONE, NONE, (1, 0), NONE], False),          # B, then A
    ([(1, 0), (0, 1), NONE, NONE, NONE], False),          # too fast
    ([(1, 0), NONE, NONE, NONE, NONE, (0, 1)], False),    # too slow
    ([(1, 1), NONE, NONE, NONE], False),                  # together
    ([(1, 0), NONE, NONE, NONE, NONE], False),            # A alone
])
def test_a_delay_line_fires_only_for_its_order_and_interval(ticks, fires):
    unit = cell_library_ext.DelayLineUnit(max_lag=8)
    fired = []
    for a, b in ticks:
        unit.push(a, b)
        fired.append(unit.check_coincidence(3, 0) > 0)
    assert any(fired) == fires


HUES = {"red": (1, 0, 0), "blue": (0, 1, 0), "yellow": (0, 0, 1), "purple": (.5, .5, 0),
        "orange": (.5, 0, .5), "green": (0, .5, .5), "neutral": (1 / 3, 1 / 3, 1 / 3)}


@pytest.mark.parametrize("scale", [0.05, 1.0, 20.0])
def test_one_receptor_tuned_to_purple_fires_for_purple_alone(scale):
    for hue, rgb in HUES.items():
        unit = cell_library_ext.ReceptorBroadcastUnit((0.5, 0.5, 0.0))
        fired = unit.step(np.array(rgb, dtype=float) * scale)[0] > 0
        assert fired == (hue == "purple"), (hue, scale)
