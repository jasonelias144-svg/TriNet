"""Ours. experiments/holography.py: light moved forward and back returns;
a hologram refocuses each point at its own depth, and without a reference
beam it does not; a quarter of the plate still shows every point, twice
as wide; a hologram of n random-phase memories reads back with a match of
about 1/sqrt(n); keys at different angles read back exactly. On a plate
of 512 pixels, so the tests take seconds."""
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "experiments"))

import holography as Hg  # noqa: E402


def test_light_moved_forward_and_back_returns_and_keeps_its_energy():
    X, Y, f2 = Hg.grid(256)
    rng = np.random.default_rng(0)
    u = Hg.view(rng.standard_normal((256, 256)) + 1j * rng.standard_normal((256, 256)), f2, 0.3)
    there = Hg.propagate(u, 5000.0, f2)
    assert np.allclose(np.linalg.norm(there), np.linalg.norm(u))
    assert np.allclose(Hg.propagate(there, -5000.0, f2), u)


def test_each_point_comes_back_at_its_own_depth_but_not_without_a_reference():
    out, points = Hg.part_a(512)
    with_ref, without = out[True], out[False]
    assert np.all(np.diag(with_ref) > 0.9) and np.all(with_ref[~np.eye(3, dtype=bool)] < 0.1)
    assert np.diag(without).mean() < 0.3


def test_a_quarter_of_the_plate_shows_every_point_twice_as_wide():
    for pz, whole, cut, _, _, shown in Hg.part_b(512):
        assert 1.6 <= cut / whole <= 2.4, (pz, whole, cut)
        assert shown > 0.5


def test_random_phase_memories_read_back_at_one_over_root_n():
    rng = np.random.default_rng(1)
    score, found = Hg.random_keys(rng, 4, 1024)
    assert abs(score - 0.5) < 0.05 and found == 1.0


@pytest.mark.parametrize("rule", ["sum", "delta"])
def test_a_matrix_memory_returns_one_memory_exactly(rule):
    score, found = Hg.matrix_memory(np.random.default_rng(2), 1, 32, rule, trials=3)
    assert score == pytest.approx(1.0) and found == 1.0


def test_the_delta_rule_returns_the_last_memory_written_exactly_and_the_sum_does_not():
    rng = np.random.default_rng(4)
    keys, vals = Hg.unit_phase(rng, (8, 32)), Hg.unit_phase(rng, (8, 32))
    last = {rule: keys[-1] @ Hg.write_matrix(keys, vals, rule) / 32 for rule in ("sum", "delta")}
    assert np.allclose(last["delta"], vals[-1]) and not np.allclose(last["sum"], vals[-1])


def test_keys_at_different_angles_read_back_exactly():
    err, share = Hg.angle_keys(np.random.default_rng(3), m=128)
    assert err < 1e-10 and 9 * share < 0.1
