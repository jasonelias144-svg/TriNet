"""Ours. experiments/holography.py: light moved forward and back returns;
a hologram refocuses each point at its own depth, and the control without
a reference beam does not; a quarter of the plate still shows every
point, about twice as wide and about a sixteenth as bright; a hologram of
n random-phase memories reads back with a match of about 1/sqrt(n), or
1/sqrt(2n) recorded as intensity; the matrices' rules differ as they
should; keys at different angles read back exactly, recorded or not. On a
plate of 512 pixels, so the tests take seconds."""
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
    out, points = Hg.part_a(768)  # (on 512 pixels the control's nearest point is not yet dim enough)
    with_ref, without = out[True], out[False]
    assert np.all(np.diag(with_ref) > 0.9) and np.all(with_ref[~np.eye(3, dtype=bool)] < 0.1)
    assert np.all(np.diag(without) < 0.5)  # no point is the brightest in its own plane
    # ... and not because the control reads darkness: it carries light
    lit, _ = Hg.write_and_read(768)
    control, _ = Hg.write_and_read(768, reference=False)
    assert all(control[z].max() > 1e-3 * lit[z].max() for z in lit)


def test_a_quarter_of_the_plate_shows_every_point_twice_as_wide():
    for r in Hg.part_b(512):
        for w, q in zip(r["whole"], r["quarter"]):  # along x and along y
            assert 1.6 <= q / w <= 2.4, r
        assert all(abs(w / r["expect_whole"] - 1) < 0.15 for w in r["whole"]), r
        assert all(abs(q / r["expect_quarter"] - 1) < 0.2 for q in r["quarter"]), r
        assert 0.04 < r["brightness"] < 0.1, r
        assert r["offset"][0] <= 1 and r["offset"][1] <= r["expect_quarter"] / 4, r  # each spot sits on its point


def test_random_phase_memories_read_back_at_one_over_root_n():
    rng = np.random.default_rng(1)
    score, found = Hg.random_keys(rng, 4, 1024, trials=20)
    assert abs(score - 0.5) < 0.05 and found == 1.0
    score, _ = Hg.random_keys(rng, 4, 1024, trials=20, recorded=True)  # the twins add as much noise again
    assert abs(score - 1 / np.sqrt(8)) < 0.05
    _, found = Hg.random_keys(rng, 1024, 256, trials=20)  # far too many to tell apart
    assert found < 0.5


@pytest.mark.parametrize("rule", ["sum", "delta"])
def test_a_matrix_memory_returns_one_memory_exactly(rule):
    score, found = Hg.matrix_memory(np.random.default_rng(2), 1, 32, rule, trials=3)
    assert score == pytest.approx(1.0) and found == 1.0


def test_the_three_matrix_rules_differ_as_they_should():
    # least squares returns every memory up to as many as a key has numbers;
    # the one-pass delta rule favours the newest; the sum is the same for all
    score = {rule: Hg.matrix_memory(np.random.default_rng(5), 24, 32, rule, trials=10)[0]
             for rule in ("sum", "delta", "lstsq")}
    assert score["lstsq"] == pytest.approx(1.0) and score["sum"] != score["delta"]
    old, *_, new = Hg.delta_by_age(np.random.default_rng(6), 48, m=32, trials=5)
    assert new > old + 0.2


def test_the_delta_rule_returns_the_last_memory_written_exactly_and_the_sum_does_not():
    rng = np.random.default_rng(4)
    keys, vals = Hg.unit_phase(rng, (8, 32)), Hg.unit_phase(rng, (8, 32))
    last = {rule: keys[-1] @ Hg.write_matrix(keys, vals, rule) / 32 for rule in ("sum", "delta")}
    assert np.allclose(last["delta"], vals[-1]) and not np.allclose(last["sum"], vals[-1])


@pytest.mark.parametrize("recorded", [False, True])
def test_keys_at_different_angles_read_back_exactly(recorded):
    err, share = Hg.angle_keys(np.random.default_rng(3), recorded=recorded)
    assert err < 1e-10 and 9 * share < 0.02
