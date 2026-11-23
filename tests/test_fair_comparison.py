"""Ours. experiments/ablations.py, gated_deltanet.py and fair_comparison.py:
each part taken out does nothing and stays so through training; the rest
of the model is untouched; the gated delta-rule memory is causal and the
size of a TriNet; the plain TriNet is built exactly as in stage_a.py."""
import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "experiments"))

import ablations  # noqa: E402
import fair_comparison as FC  # noqa: E402
import stage_a  # noqa: E402
from gated_deltanet import GatedDeltaLM  # noqa: E402


def trained(name, steps=3):
    """A model of fair_comparison after a few training steps on the harder
    task, and the same model before them."""
    torch.manual_seed(0)
    model, logits_of, data = FC.build(name)
    before = {k: v.detach().clone() for k, v in model.state_dict().items()}
    opt = torch.optim.Adam([p for p in model.parameters() if p.requires_grad], lr=3e-2)
    rng = __import__("numpy").random.default_rng(0)
    for _ in range(steps):
        x, metas = data.batch(4, 2, 4, rng)
        got, want, _, _ = stage_a.A.answers(logits_of(x), metas)
        opt.zero_grad()
        torch.nn.functional.cross_entropy(got, want).backward()
        opt.step()
    return model, before


def test_the_plain_trinet_is_stage_as():
    torch.manual_seed(7)
    ours = FC.build("trinet")[0].state_dict()
    torch.manual_seed(7)
    theirs = stage_a.build("trinet", "harder")[0].state_dict()
    assert ours.keys() == theirs.keys() and all(torch.equal(ours[k], theirs[k]) for k in ours)


def test_without_surprise_the_gates_never_learn_to_read_it():
    model, before = trained("trinet-no-surprise")
    moved = []
    for i, layer in enumerate(model.model.layers):
        n = layer.mod_in.in_features - 1
        assert layer.mod_in.weight[:, n].abs().max() == 0
        assert layer.yellow_to_write.weight[:, n].abs().max() == 0
        assert layer.yellow_to_retain.weight[:, n].abs().max() == 0
        for gate in ("mod_in", "yellow_to_write", "yellow_to_retain"):
            w = getattr(layer, gate).weight[:, :n]
            moved.append(not torch.equal(w, before[f"model.layers.{i}.{gate}.weight"][:, :n]))
    assert any(moved)  # ... while the rest of the gates learns


def test_fixed_gates_and_no_fast_memory_stay_fixed():
    for name in ("trinet-fixed-gates", "trinet-no-fast"):
        model, before = trained(name)
        for i, layer in enumerate(model.model.layers):
            assert layer.yellow_to_write.weight.abs().max() == 0
            assert not torch.equal(layer.W_base, before[f"model.layers.{i}.W_base"])  # the rest learns


@torch.no_grad()
def test_with_no_fast_memory_nothing_is_written():
    model = FC.build("trinet-no-fast")[0]
    out = model(torch.randint(4, 256, (2, 30)), use_cache=True)
    assert all(f.abs().max() == 0 for f in out.past_key_values.fast_weights)


@torch.no_grad()
def test_with_no_bus_the_bus_gives_the_same_whatever_it_hears():
    model = FC.build("trinet-no-bus")[0]
    bus = model.model.layers[0].receptor_bus
    a, b = bus(torch.randn(2, 5, 32)), bus(torch.randn(2, 5, 32))
    assert torch.equal(a, b) and a.shape == (2, 5, 8)


@torch.no_grad()
def test_with_no_shunt_the_shunt_is_exactly_zero():
    model = FC.build("trinet-no-shunt")[0]
    assert all(torch.nn.functional.softplus(layer.shunt_w).max() == 0 for layer in model.model.layers)


@torch.no_grad()
def test_the_gated_delta_memory_is_causal():
    torch.manual_seed(0)
    model = GatedDeltaLM().eval()
    x = torch.randint(0, 256, (2, 12))
    y = x.clone()
    y[:, 8:] = torch.randint(0, 256, (2, 4))
    assert torch.allclose(model(x)[:, :8], model(y)[:, :8])
    assert not torch.allclose(model(x)[:, 8:], model(y)[:, 8:])


def test_every_model_is_about_the_size_of_one_trinet():
    size = FC.trained_parameters("trinet")
    for name in FC.MODELS:
        if name in ("transformer", "transformer-nope"):
            continue  # answers_only.py's as they are; transformer-sized is the matched one
        n = FC.trained_parameters(name)
        if name == "trinet-no-delay":  # smaller by the delay line's three earlier taps, in both layers
            assert n == size - 2 * 3 * 32 * 32, (name, n)
        else:
            assert abs(n - size) / size <= 0.03, (name, n, size)


@pytest.mark.parametrize("name", FC.MODELS)
def test_every_model_runs_a_step_of_the_comparison(name, capsys):
    results = stage_a.run(name, "harder", seed=0, steps=1, eval_batches=1, builder=FC.build)
    assert set(results) == set(stage_a.TESTS["harder"])
