"""Ours. experiments/complex_retention.py: with its rotation at zero it is
TriNetLayer bit for bit; with rotation it differs; stepping token by token
with the cache gives what one full pass gives; and making a model complex
leaves the random state as it was."""
import copy
import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "experiments"))

from complex_retention import ComplexRetentionLayer, make_complex  # noqa: E402
from trinet.configuration_trinet import TriNetConfig  # noqa: E402
from trinet.modeling_trinet import TriNetForCausalLM  # noqa: E402


def plain_model(residual_error=False):
    torch.manual_seed(0)
    cfg = TriNetConfig(vocab_size=256, d_model=48, d_key=24, d_val=24, n_layers=2, n_receptors=8,
                       residual_error=residual_error)
    m = TriNetForCausalLM(cfg).eval()
    with torch.no_grad():  # gates that write and keep, so the memory matters (as in test_model.py)
        for layer in m.model.layers:
            layer.mod_in.bias.fill_(0.3)
            layer.yellow_to_write.weight.normal_(0.0, 0.5)
            layer.yellow_to_retain.weight.normal_(0.0, 0.5)
            layer.yellow_to_write.bias.fill_(0.35)
            layer.yellow_to_retain.bias.fill_(0.80)
        m.lm_head.weight.normal_(0.0, 0.5)
    return m


def rotating(m):
    c = make_complex(copy.deepcopy(m))
    with torch.no_grad():
        for layer in c.model.layers:
            layer.w_im.normal_(0.0, 1.0)
    return c


@pytest.fixture(scope="module")
def ids():
    torch.manual_seed(1)
    return torch.randint(4, 256, (2, 20))


@torch.no_grad()
@pytest.mark.parametrize("residual_error", [False, True])
def test_without_rotation_it_is_trinet_bit_for_bit(ids, residual_error):
    plain = plain_model(residual_error)
    c = rotating(plain)  # w_im nonzero: it must still not matter
    for layer in c.model.layers:
        assert isinstance(layer, ComplexRetentionLayer)
        layer.theta.zero_()
    assert torch.equal(c(ids, use_cache=False).logits, plain(ids, use_cache=False).logits)


@torch.no_grad()
def test_without_rotation_the_imaginary_part_stays_zero(ids):
    c = rotating(plain_model())
    for layer in c.model.layers:
        layer.theta.zero_()
    out = c(ids, use_cache=True)
    fast = out.past_key_values.fast_weights
    assert all(f.shape[-1] == 2 * 24 and f[..., 24:].abs().max().item() == 0.0 for f in fast)
    assert any(f[..., :24].abs().max().item() > 0.0 for f in fast)  # (the first layer's gates write nothing here)


@torch.no_grad()
def test_rotation_changes_what_the_model_says(ids):
    plain = plain_model()
    c = rotating(plain)
    gap = (c(ids, use_cache=False).logits - plain(ids, use_cache=False).logits).abs().max().item()
    assert gap > 1e-3
    out = c(ids, use_cache=True)
    assert any(f[..., 24:].abs().max().item() > 0.0 for f in out.past_key_values.fast_weights)


@torch.no_grad()
def test_token_by_token_with_the_cache_equals_one_full_pass(ids):
    c = rotating(plain_model())
    full = c(ids, use_cache=False).logits
    past, steps = None, []
    for t in range(ids.shape[1]):
        out = c(ids[:, t : t + 1], past_key_values=past, use_cache=True)
        past = out.past_key_values
        steps.append(out.logits)
    torch.testing.assert_close(torch.cat(steps, dim=1), full, rtol=1e-5, atol=1e-5)


def test_making_a_model_complex_leaves_the_random_state_alone():
    m = plain_model()
    torch.manual_seed(5)
    expected = torch.rand(3)
    torch.manual_seed(5)
    make_complex(m)
    assert torch.equal(torch.rand(3), expected)
    angles = torch.cat([layer.theta for layer in m.model.layers])
    assert angles.min().item() >= 0.0 and angles.max().item() < torch.pi / 10 and angles.std().item() > 0
