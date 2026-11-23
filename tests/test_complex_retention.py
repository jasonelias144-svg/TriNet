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

from complex_retention import THETA_MAX, THETA_MAX_WIDE, ComplexRetentionLayer, angles, make_complex  # noqa: E402
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
    c = make_complex(copy.deepcopy(m), THETA_MAX_WIDE)  # wide angles, so rotation shows
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


@pytest.mark.parametrize("theta_max", [THETA_MAX, THETA_MAX_WIDE])
def test_the_starting_angles_fill_their_range_and_w_im_starts_at_zero(theta_max):
    m = make_complex(plain_model(), theta_max)
    theta = torch.cat([layer.theta for layer in m.model.layers])
    assert theta.min().item() >= 0.0 and theta.max().item() < theta_max
    assert theta.max().item() > theta_max / 2 and theta.std().item() > 0
    assert all(layer.w_im.abs().max().item() == 0.0 for layer in m.model.layers)
    assert angles(m).count("layer ") == len(m.model.layers)


@torch.no_grad()
def test_starting_near_zero_the_model_starts_close_to_trinet_and_wide_does_not(ids):
    plain = plain_model()
    base = plain(ids, use_cache=False).logits
    gap = {}
    for theta_max in (THETA_MAX, THETA_MAX_WIDE):
        c = make_complex(copy.deepcopy(plain), theta_max)
        for layer in c.model.layers:
            layer.w_im.fill_(1.0)  # so the imaginary part is read
        gap[theta_max] = (c(ids, use_cache=False).logits - base).abs().max().item()
    assert 0.0 < gap[THETA_MAX] < gap[THETA_MAX_WIDE] / 10


def test_at_exactly_zero_the_angles_get_no_gradient_and_near_zero_they_do(ids):
    plain = plain_model()
    grads = {}
    for start in (0.0, THETA_MAX):
        c = make_complex(copy.deepcopy(plain), THETA_MAX)
        with torch.no_grad():
            for layer in c.model.layers:
                if start == 0.0:
                    layer.theta.zero_()
        c(ids, use_cache=False).logits.square().mean().backward()
        grads[start] = [(layer.theta.grad.abs().max().item(), layer.w_im.grad.abs().max().item())
                        for layer in c.model.layers]
    assert all(g == (0.0, 0.0) for g in grads[0.0])
    assert any(t > 0.0 and w > 0.0 for t, w in grads[THETA_MAX])
