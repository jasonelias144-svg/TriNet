"""Ours, not Gemini's. What TriNetForCausalLM promises, checked on a small
model: stepping token by token with the cache gives what one full pass
gives; each of the cache's three states matters; generate() feeds the
model one new token at a time; the cache stays the same size however long
the text; and the model survives save_pretrained and
AutoModelForCausalLM.from_pretrained.

At its first values the model barely uses its yellow state, so a cache
that forgot it would pass. The fixture sets the yellow gates and the head
to values that make every state count.
"""
import json

import pytest
import torch
from transformers import AutoModelForCausalLM

from trinet.configuration_trinet import TriNetConfig
from trinet.modeling_trinet import TriNetForCausalLM

STATES = ("fast_weights", "yellow_tonic", "delay_buffers")


def close(a, b):
    torch.testing.assert_close(a, b, rtol=1e-5, atol=1e-5)


@pytest.fixture(scope="module")
def model():
    torch.manual_seed(0)
    cfg = TriNetConfig(vocab_size=256, d_model=48, d_key=24, d_val=24, n_layers=2, n_receptors=8)
    m = TriNetForCausalLM(cfg).eval()
    with torch.no_grad():
        for layer in m.model.layers:
            layer.mod_in.bias.fill_(0.3)
            layer.yellow_to_write.weight.normal_(0.0, 0.5)
            layer.yellow_to_retain.weight.normal_(0.0, 0.5)
        m.lm_head.weight.normal_(0.0, 0.5)
    return m


@pytest.fixture(scope="module")
def ids():
    torch.manual_seed(1)
    return torch.randint(4, 256, (2, 20))


@torch.no_grad()
def test_token_by_token_with_the_cache_equals_one_full_pass(model, ids):
    full = model(ids, use_cache=False).logits
    past, steps = None, []
    for t in range(ids.shape[1]):
        out = model(ids[:, t : t + 1], past_key_values=past, use_cache=True)
        past = out.past_key_values
        steps.append(out.logits)
    close(torch.cat(steps, dim=1), full)


@torch.no_grad()
def test_a_prompt_then_one_token_at_a_time_equals_one_full_pass(model, ids):
    full = model(ids, use_cache=False).logits
    out = model(ids[:, :8], use_cache=True)
    parts, past = [out.logits], out.past_key_values
    for t in range(8, ids.shape[1]):
        out = model(ids[:, t : t + 1], past_key_values=past, use_cache=True)
        past = out.past_key_values
        parts.append(out.logits)
    close(torch.cat(parts, dim=1), full)


@torch.no_grad()
@pytest.mark.parametrize("state", STATES)
def test_each_state_in_the_cache_matters(model, ids, state):
    # A cache that forgot this state at every step would drift from the
    # full pass by far more than the stepping tests allow: measured on
    # 2026-11-22, 0.31 (fast weights), 0.31 (yellow; a different drift,
    # peaking at the same token), 2.9 (delay line), against 3.3e-6 when
    # nothing is forgotten.
    full = model(ids, use_cache=False).logits
    past, steps = None, []
    for t in range(ids.shape[1]):
        out = model(ids[:, t : t + 1], past_key_values=past, use_cache=True)
        past = out.past_key_values
        steps.append(out.logits)
        for tensor in getattr(past, state):
            tensor.zero_()
    assert (torch.cat(steps, dim=1) - full).abs().max().item() > 1e-4, state


@torch.no_grad()
def test_generate_feeds_the_model_one_new_token_at_a_time(model, ids):
    lengths = []
    forward = model.model.forward

    def spy(*args, **kwargs):
        lengths.append(kwargs["input_ids"].shape[1])
        return forward(*args, **kwargs)

    model.model.forward = spy
    try:
        model.generate(ids[:1, :6], max_new_tokens=5, do_sample=False, use_cache=True)
    finally:
        del model.model.forward
    assert lengths == [6, 1, 1, 1, 1]


@torch.no_grad()
def test_the_cache_is_the_same_size_however_long_the_text(model):
    sizes = []
    for length in (8, 64, 256):
        out = model(torch.randint(4, 256, (1, length)), use_cache=True)
        assert out.past_key_values.get_seq_length() == length
        sizes.append(out.past_key_values.memory_bytes())
    assert len(set(sizes)) == 1


@torch.no_grad()
def test_save_and_load_through_auto_model_for_causal_lm(model, ids, tmp_path):
    model.save_pretrained(tmp_path, safe_serialization=True)
    config = json.loads((tmp_path / "config.json").read_text())
    assert config["model_type"] == "trinet"
    assert "AutoModelForCausalLM" in config["auto_map"]
    assert (tmp_path / "model.safetensors").exists()

    loaded = AutoModelForCausalLM.from_pretrained(tmp_path, trust_remote_code=True).eval()
    assert torch.equal(loaded(ids).logits, model(ids).logits)

    prompt = ids[:1, :6]
    cached = loaded.generate(prompt, max_new_tokens=12, do_sample=False, use_cache=True)
    uncached = loaded.generate(prompt, max_new_tokens=12, do_sample=False, use_cache=False)
    assert torch.equal(cached, uncached)
    assert torch.equal(cached, model.generate(prompt, max_new_tokens=12, do_sample=False))


def test_the_model_starts_from_pytorchs_own_values_but_for_the_gate_biases():
    # Change 5 (provenance/MANIFEST.json): Gemini's code drew every weight
    # from N(0, 0.02); with PyTorch's own values the model learns its
    # in-context task (experiments/README.md).
    torch.manual_seed(0)
    m = TriNetForCausalLM(TriNetConfig(vocab_size=256, d_model=48, d_key=24, d_val=24, n_layers=2))
    for layer in m.model.layers:
        assert layer.out_proj.weight.std().item() > 0.05
        assert layer.delay_conv.weight.std().item() > 0.03  # 0.042 here; 0.02 as Gemini drew it
        assert torch.all(layer.yellow_to_write.bias == 0.25)
        assert torch.all(layer.yellow_to_retain.bias == 0.90)
    assert m.model.embed_tokens.weight.std().item() > 0.5

