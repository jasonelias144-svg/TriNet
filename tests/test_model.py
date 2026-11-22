"""Ours, not Gemini's. What TriNetForCausalLM promises, checked on a small
model: stepping token by token with the cache gives what one full pass
gives; the cache stays the same size however long the text; and the model
survives save_pretrained and AutoModelForCausalLM.from_pretrained.
"""
import json

import pytest
import torch
from transformers import AutoModelForCausalLM

from trinet.configuration_trinet import TriNetConfig
from trinet.modeling_trinet import TriNetForCausalLM

TOL = 1e-4


@pytest.fixture(scope="module")
def model():
    torch.manual_seed(0)
    cfg = TriNetConfig(vocab_size=256, d_model=48, d_key=24, d_val=24, n_layers=2, n_receptors=8)
    return TriNetForCausalLM(cfg).eval()


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
    assert (torch.cat(steps, dim=1) - full).abs().max().item() < TOL


@torch.no_grad()
def test_a_prompt_then_one_token_at_a_time_equals_one_full_pass(model, ids):
    full = model(ids, use_cache=False).logits
    out = model(ids[:, :8], use_cache=True)
    parts, past = [out.logits], out.past_key_values
    for t in range(8, ids.shape[1]):
        out = model(ids[:, t : t + 1], past_key_values=past, use_cache=True)
        past = out.past_key_values
        parts.append(out.logits)
    assert (torch.cat(parts, dim=1) - full).abs().max().item() < TOL


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
    assert (loaded(ids).logits - model(ids).logits).abs().max().item() == 0.0

    prompt = ids[:1, :6]
    cached = loaded.generate(prompt, max_new_tokens=12, do_sample=False, use_cache=True)
    uncached = loaded.generate(prompt, max_new_tokens=12, do_sample=False, use_cache=False)
    assert torch.equal(cached, uncached)
    assert torch.equal(cached, model.generate(prompt, max_new_tokens=12, do_sample=False))
