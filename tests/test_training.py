"""Ours, not Gemini's. The training script's presets have the sizes Gemini's
table gives (conversation line 41370), and a few steps of offline training
run end to end: both models train, the trained model's cache passes the
script's own check, and the checkpoint saved is the model trained.
"""
import json

import pytest
import torch
from transformers import AutoModelForCausalLM

import train_trinet
from trinet.configuration_trinet import TriNetConfig
from trinet.modeling_trinet import TriNetForCausalLM

# preset: (TriNetForCausalLM, MatchedTransformerLM), as Gemini's table gives them
TABLE = {
    "smoke": (59_708, 71_648),
    "1m": (837_512, 836_320),
    "4m": (3_797_460, 3_797_280),
    "10m": (10_330_288, 10_312_832),
}


def count(model):
    return sum(p.numel() for p in model.parameters())


@pytest.mark.parametrize("preset", sorted(TABLE))
def test_each_preset_has_the_size_the_table_gives(preset):
    cfg = TriNetConfig(vocab_size=256, **train_trinet.PRESETS[preset])
    trinet = count(TriNetForCausalLM(cfg))
    baseline = count(train_trinet.build_matched_transformer(trinet, cfg, use_pos_emb=True, max_len=512))
    assert (trinet, baseline) == TABLE[preset]


def test_a_short_offline_run_trains_both_models_and_saves_a_checkpoint(tmp_path):
    args = train_trinet.build_arg_parser().parse_args([
        "--preset", "smoke", "--smoke-test", "--seq-len", "48", "--batch-size", "4",
        "--steps", "6", "--warmup-steps", "2", "--log-every", "3", "--save-every", "0",
        "--reasoning-mix", "0.5", "--device", "cpu", "--output-dir", str(tmp_path),
    ])
    summary = train_trinet.run_training(args)
    assert summary == json.loads((tmp_path / "training_metrics.json").read_text())
    assert (summary["trinet_params"], summary["baseline_params"]) == TABLE["smoke"]
    # Both models were trained, and their losses recorded at every log step.
    assert [h["step"] for h in summary["history"]] == [1, 3, 6]
    for h in summary["history"]:
        assert h["trinet_loss"] > 0 and h["baseline_loss"] > 0
    # The script's own check that the trained model's cache gives what a
    # full pass gives.
    assert summary["o1_cache_check"]["max_logit_diff"] < 1e-4
    # The checkpoint saved is the model trained: it gives the same text.
    loaded = AutoModelForCausalLM.from_pretrained(tmp_path / "trinet_hf", trust_remote_code=True).eval()
    assert count(loaded) == TABLE["smoke"][0]
    tokenizer = train_trinet.ByteTokenizer()
    prompt = torch.tensor([tokenizer.encode("In a quiet workshop", add_bos=True)])
    out = loaded.generate(prompt, max_new_tokens=32, do_sample=False, use_cache=True)[0].tolist()
    assert tokenizer.decode(out) == summary["sample_generation"]
