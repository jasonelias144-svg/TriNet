"""Ours, not Gemini's. Checks for Stage A: the harder task, Gemini's
BicameralTriNet (bicameral/bicameral_stage_a.py), the residual_error switch
(change 7), and experiments/stage_a.py."""
import collections
import re
import sys
import tempfile
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "experiments"))

import harder_task as H  # noqa: E402
from bicameral.bicameral_stage_a import BicameralTriNetConfig, BicameralTriNetForCausalLM  # noqa: E402
from trinet.configuration_trinet import TriNetConfig  # noqa: E402
from trinet.modeling_trinet import TriNetForCausalLM  # noqa: E402

SMALL = dict(vocab_size=256, d_model=32, d_key=20, d_val=20, n_layers=2, delay_lags=4, n_receptors=8)
HALF = dict(vocab_size=256, d_model=32, d_hemi=20, d_key=12, d_val=12, n_layers=2, delay_lags=4,
            n_receptors=8, callosum_dim=8, hemisphere_dropout=0.0)


def read_episode(text):
    """An interpreter written apart from the generator: reads the
    assignments in order (a pointer copies its target's value then) and
    answers the queries."""
    head, _, tail = text.partition("?")
    value, history = {}, collections.defaultdict(list)
    for name, right in re.findall(r"([a-h])=([0-9a-h]);", head):
        value[name] = value[right] if right.isalpha() else right
        if right.isdigit():
            history[name].append(right)
    answers = []
    for name, mark, given in re.findall(r"([a-h])([<>])(\d);", "?" + tail):
        answers.append((given, value[name] if mark == ">" else history[name][-2]))
    return answers


def test_every_answer_of_the_harder_task_follows_from_its_text():
    rng = np.random.default_rng(0)
    for lo, hi in [(0, 0), (2, 10), (24, 24), (74, 74)]:
        for _ in range(500):
            text, meta, _ = H.make_episode(int(rng.integers(lo, hi + 1)), int(rng.integers(lo, hi + 1)), rng)
            pairs = read_episode(text)
            assert len(pairs) == len(meta) and all(given == right for given, right in pairs), text
            for pos, ans, kind in meta:  # the answer follows the ">" or "<"
                assert text[pos] in "<>" and text[pos + 1] == chr(ans) and text[pos - 2] == "?"


def test_the_harder_task_varies_what_a_rule_could_fix():
    rng = np.random.default_rng(1)
    targets, depths, kinds = collections.Counter(), collections.Counter(), collections.Counter()
    copies_differ = 0
    for _ in range(2000):
        text, meta, rec = H.make_episode(4, 4, rng)
        for p, (target, _) in rec["pointers"].items():
            targets["pointer" if target in rec["pointers"] else "bound"] += 1
            bound = target
            while bound in rec["pointers"]:
                bound = rec["pointers"][bound][0]
            copies_differ += rec["pointers"][p][1] != rec["history"][bound][-1]
        for v in rec["bound"]:
            depths[len(rec["history"][v]) - 1] += 1
        kinds.update(k for _, _, k in meta)
    assert targets["pointer"] > 0 and targets["bound"] > 0  # chains happen
    assert set(depths) == {0, 1, 2, 3, 4}  # from no overwrite up to four
    assert set(kinds) == {1, 2, 3, 4}
    assert copies_differ > 0  # a pointer's value can differ from its target's value now


def test_no_simple_rule_scores_near_full_marks():
    rng = np.random.default_rng(2)
    scores = H.shortcut_scores([H.make_episode(6, 6, rng) for _ in range(500)])
    assert max(scores.values()) < 70, scores


def test_bicameral_stepping_matches_one_pass_and_generate_steps_one_token():
    torch.manual_seed(0)
    x = torch.randint(0, 256, (2, 10))
    for switches in ({}, {"symmetric": False}, {"callosum": False}):
        model = BicameralTriNetForCausalLM(BicameralTriNetConfig(**{**HALF, **switches})).eval()
        with torch.no_grad():
            full = model(x, use_cache=False).logits
            cache, steps = None, []
            for t in range(x.shape[1]):
                out = model(x[:, t:t + 1], past_key_values=cache, use_cache=True)
                cache, steps = out.past_key_values, steps + [out.logits]
        assert torch.allclose(full, torch.cat(steps, 1), atol=1e-5), switches
        lengths = []
        hook = model.register_forward_pre_hook(
            lambda m, a, kw: lengths.append((kw.get("input_ids") if kw.get("input_ids") is not None else a[0]).shape[1]),
            with_kwargs=True)
        with torch.no_grad():
            model.generate(x[:1, :6], max_new_tokens=4, do_sample=False)
        hook.remove()
        assert lengths == [6, 1, 1, 1], (switches, lengths)


def test_bicameral_saves_and_loads():
    torch.manual_seed(0)
    model = BicameralTriNetForCausalLM(BicameralTriNetConfig(**HALF, symmetric=False)).eval()
    x = torch.randint(0, 256, (1, 8))
    with tempfile.TemporaryDirectory() as d, torch.no_grad():
        model.save_pretrained(d)
        again = BicameralTriNetForCausalLM.from_pretrained(d).eval()
        assert torch.equal(again(x, use_cache=False).logits, model(x, use_cache=False).logits)


def test_the_symmetric_switch_changes_only_one_stacks_retention_floor():
    sym = BicameralTriNetForCausalLM(BicameralTriNetConfig(**HALF))
    asym = BicameralTriNetForCausalLM(BicameralTriNetConfig(**HALF, symmetric=False))
    floors = lambda m: [(a.min_retention, b.min_retention) for a, b in zip(m.model.layers_a, m.model.layers_b)]
    assert floors(sym) == [(0.5, 0.5)] * 2 and floors(asym) == [(0.95, 0.5)] * 2
    assert sum(p.numel() for p in sym.parameters()) == sum(p.numel() for p in asym.parameters())


def test_residual_error_is_off_by_default_and_changes_the_model_when_on():
    assert TriNetConfig().residual_error is False
    x = torch.randint(0, 256, (2, 12))
    torch.manual_seed(0)
    off = TriNetForCausalLM(TriNetConfig(**SMALL)).eval()
    torch.manual_seed(0)
    on = TriNetForCausalLM(TriNetConfig(**SMALL, residual_error=True)).eval()
    assert all(layer.residual_error for layer in on.model.layers)
    assert all(torch.equal(a, b) for a, b in zip(off.parameters(), on.parameters()))  # same weights
    with torch.no_grad():
        assert not torch.allclose(off(x, use_cache=False).logits, on(x, use_cache=False).logits)
    with tempfile.TemporaryDirectory() as d:
        on.save_pretrained(d)
        assert TriNetForCausalLM.from_pretrained(d).config.residual_error is True


def test_the_stage_a_script_runs_every_model_on_both_tasks(capsys):
    import stage_a
    for task in stage_a.TASKS:
        for name in ("trinet-residual", "bicam-nocal", "transformer"):
            results = stage_a.run(name, task, seed=0, steps=2, eval_batches=1)
            assert set(results) == {"~52 characters", "~100 characters", "~200 characters"}
    assert "before last overwrite" in capsys.readouterr().out
