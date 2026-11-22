"""Ours. The experiment script runs, for every model it names; its results
are in experiments/README.md, and take minutes, so here it only runs."""
import importlib.util
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parent.parent / "experiments" / "answers_only.py"
spec = importlib.util.spec_from_file_location("answers_only", SCRIPT)
answers_only = importlib.util.module_from_spec(spec)
spec.loader.exec_module(answers_only)


@pytest.mark.parametrize("model", ["tri_lm", "trinet", "trinet-gemini", "transformer"])
def test_the_experiment_runs(model):
    [results] = answers_only.main([model, "--steps", "2", "--seeds", "0", "--eval-batches", "1"])
    assert set(results) == set(answers_only.TESTS)
    assert all(0.0 <= v <= 100.0 for v in results.values())


def test_gemini_initial_values_are_what_their_init_drew():
    import torch
    model, _, _ = answers_only.build("trinet-gemini")
    layer = model.model.layers[0]
    for weight in (layer.out_proj.weight, layer.delay_conv.weight, layer.W_key.weight,
                   model.model.embed_tokens.weight, model.lm_head.weight):
        assert abs(weight.std().item() - 0.02) < 0.005
    assert torch.all(layer.yellow_to_write.bias == 0.25)
    assert torch.all(layer.yellow_to_retain.bias == 0.90)


def test_each_answer_is_read_at_the_gt_that_comes_before_it():
    import numpy as np
    import torch
    task = answers_only.CharTask(bytes_as_tokens=True)
    x, metas = task.batch(8, 2, 10, np.random.default_rng(0))
    positions = torch.arange(x.shape[1], dtype=torch.float).view(1, -1, 1).expand(x.shape[0], -1, 3)
    got, want, kinds, olds = answers_only.answers(positions, metas)
    rows = [b for b, meta in enumerate(metas) for _ in meta]
    cols = got[:, 0].long()
    assert torch.all(x[rows, cols] == ord(">"))
    assert torch.all(x[rows, cols + 1] == want)
    assert torch.all(olds[kinds == 2] != want[kinds == 2])
