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
    assert abs(layer.out_proj.weight.std().item() - 0.02) < 0.005
    assert torch.all(layer.yellow_to_write.bias == 0.25)
