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


def roles(make, seeds=range(300)):
    """Which binding the pointer names, and which is overwritten, by position."""
    import numpy as np
    seen = set()
    for seed in seeds:
        text, _ = make(3, 4, 4, np.random.default_rng(seed))
        bound = [text[i] for i in range(0, 12, 4)]
        pointer, overwrite = text[16:20], text[20:24]  # after four characters of filler
        seen.add((bound.index(pointer[2]), bound.index(overwrite[0])))
    return seen


def test_gemini_task_as_given_always_points_to_the_first_and_overwrites_the_second():
    spec = importlib.util.spec_from_file_location("pointer_check", SCRIPT.parent / "pointer_check.py")
    pointer_check = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(pointer_check)
    assert roles(pointer_check.TASKS["as given"]) == {(0, 1)}
    # Change 6: every pair of different bindings occurs.
    assert roles(answers_only.train_trinet.make_reasoning_example) == {
        (a, b) for a in range(3) for b in range(3) if a != b}


def test_the_pointer_check_runs():
    spec = importlib.util.spec_from_file_location("pointer_check", SCRIPT.parent / "pointer_check.py")
    pointer_check = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(pointer_check)
    [results] = pointer_check.main(["trinet", "--steps", "2", "--seeds", "0"])
    assert {task for task, _ in results} == {"as given", "change 6"}
    assert answers_only.train_trinet.make_reasoning_example is pointer_check.TASKS["change 6"]
