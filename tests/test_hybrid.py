"""Ours. Checks for the executive-controlled TriNet + Transformer prototype
(hybrid/model.py), the recall task and experiments/hybrid_prototype.py."""
import re
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "experiments"))

import recall_task as R  # noqa: E402
from hybrid.model import BicameralHybrid, HybridConfig  # noqa: E402

SMALL = dict(tri_d=16, tri_key=8, tf_d=16, tf_ff=32, exec_d=4)


def test_no_model_sees_the_future():
    torch.manual_seed(0)
    x = torch.randint(0, 256, (2, 12))
    y = x.clone()
    y[:, 7:] = torch.randint(0, 256, (2, 5))  # change only positions 7 onwards
    for sides in ("both", "trinet", "transformer"):
        model = BicameralHybrid(HybridConfig(sides=sides, **SMALL)).eval()
        with torch.no_grad():
            a, b = model(x), model(y)
        for key in ("logp", "logp_tri", "logp_tf", "alpha"):
            if key in a:
                assert torch.allclose(a[key][:, :7], b[key][:, :7], atol=1e-6), (sides, key)
                assert not torch.allclose(a[key][:, 7:], b[key][:, 7:]), (sides, key)


def test_the_window_limits_what_attention_sees():
    torch.manual_seed(0)
    model = BicameralHybrid(HybridConfig(sides="transformer", tf_layers=1, tf_window=3, **SMALL)).eval()
    x = torch.randint(0, 256, (1, 10))
    y = x.clone()
    y[0, 0] = (x[0, 0] + 1) % 256  # change only the first token
    with torch.no_grad():
        a, b = model(x)["logp"], model(y)["logp"]
    assert not torch.allclose(a[0, :3], b[0, :3])  # positions 0 to 2 see it
    assert torch.allclose(a[0, 3:], b[0, 3:], atol=1e-6)  # later ones do not


def test_the_mixture_is_alpha_of_trinet_plus_the_rest_of_the_transformer():
    torch.manual_seed(0)
    x = torch.randint(0, 256, (2, 9))
    for mix in ("executive", "fixed"):
        out = BicameralHybrid(HybridConfig(mix=mix, **SMALL)).eval()(x)
        a = out["alpha"]
        want = a * out["logp_tri"].exp() + (1 - a) * out["logp_tf"].exp()
        assert torch.allclose(out["logp"].exp(), want, atol=1e-5), mix
        assert torch.allclose(out["logp"].exp().sum(-1), torch.ones(2, 9), atol=1e-4)
        if mix == "fixed":
            assert torch.all(a == 0.5)
        else:
            assert torch.all((a > 0) & (a < 1)) and a.std() > 0  # it varies by token


def test_the_executive_does_not_steer_the_sides_unless_attached():
    x = torch.randint(0, 256, (2, 9))
    for detach in (True, False):
        torch.manual_seed(0)
        model = BicameralHybrid(HybridConfig(detach_executive=detach, **SMALL))
        model(x)["alpha"].sum().backward()
        side_grads = [p.grad for n, p in model.named_parameters() if n.startswith(("tri.", "tf.", "embed."))]
        moved = any(g is not None and g.abs().sum() > 0 for g in side_grads)
        assert moved == (not detach)
        assert all(p.grad is not None for p in model.executive.parameters())


def test_each_side_is_trained_on_its_own_prediction_too():
    torch.manual_seed(0)
    model = BicameralHybrid(HybridConfig(**SMALL))
    x = torch.randint(0, 256, (2, 9))
    out = model(x)
    pick = lambda t: t[[0, 1], [3, 5]]
    want = torch.tensor([65, 66])
    alone = torch.nn.functional.nll_loss(pick(out["logp"]), want)
    assert model.loss(out, pick, want) > alone
    model.cfg.side_loss_weight = 0.0
    assert torch.isclose(model.loss(out, pick, want), alone)


def test_every_recall_answer_follows_from_its_text():
    rng = np.random.default_rng(0)
    for n in (1, 4, 8, 24, 40):
        for _ in range(200):
            text, meta, values = R.make_episode(n, int(rng.integers(0, 12)), rng)
            bound = dict(re.findall(r"([a-p]{2})=(\d);", text.split("?")[0]))
            assert len(bound) == n == len(values)
            for key, given in re.findall(r"\?([a-p]{2})>(\d);", text):
                assert bound[key] == given
            for pos, ans, kind in meta:
                assert text[pos] == ">" and text[pos + 1] == chr(ans) and kind == R.KIND


def test_the_four_models_are_the_same_size_within_two_percent():
    import hybrid_prototype as P
    sizes = [sum(p.numel() for p in BicameralHybrid(c).parameters()) for c in P.MODELS.values()]
    assert max(sizes) / min(sizes) < 1.02, sizes


def test_the_prototype_script_runs_every_model(tmp_path):
    import hybrid_prototype as P
    for name in P.MODELS:
        results = P.run(name, seed=0, steps=2, eval_batches=1, save=tmp_path)
        assert set(results) == set(P.TESTS)
    assert len(list(tmp_path.glob("*.npz"))) == len(P.MODELS)
