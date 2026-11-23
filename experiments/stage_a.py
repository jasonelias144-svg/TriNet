"""Ours, not Gemini's. Stage A: does a model of two TriNet stacks
(bicameral/bicameral_stage_a.py, Gemini's) do better than one TriNet?

Each model is trained on the answers only, as in answers_only.py, on one
of two tasks:
  answers   the task of answers_only.py (train_trinet.make_reasoning_example)
  harder    the task of harder_task.py: more names, pointers that copy a
            value and can chain, several overwrites, and queries for the
            value before the last overwrite

Models, all of about the same size (33,000 to 34,200 parameters) except
the Transformer:
  trinet           one TriNet (TriNetForCausalLM), as in answers_only.py
  trinet-residual  the same, with residual_error=True (change 7): the fast
                   memory stores adjustments to W_base
  bicam-sym        two identical stacks, joined by the callosum
  bicam-asym       the same, but stack A keeps its memory longer
                   (retention floor 0.95 against 0.5)
  bicam-nocal      two identical stacks with no callosum, joined only at
                   the output
  transformer      answers_only.py's small Transformer

    python experiments/stage_a.py MODEL [--task harder] [--steps 1500] [--seeds 0 1 2]

Results of 2026-11-22 are in experiments/README.md.
"""
import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import answers_only as A  # noqa: E402
import harder_task as H  # noqa: E402
from bicameral.bicameral_stage_a import BicameralTriNetConfig, BicameralTriNetForCausalLM  # noqa: E402
from trinet.configuration_trinet import TriNetConfig  # noqa: E402
from trinet.modeling_trinet import TriNetForCausalLM  # noqa: E402

CHARS = sorted(set(H.VARS + H.VALS + H.FILLER + list("=;?><")))
# Two stacks of width 20 with keys and values of 12 come within 2% of one
# TriNet of answers_only.SMALL (33,748 parameters).
HALF = dict(vocab_size=256, d_model=32, d_hemi=20, d_key=12, d_val=12, n_layers=2, delay_lags=4,
            n_receptors=8, callosum_dim=8, hemisphere_dropout=0.0)
BICAMERAL = {"bicam-sym": {}, "bicam-asym": {"symmetric": False}, "bicam-nocal": {"callosum": False}}
MODELS = ["trinet", "trinet-residual", *BICAMERAL, "transformer"]
TASKS = ["answers", "harder"]
# Filler lengths as in answers_only.TESTS; the harder task's episodes are longer.
TESTS = {"answers": A.TESTS,
         "harder": {"~125 characters": (2, 10), "~160 characters": (24, 24), "~260 characters": (74, 74)}}


class HarderTask:
    """Episodes of harder_task.py as token ids. For each query: where it is,
    the answer, its kind, and the value a confused model would give: the
    value before the last overwrite for a current-value query, and the
    current value for a before-the-last-overwrite query."""

    def __init__(self, bytes_as_tokens):
        self.ids = (lambda c: ord(c)) if bytes_as_tokens else (lambda c: CHARS.index(c) + 1)

    def batch(self, size, lo, hi, rng):
        seqs, metas = [], []
        for _ in range(size):
            text, meta, rec = H.make_episode(int(rng.integers(lo, hi + 1)), int(rng.integers(lo, hi + 1)), rng)
            seqs.append([self.ids(c) for c in text])
            rows = []
            for pos, ans, kind in meta:
                hist = rec["history"].get(text[pos - 1], [])
                other = hist[-2] if kind == 2 else hist[-1] if kind == 4 else " "
                rows.append((pos, self.ids(chr(ans)), kind, self.ids(other)))
            metas.append(rows)
        x = np.zeros((size, max(map(len, seqs))), dtype=np.int64)
        for i, s in enumerate(seqs):
            x[i, : len(s)] = s
        return torch.from_numpy(x), metas


def build(name, task):
    """A model, a function from token ids to logits, and the task."""
    bytes_as_tokens = name != "transformer"
    data = HarderTask(bytes_as_tokens) if task == "harder" else A.CharTask(bytes_as_tokens)
    if name in ("trinet", "trinet-residual"):
        model = TriNetForCausalLM(TriNetConfig(**A.SMALL, residual_error=(name == "trinet-residual")))
    elif name in BICAMERAL:
        model = BicameralTriNetForCausalLM(BicameralTriNetConfig(**{**HALF, **BICAMERAL[name]}))
    elif name == "transformer":
        model = A.SmallTransformer(len(CHARS if task == "harder" else A.CHARS) + 1)
        return model, model, data
    else:
        raise SystemExit(f"unknown model {name!r}")
    return model, (lambda x: model(x, use_cache=False).logits), data


def curve_point(logits_of, data, step):
    """Accuracy at the training length on 128 fixed episodes, for the
    learning curve. Its own random generator, so the training is the same
    with or without it."""
    with torch.no_grad():
        x, metas = data.batch(128, 2, 10, np.random.default_rng(999))
        got, want, _, _ = A.answers(logits_of(x), metas)
    return f"{step}: {100 * (got.argmax(-1) == want).float().mean():.1f}"


def run(name, task, seed, steps, eval_batches=8, curve_every=0):
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    model, logits_of, data = build(name, task)
    opt = torch.optim.Adam(model.parameters(), lr=3e-3)
    start, curve = time.time(), []
    for step in range(1, steps + 1):
        x, metas = data.batch(32, 2, 10, rng)
        got, want, _, _ = A.answers(logits_of(x), metas)
        loss = F.cross_entropy(got, want)
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        if curve_every and step % curve_every == 0:
            curve.append(curve_point(logits_of, data, step))
    params = sum(p.numel() for p in model.parameters())
    print(f"{name} on {task}, seed {seed}, {steps} steps, {params:,} parameters ({time.time() - start:.0f} s)",
          flush=True)
    if curve:
        print("    learning curve, % right at the training length, by step: " + "; ".join(curve), flush=True)
    kinds = H.KINDS if task == "harder" else {1: "direct", 2: "overwrite", 3: "two hops"}
    results = {}
    with torch.no_grad():
        for label, (lo, hi) in TESTS[task].items():
            erng = np.random.default_rng(1000 + lo)
            right, confused, kind = [], [], []
            for _ in range(eval_batches):
                x, metas = data.batch(64, lo, hi, erng)
                got, want, k, other = A.answers(logits_of(x), metas)
                right.append((got.argmax(-1) == want).numpy())
                confused.append((got.argmax(-1) == other).numpy())
                kind.append(k)
            right, confused, kind = np.concatenate(right), np.concatenate(confused), np.concatenate(kind)
            results[label] = 100 * right.mean()
            per = ", ".join(f"{kinds[k]} {100 * right[kind == k].mean():5.1f}" for k in sorted(kinds))
            mixed = ", ".join(f"{kinds[k]} {100 * confused[kind == k].mean():4.1f}" for k in (2, 4) if k in kinds)
            print(f"    {label:16s} all {100 * right.mean():5.1f}%  ({per})  [other value: {mixed}]", flush=True)
    return results


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("model", choices=MODELS)
    p.add_argument("--task", choices=TASKS, default="harder")
    p.add_argument("--steps", type=int, default=1500)
    p.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    p.add_argument("--threads", type=int, default=2)
    p.add_argument("--eval-batches", type=int, default=8, help="batches of 64 episodes per test length")
    p.add_argument("--curve-every", type=int, default=250, help="steps between learning-curve points (0: none)")
    args = p.parse_args(argv)
    torch.set_num_threads(args.threads)
    return [run(args.model, args.task, seed, args.steps, args.eval_batches, args.curve_every)
            for seed in args.seeds]


if __name__ == "__main__":
    main()
