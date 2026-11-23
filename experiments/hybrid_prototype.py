"""Ours. Is a TriNet + Transformer pair, with an executive between them,
worth developing? (hybrid/model.py)

Four models of about the same size are trained on the answers only, on a
mix of two tasks, half of each batch from each:
  track   harder_task.py: values that change, pointers, the value before the
          last change; TriNet's strength
  recall  recall_task.py: up to 24 keys bound once, then asked; what
          attention should do better than a memory of fixed size

Models:
  trinet        the TriNet side alone
  transformer   the Transformer side alone
  hybrid-fixed  both sides, each trusted half (alpha = 0.5)
  hybrid        both sides and the executive

The test that decides whether to develop it further, set before the runs:
the hybrid must beat both sides alone, by more than the spread between
seeds, and the executive must beat the fixed mix. `--save DIR` keeps every
question's result, so oracle.py can say how much any router could gain.

    python experiments/hybrid_prototype.py MODEL [--steps 1500] [--seeds 0 1 2] [--save DIR]
"""
import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import answers_only as A  # noqa: E402
import harder_task as H  # noqa: E402
import recall_task as R  # noqa: E402
from hybrid.model import BicameralHybrid, HybridConfig  # noqa: E402

# Sizes chosen so that every model has 44,200 to 44,800 parameters (printed with each run).
MODELS = {
    "trinet": HybridConfig(sides="trinet", tri_d=40, tri_key=20),
    "transformer": HybridConfig(sides="transformer", tf_d=40, tf_ff=68),
    "hybrid-fixed": HybridConfig(mix="fixed"),
    "hybrid": HybridConfig(),
}
TESTS = {  # name: (task, filler lo, filler hi, keys)
    "track, ~125 characters": ("track", 2, 10, None),
    "track, ~160 characters": ("track", 24, 24, None),
    "track, ~260 characters": ("track", 74, 74, None),
    "recall, 8 keys": ("recall", 2, 10, 8),
    "recall, 16 keys": ("recall", 2, 10, 16),
    "recall, 24 keys": ("recall", 2, 10, 24),
}
KINDS = {**H.KINDS, **R.KINDS}


def episode(task, lo, hi, keys, rng):
    if task == "track":
        text, meta, _ = H.make_episode(int(rng.integers(lo, hi + 1)), int(rng.integers(lo, hi + 1)), rng)
    else:
        text, meta, _ = R.make_episode(keys if keys else int(rng.integers(4, 25)), int(rng.integers(lo, hi + 1)), rng)
    return text, meta


def batch(episodes):
    """Byte ids, padded with 0, and (position, answer, kind, 0) per query, as answers_only.answers takes."""
    x = np.zeros((len(episodes), max(len(t) for t, _ in episodes)), dtype=np.int64)
    metas = []
    for i, (text, meta) in enumerate(episodes):
        x[i, : len(text)] = [ord(c) for c in text]
        metas.append([(pos, ans, kind, 0) for pos, ans, kind in meta])
    return torch.from_numpy(x), metas


def run(name, seed, steps, eval_batches=4, save=None):
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    model = BicameralHybrid(MODELS[name])
    opt = torch.optim.Adam(model.parameters(), lr=3e-3)
    start = time.time()
    for _ in range(steps):
        x, metas = batch([episode("track", 2, 10, None, rng) for _ in range(16)]
                         + [episode("recall", 2, 10, None, rng) for _ in range(16)])
        out = model(x)
        rows = [b for b, m in enumerate(metas) for _ in m]
        cols = [p for m in metas for p, _, _, _ in m]
        want = torch.tensor([a for m in metas for _, a, _, _ in m])
        loss = model.loss(out, lambda t: t[rows, cols], want)
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
    print(f"{name}, seed {seed}, {steps} steps, {model.describe()} ({time.time() - start:.0f} s)", flush=True)
    results, kept = {}, {}
    with torch.no_grad():
        for label, (task, lo, hi, keys) in TESTS.items():
            erng = np.random.default_rng(2000 + lo + (keys or 0))
            right, kinds, alphas = [], [], []
            for _ in range(eval_batches):
                x, metas = batch([episode(task, lo, hi, keys, erng) for _ in range(64)])
                out = model(x)
                got, want, kind, _ = A.answers(out["logp"], metas)
                right.append((got.argmax(-1) == want).numpy())
                kinds.append(kind)
                if "alpha" in out:
                    alphas.append(A.answers(out["alpha"], metas)[0].squeeze(-1).numpy())
            right, kinds = np.concatenate(right), np.concatenate(kinds)
            results[label] = 100 * right.mean()
            kept[label] = right
            per = ", ".join(f"{KINDS[k]} {100 * right[kinds == k].mean():5.1f}" for k in sorted(set(kinds)))
            trust = ""
            if alphas:
                a = np.concatenate(alphas)
                trust = "  [trust in TriNet: " + ", ".join(
                    f"{KINDS[k]} {a[kinds == k].mean():.2f}" for k in sorted(set(kinds))) + "]"
            print(f"    {label:24s} {100 * right.mean():5.1f}%  ({per}){trust}", flush=True)
    if save:
        Path(save).mkdir(parents=True, exist_ok=True)
        np.savez(Path(save) / f"{name}-seed{seed}.npz", **{k.replace(", ", "_").replace(" ", "-"): v
                                                           for k, v in kept.items()})
    return results


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("model", choices=list(MODELS))
    p.add_argument("--steps", type=int, default=1500)
    p.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    p.add_argument("--threads", type=int, default=2)
    p.add_argument("--eval-batches", type=int, default=4, help="batches of 64 episodes per test")
    p.add_argument("--save", help="directory for every question's result, for oracle.py")
    args = p.parse_args(argv)
    torch.set_num_threads(args.threads)
    return [run(args.model, s, args.steps, args.eval_batches, args.save) for s in args.seeds]


if __name__ == "__main__":
    main()
