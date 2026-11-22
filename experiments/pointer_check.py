"""Ours, not Gemini's. Does a model trained on Gemini's task as given follow
the pointer, or answer with the first value bound?

In Gemini's make_reasoning_example the pointer always named the first
variable bound, and the overwrite always fell on the second (change 6 in
provenance/MANIFEST.json draws both at random). This script trains a model
as answers_only.py does, on Gemini's task as given, then tests it on that
task and on the task with change 6. It scores the pointer queries, and the
first query of each episode alone, which no earlier answer in the text can
help with.

    python experiments/pointer_check.py MODEL [--steps 1500] [--seeds 0]

Results of 2026-11-22 are in experiments/README.md.
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "experiments"))

import answers_only as A  # noqa: E402


def as_given():
    """Gemini's make_reasoning_example as the conversation gives it: this
    repository's train_trinet.py with change 6 undone, as the provenance
    test undoes it."""
    manifest = json.loads((ROOT / "provenance" / "MANIFEST.json").read_text())
    [change] = [c for c in manifest["changes"] if c["file"] == "train_trinet.py"]
    source = (ROOT / "train_trinet.py").read_text()
    assert source.count(change["new"]) == change["count"]
    namespace = {"__name__": "train_trinet_as_given"}
    exec(compile(source.replace(change["new"], change["old"]), "train_trinet.py (as given)", "exec"), namespace)
    return namespace["make_reasoning_example"]


TASKS = {"as given": as_given(), "change 6": A.train_trinet.make_reasoning_example}


def using(task):
    A.train_trinet.make_reasoning_example = TASKS[task]


def run(name, seed, steps, train_task="as given"):
    using(train_task)
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    model, logits_of, task = A.build(name)
    opt = torch.optim.Adam(model.parameters(), lr=3e-3)
    start = time.time()
    for _ in range(steps):
        x, metas = task.batch(32, 2, 10, rng)
        got, want, _, _ = A.answers(logits_of(x), metas)
        loss = F.cross_entropy(got, want)
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
    print(f"{name} seed {seed}, {steps} steps on the task {train_task} ({time.time() - start:.0f} s)", flush=True)
    results = {}
    with torch.no_grad():
        for test_task in TASKS:
            using(test_task)
            for label, (lo, hi) in A.TESTS.items():
                erng = np.random.default_rng(1000 + lo)
                right, kinds, first = [], [], []
                for _ in range(8):
                    x, metas = task.batch(64, lo, hi, erng)
                    got, want, kind, _ = A.answers(logits_of(x), metas)
                    right.append((got.argmax(-1) == want).numpy())
                    kinds.append(kind)
                    first.append(np.array([i == 0 for meta in metas for i in range(len(meta))]))
                right, kinds, first = map(np.concatenate, (right, kinds, first))
                hops, alone = right[kinds == 3].mean(), right[(kinds == 3) & first].mean()
                results[test_task, label] = 100 * hops
                print(f"    tested on the task {test_task:9s} {label:16s} pointer queries {100 * hops:5.1f}%"
                      f"  (first query of the episode: {100 * alone:5.1f}%, all kinds {100 * right[first].mean():5.1f}%)",
                      flush=True)
    using("change 6")
    return results


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("model", choices=["tri_lm", "trinet", "trinet-gemini", "transformer"])
    p.add_argument("--steps", type=int, default=1500)
    p.add_argument("--seeds", type=int, nargs="+", default=[0])
    p.add_argument("--train-task", choices=list(TASKS), default="as given")
    p.add_argument("--threads", type=int, default=2)
    args = p.parse_args(argv)
    torch.set_num_threads(args.threads)
    return [run(args.model, seed, args.steps, args.train_task) for seed in args.seeds]


if __name__ == "__main__":
    main()
