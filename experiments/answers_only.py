"""Ours, not Gemini's. Can a model learn to bind, overwrite and follow a
pointer in context, and keep doing it on longer texts?

The task is the one train_trinet.py mixes into training
(make_reasoning_example): three values bound, then filler, a pointer to
one of them, an overwrite of another, more filler, and four queries. Each
model is trained on the answers only (the character after each "?x>",
predicted at the ">"),
with filler of 2 to 10 characters, and tested on 512 new episodes at that
length (about 52 characters in all), and with 24 and 74 characters of
filler (about 100 and 200). Runs on a CPU: a few minutes per model and
seed.

    python experiments/answers_only.py MODEL [--steps 1500] [--seeds 0 1]

MODEL is one of:
  tri_lm        Gemini's prototype, prototypes/tri_lm.py, unchanged
  trinet        TriNetForCausalLM as in this repository (bytes as tokens)
  trinet-gemini TriNetForCausalLM with Gemini's initial values: every
                linear, embedding and convolution weight from N(0, 0.02),
                as their _init_weights did before change 5
  transformer   a small causal Transformer, ours, for comparison

Results of 2026-11-22 are in experiments/README.md.
"""
import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "prototypes"))

import train_trinet  # noqa: E402
from trinet.configuration_trinet import TriNetConfig  # noqa: E402
from trinet.modeling_trinet import TriNetForCausalLM  # noqa: E402

CHARS = sorted(set(train_trinet._VARS + train_trinet._VALS + train_trinet._FILLER + list("=;?>")))
SMALL = dict(vocab_size=256, d_model=32, d_key=20, d_val=20, n_layers=2, delay_lags=4, n_receptors=8)
TESTS = {"~52 characters": (2, 10), "~100 characters": (24, 24), "~200 characters": (74, 74)}


class CharTask:
    """Episodes as token ids, with where each answer is and what kind."""

    def __init__(self, bytes_as_tokens):
        self.ids = (lambda c: ord(c)) if bytes_as_tokens else (lambda c: CHARS.index(c) + 1)

    def batch(self, size, lo, hi, rng):
        seqs, metas = [], []
        for _ in range(size):
            text, meta = train_trinet.make_reasoning_example(
                3, int(rng.integers(lo, hi + 1)), int(rng.integers(lo, hi + 1)), rng)
            seqs.append([self.ids(c) for c in text])
            old = {text[i]: text[i + 2] for i in range(0, 12, 4)}  # the first three bindings, "h=3;a=0;b=0;"
            metas.append([(pos, self.ids(chr(ans)), kind, self.ids(old.get(text[pos - 1], " ")))
                          for pos, ans, kind in meta])
        x = np.zeros((size, max(map(len, seqs))), dtype=np.int64)
        for i, s in enumerate(seqs):
            x[i, : len(s)] = s
        return torch.from_numpy(x), metas


def answers(logits, metas):
    """The logits at each ">" of "?x>", which predict the answer after it."""
    rows, cols, targets, kinds, olds = [], [], [], [], []
    for b, meta in enumerate(metas):
        for pos, target, kind, old in meta:
            rows.append(b)
            cols.append(pos)
            targets.append(target)
            kinds.append(kind)
            olds.append(old)
    return logits[rows, cols], torch.tensor(targets), np.array(kinds), torch.tensor(olds)


class SmallTransformer(nn.Module):
    def __init__(self, vocab, d=32, layers=2, heads=4, ff=96, max_len=512):
        super().__init__()
        self.emb, self.pos = nn.Embedding(vocab, d), nn.Embedding(max_len, d)
        layer = nn.TransformerEncoderLayer(d, heads, ff, dropout=0.0, batch_first=True, norm_first=True)
        self.enc, self.head = nn.TransformerEncoder(layer, layers), nn.Linear(d, vocab)

    def forward(self, x):
        t = x.shape[1]
        mask = torch.triu(torch.full((t, t), float("-inf")), 1)
        return self.head(self.enc(self.emb(x) + self.pos(torch.arange(t)), mask=mask))


def gemini_initial_values(model):
    for module in model.modules():
        if isinstance(module, (nn.Linear, nn.Embedding, nn.Conv1d)):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if getattr(module, "bias", None) is not None:
                nn.init.zeros_(module.bias)
    return model


def build(name):
    """A model, a function from token ids to logits, and the task's tokens."""
    if name == "tri_lm":
        import tri_lm
        model = tri_lm.TriLanguageModel(vocab_size=len(CHARS) + 1)
        return model, model, CharTask(bytes_as_tokens=False)
    if name in ("trinet", "trinet-gemini"):
        model = TriNetForCausalLM(TriNetConfig(**SMALL))
        if name == "trinet-gemini":
            gemini_initial_values(model)
            for layer in model.model.layers:  # the gate biases, as _init_weights sets them
                nn.init.constant_(layer.yellow_to_write.bias, 0.25)
                nn.init.constant_(layer.yellow_to_retain.bias, 0.90)
        return model, (lambda x: model(x, use_cache=False).logits), CharTask(bytes_as_tokens=True)
    if name == "transformer":
        model = SmallTransformer(len(CHARS) + 1)
        return model, model, CharTask(bytes_as_tokens=False)
    raise SystemExit(f"unknown model {name!r}")


def run(name, seed, steps, eval_batches=8):
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    model, logits_of, task = build(name)
    opt = torch.optim.Adam(model.parameters(), lr=3e-3)
    start = time.time()
    for _ in range(steps):
        x, metas = task.batch(32, 2, 10, rng)
        got, want, _, _ = answers(logits_of(x), metas)
        loss = F.cross_entropy(got, want)
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
    params = sum(p.numel() for p in model.parameters())
    print(f"{name} seed {seed}, {steps} steps, {params:,} parameters ({time.time() - start:.0f} s)", flush=True)
    results = {}
    with torch.no_grad():
        for label, (lo, hi) in TESTS.items():
            erng = np.random.default_rng(1000 + lo)
            right, stale, kinds = [], [], []
            for _ in range(eval_batches):
                x, metas = task.batch(64, lo, hi, erng)
                got, want, kind, old = answers(logits_of(x), metas)
                right.append((got.argmax(-1) == want).numpy())
                stale.append((got.argmax(-1) == old).numpy())
                kinds.append(kind)
            right, stale, kinds = np.concatenate(right), np.concatenate(stale), np.concatenate(kinds)
            results[label] = 100 * right.mean()
            print(f"    {label:16s} all {100 * right.mean():5.1f}%  (direct {100 * right[kinds == 1].mean():5.1f},"
                  f" overwrite {100 * right[kinds == 2].mean():5.1f} [old value {100 * stale[kinds == 2].mean():4.1f}],"
                  f" two hops {100 * right[kinds == 3].mean():5.1f})", flush=True)
    return results


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("model", choices=["tri_lm", "trinet", "trinet-gemini", "transformer"])
    p.add_argument("--steps", type=int, default=1500)
    p.add_argument("--seeds", type=int, nargs="+", default=[0, 1])
    p.add_argument("--threads", type=int, default=2)
    p.add_argument("--eval-batches", type=int, default=8, help="batches of 64 episodes per test length")
    args = p.parse_args(argv)
    torch.set_num_threads(args.threads)
    return [run(args.model, seed, args.steps, args.eval_batches) for seed in args.seeds]


if __name__ == "__main__":
    main()
