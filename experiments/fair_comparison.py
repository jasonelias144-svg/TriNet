"""Ours. The fair comparison of STATUS.md, Next 1, on the harder task of
harder_task.py, trained and scored exactly as in stage_a.py (answers
only, Adam, 1,500 steps of 32 episodes, three test lengths).

Models, each about the size of one TriNet of answers_only.SMALL (33,748
parameters), unless said:
  trinet                 one TriNet, as in stage_a.py
  trinet-no-bus          ... with the parts of ablations.py taken out,
  trinet-no-surprise         one at a time
  trinet-fixed-gates
  trinet-no-shunt
  trinet-no-fast
  trinet-no-delay        ... built with delay_lags=1 (6,144 parameters fewer:
                         the delay line's three earlier taps, in both layers)
  gated-delta            a plain gated delta-rule memory (gated_deltanet.py)
  transformer            answers_only.py's small Transformer, learned positions
  transformer-nope       the same without positions (causal attention alone
                         can tell order)
  transformer-sized      without positions, width 36 and feed-forward 144:
                         the size of one TriNet (34,011 parameters)

    python experiments/fair_comparison.py MODEL [--seeds 0 1 2 3 4] [--lr 3e-3] [--steps 1500] [--threads 2]

The learning rate and the number of steps are options so that the
Transformer can be tuned: its best setting is the one compared.
"""
import argparse
import sys
from pathlib import Path

import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import ablations  # noqa: E402
import answers_only as A  # noqa: E402
import stage_a as S  # noqa: E402
from gated_deltanet import GatedDeltaLM  # noqa: E402
from trinet.configuration_trinet import TriNetConfig  # noqa: E402
from trinet.modeling_trinet import TriNetForCausalLM  # noqa: E402

MODELS = ["trinet", *ablations.ABLATIONS, "trinet-no-delay", "gated-delta",
          "transformer", "transformer-nope", "transformer-sized"]


class NoPositions(nn.Module):
    """answers_only.SmallTransformer with its position table unused."""

    def __init__(self, inner):
        super().__init__()
        self.inner = inner
        del inner.pos

    def forward(self, x):
        t = x.shape[1]
        mask = torch.triu(torch.full((t, t), float("-inf")), 1)
        return self.inner.head(self.inner.enc(self.inner.emb(x), mask=mask))


def build(name, task="harder"):
    """A model, a function from token ids to logits, and the task."""
    if name not in MODELS:
        raise SystemExit(f"unknown model {name!r}")
    if name.startswith("trinet"):
        cfg = dict(A.SMALL)
        if name == "trinet-no-delay":
            cfg["delay_lags"] = 1
        model = TriNetForCausalLM(TriNetConfig(**cfg))
        if name in ablations.ABLATIONS:
            ablations.ABLATIONS[name](model)
        return model, (lambda x: model(x, use_cache=False).logits), S.HarderTask(bytes_as_tokens=True)
    if name == "gated-delta":
        model = GatedDeltaLM()
        return model, model, S.HarderTask(bytes_as_tokens=True)
    if name == "transformer-sized":
        model = NoPositions(A.SmallTransformer(len(S.CHARS) + 1, d=36, layers=2, heads=4, ff=144))
    else:
        model = A.SmallTransformer(len(S.CHARS) + 1)
    if name == "transformer-nope":
        model = NoPositions(model)
    return model, model, S.HarderTask(bytes_as_tokens=False)


def trained_parameters(name):
    model = build(name)[0]
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("model", choices=MODELS)
    p.add_argument("--steps", type=int, default=1500)
    p.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    p.add_argument("--lr", type=float, default=3e-3)
    p.add_argument("--threads", type=int, default=2)
    p.add_argument("--eval-batches", type=int, default=8)
    p.add_argument("--curve-every", type=int, default=250)
    args = p.parse_args(argv)
    torch.set_num_threads(args.threads)
    print(f"{args.model}: {trained_parameters(args.model):,} parameters trained, learning rate {args.lr}",
          flush=True)
    return [S.run(args.model, "harder", seed, args.steps, args.eval_batches, args.curve_every,
                  lr=args.lr, builder=build) for seed in args.seeds]


if __name__ == "__main__":
    main()
