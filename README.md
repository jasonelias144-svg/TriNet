# TriNet

A small language model built from a different kind of artificial neuron,
packaged so that anyone can load it from Hugging Face.

The idea is the founder's. A neuron in most artificial networks sends one
number. A real neuron sends several chemical signals, which do different
things. TriNet's unit sends three, named by colour: **red** drives the next
unit, **blue** damps it by division (shunting), and **yellow** modulates
it: it decides what is written to a fast, temporary memory and how quickly
that memory dissolves back to its resting state. The founder worked the
idea out with Gemini in the conversation
[`012-08-04-GMNI-Patterns_Multiverses.txt`](https://github.com/jasonelias144-svg/open-worlds/blob/main/sketches/records/012-08-04-GMNI-Patterns_Multiverses.txt),
from line 35595, and Gemini wrote the code that is here. ChatGPT reviewed
the conversation and this repository, in their own conversation with the
founder. Who wrote what: [`CONTRIBUTIONS.md`](CONTRIBUTIONS.md).

What it is for, in the founder's words (line 37647): "to produce a model
and share on Hugging Face with all other models that are being developed
and shared on the site. Let's first develop this architecture into
something that does what we want, and then process the model into a
package format that meets the requirements of Hugging Face".

## State

**The code runs; there is no trained model yet.** The package loads and
saves through Hugging Face's `AutoModelForCausalLM`, generates text one
token at a time with a memory of fixed size, and trains on a CPU. It has
been trained here only in small test runs on a CPU, and nothing has been
published on Hugging Face.

**What it was built for, it can learn, when trained on the answers.** On a
task of binding values, overwriting one and following a pointer, trained on
the answers alone, the model reaches 100% and keeps that at four times the
length it was trained on, with a memory of fixed size
(`experiments/README.md`). Two changes made that true and measurable:
- **Change 5:** starting from PyTorch's own initial values instead of the
  small ones Gemini's code drew.
- **Change 6:** a task in which the pointer names a random variable. In
  Gemini's task it always named the first, and the models learned that
  shortcut instead of following the pointer.

Trained on every next character, as `train_trinet.py` trains, it learns the
task only in part so far (34% at best, one seed of three), and the small
Transformer beside it does better in most runs.

## What is here

| File | What it is |
|---|---|
| `trinet/configuration_trinet.py`, `trinet/modeling_trinet.py` | The model, as a Hugging Face custom architecture (`TriNetForCausalLM`), with its fixed-size cache (`TriNetCache`) |
| `train_trinet.py` | Byte-level training, beside a Transformer of the same size, on TinyStories, a local text, or an offline sample; with a mix of in-context tasks (bind a value, overwrite it, follow a pointer) |
| `verify_hf_roundtrip.py` | Gemini's check: save, load through `AutoModelForCausalLM`, and generate with and without the cache |
| `app.py`, `requirements.txt` | A Gradio app that shows the model's yellow gates and fast memory, token by token |
| `prototypes/` | Gemini's earlier prototypes, kept as given, for checking what they reported ([README](prototypes/README.md)) |
| `bicameral/` | Gemini's BicameralTriNet for Stage A: two TriNet stacks, with a narrow "callosum" between them, kept as given ([experiments/README.md](experiments/README.md), "Stage A") |
| `experiments/` | Ours: experiments that anyone can run again, with their results ([README](experiments/README.md)) |
| `docs/` | Ours: [a catalog of neuron types](docs/NEURON-CATALOG.md), toward one general cell whose dials can set it to act like any of them |
| `provenance/` | Where each of Gemini's files was taken from, the changes made here, and Gemini's draft model card |
| `tests/` | Ours: the checks below |
| `CONTRIBUTIONS.md` | Who wrote what, and where the quotations come from |

## Run and test

```sh
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install transformers safetensors numpy pytest
python -m pytest -q                      # 46 tests, under a minute
python verify_hf_roundtrip.py            # Gemini's own check
python train_trinet.py --preset smoke --smoke-test --seq-len 72 --steps 100
python experiments/answers_only.py trinet  # minutes; see experiments/README.md
```

CI runs the tests and Gemini's check with transformers 4.46.3 and 5.24.0.

## What has been checked, and what has not

Checked here, on 2026-11-22:

- **Seven changes have been made to Gemini's code.** Each is listed with
  its reason in `provenance/MANIFEST.json`; a test undoes them and checks
  that the rest is, byte for byte, what Gemini gave.
  - Changes 1 to 4 make the model work with the real transformers library:
    Gemini's sandbox did not have it, and Gemini tested against a stand-in
    of their own (they said so, line 41318). Three made it run at all; the
    fourth, found by this repository's first review, made `generate()` on
    transformers 5.x use the cache instead of reading the whole text again
    at every step.
  - Change 5 lets the model learn: it starts from PyTorch's own initial
    values.
  - Change 6 removes a shortcut from the training task.
  - Change 7 is a switch, off by default, for the founder's two classes of
    weights: with `residual_error=True` the fast memory stores adjustments
    to the lasting weights `W_base`, not whole associations.
- **Stepping token by token with the cache gives what one full pass
  gives**, each of the cache's three states matters, `generate()` feeds the
  model one new token at a time, and the cache stays the same size, however
  long the text (`tests/test_model.py`).
- **The four presets have exactly the sizes in Gemini's table**, for both
  TriNet and the Transformer beside it (`tests/test_training.py`). At the
  smallest preset the Transformer is 20% larger (71,648 parameters against
  59,708); at the others they are within 0.2%.
- **Training runs**, and its checkpoint loads again.
- **The app** draws its three figures from the untrained model (by hand;
  it is not in CI).
- **The model learns its in-context task** when trained on the answers, as
  above, and so does Gemini's prototype, at about the pace they reported, on
  a different set-up (`experiments/README.md`). A few smaller things Gemini said of their
  prototypes hold too (`prototypes/README.md`).

Not checked: most of Gemini's reported results from runs in their own
sandbox (comparisons with GRUs and tuned Transformers, the "crucible" of
the micro-brain). The code for them is not in the conversation. Our own
Transformer for comparison is small and untuned, so the comparison in
`experiments/README.md` shows that TriNet learns the task, not that it
beats a good Transformer. Gemini's draft model
card (`provenance/model-card-draft.md`) repeats them; it is kept as Gemini
wrote it and is not this repository's model card.

## How it relates to known work

The layer in `modeling_trinet.py` keeps a fast-weight matrix that it
updates by a variant of the gated delta rule: each step it decays the
matrix (yellow's retention gate) and writes the difference between the new
value and what the memory predicted before the decay (yellow's write gate,
which can reach 1.5; the matrix is clipped to ±6). That is the family of
fast-weight and linear-attention models:

- Ba et al., *Using Fast Weights to Attend to the Recent Past* (2016),
  [arXiv:1610.06258](https://arxiv.org/abs/1610.06258)
- Katharopoulos et al., *Transformers are RNNs* (2020),
  [arXiv:2006.16236](https://arxiv.org/abs/2006.16236)
- Schlag, Irie and Schmidhuber, *Linear Transformers Are Secretly Fast
  Weight Programmers* (2021), [arXiv:2102.11174](https://arxiv.org/abs/2102.11174)
- Yang et al., *Parallelizing Linear Transformers with the Delta Rule over
  Sequence Length* (2024), [arXiv:2406.06484](https://arxiv.org/abs/2406.06484)
- Yang, Kautz and Hatamizadeh, *Gated Delta Networks* (2024),
  [arXiv:2412.06464](https://arxiv.org/abs/2412.06464)

Changing TriNet's settings does not turn it into linear attention. ChatGPT's
review lists why (record 012-09-05, lines 16368 to 16372):

> * Its memory write always uses a Delta error: it subtracts the value already predicted by the memory before writing.
> * Its Blue normalization depends on the retrieved values.
> * Its cache does not contain the separate key-mass vector z.
> * Its key and query features can be negative.
> * It reads before the current write and clips the resulting fast-weight matrix.

Checked here against `trinet/modeling_trinet.py`: all five hold. The
clipping is at `max_fast_weight` (6.0 by default), which is a setting; the
other four are not.

What TriNet adds to that family is its own way of making the gates: a
"receptor" layer that reads a three-colour signal, a slow yellow state that
carries across steps, divisive (shunting) normalization of what is read,
and a short causal convolution in front (the "delay line"). Whether these
make a better model than the plain family is what training will show. The
conversation's last turns ask whether two TriNets, set differently, could
work side by side like two hemispheres; models that run attention and
recurrent heads side by side already exist, for example *Hymba* (2024),
[arXiv:2411.13676](https://arxiv.org/abs/2411.13676).

## Licence and rules

[CC0 1.0](LICENSE): a public-domain dedication. This repository follows the
[policy handbook](https://github.com/jasonelias144-svg/policy-handbook);
`AGENTS.md` has its rules here.
