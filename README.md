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
from line 35595, and Gemini wrote the code that is here.

What it is for, in the founder's words (line 37647): "to produce a model
and share on Hugging Face with all other models that are being developed
and shared on the site. Let's first develop this architecture into
something that does what we want, and then process the model into a
package format that meets the requirements of Hugging Face".

## State

**The code runs; there is no trained model yet.** The package loads and
saves through Hugging Face's `AutoModelForCausalLM`, generates text one
token at a time with a memory of fixed size, and trains on a CPU. It has
been trained here only in small test runs (the smallest preset, up to
2,000 steps, on CPU), and nothing has been published on Hugging Face.

## What is here

| File | What it is |
|---|---|
| `trinet/configuration_trinet.py`, `trinet/modeling_trinet.py` | The model, as a Hugging Face custom architecture (`TriNetForCausalLM`), with its fixed-size cache (`TriNetCache`) |
| `train_trinet.py` | Byte-level training, beside a Transformer of the same size, on TinyStories, a local text, or an offline sample; with a mix of in-context tasks (bind a value, overwrite it, follow a pointer) |
| `verify_hf_roundtrip.py` | Gemini's check: save, load through `AutoModelForCausalLM`, and generate with and without the cache |
| `app.py`, `requirements.txt` | A Gradio app that shows the model's yellow gates and fast memory, token by token |
| `provenance/` | Where each of Gemini's files was taken from, the changes made here, and Gemini's draft model card |
| `tests/` | Ours: the checks below |

## Run and test

```sh
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install transformers safetensors numpy pytest
python -m pytest -q                      # 15 tests
python verify_hf_roundtrip.py            # Gemini's own check
python train_trinet.py --preset smoke --smoke-test --seq-len 72 --steps 100
```

CI runs the tests and Gemini's check with transformers 4.46.3 and 5.24.0.

## What has been checked, and what has not

Checked here, on 2026-11-22:

- **Four small changes were needed** for the model to work with the real
  transformers library: Gemini's sandbox did not have it, and Gemini tested
  against a stand-in of their own (they said so, line 41318). Three made it
  run at all; the fourth, found by this repository's first review, made
  `generate()` on transformers 5.x use the cache instead of reading the
  whole text again at every step. Each change is listed with its reason in
  `provenance/MANIFEST.json`; a test undoes them and checks that the rest
  is, byte for byte, what Gemini gave.
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

Not checked: Gemini reports many results from runs in their own sandbox
(TriNet at 100% on tasks where Transformers and GRUs score 30 to 70%). The
code for most of those runs is not in the conversation, so they cannot be
repeated here yet, and none of them is claimed here. Gemini's draft model
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
