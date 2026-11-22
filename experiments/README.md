# Experiments

## Binding, overwriting and following a pointer, on longer texts

**Question.** Can a model hold values given in its text, replace one when it
is given again, follow a pointer from one name to another, and keep doing
so when the text grows longer than any it was trained on? Gemini built
TriNet for this, and reported 100% for their prototype where Transformers
scored much less (conversation lines 37465 to 37639).

**Task.** The one `train_trinet.py` mixes into training
(`make_reasoning_example`). For example, with 6 characters of each
filler (`numpy.random.default_rng(3)`):

```
h=3;a=0;b=0;.dxxz e=h;a=1;,z.-xx?h>3;?e>3;?a>1;?b>0;
```

Three values are bound (`h`, `a` and `b`). Filler follows, which can hold
stray names (`d` here). Then a pointer (`e=h`), an overwrite (`a=1`), more
filler, and four queries in random order: `?e>` must follow the pointer to
`h`, and `?a>` must give the new value. Chance is one in six.

**Training.** On the answers only (the character after each `?x>`), with 2
to 10 characters of each filler. Adam, learning rate 0.003, batches of 32,
1,500 steps, on a CPU.

**Test.** 512 new episodes at the trained length (about 52 characters in
all), and with 24 and 74 characters of each filler (about 100 and 200).

**Run.** `python experiments/answers_only.py MODEL` (two seeds, a few
minutes each).

### Results, 2026-11-22 (seeds 0 and 1)

| Model | Parameters | ~52 characters | ~100 characters | ~200 characters |
|---|---|---|---|---|
| `tri_lm`: Gemini's prototype, unchanged | 18,358 | 100.0 / 100.0% | 100.0 / 100.0% | 99.9 / 100.0% |
| `trinet`: `TriNetForCausalLM` as in this repository | 33,748 | 100.0 / 100.0% | 100.0 / 99.7% | 100.0 / 99.0% |
| `trinet-gemini`: the same, with Gemini's initial values | 33,748 | 71.4 / 71.0% | 69.3 / 69.7% | 69.1 / 68.2% |
| `transformer`: a small causal Transformer, ours | 39,322 | 68.8 / 76.9% | 57.2 / 60.1% | 55.1 / 59.3% |

The script prints each kind of query separately. Two lines stand out:

- **The Transformer breaks on overwrites.** At about 100 and 200
  characters its overwrite queries fall to 10.7 to 13.5%, below chance: it
  answers with the old value. Gemini reported 13.5% for theirs.
- **`trinet-gemini` stalls on overwrites too,** at 50.6 to 59.8%,
  where `trinet` gets 99.8 to 100%.

### What this shows

- **Gemini's prototype does what they said it would.** It learns all
  three kinds of query, and loses nothing at four times the length it was
  trained on. Its memory is the same size at any length.
- **It is slower to learn than Gemini reported.** They report 100% by step
  250. Here, at 450 steps (`--steps 450`), it had 70.7 and 74.7% at the
  trained length.
- **The packaged model learns it too, once it starts from PyTorch's own
  initial values.**
  - Gemini's `_init_weights` drew every weight from N(0, 0.02), as Hugging
    Face models usually do. With those values, the model stalls near 70%
    at every length.
  - With PyTorch's values, it reaches 99 to 100%. That is change 5 in
    `provenance/MANIFEST.json`.
  - Before that was found, we let the gates also see the local context
    (they see it only through the three-colour receptors). That changed
    nothing: 69 to 71%, two seeds, not kept here.
- **The Transformer here is a weak comparison.**
  - It is ours, untuned, and twice the prototype's size.
  - It learns its positions, so beyond the trained length it meets
    positions it never saw. That is part of why it does worse on longer
    texts.
  - A fair comparison needs a tuned Transformer of the same size,
    including one without learned positions. Gemini ran one; we have not.

## With `train_trinet.py` as written

The training script trains on every next character, with half the
batches from the task above and half from a paragraph of prose, at the
smallest preset (59,708 parameters, against a Transformer of 71,648):

```
python train_trinet.py --preset smoke --smoke-test --seq-len 96 --steps 2000 \
  --warmup-steps 100 --reasoning-mix 0.5 --log-every 500 --save-every 0 --seed 42
```

Its own test of the task (one seed, 2026-11-22):

| | 6 characters of filler | 24 characters of filler |
|---|---|---|
| TriNet, Gemini's initial values (before change 5) | 20.3% | 14.6% |
| TriNet, PyTorch's initial values (change 5) | 32.8% | 34.4% |
| Transformer beside it, first run / second run | 48.4% / 38.5% | 18.0% / 16.4% |

The Transformer's two runs differ because the model built before it
draws a different amount from the random number generator in each case.

- **Change 5 helps here too.** It is far from 100%, because in this
  script the answers are a few characters among many to predict, and
  2,000 small steps teach the task only in part.
- **Not repeated:** a second seed of these runs, which did not finish in
  its time limit.
