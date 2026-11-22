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

**Training.** On the answers only: the character after each `?x>`,
predicted at the `>`. Filler of 2 to 10 characters. Adam, learning rate
0.003, batches of 32, 1,500 steps, on a CPU.

**Test.** 512 new episodes at the trained length (about 52 characters in
all), and with 24 and 74 characters of each filler (about 100 and 200).

**Run.** `python experiments/answers_only.py MODEL` (two seeds, a few
minutes each).

### Results, 2026-11-22 (seeds 0 and 1)

| Model | Parameters | ~52 characters | ~100 characters | ~200 characters |
|---|---|---|---|---|
| `tri_lm`: Gemini's prototype, unchanged | 18,358 | 100.0 / 100.0% | 100.0 / 100.0% | 100.0 / 100.0% |
| `trinet`: `TriNetForCausalLM` as in this repository | 33,748 | 100.0 / 100.0% | 100.0 / 100.0% | 100.0 / 100.0% |
| `trinet-gemini`: the same, with Gemini's initial values | 33,748 | 92.5 / 81.3% | 90.3 / 79.0% | 91.1 / 78.6% |
| `transformer`: a small causal Transformer, ours | 39,322 | 69.7 / 69.0% | 56.0 / 57.6% | 56.9 / 51.3% |

The script prints each kind of query separately, and for the overwrite
queries how often the answer is the old value.

- **The Transformer fails the overwrites on longer texts.** At about 100
  and 200 characters it gives the new value 8.6 to 13.5% of the time,
  below chance (one in six, 16.7%). It does not keep to the old value
  either: it gives that 15.4 to 19.3% of the time, about chance. Gemini
  reported 13.5% on overwrites for theirs.
- **`trinet-gemini` is held back on every kind of query,** not on one: 75
  to 91% direct, 79 to 97% overwrite, 79 to 90% two hops.

### What this shows

- **Gemini's prototype does what they said it would.** It learns all
  three kinds of query, and loses nothing at four times the length it was
  trained on. Its memory is the same size at any length.
- **It learns about as fast as they reported.** At 450 steps
  (`--steps 450`) it has 100.0 and 99.9% at the trained length, and at
  250 steps 100.0 and 99.4%. Gemini reports 100% by step 250, but their training
  differs: they trained on every next character ("purely from
  next-character prediction", conversation line 37466) with 22 characters,
  learning rate and batch size not given; here it trains on the answers
  only, with 25 characters and padding, at the settings above.
- **The packaged model learns it completely once it starts from PyTorch's
  own initial values.**
  - Gemini's `_init_weights` drew every weight from N(0, 0.02), as Hugging
    Face models usually do. With those values, the model reaches 78.6 to 92.5%.
  - With PyTorch's values, it reaches 100%. That is change 5 in
    `provenance/MANIFEST.json`.
- **The Transformer here is a weak comparison.**
  - It is ours, untuned, and twice the prototype's size.
  - It learns its positions, so beyond the trained length it meets
    positions it never saw. That is part of why it does worse on longer
    texts.
  - A fair comparison needs a tuned Transformer of the same size,
    including one without learned positions. Gemini ran one; we have not.

### Corrected after review

Until the independent review of #2, the script read each answer at the
query's name, one character before the `>`. All four models were trained
and tested that way, so the comparison held, but two numbers were lower
than they are now: `trinet-gemini` reached about 70%, and `tri_lm` at 450
steps 70.7 and 74.7%, which we took to mean it learns more slowly than
Gemini said. The numbers above are from the corrected script. One result
from before was not repeated: letting the gates also see the local
context (they see it only through the three-colour receptors) changed
nothing then (69 to 71% with Gemini's initial values, two seeds; not kept
here).

## With `train_trinet.py` as written

The training script trains on every next character, with half the
batches from the task above and half from a paragraph of prose, at the
smallest preset (59,708 parameters, against a Transformer of 71,648):

```
python train_trinet.py --preset smoke --smoke-test --seq-len 96 --steps 2000 \
  --warmup-steps 100 --reasoning-mix 0.5 --log-every 500 --save-every 0 --seed SEED
```

Its own test of the task, with 6 / 24 characters of filler (2026-11-22;
chance is 16.7%):

| | seed 0 | seed 1 | seed 42 |
|---|---|---|---|
| TriNet, Gemini's initial values (before change 5) | 17.4 / 15.1% | 16.7 / 15.6% | 20.3 / 14.6% |
| TriNet, PyTorch's initial values (change 5) | 99.7 / 100.0% | 19.3 / 17.4% | 66.9 / 67.4% |
| Transformer beside it, before change 5 | 50.8 / 37.5% | 42.7 / 20.3% | 48.4 / 18.0% |
| Transformer beside it, with change 5 | 36.5 / 25.8% | 58.6 / 53.4% | 18.0 / 18.8% |

- **Change 5 helps here too, but not every time.** With it, TriNet learned
  the task in two runs of three, once completely; with Gemini's values,
  in none.
- **Runs differ widely from seed to seed,** so three seeds are a start,
  not a measure. An earlier form of change 5 in this pull request, which
  drew the same kind of values in a different order, gave 32.8 and 34.4%
  at seed 42.
- **The Transformer's runs differ before and after change 5** because the
  TriNet built before it draws a different amount from the random number
  generator.
- Seeds 0 and 1 ran on 2 threads (`OMP_NUM_THREADS=2`), seed 42 on 4
  (PyTorch's default on this machine).
