# Experiments

## Binding, overwriting and following a pointer, on longer texts

**Question.** Can a model hold values given in its text, replace one when it
is given again, follow a pointer from one name to another, and keep doing
so when the text grows longer than any it was trained on? Gemini built
TriNet for this, and reported 100% for their prototype where Transformers
scored much less (conversation lines 37465 to 37639).

**Task.** The one `train_trinet.py` mixes into training
(`make_reasoning_example`), with change 6. For example, with 6 characters
of each filler (`numpy.random.default_rng(4)`):

```
d=0;f=2;g=3;.gy - e=f;g=5;.hz.xx?g>5;?e>2;?d>0;?f>2;
```

Three values are bound (`d`, `f` and `g`). Filler follows, which can hold
stray names (`g` here). Then a pointer to one of the bound names, chosen at
random (`e=f`), an overwrite of a different one (`g=5`), more filler, and
four queries in random order: `?e>` must follow the pointer to `f`, and
`?g>` must give the new value. Chance is one in six.

Before change 6 the pointer always named the first name bound and the
overwrite always fell on the second; see "A shortcut in Gemini's task"
below.

**Training.** On the answers only: the character after each `?x>`,
predicted at the `>`. Filler of 2 to 10 characters. Adam, learning rate
0.003, batches of 32, 1,500 steps, on a CPU.

**Test.** 512 new episodes at the trained length (about 52 characters in
all), and with 24 and 74 characters of each filler (about 100 and 200).

**Run.** `python experiments/answers_only.py MODEL` (two seeds, a few
minutes each).

**Where the tables on this page ran** (2026-11-22): a CPU (Intel Xeon,
2.10 GHz, 4 cores), on 2 threads unless a table says otherwise (the
scripts' `--threads` default; `OMP_NUM_THREADS=2` for `train_trinet.py`),
with Python 3.11.17, PyTorch 2.15.0 (CPU build), transformers 5.24.0 and
NumPy 2.4.6. PyTorch does not promise the same numbers on another number
of threads.

### Results, 2026-11-22 (task with change 6, seeds 0 and 1)

| Model | Parameters | ~52 characters | ~100 characters | ~200 characters |
|---|---|---|---|---|
| `tri_lm`: Gemini's prototype, unchanged | 18,358 | 99.6 / 100.0% | 99.5 / 100.0% | 99.5 / 100.0% |
| `trinet`: `TriNetForCausalLM` as in this repository | 33,748 | 100.0 / 100.0% | 100.0 / 100.0% | 100.0 / 100.0% |
| `trinet-gemini`: the same, with Gemini's initial values | 33,748 | 99.9 / 69.9% | 99.9 / 71.1% | 99.8 / 69.5% |
| `transformer`: a small causal Transformer, ours | 39,322 | 49.7 / 51.8% | 37.2 / 34.2% | 37.5 / 34.2% |

The script prints each kind of query separately, and for the overwrite
queries how often the answer is the old value.

- **The Transformer fails the overwrites on longer texts, and keeps to the
  old value.** At about 100 and 200 characters it gives the new value 6.1
  to 10.0% of the time and the old value 47.9 to 54.9%. Its pointer
  queries are at 40.6 to 48.6%. Gemini reported 13.5% on overwrites for
  theirs.
- **`trinet-gemini` learns the task in one seed of two.** Seed 0 reaches
  99.8 to 99.9%; seed 1 stays near 70% at every length, held back on every
  kind of query (direct 70 to 71%, overwrite 79 to 84%, pointer 55 to 58%).
- **Gemini's prototype at fewer steps:** 99.4 to 99.9% at 250 steps
  (`--steps 250`), and 99.9 to 100% at 450.

### A shortcut in Gemini's task (change 6)

In Gemini's `make_reasoning_example` the pointer always named the first
name bound, and the overwrite always fell on the second. A model could
answer every pointer query with the first value bound, without following
the pointer. ChatGPT's review pointed this out (record 012-09-05, line
16522). `experiments/pointer_check.py` trains a model on the task as given
and tests it on that task and on the task with change 6 (seeds 0 and 1,
1,500 steps, as above):

| Model | Pointer queries, task as given | Pointer queries, change 6 (~52 / ~100 / ~200) |
|---|---|---|
| `trinet` | 99.8 to 100.0% | 51.2 / 40.6 / 45.9%, both seeds |
| `tri_lm` | 100.0% | 51.2 / 40.6 / 45.9%, both seeds |
| `transformer` | 72.5 to 85.2% | 37.1 to 43.8% |

- **Both TriNet models learned the shortcut.** Their pointer scores with
  change 6 are the same to the decimal, for both models and both seeds, and
  exactly what the rule "answer with the first value bound" scores on the
  same test episodes: 51.2 / 40.6 / 45.9%. (The rule is right when the
  pointer names the first name, or another holding the same value: about
  44% on average.) Counting every kind of query together, on the first
  query of each episode, they scored 82.6 to 88.1%.
- **Trained on the task with change 6, both follow the pointer** (the
  results above).
- **Earlier answers in the text do not explain the scores.** ChatGPT's
  review raised this too (line 16519). Scoring only the first query of
  each episode, which no earlier answer can help, gives the same picture
  (pointer queries 53.1 / 41.1 / 45.7% with change 6).

### What this shows

- **Gemini's prototype does what they said it would.** It learns all
  three kinds of query, and loses nothing at four times the length it was
  trained on. Its memory is the same size at any length. It learns about
  as fast as they reported: 100% by step 250 in their report, 99.4 to 99.9%
  here. Their training differs: they trained on every next character
  ("purely from next-character prediction", conversation line 37466) with
  22 characters, learning rate and batch size not given; here it trains on
  the answers only, with 25 characters and padding, at the settings above.
- **The packaged model learns the task completely once it starts from
  PyTorch's own initial values** (change 5 in `provenance/MANIFEST.json`):
  100% in both seeds. Gemini's `_init_weights` drew every weight from
  N(0, 0.02), as Hugging Face models usually do; with those values it
  learned the task in one seed of two.
- **Until change 6, "follows a pointer" was not shown.** The task as given
  could be passed without following the pointer, and the models did pass
  it that way. With change 6 it is shown, trained on the answers.
- **The Transformer here is a weak comparison.**
  - It is ours, untuned, and twice the prototype's size.
  - It learns its positions, so beyond the trained length it meets
    positions it never saw. That is part of why it does worse on longer
    texts.
  - A fair comparison needs a tuned Transformer of the same size,
    including one without learned positions. Gemini ran one; we have not.

### Corrected after review

- Until the independent review of #2, the script read each answer at the
  query's name, one character before the `>`. All four models were trained
  and tested that way, so the comparison held, but two numbers were lower:
  `trinet-gemini` reached about 70%, and `tri_lm` at 450 steps 70.7 and
  74.7%, which we took to mean it learns more slowly than Gemini said. One
  result from that time was not repeated: letting the gates also see the
  local context (they see it only through the three-colour receptors)
  changed nothing then (69 to 71% with Gemini's initial values, two seeds;
  not kept here).
- Until #3, every result on this page used Gemini's task with its fixed
  roles. On that task `trinet-gemini` reached 78.6 to 92.5% and the
  Transformer 51.3 to 69.7%; the shortcut section shows what it hid. The
  results above are from the task with change 6; the earlier ones are in
  this file's history.

## With `train_trinet.py` as written

The training script trains on every next character, with half the
batches from the task above and half from a paragraph of prose, at the
smallest preset (59,708 parameters, against a Transformer of 71,648):

```
python train_trinet.py --preset smoke --smoke-test --seq-len 96 --steps 2000 \
  --warmup-steps 100 --reasoning-mix 0.5 --log-every 500 --save-every 0 --seed SEED
```

Its own test of the task, with 6 / 24 characters of filler (2026-11-22;
chance is 16.7%; 2 threads, `OMP_NUM_THREADS=2`):

| Task with change 6 | seed 0 | seed 1 | seed 42 |
|---|---|---|---|
| TriNet, PyTorch's initial values (change 5) | 33.6 / 33.9% | 19.0 / 16.4% | 16.9 / 20.1% |
| TriNet, Gemini's initial values | 15.9 / 16.9% | 17.7 / 15.9% | 15.9 / 19.0% |
| Transformer beside it, with change 5 | 25.0 / 20.3% | 36.5 / 27.6% | 33.6 / 27.6% |
| Transformer beside it, without change 5 | 39.8 / 28.9% | 38.3 / 28.6% | 34.1 / 34.6% |

- **Trained this way, TriNet learns the task only in part.** With change 5
  it reached 34% in one seed of three (overwrites 44 to 51%) and stayed
  near chance in the other two. With Gemini's values it stayed near chance
  in all three.
- **The Transformer beside it does better in five runs of six** (20 to
  40%).
- **Much of the earlier success came from the shortcut.** On Gemini's task
  as given, the same runs with change 5 gave 99.7 / 100.0% (seed 0), 19.3 /
  17.4% (seed 1) and 66.9 / 67.4% (seed 42; on 4 threads), and 15 to 20%
  without it.
- So TriNet can learn the task, trained on the answers (above), but not
  yet from every next character in 2,000 small steps. That is STATUS
  "Next" 1.
- The Transformer's runs differ with and without change 5 because the
  TriNet built before it draws a different amount from the random number
  generator.
