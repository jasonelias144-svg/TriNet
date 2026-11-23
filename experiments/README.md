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
2.10 GHz, 4 cores), on 2 threads unless stated otherwise (the
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
  review pointed out that later queries can see earlier answers (record
  012-09-05, line 16519), so `pointer_check.py` also scores the first
  query of each episode alone, which no earlier answer can help. It gives
  the same picture (pointer queries 53.1 / 41.1 / 45.7% with change 6).

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

## Writing and reading a hologram

**Question.** The founder asked to look at how holograms are written and
read, as physics, not as a metaphor (STATUS.md, Decided, 2026-11-23).
`docs/NEURON-CATALOG.md` ("Writing and reading a hologram") sets out the
optics and proposed this simulation: does the textbook account hold when
it is computed, and how does a hologram compare, as a memory, with the
matrix memories TriNet is built on?

**Method** (`experiments/holography.py`, numpy only). Light of one colour
(633 nm) as a scalar field on a plate 1,024 pixels of 1 micrometre wide.
Light moves by the angular spectrum method (each spatial frequency
changes only its phase). Writing records the plate's intensity,
`|O + R|²`, for an object wave O and a reference R, a plane wave at 8.6
degrees. Reading lights the plate with R again and keeps only the light
travelling the object's way (an aperture in the focal plane of a lens),
then moves it back to each depth.

**Run.** `python experiments/holography.py` (about 10 seconds; run on
2026-11-23 on 1 thread, with NumPy 2.4.6, on the machine above).

**A. Each point comes back at its own depth.** Three points of light, at
20, 30 and 40 mm behind the plate. Brightness at each point's place, as a
share of the brightest value in each refocused plane:

| Refocused at | point at 20 mm | point at 30 mm | point at 40 mm |
|---|---|---|---|
| 20 mm | **1.00** | 0.00 | 0.00 |
| 30 mm | 0.00 | **1.00** | 0.00 |
| 40 mm | 0.00 | 0.00 | **1.00** |
| *without a reference beam:* | | | |
| 20 mm | 0.03 | 0.05 | 0.01 |
| 30 mm | 0.37 | 0.03 | 0.02 |
| 40 mm | 0.15 | 0.05 | 0.05 |

With the reference, each point is sharp in its own plane and absent from
the others. Recorded without it (the intensity of the object's light
alone, as film records), no point comes back at its depth: the depth was
in the phase, and only the reference keeps the phase.

**B. A quarter of the plate still shows every point.** Spot widths at
half maximum, in micrometres, with diffraction's 0.886 λz/D in brackets
(D the window's width; pixels are 1 micrometre):

| Point at | whole plate | one quarter | how much wider | quarter's image: share of the brightest |
|---|---|---|---|---|
| 20 mm | 12 (11) | 24 (22) | 2.0 | 1.00 |
| 30 mm | 18 (16) | 36 (33) | 2.0 | 0.96 |
| 40 mm | 23 (22) | 43 (44) | 1.9 | 0.98 |

Cutting the plate to a quarter (half the width each way) loses no point;
each spot is about twice as wide, as a window half as wide should make
it. Damage cost sharpness, not points.

**C. Many memories in one store.** n memories of unit phases, each read
back with its key. "Score" is the mean match between the read-out and the
memory asked for (1 is perfect); "found" is how often that memory is the
best match among all those stored. Every store holds 4,096 numbers: a
hologram (a point-by-point product of value and conjugate key, summed)
with memories of 4,096 numbers, and 64 × 64 matrices (TriNet's kind of
memory) with memories of 64 numbers, written as a plain sum of outer
products or by the delta rule. 20 trials each.

| Memories | hologram: score, found | sum of outer products | delta rule |
|---|---|---|---|
| 1 | 1.000, 100% | 1.000, 100% | 1.000, 100% |
| 2 | 0.708, 100% | 0.993, 100% | 0.996, 100% |
| 8 | 0.354, 100% | 0.950, 100% | 0.969, 100% |
| 32 | 0.179, 100% | 0.825, 100% | 0.841, 100% |
| 64 | 0.124, 100% | 0.702, 100% | 0.620, 90% |
| 128 | 0.091, 100% | 0.587, 100% | 0.494, 65% |
| 512 | 0.045, 70% | 0.335, 60% | 0.252, 20% |

- **The hologram's read-out falls as 1/√n** (0.707, 0.354, 0.177, 0.125,
  0.088, 0.044), as the catalog says: the other memories come back as
  noise spread over the whole field. But the memory asked for is still
  the best match up to 128 memories, and 70% of the time at 512.
- **The matrices give much cleaner read-outs** (0.83 against 0.18 at 32
  memories), because reading takes an inner product with the key, which
  averages the other memories down; but their memories are 64 numbers,
  not 4,096, and past 64 memories (as many as a key has numbers) they
  are found less often than the hologram's.
- **The delta rule helps a little up to 32 memories and hurts beyond
  64**: with random keys there is little crosstalk for it to remove, and
  once keys outnumber their dimensions each write disturbs the earlier
  ones. TriNet's keys are learned, not random; that is the next test.
- **Keys at different angles** (part C's last line): 9 pages, each using
  0.5% of the plate's spatial frequencies, read back with a largest
  relative error of 9.4 × 10⁻¹⁵: exact, because each page's unwanted
  terms leave in other directions. Such pages share the plate's
  frequencies, so their number is capped by the plate's size.

**What this shows.** The account in the catalog holds when computed: the
reference beam is what keeps depth; a piece of a hologram shows the
whole, less sharp; and many memories share one plate, kept apart by
direction or traded against noise. As a memory, a hologram holds big
memories noisily in few numbers; a matrix holds small memories cleanly in
many. Which serves a network better is the catalog's next step: the same
operations inside a TriNet layer, on the recall task.

**Not checked.** Light of more than one colour; a thick (volume)
hologram; noise in the recording; objects other than points (a smooth
object's light gathers near the plate's centre, so a quarter away from it
would show mostly edges).
