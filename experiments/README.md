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
minutes each, on 2 threads; on another number of threads a run can
differ by a point or two, as Stage A below found).

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
  "Next" 2.
- The Transformer's runs differ with and without change 5 because the
  TriNet built before it draws a different amount from the random number
  generator.

## Stage A: two TriNet stacks against one

**Question.** Does Gemini's BicameralTriNet
(`bicameral/bicameral_stage_a.py`) do better than one TriNet of the same
size? It runs two TriNet stacks side by side, joined by a narrow
"callosum" between their layers and mixed at the output.

**Models** (`experiments/stage_a.py`):

| Model | What it is | Parameters |
|---|---|---|
| `trinet` | one TriNet, as above | 33,748 |
| `trinet-residual` | the same, with `residual_error=True` (change 7): the fast memory stores adjustments to `W_base` | 33,748 |
| `bicam-sym` | two identical stacks, with the callosum | 34,193 |
| `bicam-asym` | the same, but stack A keeps its memory longer (retention floor 0.95 against 0.5) | 34,193 |
| `bicam-nocal` | two identical stacks with no callosum, joined only at the output | 33,056 |
| `transformer` | the small Transformer above | 39,387 (39,322 on the task above, whose characters are fewer) |

The two stacks are each 20 wide, with keys and values of 12, so that the
pair comes within 2.1% of one TriNet. Gemini's random dropping of a whole
stack in training (`hemisphere_dropout`) is off.

**Training and test** as above: on the answers only, Adam at 0.003,
batches of 32, 1,500 steps; tested on 512 new episodes at each length.

### On the task above: no room to show a difference

**Run.** `python experiments/stage_a.py MODEL --task answers --threads 1`
(seeds 0, 1 and 2). The runs below used commit 2970da8 (its later
changes are to docstrings only), two at a time, on 1 thread each, on
2026-11-23. The learning curve gives the score at the training length
(about 52 characters) every 250 steps.

| Model | ~52 characters | ~100 characters | ~200 characters | First curve point at 95% or more (step) |
|---|---|---|---|---|
| `trinet` | 100.0 / 99.2 / 99.8% | 100.0 / 99.2 / 99.9% | 100.0 / 98.2 / 99.6% | 500 / 250 / 250 |
| `trinet-residual` | 100.0 / 100.0 / 100.0% | 100.0 / 100.0 / 99.9% | 100.0 / 99.7 / 99.8% | 250 / 250 / 250 |
| `bicam-sym` | 100.0 / 99.6 / 99.6% | 100.0 / 99.1 / 99.8% | 100.0 / 99.5 / 99.4% | 250 / 500 / 500 |
| `bicam-asym` | 100.0 / 99.9 / 99.8% | 100.0 / 100.0 / 99.7% | 99.9 / 100.0 / 99.1% | 250 / 500 / 500 |
| `bicam-nocal` | 100.0 / 100.0 / 100.0% | 99.9 / 99.9 / 99.9% | 99.7 / 99.9 / 100.0% | 500 / 250 / 500 |
| `transformer` | 49.5 / 51.1 / 51.2% | 37.6 / 33.7 / 36.7% | 37.2 / 34.2 / 36.7% | never in any seed |

- **Every TriNet model, one stack or two, scores 98.2 to 100%** at every
  length in every seed, and passes 95% on the learning curve by step
  500. A pair of stacks has no room to do better here.
- **The number of threads alone moved a run by 1.8 points,** in the one
  case compared. On the default 2 threads, `trinet` seed 1 scores 100% at
  every length (an earlier run, and the independent review of #7); on 1
  thread, as in the table, 98.2 to 99.2%. PyTorch adds numbers in a
  different order on a different number of threads, and training
  amplifies the difference. So a single run's outcome carries noise
  beside the seed's; how much on the harder task below was not measured,
  and there it could decide whether a run stops short.
- The small Transformer scores 34 to 51%, less as the text grows, as in
  the results above.
- Time: 228 to 244 seconds a run for one TriNet, 403 to 447 for two
  stacks, 36 to 40 for the Transformer.

### A harder task

`experiments/harder_task.py`, for example:

```
h=4;g=3;f=1;c=4;e=5;z-x,,-d=h;f=3; g=5;z.b=c;--e=0;xf=1;e=4;c=2; c,- f?b>4;?f>1;?c>2;?f<3;?h>4;?g<3;?e>4;?c<4;?d>4;?g>5;?e<0;
```

- Five names are bound to values.
- Then, in random order: two pointers (`d=h`), each copying the value its
  target has at that moment, the second possibly naming the first; and
  six overwrites of the first five names (`f=3`).
- Then about eleven queries: `?x>` for the current value of x, and `?x<`
  for the value x had before its last overwrite.

Four kinds of query: a current value never overwritten, a current value
overwritten, a pointer, and the value before the last overwrite. Chance is
one in six. No simple rule passes it: on the 512 test episodes at the
training length, "the first value given" is right for 41.9% of the
non-pointer queries, and for pointers "the target's value now" 58.0% and
"the target's first value" 60.9%.

The test lengths are about 125 characters (the training length), 160 and
260. The script also prints a learning curve: the score at the training
length every 250 steps, on 128 fixed episodes.

**Run.** `python experiments/stage_a.py MODEL --task harder` (seeds 0, 1
and 2; about 9 minutes a seed for one TriNet, 17 for two stacks, 1 for the
Transformer). The runs below used commit ac96fb5, two at a time, on 2
threads each.

The task was changed once before these runs. Its first version, with
four names and four overwrites, gave one TriNet 96 to 97% at seed 0. The
version here, with five names and six overwrites, gave it 99.1 to 99.5%
at seed 0 (more overwrites are more practice; commit ac96fb5). All the
runs below used only this version.

### Results, 2026-11-22 (harder task, seeds 0 / 1 / 2)

| Model | ~125 characters | ~160 characters | ~260 characters | First curve point at 95% or more (step) |
|---|---|---|---|---|
| `trinet` | 99.5 / 99.2 / 88.9% | 99.1 / 99.3 / 88.9% | 99.2 / 98.8 / 89.8% | 1000 / 1500 / never |
| `trinet-residual` | 86.6 / 99.6 / 99.8% | 86.6 / 99.7 / 99.9% | 86.5 / 99.7 / 99.8% | never / 750 / 750 |
| `bicam-sym` | 99.5 / 99.2 / 93.2% | 99.3 / 99.3 / 93.4% | 99.0 / 99.2 / 93.7% | 1000 / 750 / never |
| `bicam-asym` | 95.8 / 99.6 / 92.8% | 96.5 / 99.6 / 92.1% | 96.0 / 99.2 / 92.2% | 1500 / 750 / never |
| `bicam-nocal` | 98.5 / 99.3 / 99.5% | 98.9 / 99.5 / 99.7% | 98.6 / 99.5 / 99.5% | 1250 / 750 / 750 |
| `transformer` | 33.0 / 33.8 / 33.3% | 29.0 / 28.5 / 28.5% | 24.5 / 24.0 / 24.8% | never in any seed |

- **The difference between seeds is larger than the difference between
  models.** Every TriNet model scores 98.5 to 99.9% in some seeds, and five
  of the fifteen runs stop short (86.5 to 96.5%). Averaged over the three
  seeds at about 160 characters: `trinet` 95.8%, `trinet-residual` 95.4%,
  `bicam-sym` 97.3%, `bicam-asym` 96.1%, `bicam-nocal` 99.4%. The runs
  that stop short decide these averages (two for `bicam-asym`, one for
  each of the first three, none for `bicam-nocal`), so three seeds do not
  show that any of these differs from another.
- **A run that stops short fails one kind of query.** `trinet` seed 2,
  `bicam-sym` seed 2 and `bicam-asym` seeds 0 and 2 answer "the value
  before the last overwrite" 69 to 92% right, and give the current value
  instead 7 to 23% of the time. `trinet-residual` seed 0 follows pointers
  29% of the time; everything else it answers 99 to 100% right.
- **Only `bicam-nocal` learned the task in all three seeds**, and it is
  the pair *without* the callosum. If anything helps it, which three
  seeds do not show, it is not the connection between the stacks. One
  reading, not tested: two stacks that learn on their own are two tries
  at the task, joined at the output.
- **The longer retention of one stack (`bicam-asym`) did not help.** It
  scored below `bicam-sym` in two seeds of three.
- **Change 7 (`trinet-residual`) learns fastest when it learns:** at step
  500 it is at 93.0 and 90.2% in seeds 1 and 2, where no other model is
  above 86.6% in any seed. Its final scores, 99.7 and 99.9% at about 160
  characters, are among the best (`bicam-nocal` seed 2 also has 99.7%).
  In seed 0 it never learned pointers. Not shown to be better; it stays
  off by default.
- **Seed 2 was hard for three models.** With the same seed every model
  sees the same episodes in the same order (the episodes come from
  `numpy.random.default_rng(seed)`), though not the same initial weights.
  `trinet`, `bicam-sym` and `bicam-asym` all stopped short in seed 2;
  `bicam-nocal` and `trinet-residual` did not.
- **Nothing is lost at longer texts.** Every TriNet model scores the same
  at 260 characters as at 125, within a point.
- **Two stacks take twice as long:** 961 to 1,035 seconds a run, against
  504 to 559 for one TriNet.
- **The small Transformer fails the task** (24 to 34%), and does worse as
  the text grows. It learns its positions and is untuned (STATUS "Next"
  1); it is no measure of what attention can do here.

**Not checked:**
- More seeds, and the thread noise above, for the harder task.
- Gemini's own advice for Stage A (record 012-10-06, line 3893): to run
  `hemisphere_dropout` at 0.1 as well as 0.0, and to match parameter
  counts with and without the callosum (`bicam-nocal` has 1,137 fewer
  parameters than `bicam-sym`).
- Whether a model answers later queries from earlier answers in the same
  episode. On the original task, `pointer_check.py` scores the first
  query alone for this; the harder task has no such check yet.

### What Stage A shows

Two TriNet stacks did not do measurably better than one, on either task.
One guess, not tested: the two halves are too alike to bring different
strengths (though `bicam-asym`, whose halves differ in how long they
keep their memory, did not help either). A pair of different halves, a
TriNet and a Transformer with an executive network between them, is to
be one of the models of the fair comparison (STATUS "Next" 1). Its code
is on its own branch (`claude/hybrid-prototype`, not merged); its results
will come with its own pull request.

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
changes only its phase), on a grid twice as wide so that little of the
light leaving the plate's edge wraps round (the FFT's grid is periodic;
on wider grids still, the numbers below move by a few hundredths, or a
pixel, at most). Writing records the plate's
intensity, `|O + R|²`, for an object wave O and a reference R, a plane
wave at 8.6 degrees. Reading lights the plate with R again and keeps only
the light travelling the object's way (an aperture in the focal plane of
a lens), then moves it back to each depth.

**Run.** `python experiments/holography.py` (about 90 seconds; run on
2026-11-23 on 1 thread, with NumPy 2.4.6, on the machine above).

**A. Each point comes back at its own depth.** Three points of light, at
20, 30 and 40 mm behind the plate. Brightness at each point's place, as a
share of the brightest value in each refocused plane:

| Refocused at | point at 20 mm | point at 30 mm | point at 40 mm |
|---|---|---|---|
| 20 mm | **1.00** | 0.00 | 0.00 |
| 30 mm | 0.00 | **1.00** | 0.00 |
| 40 mm | 0.00 | 0.00 | **1.00** |
| *control, without a reference beam:* | | | |
| 20 mm | 0.33 | 0.16 | 0.22 |
| 30 mm | 0.07 | 0.18 | 0.00 |
| 40 mm | 0.00 | 0.01 | 0.01 |

With the reference, each point is sharp in its own plane and absent from
the others. The control records the intensity of the object's light
alone, as film does without holography, then lights that plate straight
on and refocuses it in the same way: no point is the brightest thing in
its own plane. Intensity alone keeps how the points' waves differ in
phase from one another, but not their phase against a known wave, so the
wave cannot be rebuilt; the reference gives that known wave.

**B. A quarter of the plate still shows every point.** Spot widths at
half maximum along x and y, in micrometres, with diffraction's 0.886 λz/D
for a square window of width D in brackets:

| Point at | whole plate, x / y | one quarter, x / y | how much wider, x / y | quarter's spot, brightness beside the whole plate's |
|---|---|---|---|---|
| 20 mm | 11.3 / 10.8 (11.0) | 22.5 / 19.8 (21.9) | 1.99 / 1.84 | 0.058 |
| 30 mm | 17.0 / 16.6 (16.4) | 34.6 / 30.7 (32.9) | 2.04 / 1.85 | 0.057 |
| 40 mm | 22.3 / 22.1 (21.9) | 43.4 / 42.8 (43.8) | 1.95 / 1.94 | 0.068 |

Cutting the plate to a quarter (half the width each way) loses none of
the three points: each spot is about twice as wide, as a window half as
wide should make it, and about a sixteenth as bright, as a quarter of the
area gathers a quarter of the light into a spot four times the area.
Each spot's peak lies within 0 to 1 pixel of its point on the whole
plate, and 0 to 3 on the quarter. **Our reading:** damage cost
sharpness and light, not points, for these three isolated points.

**C. Many memories in one store.** n memories of unit phases, each read
back with its key, 100 trials each. "Score" is the mean match between the
read-out and the memory asked for (1 is perfect); "found" is how often
that memory is the best match among those stored: picking it from a
known list, not recalling it. Every store holds 4,096 complex numbers:

- a hologram whose memories are 4,096 numbers each, either as the
  complex sum of the term a hologram is read for (value times conjugate
  key), or recorded on a plate in intensity, one exposure per memory,
  each exposure's mean taken off (so its twin terms remain). Taking it
  off is what digital holography does with a camera's record; a film
  plate keeps it, and it would come back as more noise, which is not
  simulated here;
- 64 × 64 matrices (TriNet's kind of memory) whose memories are 64
  numbers each: the plain sum of outer products; the delta rule in one
  pass at full step, without the decay, clipping and learned gates of
  TriNet's fast memory (`F ← λF + η·kᵀ(v − kF)`); and least squares,
  fitted to all memories at once: the best fit in squared error, which
  for up to 64 memories returns each exactly.

| Memories | hologram, complex sum | hologram, recorded | matrix: sum | matrix: delta rule | matrix: least squares |
|---|---|---|---|---|---|
| 1 | 1.000, 100% | 0.707, 100% | 1.000, 100% | 1.000, 100% | 1.000, 100% |
| 2 | 0.707, 100% | 0.501, 100% | 0.993, 100% | 0.995, 100% | 1.000, 100% |
| 8 | 0.354, 100% | 0.250, 100% | 0.951, 100% | 0.971, 100% | 1.000, 100% |
| 32 | 0.179, 100% | 0.126, 100% | 0.823, 100% | 0.834, 100% | 1.000, 100% |
| 64 | 0.126, 100% | 0.090, 100% | 0.715, 100% | 0.682, 98% | 1.000, 100% |
| 128 | 0.089, 100% | 0.061, 97% | 0.580, 100% | 0.476, 60% | 0.706, 100% |
| 512 | 0.047, 69% | 0.033, 17% | 0.347, 59% | 0.185, 11% | 0.369, 66% |

- **The hologram's read-out falls as 1/√n** (0.707, 0.354, 0.177, 0.125,
  0.088, 0.044), as the catalog says: the other memories come back as
  noise spread over the whole field. Recorded on a plate, the twins add
  as much noise again, and the match falls as 1/√(2n). The memory asked
  for is still the one found, every time, up to 128 memories as a
  complex sum and up to 64 when recorded.
- **The matrices read far more cleanly** (0.82 against 0.18 at 32
  memories), because reading takes an inner product with the key, which
  averages the other memories down; least squares returns every memory
  exactly up to 64, as many as a key has numbers. But their memories are
  64 numbers, not 4,096, so "found" does not compare like with like.
  With 100 trials (a standard error of about 5 points at 512), the sum
  of outer products finds the memory as often as the hologram up to 128,
  and a little less at 512 (59% against 69%).
- **The one-pass delta rule favours the newest memories.** Its match by
  age, oldest quarter to newest, on 64 × 64:

  | Memories | oldest quarter | second | third | newest quarter |
  |---|---|---|---|---|
  | 32 | 0.74 | 0.81 | 0.88 | 0.96 |
  | 64 | 0.48 | 0.59 | 0.73 | 0.91 |
  | 128 | 0.21 | 0.32 | 0.50 | 0.81 |

  Each write at full step makes its own key exact and disturbs the
  earlier ones, so on average it beats the sum only while there are few
  memories, and falls behind from about 50. This is the delta rule
  without TriNet's decay, clipping and gates, on random keys; TriNet's
  keys are learned, which is the next test.
- **Keys at different angles read back exactly**, both as a complex sum
  (largest relative error 4.5 × 10⁻¹⁵) and recorded on a plate in
  intensity, one exposure per page (1.3 × 10⁻¹³). There are nine pages,
  each using 0.14% of the plate's spatial frequencies. The carriers are
  all on one side and apart by more than twice each page's band, so that
  each page's unwanted terms, and on the recorded plate its zero order
  and twin, travel in other directions, where the aperture leaves them
  out (the zero order also needs each carrier more than three times the
  band from the axis, which these are: at least 26/256, against a band
  of 0.02). Such pages share the plate's frequencies, so their number is
  capped by the plate's size.

**What this shows.** The account in the catalog holds when computed: the
reference beam is what lets the wave, and so the depth, be rebuilt; a
piece of the plate shows the whole, less sharp and less bright; and many
memories share one plate, kept apart by direction, or traded against
noise. As a memory, a hologram holds big memories noisily in few
numbers; a matrix holds small memories cleanly. Which serves a network
better would be shown by the same operations inside a TriNet layer, with
learned keys, on the recall task: the catalog's holographic layer, set
aside for now (STATUS.md, Decided, 2026-11-23).

**Not checked.** Light of more than one colour; a thick (volume)
hologram; noise in the recording; objects other than isolated points (a
smooth object's light gathers near the plate's centre, so a quarter away
from it would show mostly edges); and learned keys.
