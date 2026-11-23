# Unit 0: a basic unit designed from the ground up

Written by Claude (technical lead), 2026-11-23, at the founder's
invitation. A design page: the design comes first, with its predictions,
and code comes after. "Unit 0" is a working name.

Claims carry the labels of `docs/NEURON-CATALOG.md`: **Established**,
**Reported**, **Leading theory**, **Contested**, **Open**, **Our
reading**, **Our proposal**. Results from this repository give the
section of `experiments/README.md` they come from.

## Why this page

The founder, 2026-11-23, in conversation with Claude:

> "Also, consider TriNet was just one idea. We can explore others. Maybe
> start with a two-channel. My point is that we do not need to be locked
> into TriNet. The whole fault I am exploiting is this very fact that
> once a particular neuron architecture was established as standard, it
> became costly to change and everyone continued build long from there.
> I don't want us to make the same mistake of investing so much in
> TriNet that we forget it as one option mentioned casually."

> "I have you my ideas of how to build a basic neuron in different ways,
> but I would rather, at this point, let you design it from the ground
> up. Not starting with TriNet, but starting with an invitation to build
> something from scratch that takes everything we have learned and
> utilizes that knowledge to design a new basic neural processing unit.
> And then we can build from there."

Earlier the same day, on images:

> "the three-channel neuron was meant to be efficient for image
> processing because of the ability to associate a pixel with a neuron.
> But if the pixels have already been flattened in the data, to
> accommodate single channel neurons, then we would not expect to see the
> benefit. I would be curious with the same image data, but for TriNet,
> the training data is formatted to take advantage of its architecture,
> instead of training with data that was processed for a different neuron
> topology."

The fault the founder names has a name in the field: the hardware
lottery (Hooker, 2021), when a research idea wins because it suits the
software and hardware of the day, not because it is better than the
alternatives.

## What we have learned

The evidence this design starts from:

1. **An error-correcting fast memory learns recall in the text.**
   TriNet's memory is written by the delta rule: it writes the
   difference between what it sees and what it would have predicted.
   One TriNet of 33,748 parameters learns the harder task, where a small
   Transformer, untuned, did not ("Stage A"). The rule is the
   Widrow–Hoff rule (1960). As a memory inside a sequence model it is a
   "fast weight programmer" (Schlag, Irie & Schmidhuber, 2021), and
   DeltaNet and Gated DeltaNet train it at scale (Yang et al., 2024;
   Yang, Kautz & Hatamizadeh, 2024). **Established** for the rule; for
   this task, our result.
2. **Seeds differ more than models.** With three seeds no TriNet variant
   was shown better than another ("What Stage A shows"). Every
   comparison here needs five seeds or more.
3. **A loop over characters is slow on a GPU.** TriNet's step time
   hardly changes with the batch, and its memory runs out first ("Speed
   on a GPU"). If the gates read only the input, not the memory, the
   update can be computed in parallel chunks (Yang et al., 2024).
4. **A phase can be learned, but read as Re(qF) it costs on longer
   texts.** Complex retention learned the task when its angles started
   near zero, was not better, and lost 0.8 to 11.0 points at 260
   characters, where TriNet lost at most 2.6 ("Complex retention").
   Rotary positions in Transformers also extrapolate poorly to longer
   texts (Press, Smith & Lewis, 2022; **Reported**).
5. **A matrix written by the delta rule reads more cleanly than a
   hologram's superposition.** In our simulation a hologram's read-out
   falls as 1/√n with n memories, and 64 × 64 matrices of the same
   4,096 numbers read more cleanly ("Writing and reading a hologram").
6. **Not yet known: which of TriNet's other parts matter.** The fair
   comparison now running takes out the receptor bus, the reading of
   surprise, the gates, the shunt, the delay line and the clamp, one at
   a time. Its predictions were written before any result was read
   ("The fair comparison", branch `claude/fair-comparison`).

## Principles

Each comes from the evidence above or is labelled a choice.

1. **The unit carries its own small memory, written by correcting its
   errors** (evidence 1, 5).
2. **Two inputs:** evidence from below, and context from above. In the
   cortex, pyramidal cells take input from below on their basal
   dendrites and feedback from above on their apical dendrite (Larkum,
   2013; **Established** for the anatomy, **Leading theory** for its
   role). Here the context steers what the unit looks for and how
   strongly it answers.
3. **Two outputs: what it found, and how surprised it was.** These are the
   founder's two channels, each given a job. The surprise is the size of
   the memory's error, which the unit computes anyway. A level above can
   use it to set boundaries (`docs/NEURON-CATALOG.md`, "Levels of
   text").
4. **Divide as well as add.** Divisive normalization, a response divided
   by the pooled activity around it, is found in many systems of the
   brain (**Established**); that it is a canonical computation is a
   **Leading theory** (Carandini & Heeger, 2012). It gives invariance to
   brightness and scale. Multiplication gives gating (Dauphin et al.,
   2017; Shazeer, 2020).
5. **Fit the hardware without being captured by it.** Every part is
   written as matrix products and parallel scans. Anything that must wait
   for the step before it waits only at chunk boundaries: for example,
   context from above arrives one chunk late (evidence 3). A choice.
6. **Every feature is a dial whose "off" is a simpler unit,** so each
   part can be taken out and measured (evidence 2, 6). A choice.
7. **One interface at every level,** so units compose into larger units:
   the fractal design of `docs/NEURON-CATALOG.md`. A choice.
8. **A phase only where it pays,** tested on texts longer than training
   (evidence 4).

## What the unit is

A unit here is a small population of neurons sharing one memory, not a
single neuron. **Our reading:** a single artificial neuron is a weighted
sum and a nonlinearity, and changing the nonlinearity alone (ReLU, GELU,
SwiGLU) has moved results by small amounts; what a population stores
between steps, and how, has moved them far more (evidence 1). The
founder's cells (`docs/NEURON-CATALOG.md`, "The general cell") stay in
play as the unit's nonlinearity, one dial among the others.

### Inputs and outputs

- In: `x_t`, evidence from below (width d), and `c_t`, context from
  above (width d_c; zero where there is no level above).
- Out: `y_t`, what it found (width d), and `s_t`, its surprise (one
  number).

### One step

With a memory S (d_k × d_v), starting at zero:

    h_t  = SiLU(conv4(x)_t)                   short causal convolution
    k_t  = l2(SiLU(W_k h_t))                  key, length 1
    v_t  = W_v h_t                            value
    q_t  = l2(SiLU(W_q h_t + U_q c_t))        query: context steers it
    a_t  = sigmoid(A h_t + b_a)               decay, one per key row
    b_t  = sigmoid(w_b·h_t + u_b·c_t + b_b)   write strength, one number

    D_t  = diag(a_t) S_{t-1}                  the memory, faded
    e_t  = v_t − k_t D_t                      the error: what it missed
    S_t  = D_t + b_t · k_tᵀ e_t               the delta rule
    s_t  = |e_t| / √d_v                       surprise

    o_t  = q_t S_t                            the read
    n_t  = o_t / √(σ² + mean(o_t²))           divided by pooled activity
    y_t  = W_o (n_t ⊙ SiLU(W_g h_t + U_g c_t))   gain from context

`l2(·)` scales a vector to length 1; σ is learned. A block adds `y_t`
to its input and a small MLP after it, as Gated DeltaNet's blocks do.

The gates (a_t, b_t) read the input and the context, never the memory,
so the whole sequence can be computed in chunks (principle 5). The
surprise is an output, not an input to the gates. That is the
difference from TriNet, whose gates read surprise; the fair comparison's
`trinet-no-surprise` arm tests whether the difference costs anything.

### The three channels

Between two levels the unit has three kinds of connection: content up
(`y`), surprise up (`s`), and context down (`c`, the level above's `y`).
**Our reading:** this is one way to give the founder's three channels
each a job. It is not TriNet's red, blue and yellow, and it is not
claimed to be.

### Levels

A second unit reads spans of the first one's output: it takes the mean
of `y` over each span, and its own `y` returns to the first unit as `c`
for the next span. A span ends where the first unit's surprise passes a
threshold (learned), or, as the control, every n characters. This is the
experiment of `docs/NEURON-CATALOG.md`, "Levels of text", with the
boundaries set by surprise.

### What is not new

**Our reading.** The memory is the gated delta rule (evidence 1; Yang,
Kautz & Hatamizadeh, 2024), with a decay per key row as in RWKV-7 (Peng et
al., 2025); the short convolution, the normalized keys and the output gate
are Gated DeltaNet's too. What may be new is the interface: context from
above in the query, the gates and the gain; surprise as an output;
boundaries from surprise; and the input formats for images. Most new units
lose to tuned standard ones, so this design is built to find out quickly
and cheaply which of these ideas earn their place.

## Dials

| Dial | Off (simpler) | On | Tests |
|---|---|---|---|
| Memory | no memory: a gated MLP | the delta-rule memory | T1 |
| Write | adds `k_tᵀ v_t` (Hebbian) | corrects the error (delta rule) | T1 |
| Decay | one number per step | one per key row | T1, T2 |
| Convolution | none | over the last 4 steps | T1 |
| Context | `c_t = 0` | from the level above | T4 |
| Division | none | divided by the pooled activity (σ learned): the read, and for images each pixel's channels | T3 |
| Phase | real memory | rotating memory (as complex retention) | T2 |
| Cell | SiLU | the founder's cells, from the catalog | T1, T3 |
| Boundaries | every n characters | where surprise passes a threshold | T4 |

## Tests, in order

Each against the plain unit (the delta-rule memory, with every other dial
off), the `gated-delta` model, a tuned Transformer of the same size and
TriNet, on the same harness, five seeds or more.

- **T1, recall:** the harder task, at Stage A's size (about 33,700
  parameters), on Kaggle's CPU sessions.
- **T2, length:** trained on texts of about 125 characters, tested at
  260 and 500.
- **T3, images:** CIFAR-10 at full size and with 1,000 and 5,000
  training images. Two formats of the same images: each pixel's red,
  green and blue kept as three channels with their own weights (the
  founder's format), and the usual mixed one. Then colour under
  brightness from 0.05 to 20 times.
- **T4, levels:** real text at the character level, one unit against
  two levels, with spans set by surprise or fixed; bits per character
  at the same compute.
- **T5, speed:** the chunked form on Kaggle's T4, against the step form
  (equal to it within rounding, tested) and the Transformer, as in
  "Speed on a GPU".

## Predictions, written before any code

**Our proposal**, each to be checked:

1. On T1, unit 0 is within the noise of `gated-delta` (as the fair
   comparison's predictions define it): its memory is the same rule.
2. With the Hebbian write it scores more than 10 points below the delta
   rule. (Fairly confident: a sum forgets an old value only by fading
   everything, so it trades the overwritten values against the old
   ones.)
3. A decay per key row and the convolution are each within the noise on
   T1. (Weakly held; the fair comparison's `trinet-no-delay` bears on
   the second.)
4. On T2, unit 0 loses at most 3 points from 125 to 260 characters, as
   TriNet does. With the phase dial on, it loses more, as complex
   retention did.
5. On T3's brightness test, with each pixel's channels divided by their
   pooled activity, the accuracy changes by less than 2 points from 0.05
   to 20 times; with no division and no other normalization of the
   input, it falls by more than 20 points at the extremes. (A sanity
   check: the division cancels a common scale, up to the small σ.)
6. On T3, the founder's format does better than the mixed one with
   1,000 training images, by more than the spread over seeds. (About
   even chances.)
7. On T4, spans set by surprise give fewer bits per character than fixed
   spans at the same compute. (About even chances.)
8. On T5, at the `1m` size and 512 characters, the chunked form trains
   within 3 times the Transformer's characters a second, where the
   Transformer trained 86 times faster than TriNet's loop.

## Guard rails

- Every unit is compared with the plain unit, on the same harness and
  seeds.
- No speed work beyond the chunked form until a unit wins on T1 to T4.
- Predictions are written before results, and results that go against
  them are kept and reported.
- TriNet stays one option among others: it is an entry in every
  comparison, not the starting point of this design.
- What the work is for, its names, and anything published stay the
  founder's decisions.

## Order of work

1. The fair comparison's results, which may add dials (for example, if
   the receptor bus is shown to help).
2. Unit 0 in PyTorch, one step at a time, with tests: causal, the cache
   equal to a full pass, each dial's "off" equal to the simpler unit.
3. T1 and T2 on Kaggle's CPU sessions.
4. The chunked form, tested equal to the step form, then T5.
5. T3 and T4.

The code will live in `unit/` in this repository, beside TriNet, so it
can use the same tasks, scoring and Kaggle scripts. Whether it moves to
a repository of its own, and under what name, is the founder's decision.

## Sources

- Carandini, M., & Heeger, D. J. (2012). Normalization as a canonical
  neural computation. *Nature Reviews Neuroscience*, 13(1), 51–62.
- Dauphin, Y. N., Fan, A., Auli, M., & Grangier, D. (2017). Language
  modeling with gated convolutional networks. *ICML 2017*.
  arXiv:1612.08083.
- Hooker, S. (2021). The hardware lottery. *Communications of the ACM*,
  64(12), 58–65. arXiv:2009.06489.
- Larkum, M. (2013). A cellular mechanism for cortical associations: an
  organizing principle for the cerebral cortex. *Trends in
  Neurosciences*, 36, 141–151.
- Peng, B., et al. (2025). RWKV-7 "Goose" with expressive dynamic state
  evolution. arXiv:2503.14456.
- Press, O., Smith, N. A., & Lewis, M. (2022). Train short, test long:
  attention with linear biases enables input length extrapolation.
  *ICLR 2022*. arXiv:2108.12409.
- Schlag, I., Irie, K., & Schmidhuber, J. (2021). Linear Transformers
  are secretly fast weight programmers. *ICML 2021*. arXiv:2102.11174.
- Shazeer, N. (2020). GLU variants improve Transformer. arXiv:2002.05202.
- Widrow, B., & Hoff, M. E. (1960). Adaptive switching circuits. *IRE
  WESCON Convention Record*, Part 4, 96–104.
- Yang, S., Wang, B., Zhang, Y., Shen, Y., & Kim, Y. (2024).
  Parallelizing linear Transformers with the delta rule over sequence
  length. *NeurIPS 2024*. arXiv:2406.06484.
- Yang, S., Kautz, J., & Hatamizadeh, A. (2024). Gated delta networks:
  improving Mamba2 with delta rule. arXiv:2412.06464.
