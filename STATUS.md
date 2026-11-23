# Status: TriNet

<!-- The working memory of this repository (handbook P1). Read it at the start
of every session; update it on your branch at the end. Keep it short and true. -->

**What this is:** a small language model built from the founder's three-channel neuron, packaged for Hugging Face; details in README.md
**State:** active since 2026-11-22
**Updated:** 2026-11-23 by Claude

## Live

<!-- What users get right now (handbook P6). -->
- Where: not deployed. Nothing is on Hugging Face.
- Version: none
- Last check: 2026-11-23, the 56 tests and Gemini's round-trip check pass with transformers 4.46.3 and 5.24.0

## Now

- Claude: six proposals of the founder's in the catalog: signals like light, amplitude and phase, ports and receptors, the 5-HT2A receptor, the canyon, the connection ledger (`docs/NEURON-CATALOG.md`, #9)
- Claude: complex retention, the first test of complex signals (branch
  `claude/complex-retention`; its pull request to come)

## Next

1. **A fair comparison**, the main line (Decided, 2026-11-23): what
   every other proposal is measured against. On the harder task
   (`experiments/harder_task.py`), where Stage A found the seeds differ
   more than the models, with more seeds and a fixed number of threads:
   a tuned Transformer of the same size, including one without learned
   positions; a plain gated delta-rule memory (Gated DeltaNet) of the
   same size; and TriNet with its own parts removed one at a time (the
   receptor bus, the yellow gates, the shunt, the delays), to show what
   they add. Then the hybrid of `experiments/README.md` ("What Stage A
   shows"), a TriNet and a Transformer with an executive network between
   them, once a Transformer learns the task.
2. **Teach the in-context task under the training script.** Trained on
   the answers, the model learns the task with change 6 completely (100%,
   two seeds, on 2 threads; 98.2 to 100% in Stage A's three seeds on 1
   thread; `experiments/README.md`). Under `train_trinet.py`, which
   trains on every next character with half the batches from the task, it
   learns it only in part: 34% in one seed of three, near chance in the
   others, and the Transformer beside it does better in five runs of six.
   (Before change 6 it seemed to do better, 99.7% at seed 0, but much of
   that was the shortcut.) Find what decides it (weighting the answers in
   the loss, training longer, more seeds) and measure.
3. A real training run (the `1m` preset on TinyStories) on Kaggle's free
   GPUs (Decided, 2026-11-23), after Next 1 (the technical lead's order);
   first a short test of TriNet's speed there. The founder's Kaggle token
   is among this environment's network secrets (never in the
   repository); on 2026-11-23 a call that needs it returned the account's
   GPU quota, 30 hours a week. Kaggle's own command-line tool cannot sign
   in with a token held this way (its check sends the token in the
   request body), so the calls go to Kaggle's API directly.
4. This repository's own model card, from what has been measured here.
5. Publishing on Hugging Face, when the founder decides (P5).

Beside these, not after them (Decided, 2026-11-23): the general cell, by
the order of work in `docs/NEURON-CATALOG.md` ("The general cell").
Items 1 and 2 test what TriNet already is, and their results are what the
general cell is compared against. Also beside them (Decided, 2026-11-23):
as the first test of complex signals, complex retention, a fast memory
that fades and rotates, so a trace's phase tells its age (in its own pull
request); then the canyon and the connection ledger as one experiment
(`docs/NEURON-CATALOG.md`, "The connection ledger"), the first step
toward temporary weights that become lasting ones. Set aside for now,
each kept in the catalog with its test: optical hardware, the 5-HT2A
dial, new port types, the holographic layer, and the receptor bus's
colour signal written as a complex number.

## Before publishing

The founder's word (Decided, 2026-11-22): publishing comes last, and every
change moves toward it. What stands between this repository and a model
on Hugging Face:

- [x] A task without the fixed-roles shortcut (change 6, #3).
- [ ] The comparisons of Next 1, so the model card says honestly what
      TriNet adds.
- [ ] A model trained on real text (Next 3), judged on text it was not
      trained on (see Noticed: validation).
- [ ] `app.py` made safe for a public Space (see Noticed).
- [ ] This repository's own model card, from measured results only
      (Next 4).
- [ ] The founder's decision on where and under which account (P5).

## Open questions

- For the founder: under which account a model would be published. Not
  needed until Next item 5. (asked 2026-11-22; where GPU runs happen was
  answered on 2026-11-23, Decided)

## Decided

- 2026-11-23, founder, on GPU training, after asking the technical lead
  what it needs and costs: "If you could provide a list of options at
  different price levels, and the benefits associated and I will get the
  funding for the GPU training. But let's waiting until we know we have
  something worth taking to the next development stage. Until then, let's
  maximize the resources at our disposal." Then, from the technical lead's
  list: "Ok, Kaggle it is. 30 GPU-hours per week is plenty if we utilize
  that time well." And: "I have a Kaggle account." Hence Kaggle's free
  GPUs, under the founder's own Kaggle account, are a resource to use from
  now on, starting with a test of TriNet's speed there; paid GPUs wait
  until there is something worth taking to the next stage, and for the
  founder's funding. That Next 3's training on real text comes after Next
  1 is the technical lead's order of work, not the founder's words: it is
  to show what TriNet adds before a long run is spent on it.
- 2026-11-23, founder, after offering the day's proposals for
  discussion: "Honestly, my goal was to throw out so many ideas that we
  could move in any of a variety of directions and then let you decide
  what to do from there. … What to work on next is up to you. I mention
  the 5-HT2A receptor because I have always been curious, but it may have
  nothing to do with our work. And that goes for the other ideas. But if
  something of interest strikes you as worth further development, let's
  move in that direction. Thoughts on where to move forward next?"
  Hence, decided by the technical lead (Claude)
  the same day: the fair comparison becomes the main line (Next 1, moved
  up from 2), since it is what every other proposal is measured against;
  the canyon and the connection ledger become one experiment, after
  complex retention (the question asked here earlier that day, now
  closed); optical hardware, the 5-HT2A dial, new port types and the
  holographic layer and the receptor bus's colour signal written as a
  complex number are set aside for now, each kept in
  `docs/NEURON-CATALOG.md` with its test. Of the hologram simulation
  agreed below ("Yes to both"), the first step is done (#8) and the
  second, the holographic layer inside TriNet, is the one set aside.
  Under which account a model is published stays the founder's (Open
  questions); where GPU runs happen was answered later that day (above).
- 2026-11-23, founder, answering two questions: does the hologram
  simulation run beside the items under "Next", and should the first test
  of complex signals be complex retention (a fast memory that fades and
  rotates, so a trace's phase tells its age)? "Yes to both, alongside is
  fine." Hence `experiments/holography.py` and the paragraph under Next.
- 2026-11-23, founder, answering the two questions then open (may
  passages of the stage texts 003 to 005 on "levels of magnification" be
  quoted; does the general-cell work come before Next, or beside it):
  "Yes, quote them, and alongside is fine." Hence the quoted passages in
  `docs/NEURON-CATALOG.md` ("Levels of magnification"), the stage texts
  themselves staying private, and the paragraph under Next. The same
  day, on holographic memory: "I do not just mean that as a metaphor, I
  would like to investigate how two-dimensional wave interference
  patterns store three-dimensional information", and "I would like to
  look at reading and writing holographic information." Offered for
  discussion (the first ends "Thoughts?", and the second restates it):
  hence "Writing and reading a hologram"
  there, with a simulation proposed, not decided.
- 2026-11-23, founder, on where TriNet is going: "first we should catalog
  the existing known types of neurons in the human brain. … My thought is
  to map the differences in shape and location and configuration to the
  known functions of the brain in certain regions. By doing so, I would
  hope to understand how the brain adapts the neuron to the system in
  which it is functioning." And: "The end goal is a general neural stem
  cell with all the dials and switches needed to change parameters."
  Hence `docs/NEURON-CATALOG.md` (#5). The founder's further proposals of
  that day, each offered for discussion with "Thoughts?", are quoted
  there in their own sections, not recorded here as decisions:
  connections that fade and come back, a cell that chooses, dials,
  switches and buttons, two levels of learning, shared-benefit
  optimization, and fractal design with the Transformer as a black box.
- 2026-11-22, founder, on credit: "Our records should allow others to
  recreate our results, in theory. That is the goal." On quoting ChatGPT:
  "Yes, feel free to quote ChatGPT, as they are aware of the public nature
  of the project, having been there since the start." And: "Credit in
  terms of final product, but we don't have to get into credit in terms of
  idea origin. … only authorship credit in terms of who wrote what code or
  what part for the repo. … And if a quote is already being reproduced in
  the records, then provide the source of the quote." Hence
  `CONTRIBUTIONS.md` (who wrote what, and where quotations come from) and
  where each result table ran (`experiments/README.md`).
- 2026-11-22, founder: "Publishing comes last but should be an intention
  from the beginning. We have some work to do before publishing the model,
  but I wanted to make sure we would always be moving in that direction."
  Hence the list "Before publishing" above.
- 2026-11-22, founder, on TriNet's dissolving weights (Gemini's record
  012-08-04, line 43649 of a later copy: 1,843,004 bytes, SHA-256
  `d3a9e7edfd8004433f935772234a706512e2d8c15dceaef8fe6c53d305449406`, not
  yet in open-worlds, whose copy ends earlier and differs from line
  43643): "I meant those to be a
  class of weights, short-term or temporary weights. Those would work in
  addition to and in conjunction with long-term weights that act as more
  stable default or baseline values or states. Think of Aristotle’s
  Nicomachean Ethics and the way character has to develop from a stable
  base of habitual behavior. Also consider Cybernetics." TriNet reads
  through `W_base + F_fast`, which is this; how temporary weights would
  become lasting ones (consolidation) is not built.
- 2026-11-22, founder: "I just created a new repo: TriNet, which is public
  without any license chosen." The licence is the founder's standing one,
  CC0 1.0 (handbook decision 0003). The repository was empty, so the
  founder chose that Claude push one commit to `main` to create it, the
  licence alone; everything after comes through pull requests.

## Noticed

- `attention_mask` is accepted and ignored (`modeling_trinet.py`), so left
  padding feeds pad tokens into the recurrence: a padded batch differs from
  the same texts run alone. Skip the state update where the mask is 0, or
  say that padded batches are not supported. (Independent review of #1.)
- On transformers 5.x, `TriNetCache` leaves its base class not set up, so
  `repr`, `batch_size` and the like on it fail; and on both versions a
  `TriNetCache` made by hand cannot be passed to `generate()`. Generation
  itself works. (Independent review of #1.)
- **Before any Space:** `app.py` passes whatever repository or path a
  visitor types to `from_pretrained(trust_remote_code=True)`, which on a
  public Space would run code from any repository; and its cache of models
  grows without bound. Load only `TRINET_REPO_ID`, at a pinned revision.
  (Independent review of #1.)
- With `--smoke-test` or `--local-text`, `train_trinet.py` validates on the
  same text it trains on, so its BPB is training BPB. (Independent review
  of #1.)
- A weight missing from a checkpoint: since change 5, a linear,
  convolution, embedding or norm weight starts from PyTorch's own values
  (tested). Still not set, as in Gemini's code: `TriNetLayer`'s own
  parameters (`W_base`, `shunt_w`, `affinity_logits`, `log_sigma`,
  `log_g0`, `tonic_decay_logit`), which load as whatever memory held. A
  missing gate bias gets PyTorch's value on transformers 5.x and 0.25 or
  0.90 on 4.46. Only matters for a checkpoint that lacks weights.
  (Independent review of #2.)
- `provenance/MANIFEST.json` gives the record 43,646 lines: the pieces
  from splitting on newlines. `wc -l` gives 43,645, as the file ends with a
  newline. Say which count it is, in a later change. (Independent review of
  #4.)
- `config.initializer_range` no longer sets any weight (change 5). Say so
  in the config, or drop it, in a later change. (Independent review of #2.)
- Gemini's Colab script (conversation lines 42767 to 43561) did not come
  through whole: its code lost its indentation and the underscores of
  names like `__init__`, and it begins partway in. It is not here; its
  steps (bundle, train, check, push) can be done with the files that are.
- Gemini's draft model card (`provenance/model-card-draft.md`) lost some
  of its formatting in the same way, and repeats results that have not
  been repeated here.

## Log

<!-- Newest first. One line per session: date, who, what, pull request. -->
- 2026-11-23 Claude: six proposals of the founder's in the catalog, each quoted and offered for discussion, with what is established, what is theory, and a test: signals like light (optoelectronics, optical multiplexing, colour television as the example), amplitude and phase (complex signals), ports and receptors, the 5-HT2A receptor, the canyon (learning carves paths; William James, 1890), and the connection ledger (with glia); 37 sources added; at the founder's word, what comes next decided by the technical lead: the fair comparison first, the canyon and the ledger as one experiment after complex retention, five proposals set aside; the founder's word on GPU training (Kaggle, when the CPU experiments are worth it); after independent review, a training claim, where 5-HT2A sits, a theory's label and what the receptor bus feeds corrected (#9)
- 2026-11-23 Claude: at the founder's word, the hologram simulation (`experiments/holography.py`, 10 tests): each of three points comes back at its own depth with a reference beam, and not in a control without one; a quarter of the plate shows every point, about twice as wide and a sixteenth as bright; as a memory, a hologram's read-out falls as 1/sqrt(n) (1/sqrt(2n) recorded on a plate) but finds the memory asked for up to 128 memories in 4,096 numbers, where 64 x 64 matrices read more cleanly, the one-pass delta rule favouring the newest; keys at different angles read back exactly, recorded or not; after independent review, a fair control, plates recorded in intensity, padding and more trials (#8)
- 2026-11-23 Claude: Stage A: Gemini's BicameralTriNet (record 012-10-06) kept as given in `bicameral/`, a TriNet of two stacks joined by a callosum, measured against one TriNet of the same size on two tasks, three seeds each: on the original task every TriNet model scores 98.2 to 100%; on a harder task (`experiments/harder_task.py`) the difference between seeds is larger than between models, and only the pair without the callosum learned it in all three seeds; change 7 (`residual_error`, off by default) learns fastest when it learns; the number of threads alone moved a run by 1.8 points in the one case compared; after independent review, the cause of the difference from the earlier "100%" corrected (threads), the hybrid's unrecorded first check taken out, and two tests made stronger (#7)
- 2026-11-23 Claude: at the founder's word, the stage texts' passages on levels of magnification quoted with stage file and line, and read as a specification for the fractal network; memory spread through the network (holographic memory and coded backups, the founder's proposal); writing and reading a hologram (the optics, Fourier's part, what a hologram holds, and a simulation proposed); 18 sources added; the two review notes of #5 taken (#6)
- 2026-11-23 Claude: at the founder's word, a catalog of ten neuron families (shape, place, wiring, function, how sure, nearest TriNet part), five functions TriNet lacks as candidate dials, and the founder's proposals for a general cell (dials, switches and buttons; two levels of learning; connections that fade and come back; a cell that chooses; shared-benefit optimization; fractal design), with an order of work; after independent review, the census counts, the archetype numbering, the labels and five other claims corrected (#5)
- 2026-11-22 Claude: at the founder's word on credit, `CONTRIBUTIONS.md` (who wrote each part, and the records quotations come from); ChatGPT named as a reviewer where their review led to a change, and quoted, with the source, for the five differences from linear attention; the machine, versions and threads for every result table (#4)
- 2026-11-22 Claude: the pointer shortcut found and removed (change 6, `experiments/pointer_check.py`): trained on Gemini's task, both TriNet models answered pointers with the first value bound; every result run again on the fair task (answers only: TriNet 100%; whole-text training: 34% at best); README claims corrected; the founder's publishing word and weight-class intent recorded; the list "Before publishing" (#3)
- 2026-11-22 Claude: the packaged model learns its in-context task once it starts from PyTorch's own initial values (change 5; 100% against 78.6 to 92.5%); Gemini's prototypes kept and checked; `experiments/answers_only.py` and its results; after independent review, the answers scored at the `>`, missing weights given PyTorch's values, and the results re-run (#2)
- 2026-11-22 Claude: the repository set up from Gemini's conversation: the model, training, the app, provenance, 15 tests, CI; after independent review, a fourth change (generate() used no cache on transformers 5.x) and stronger tests (#1)
