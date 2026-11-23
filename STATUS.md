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
- Last check: 2026-11-22, the 35 tests and Gemini's round-trip check pass with transformers 4.46.3 and 5.24.0

## Now

- Claude: a catalog of neuron types, toward a general cell (`docs/NEURON-CATALOG.md`, #5)

## Next

1. **Teach the in-context task under the training script.** Trained on
   the answers, the model learns the task with change 6 completely (100%,
   two seeds; `experiments/README.md`). Under `train_trinet.py`, which
   trains on every next character with half the batches from the task, it
   learns it only in part: 34% in one seed of three, near chance in the
   others, and the Transformer beside it does better in five runs of six.
   (Before change 6 it seemed to do better, 99.7% at seed 0, but much of
   that was the shortcut.) Find what decides it (weighting the answers in
   the loss, training longer, more seeds) and measure.
2. **A fair comparison.** A tuned Transformer of the same size, including
   one without learned positions, on the task of `experiments/` (ours is
   small and untuned); and a plain gated delta-rule memory (Gated DeltaNet)
   of the same size, with TriNet's own parts removed one at a time, to show
   what they add.
3. A real training run (the `1m` preset on TinyStories) on a GPU, with the
   founder's choice of where it runs.
4. This repository's own model card, from what has been measured here.
5. Publishing on Hugging Face, when the founder decides (P5).

## Before publishing

The founder's word (Decided, 2026-11-22): publishing comes last, and every
change moves toward it. What stands between this repository and a model
on Hugging Face:

- [x] A task without the fixed-roles shortcut (change 6, #3).
- [ ] The comparisons of Next 2, so the model card says honestly what
      TriNet adds.
- [ ] A model trained on real text (Next 3), judged on text it was not
      trained on (see Noticed: validation).
- [ ] `app.py` made safe for a public Space (see Noticed).
- [ ] This repository's own model card, from measured results only
      (Next 4).
- [ ] The founder's decision on where and under which account (P5).

## Open questions

- For the founder: may passages of the project's earlier stage texts
  (003, 004 and 005, kept privately in `ebook-press`), on "levels of
  magnification", be quoted in `docs/NEURON-CATALOG.md`, or only
  described, with the stage and line as the source? (The passage in 008
  is public already, in the open-worlds library.) The technical lead's
  recommendation: quote short passages, each with its stage and line,
  and keep the stage texts themselves private; they are private for the
  published translations they hold, not for these passages. A section
  on them follows the answer. (asked 2026-11-23)
- For the founder: does the work toward the general cell
  (`docs/NEURON-CATALOG.md`, "Order of work") come before the items under
  "Next", or run beside them? The technical lead's recommendation:
  beside. Next 1 and 2 test what TriNet already is, and their results
  are what the general cell is compared against. (asked 2026-11-23)
- For the founder: where a GPU training run should happen (Colab, a Hugging
  Face Space, or elsewhere), and under which account a model would be
  published. Not needed until Next item 3. (asked 2026-11-22)

## Decided

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
- 2026-11-23 Claude: at the founder's word, a catalog of ten neuron families (shape, place, wiring, function, how sure, nearest TriNet part), five functions TriNet lacks as candidate dials, and the founder's proposals for a general cell (dials, switches and buttons; two levels of learning; connections that fade and come back; a cell that chooses; shared-benefit optimization; fractal design), with an order of work; after independent review, the census counts, the archetype numbering, the labels and five other claims corrected (#5)
- 2026-11-22 Claude: at the founder's word on credit, `CONTRIBUTIONS.md` (who wrote each part, and the records quotations come from); ChatGPT named as a reviewer where their review led to a change, and quoted, with the source, for the five differences from linear attention; the machine, versions and threads for every result table (#4)
- 2026-11-22 Claude: the pointer shortcut found and removed (change 6, `experiments/pointer_check.py`): trained on Gemini's task, both TriNet models answered pointers with the first value bound; every result run again on the fair task (answers only: TriNet 100%; whole-text training: 34% at best); README claims corrected; the founder's publishing word and weight-class intent recorded; the list "Before publishing" (#3)
- 2026-11-22 Claude: the packaged model learns its in-context task once it starts from PyTorch's own initial values (change 5; 100% against 78.6 to 92.5%); Gemini's prototypes kept and checked; `experiments/answers_only.py` and its results; after independent review, the answers scored at the `>`, missing weights given PyTorch's values, and the results re-run (#2)
- 2026-11-22 Claude: the repository set up from Gemini's conversation: the model, training, the app, provenance, 15 tests, CI; after independent review, a fourth change (generate() used no cache on transformers 5.x) and stronger tests (#1)
