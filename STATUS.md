# Status: TriNet

<!-- The working memory of this repository (handbook P1). Read it at the start
of every session; update it on your branch at the end. Keep it short and true. -->

**What this is:** a small language model built from the founder's three-channel neuron, packaged for Hugging Face; details in README.md
**State:** active since 2026-11-22
**Updated:** 2026-11-22 by Claude

## Live

<!-- What users get right now (handbook P6). -->
- Where: not deployed. Nothing is on Hugging Face.
- Version: none
- Last check: 2026-11-22, the 33 tests and Gemini's round-trip check pass with transformers 4.46.3 and 5.24.0

## Now

- Claude: Gemini's prototypes, the experiments, and change 5 (the model's initial values) (#2)

## Next

1. **Teach the in-context task under the training script.** With change 5
   the model learns it when trained on the answers (100%, two seeds;
   `experiments/README.md`). Under `train_trinet.py`, which trains on
   every next character with half the batches from the task, it learned it
   in two runs of three (seed 0: 99.7 and 100%; seed 42: 67%; seed 1: at
   chance), and in none with Gemini's initial values. Find what decides it
   (more seeds, weighting the answers in the loss, training longer) and
   measure.
2. **A fair comparison.** A tuned Transformer of the same size, including
   one without learned positions, on the task of `experiments/` (ours is
   small and untuned).
3. A real training run (the `1m` preset on TinyStories) on a GPU, with the
   founder's choice of where it runs.
4. This repository's own model card, from what has been measured here.
5. Publishing on Hugging Face, when the founder decides (P5).

## Open questions

- For the founder: where a GPU training run should happen (Colab, a Hugging
  Face Space, or elsewhere), and under which account a model would be
  published. Not needed until Next item 3. (asked 2026-11-22)

## Decided

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
- 2026-11-22 Claude: the packaged model learns its in-context task once it starts from PyTorch's own initial values (change 5; 100% against 78.6 to 92.5%); Gemini's prototypes kept and checked; `experiments/answers_only.py` and its results; after independent review, the answers scored at the `>`, missing weights given PyTorch's values, and the results re-run (#2)
- 2026-11-22 Claude: the repository set up from Gemini's conversation: the model, training, the app, provenance, 15 tests, CI; after independent review, a fourth change (generate() used no cache on transformers 5.x) and stronger tests (#1)
