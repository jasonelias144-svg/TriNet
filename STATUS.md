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
- Last check: 2026-11-22, the tests and Gemini's round-trip check pass with transformers 4.46.3 and 5.24.0

## Now

- Claude: the repository set up, with Gemini's code as given and three changes (#1)

## Next

1. **Repeat one of Gemini's reported results here, with code in this
   repository**, before anything is published. The first to try: the
   in-context tasks of `train_trinet.py` (bind, overwrite, two hops),
   TriNet beside the Transformer of the same size, more than one seed. On
   2026-11-22, at the smallest preset and 2,000 steps, TriNet stayed at
   chance on them and the Transformer did better (#1).
2. A real training run (the `1m` preset on TinyStories) on a GPU, with the
   founder's choice of where it runs.
3. This repository's own model card, from what has been measured here.
4. Publishing on Hugging Face, when the founder decides (P5).

## Open questions

- For the founder: where a GPU training run should happen (Colab, a Hugging
  Face Space, or elsewhere), and under which account a model would be
  published. Not needed until Next item 2. (asked 2026-11-22)

## Decided

- 2026-11-22, founder: "I just created a new repo: TriNet, which is public
  without any license chosen." The licence is the founder's standing one,
  CC0 1.0 (handbook decision 0003). The repository was empty, so the
  founder chose that Claude push one commit to `main` to create it, the
  licence alone; everything after comes through pull requests.

## Noticed

- Gemini's Colab script (conversation lines 42767 to 43561) did not come
  through whole: its code lost its indentation and the underscores of
  names like `__init__`, and it begins partway in. It is not here; its
  steps (bundle, train, check, push) can be done with the files that are.
- Gemini's draft model card (`provenance/model-card-draft.md`) lost some
  of its formatting in the same way, and repeats results that have not
  been repeated here.

## Log

<!-- Newest first. One line per session: date, who, what, pull request. -->
- 2026-11-22 Claude: the repository set up from Gemini's conversation: the model, training, the app, provenance, 11 tests, CI (#1)
