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
- Last check: 2026-11-22, the 15 tests and Gemini's round-trip check pass with transformers 4.46.3 and 5.24.0

## Now

- Claude: the repository set up, with Gemini's code as given and four changes (#1)

## Next

1. **Find why the packaged model learns less than Gemini's prototype.**
   On the in-context task of `train_trinet.py` (bind three values, follow a
   pointer, overwrite one, filler, four queries), trained on the answers
   only, 1,500 steps, two seeds each, on 2026-11-22:
   - Gemini's prototype `tri_lm.py` (18,358 parameters): 100% at the
     trained length (about 52 characters), and 99.9 to 100% at about 100
     and 200 characters. A small Transformer (39,322 parameters, ours,
     untuned): 55 to 77%.
   - `TriNetForCausalLM` here (33,748 and 59,708 parameters): about 70%,
     the same at all three lengths. Letting its gates see the local
     context, as the prototype's do, did not change that.
   - With `train_trinet.py` as written (smallest preset, 2,000 steps, half
     the batches on that task, seed 42; the second seed did not finish in
     its time limit): TriNet at chance, 20.3% with 6 characters of filler
     and 14.6% with 24; the Transformer beside it, 20% larger, 48.4% and
     18.0%. Command: `python train_trinet.py --preset smoke --smoke-test
     --seq-len 96 --steps 2000 --warmup-steps 100 --reasoning-mix 0.5
     --log-every 500 --save-every 0 --seed 42`.
   The prototype and the scripts that trained it on the answers only are
   to be added here in the next pull request, so that these can be
   repeated; until then they are noted, not claimed.
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
- Gemini's Colab script (conversation lines 42767 to 43561) did not come
  through whole: its code lost its indentation and the underscores of
  names like `__init__`, and it begins partway in. It is not here; its
  steps (bundle, train, check, push) can be done with the files that are.
- Gemini's draft model card (`provenance/model-card-draft.md`) lost some
  of its formatting in the same way, and repeats results that have not
  been repeated here.

## Log

<!-- Newest first. One line per session: date, who, what, pull request. -->
- 2026-11-22 Claude: the repository set up from Gemini's conversation: the model, training, the app, provenance, 15 tests, CI; after independent review, a fourth change (generate() used no cache on transformers 5.x) and stronger tests (#1)
