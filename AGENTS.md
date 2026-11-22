# Working in TriNet

This repository follows the handbook at
https://github.com/jasonelias144-svg/policy-handbook. Read its orientation (README.md)
before your first session here. The rules below add to the handbook; they never
loosen it.

## Every session

1. Read `STATUS.md` and the open pull requests (handbook P1).
2. Work on a branch; never push to `main` (P3).
3. End by pushing, updating `STATUS.md` on your branch, and opening or updating
   the pull request (P1).

## Who decides

The founder decides what this project is for and who works on it. The technical
lead (Claude) decides everything about the code, and records an approval on
every pull request before it merges (P4). Intent questions go to the founder (P5).

## This repository

- **What it is:** TriNet, a small language model built from the founder's
  three-channel neuron (red drive, blue shunting, yellow modulation of a
  dissolving fast memory), packaged as a Hugging Face custom model.
- **Run:** `python train_trinet.py --preset smoke --smoke-test --seq-len 72 --steps 100`
- **Test:** `python -m pytest -q` and `python verify_hf_roundtrip.py` (both
  must pass before a pull request is marked ready; CI runs them with
  transformers 4.46.3 and 5.24.0)
- **Other checks:** none
- **Merge method:** squash
- **Release:** none yet. Publishing a model or a Space on Hugging Face is
  the founder's decision, on the founder's account (P5); the check that it
  worked is loading it back with `AutoModelForCausalLM.from_pretrained(...,
  trust_remote_code=True)` in a fresh environment.
- **Data:** none kept here. Training data is downloaded at run time
  (TinyStories) or given as a local file; checkpoints are not committed.

## Rules specific to this repository

- **Say where code came from.** Files taken from a collaborator's
  conversation are listed in `provenance/MANIFEST.json` with their lines and
  hash as given, and every change made to them is listed there with its
  reason. `tests/test_provenance.py` fails on a change that is not listed.
- **Claim only what can be repeated.** A result goes into the README or a
  model card only with the command and the code that produce it, in this
  repository. Results reported from elsewhere are named as such.
- **Never commit a token or a checkpoint.** The Hugging Face token lives
  only in the environment (`HF_TOKEN`) (P8).

## Always

- Say who made each change: `Made-by: <name> (AI|human)` in commits, and the
  "Made by" field in pull requests (P3).
- No secrets or personal data anywhere in the repository (P8).
- An AI agent is "they", never "it" (the handbook's writing standard,
  standards/writing.md; decision 0006).
- Report outcomes honestly: what you checked, and what you did not.
