# Gemini's prototypes

Five files Gemini wrote on the way from the founder's idea to the model in
`trinet/`, kept as the conversation gives them
(`provenance/MANIFEST.json` gives each one's lines and hash). Nothing in
`trinet/` imports them; they are here so that what Gemini reported of them
can be checked.

| File | Conversation line | What it is |
|---|---|---|
| `tri_net.py` | 36703 | `TriColumn`, in NumPy: red drive through baseline and dissolving fast weights, blue shunting, yellow from a mismatch with an expectation from above |
| `cell_library_ext.py` | 36871 | Two more kinds of unit, in NumPy: `ReceptorBroadcastUnit` (reads a shared three-colour field by its hue) and `DelayLineUnit` (exact order and timing; timed cancellation) |
| `micro_brain.py` | 37058 | `MicroBrain`, in NumPy: the six kinds of unit wired into four modules |
| `meta_micro_brain.py` | 37298 | `MetaMicroBrain`, in PyTorch: the same, trained from random starting values |
| `tri_lm.py` | 37486 | `TriLanguageModel`, in PyTorch: two layers of the units as a small character model |

## What has been checked

- **Sizes.** `MetaMicroBrain` has 928 parameters and `TriLanguageModel`
  18,098, as Gemini says (`tests/test_prototypes.py`).
- **A delay line tells order and interval.** Set for "A, then B three
  ticks later", it fires for exactly that, and not for B before A, too
  fast, too slow, both at once, or A alone (Gemini's benchmark 6A;
  `tests/test_prototypes.py`).
- **One receptor picks out purple.** A single `ReceptorBroadcastUnit`
  tuned to purple (red and blue) fires for purple and for none of the
  other six hues, at 0.05, 1 and 20 times the intensity (Gemini's
  benchmark 5B; `tests/test_prototypes.py`).
- **`TriLanguageModel` learns to bind, overwrite and follow a pointer, and
  keeps it on longer texts.** Repeated here with
  `experiments/answers_only.py tri_lm`, with more training steps than
  Gemini reports: `experiments/README.md`.

Not checked: Gemini's other reported results for these files (the
"crucible" of the micro-brain, the comparisons with GRUs). The code that
produced them is not in the conversation.
