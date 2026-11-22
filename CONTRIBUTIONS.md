# Who contributed what

The founder's rule (2026-11-22, `STATUS.md` "Decided"): name each source
for exactly what they contributed, and keep records from which anyone
could, in theory, recreate the results. This page does that for TriNet.
Each entry says where to check it: a line of a conversation record, or a
pull request here.

## The records cited

- **Gemini 012-08-04:** the founder's conversation with Gemini in which
  TriNet was designed and written. Public, kept in
  [open-worlds](https://github.com/jasonelias144-svg/open-worlds/blob/main/sketches/records/012-08-04-GMNI-Patterns_Multiverses.txt):
  1,829,108 bytes, 43,646 lines, SHA-256
  `52ae3fc6629d6e77f3e605b06e5e24cdbbe8ff611aedc14929b196573fa1eddf`.
- **Gemini 012-08-04, later copy:** the same conversation, carried
  further. Not yet in open-worlds. It differs from the copy there from
  line 43643: 1,843,004 bytes, 43,722 lines, SHA-256
  `d3a9e7edfd8004433f935772234a706512e2d8c15dceaef8fe6c53d305449406`.
- **ChatGPT 012-09-05:** the founder's conversation with ChatGPT, in which
  ChatGPT reviewed Gemini's record and this repository. It is kept
  privately; the passages below are quoted with the founder's consent
  (2026-11-22). 1,117,467 bytes, 17,203 lines, SHA-256
  `12e690d4540a1184c18da81618a613d7cd8f8d468ceee798a42bbad83a41e188`.
  (open-worlds lists an earlier, shorter copy; the line numbers here are
  of this one.)

## The founder

- **The idea.** A unit that sends several signals, as a neuron uses
  several neurotransmitters, instead of one number. Gemini 012-08-04, line
  35595: "imagine if we simplify to three neurotransmitter types, so three
  output channels from each neuron into the next, and each channel, per
  neuron, may have different weights, though possibly related. If we
  colored these neurotransmitters with red, blue, and yellow, we would see
  three colors from each neuron".
- **The aim:** a model shared on Hugging Face (line 37647; `README.md`).
- **Two classes of weights:** temporary weights working together with
  lasting ones (later copy, line 43649; said to ChatGPT too, 012-09-05,
  line 16622; the words are in `STATUS.md`). TriNet reads through
  `W_base + F_fast`, which is this.
- **Two modules,** synthetic and analytic, "sharing information in some
  central channel where decisions are made" (line 43562). This is the
  BicameralTriNet; it is not yet in this repository.
- **Publishing as the aim from the start, and its last step**
  (`STATUS.md`), and the rule this page follows.

## Gemini

- **The three-colour unit and the layer.** Answering the founder's
  question (from line 35598), Gemini gave the three colours different
  jobs: red drives the next unit, blue divides it (shunting), and yellow
  modulates a fast memory that dissolves. Gemini built these, with a
  receptor layer and a delay line, into the layer in
  `trinet/modeling_trinet.py`.
- **Every file listed in `provenance/MANIFEST.json`,** which gives each
  file's lines in the conversation and its hash:
  - the model and its configuration;
  - the training script, with its in-context task and the Transformer
    beside it;
  - the round-trip check;
  - the app;
  - a draft model card;
  - five prototypes.

  Six changes were made to them here, each listed with its reason in the
  manifest.
- **`tri_lm`, the prototype that learns the task** (line 37486). Repeated
  here: it binds, overwrites and follows a pointer, and keeps that at four
  times the length it was trained on (`experiments/README.md`).
- **The hybrid models to compare with:** Jamba, Griffin and Hymba (line
  43566). Gemini said such models "mostly stack" recurrent and attention
  layers; Hymba runs them side by side, as `README.md` says (since #1) and
  as ChatGPT also pointed out (012-09-05, line 16327).
- **Two versions of the BicameralTriNet,** in later records. They are not
  yet in this repository.

## ChatGPT

All of these are from 012-09-05. ChatGPT's review covered Gemini's whole
record and this repository as it stood during #1 (lines 16232 and 16244).

1. **The shortcut in the task.** Line 16522: "The current generator
   consistently points to the first direct variable and overwrites the
   second. Randomizing these choices would test a broader capability."
   - Here, `experiments/pointer_check.py` measured it. Trained on the task
     as given, both TriNet models answered pointer queries exactly as the
     rule "give the first value bound" does.
   - Change 6 removed it (#3).
2. **Earlier answers in the text.** Line 16519: "later queries receive
   earlier ground-truth answers, which can help with related questions.
   We should report independent-prefix or generated-answer results
   alongside that teacher-forced metric."
   - Here, `pointer_check.py` also scores the first query of each episode
     alone, which no earlier answer can help. It gives the same picture
     (#3).
3. **TriNet does not become attention by changing its settings.** Line
   16236: "the current TriNet implementation does not become a Transformer
   merely by changing the parameter settings described in that response".
   This corrects Gemini's answer (Gemini 012-08-04, line 43604) to the
   founder's question whether a TriNet can be set to behave like a
   Transformer (line 43601).
   - ChatGPT names five differences (lines 16368 to 16372):
     - the write uses a delta error;
     - blue's normalization depends on what is read;
     - the cache holds no key-mass state;
     - keys and queries can be negative;
     - the memory is read before it is written, and clipped.
   - Checked here against `trinet/modeling_trinet.py`: all five hold.
4. **How to test two branches.** Lines 16482 to 16500 set one question:
   "Does deliberate differentiation between two communicating TriNet
   branches improve historical recall, current-state tracking, and
   composition, compared with equally resourced homogeneous and symmetric
   controls?"
   - It is asked in two stages, with a symmetric pair as the essential
     control: "If two branches improve simply because parallel processing
     helps, that would still be useful—but it would support a different
     conclusion".
   - This is the plan given to Gemini for the BicameralTriNet on
     2026-11-22. It has not been run yet.
5. **Moving a temporary weight into a lasting one without counting it
   twice** (lines 16774 to 16790). With `W = B + F`, move a contribution
   `C` as `B' = B + C` and `F' = F - C`.
   - Used here to review Gemini's BicameralTriNet, whose consolidation
     counted twice.
   - Not built in this repository.
6. **Experiments on lasting learning:**
   - adaptation, recovery and durable learning (from line 16816);
   - a small experiment that keeps a learned change to the context it came
     from (from line 17003), with its code in ChatGPT's package 020.

   ChatGPT reported these. They have not been repeated here, and the
   package has not reached this repository.
7. **Also in their review:**
   - related work (lines 16325 and 16327);
   - the Transformer's larger size at the smallest preset (line 16513);
   - its learned positions on longer texts (line 16516);
   - validation on the training text (line 16525).

   The first, second and fourth were already in this repository from #1
   (`README.md` and `STATUS.md`), which ChatGPT's review read. The third
   was added here in #2 without our having seen their review, so it was
   found twice, independently.

## Claude, technical lead, and the independent reviewers

Claude set this repository up from the conversation, and keeps it. A fresh
Claude session that had no part in writing a change reviewed each one
(handbook P4).

- **#1:** the repository built from the conversation.
  - Changes 1 to 3 made Gemini's code run with the real transformers
    library (Gemini had tested against a stand-in of their own, line
    41318).
  - The tests, CI and the provenance record.
  - The independent review found change 4: on transformers 5.x,
    `generate()` read the whole text again at every step.
- **#2:**
  - Change 5, PyTorch's own initial values, with which the model learns
    its task.
  - `experiments/answers_only.py` and its results.
  - The independent review found two faults, both fixed
    (`experiments/README.md`, "Corrected after review").
- **#3:** change 6 and `experiments/pointer_check.py`, from ChatGPT's
  points 1 and 2, and every result run again.
- **Found independently of ChatGPT:** the comparison with Gated DeltaNet
  (`STATUS.md`, Next 2) and the Transformer's learned positions.

## Recreating the results

- The commands are in `experiments/README.md`. Each result table there
  gives the machine, the software versions and the number of threads it
  ran with. PyTorch does not promise the same numbers on a different
  number of threads, so use the number given.
- The code from Gemini is checked byte for byte against the record
  (`tests/test_provenance.py`).
- Asking a model again does not give the same conversation. The records
  are how what a conversation produced can be checked.
