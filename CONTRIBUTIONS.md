# Who wrote what

This page records who wrote what, not where ideas came from (the
founder's rule, `STATUS.md` "Decided").

| Part | Written by |
|---|---|
| `trinet/`, `train_trinet.py`, `verify_hf_roundtrip.py`, `app.py`, `requirements.txt`, `provenance/model-card-draft.md`, the five `.py` files in `prototypes/` | Gemini, in their conversation with the founder (record 012-08-04). `provenance/MANIFEST.json` gives each file's lines in the record and its hash. Claude made six changes to these files; each is listed in the manifest with its reason. |
| `tests/`, `experiments/`, `provenance/MANIFEST.json`, `.github/workflows/ci.yml`, `pytest.ini`, `.gitignore` | Claude, technical lead |
| `README.md`, `STATUS.md`, `AGENTS.md`, the READMEs in `experiments/` and `prototypes/`, `docs/`, and this page | Claude, technical lead |
| `CONTRIBUTING.md`, `SECURITY.md`, `CLAUDE.md`, and `.github/` apart from `ci.yml` | Unchanged copies of the policy handbook's files (handbook P9), which Claude wrote |
| `LICENSE` | Creative Commons' CC0 1.0 text, added by Claude as the founder's standing licence (handbook decision 0003) |

**Reviews.** A fresh Claude session reviewed each pull request (handbook
P4). ChatGPT also reviewed this repository and Gemini's record, in their
own conversation with the founder (record 012-09-05). Where a review led
to a change, the change says so.

## Where quotations come from

Quotations from a record give the record and line. The founder's words
to Claude, which no record holds, give their date, as `STATUS.md`
("Decided") does. The records, by size and SHA-256:

- **Gemini 012-08-04,** kept in
  [open-worlds](https://github.com/jasonelias144-svg/open-worlds/blob/main/sketches/records/012-08-04-GMNI-Patterns_Multiverses.txt):
  1,829,108 bytes,
  `52ae3fc6629d6e77f3e605b06e5e24cdbbe8ff611aedc14929b196573fa1eddf`.
- **Gemini 012-08-04, a later copy,** not yet in open-worlds. It differs
  from the copy there from line 43643: 1,843,004 bytes,
  `d3a9e7edfd8004433f935772234a706512e2d8c15dceaef8fe6c53d305449406`.
- **ChatGPT 012-09-05,** kept privately; quoted with the founder's consent
  (2026-11-22): 1,117,467 bytes,
  `12e690d4540a1184c18da81618a613d7cd8f8d468ceee798a42bbad83a41e188`.
  (open-worlds lists an earlier, shorter copy; the line numbers here are
  of this one.)
- **The stage texts 003, 004 and 005,** kept privately in `ebook-press`
  (`history/`), as kept there (a personal name and account links
  replaced; every line where it was); quoted in
  `docs/NEURON-CATALOG.md` with the founder's consent (2026-11-23).
  `003_Language.txt`: 15,770,840 bytes,
  `40a8ecb802bd539df9196d1034343487100e51715cb44daafb2e05c823458cac`.
  `004_Pattern.txt`: 13,887,988 bytes,
  `5dcb422d85b9ce9ff7e8aef43a3ef4f661f2dee39770e3005e681f3b803f8a8c`.
  `005_Method.txt`: 6,578,713 bytes,
  `a9b9add06efcb670043cbc5328410957199ac91c473b50704429e225913b7e41`.
- **The dialogue on shared-benefit optimization,** public in
  [open-worlds](https://github.com/jasonelias144-svg/open-worlds/blob/main/library/008-SBO-CGPT-GROK-CLDE-Shared_Benefit_Optimization.txt):
  397,885 bytes,
  `d235ad9062af9c47a5b595f0c5a55c5c2424e257376fd8b8238eb20ffdf26507`.

## Recreating the results

- The commands are in `experiments/README.md`. That page also gives the
  machine, the software versions and the number of threads each result
  table ran with. PyTorch does not promise the same numbers on a
  different number of threads, so use the number given.
- `tests/test_provenance.py` checks Gemini's files byte for byte against
  the record, once the listed changes are undone.
