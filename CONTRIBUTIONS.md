# Who wrote what

Who wrote each part of this repository. In this project, ideas move freely
and all of them come back to the founder, so this page does not track
where ideas came from. It records who wrote what (the founder's rule,
`STATUS.md` "Decided").

| Part | Written by |
|---|---|
| `trinet/`, `train_trinet.py`, `verify_hf_roundtrip.py`, `app.py`, `requirements.txt`, `provenance/model-card-draft.md`, `prototypes/` | Gemini, in their conversation with the founder (record 012-08-04). `provenance/MANIFEST.json` gives each file's lines in the record and its hash. Claude made six changes to these files; each is listed in the manifest with its reason. |
| `tests/`, `experiments/`, `provenance/MANIFEST.json`, `.github/`, `pytest.ini`, `.gitignore` | Claude, technical lead |
| `README.md`, `STATUS.md`, `AGENTS.md`, `CLAUDE.md`, `CONTRIBUTING.md`, `SECURITY.md`, the READMEs in `experiments/` and `prototypes/`, and this page | Claude, technical lead, following the policy handbook |
| `LICENSE` | CC0 1.0, the founder's standing licence (handbook decision 0003) |

**Reviews.** A fresh Claude session reviewed each pull request (handbook
P4). ChatGPT also reviewed this repository and Gemini's record, in their
own conversation with the founder (record 012-09-05). Where a review led
to a change, the change says so.

## Where quotations come from

Quotations in this repository give their record and line. The records:

- **Gemini 012-08-04,** kept in
  [open-worlds](https://github.com/jasonelias144-svg/open-worlds/blob/main/sketches/records/012-08-04-GMNI-Patterns_Multiverses.txt):
  1,829,108 bytes, 43,646 lines, SHA-256
  `52ae3fc6629d6e77f3e605b06e5e24cdbbe8ff611aedc14929b196573fa1eddf`.
- **Gemini 012-08-04, a later copy,** not yet in open-worlds. It differs
  from the copy there from line 43643: 1,843,004 bytes, 43,722 lines,
  SHA-256 `d3a9e7edfd8004433f935772234a706512e2d8c15dceaef8fe6c53d305449406`.
- **ChatGPT 012-09-05,** kept privately; quoted with the founder's consent
  (2026-11-22): 1,117,467 bytes, 17,203 lines, SHA-256
  `12e690d4540a1184c18da81618a613d7cd8f8d468ceee798a42bbad83a41e188`.
  (open-worlds lists an earlier, shorter copy; the line numbers here are
  of this one.)

## Recreating the results

- The commands are in `experiments/README.md`. That page also gives the
  machine, the software versions and the number of threads each result
  table ran with. PyTorch does not promise the same numbers on a
  different number of threads, so use the number given.
- `tests/test_provenance.py` checks Gemini's files byte for byte against
  the record, once the listed changes are undone.
