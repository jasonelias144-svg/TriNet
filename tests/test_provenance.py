"""Ours, not Gemini's. Every file taken from the conversation is what Gemini
gave, apart from the changes listed in provenance/MANIFEST.json.

Each change is undone, and the result must hash to what the conversation
holds. So a change made here and not listed fails this test, and so does a
listed change that was never made.
"""
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
MANIFEST = json.loads((ROOT / "provenance" / "MANIFEST.json").read_text(encoding="utf-8"))


def as_given(name):
    text = (ROOT / name).read_bytes().decode("utf-8")  # no newline translation
    for change in reversed(MANIFEST["changes"]):
        if change["file"] != name:
            continue
        assert text.count(change["new"]) == change["count"], (name, change["why"])
        text = text.replace(change["new"], change["old"])
    return text.encode("utf-8")


def test_every_file_is_what_gemini_gave_but_the_listed_changes():
    for name, entry in MANIFEST["files"].items():
        digest = hashlib.sha256(as_given(name)).hexdigest()
        assert digest == entry["sha256"], name


def test_every_change_names_a_file_taken_from_the_conversation():
    for change in MANIFEST["changes"]:
        assert change["file"] in MANIFEST["files"]
        assert change["why"].strip()
