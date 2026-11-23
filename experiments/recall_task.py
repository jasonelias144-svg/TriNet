"""Ours, not Gemini's. A recall task: many keys, each bound once, then asked.

    kd=3;bo=7;ap=0;ni=5; ... ;.x -y?bo>7;?kd>3;?ni>5;...

Each key is two letters from "a" to "p" (256 possible keys), each value a
digit. Nothing is overwritten; the difficulty is the number of keys. A
memory that keeps every token (attention) can look each one up; a memory of
fixed size (TriNet's) holds only so many associations before they blur.

Query kind 5 ("recall"), after harder_task.py's kinds 1 to 4.
"""
from typing import List, Tuple

import numpy as np

KEY_CHARS = list("abcdefghijklmnop")
VALS = list("0123456789")
FILLER = list("xyz .,-")
KIND = 5
KINDS = {KIND: "recall"}


def make_episode(n_pairs: int, l_filler: int, rng: np.random.Generator,
                 n_queries: int = 8) -> Tuple[str, List[Tuple[int, int, int]], dict]:
    """One episode: its text, and for each query (position of its ">", the
    answer's character code, kind 5). The third value maps each key to its
    value, for the checks in tests/test_hybrid.py."""
    codes = rng.choice(len(KEY_CHARS) ** 2, size=n_pairs, replace=False)
    keys = [KEY_CHARS[c // len(KEY_CHARS)] + KEY_CHARS[c % len(KEY_CHARS)] for c in codes]
    values = {k: str(rng.choice(VALS)) for k in keys}
    text = list("".join(f"{k}={values[k]};" for k in keys))
    text += [str(rng.choice(FILLER)) for _ in range(l_filler)]
    asked = rng.choice(n_pairs, size=n_queries, replace=n_queries > n_pairs)
    meta = []
    for i in asked:
        key = keys[i]
        text += ["?", key[0], key[1], ">"]
        meta.append((len(text) - 1, ord(values[key]), KIND))
        text += [values[key], ";"]
    return "".join(text), meta, values
