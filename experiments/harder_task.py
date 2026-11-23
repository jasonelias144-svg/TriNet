"""Ours, not Gemini's. A harder in-context task, with room left above one TriNet.

On the task of answers_only.py, one TriNet answers 98.2 to 100% of the
queries (three seeds; experiments/README.md, Stage A), so a model built
from two TriNets has no room to show that it does better.
This task asks more of a memory:

    h=4;g=3;f=1;c=4;e=5;z-x,,-d=h;f=3; g=5;z.b=c;--e=0;xf=1;e=4;c=2; c,- f?b>4;?f>1;?c>2;?f<3;?h>4;?g<3;?e>4;?c<4;?d>4;?g>5;?e<0;

(`make_episode(6, 6, numpy.random.default_rng(4))`.)

- Five names are bound to values.
- Then, in random order, with a little filler between them:
  - two pointers, each a new name given an earlier name (`d=h`); the second
    may name the first, making a chain;
  - six overwrites of the first five names (`f=3`). One name can be
    overwritten several times, and some are not overwritten at all.
- Then the queries, in random order:
  - `?x>` asks for the current value of x;
  - `?x<` asks for the value x had before its last overwrite (only for names
    that were overwritten).

A pointer copies the value its target has at that moment: `b=c` sets b to
c's value then, 4, and `?b>4` asks for it, although c is overwritten later
(`c=2`). So neither "the target's value now" nor "the target's first
value" is always right; `shortcut_scores` measures how often each such rule
is.

Query kinds: 1 current value, never overwritten; 2 current value,
overwritten; 3 pointer; 4 value before the last overwrite.
"""
from typing import Dict, List, Optional, Tuple

import numpy as np

VARS = list("abcdefgh")
VALS = list("012345")
FILLER = list("xyz .,-")
KINDS = {1: "current, not overwritten", 2: "current, overwritten", 3: "pointer", 4: "before last overwrite"}


def _filler(n: int, rng: np.random.Generator) -> str:
    # As in train_trinet.py: stray names appear in the filler.
    return "".join(str(rng.choice(VARS)) if rng.random() < 0.18 else str(rng.choice(FILLER)) for _ in range(n))


def make_episode(
    l1: int,
    l2: int,
    rng: np.random.Generator,
    n_bound: int = 5,
    n_pointers: int = 2,
    n_overwrites: int = 6,
) -> Tuple[str, List[Tuple[int, int, int]], Dict]:
    """One episode: its text, and for each query (position of its ">" or
    "<", the answer's character code, kind). The third value records what
    happened, for the checks in tests/test_stage_a.py."""
    names = [str(v) for v in rng.choice(VARS, size=n_bound + n_pointers, replace=False)]
    bound, pointers = names[:n_bound], names[n_bound:]

    value: Dict[str, str] = {}
    history: Dict[str, List[str]] = {}
    parts: List[str] = []
    for v in bound:
        value[v] = str(rng.choice(VALS))
        history[v] = [value[v]]
        parts.append(f"{v}={value[v]};")
    parts.append(_filler(l1, rng))

    events = ["p"] * n_pointers + ["o"] * n_overwrites
    rng.shuffle(events)
    made: List[str] = []  # pointers made so far
    pointer_log: Dict[str, Tuple[str, str]] = {}  # pointer -> (target, value copied)
    for e in events:
        if e == "o":
            v = bound[int(rng.integers(n_bound))]
            new = str(rng.choice([d for d in VALS if d != value[v]]))
            value[v] = new
            history[v].append(new)
            parts.append(f"{v}={new};")
        else:
            p = pointers[len(made)]
            target = str(rng.choice(bound + made))
            value[p] = value[target]
            pointer_log[p] = (target, value[target])
            made.append(p)
            parts.append(f"{p}={target};")
        parts.append(_filler(int(rng.integers(0, 3)), rng))
    parts.append(_filler(l2, rng))

    queries: List[Tuple[str, str, str, int]] = []  # (name, mark, answer, kind)
    for v in bound:
        overwritten = len(history[v]) > 1
        queries.append((v, ">", value[v], 2 if overwritten else 1))
        if overwritten:
            queries.append((v, "<", history[v][-2], 4))
    for p in pointers:
        queries.append((p, ">", value[p], 3))
    order = rng.permutation(len(queries))

    text = list("".join(parts))
    meta: List[Tuple[int, int, int]] = []
    for i in order:
        name, mark, ans, kind = queries[i]
        text.extend(["?", name, mark])
        meta.append((len(text) - 1, ord(ans), kind))
        text.extend([ans, ";"])
    record = {"history": history, "pointers": pointer_log, "bound": bound}
    return "".join(text), meta, record


def shortcut_scores(episodes) -> Dict[str, float]:
    """How often simple rules answer correctly, over (text, meta, record)
    episodes. A model scoring near one of these may have learned the rule."""
    hits = {"first value": [], "current value of the target (pointers)": [],
            "first value of the target (pointers)": [], "last digit before the query": []}
    for text, meta, rec in episodes:
        hist, ptrs = rec["history"], rec["pointers"]
        for pos, ans, kind in meta:
            name, mark = text[pos - 1], text[pos]
            want = chr(ans)
            if kind == 3:
                target = ptrs[name][0]
                while target in ptrs:  # a chain: follow to a bound name
                    target = ptrs[target][0]
                hits["current value of the target (pointers)"].append(hist[target][-1] == want)
                hits["first value of the target (pointers)"].append(hist[target][0] == want)
            else:
                hits["first value"].append(hist[name][0] == want)
            prev_digits = [c for c in text[: pos - 2] if c in VALS]
            hits["last digit before the query"].append(prev_digits[-1] == want if prev_digits else False)
    return {k: 100 * float(np.mean(v)) for k, v in hits.items() if v}
