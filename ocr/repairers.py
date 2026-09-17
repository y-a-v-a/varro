#!/usr/bin/env python3
"""Things that take noisy Latin and hand back Latin. The benchmark judges them;
this file is where a new one goes.

THE CONTRACT
    fit(pairs, lexicon)   shown the DEV split's (true, noisy) pairs and a word
                          count from the training corpus. Nothing else. In
                          particular not the benchmark's confusion table:
                          what the noise is has to be learned from the pairs,
                          as it would from a few hand-corrected pages.
    repair(noisy) -> str  the whole job.
    suspicion(noisy)      optional: one score per whitespace word, higher for
                          "look here". Lets detection be measured apart from
                          correction, because a model that finds damage it
                          cannot fix is still useful to a human corrector.

THREE SYSTEMS, AND THE GAP BETWEEN THE LAST TWO IS THE EXPERIMENT
    identity   returns its input. Every number it posts is a floor, and its
               `broken` count must be zero or the metrics are wrong.
    unigram    proposes attested words reachable through the fitted channel
               and picks by word frequency. No context. This is plan.md's
               frequency table, given every advantage except order.
    lii        the same proposals, the same channel, the same decision rule --
               and P(candidate + what follows | what came before) under
               LATIN-LII where unigram has a word count.

    Same candidates on purpose. If lii had a better proposer as well, a win
    would not say which part had earned it. Held to one proposer, the
    difference between the two rows is what reading the context is worth, and
    that is the only claim plan.md makes for the model.

    `lii` is a reference point, not the experiment's answer: greedy, one word
    at a time, right context still noisy. It is here so the harness has been
    run end to end on the thing it exists to judge.
"""

import math
import sys
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(Path(__file__).resolve().parent))

import channel as _channel  # noqa: E402
from metrics import LEAD_PUNCT, TRAIL_PUNCT  # noqa: E402

# How many proposals per word go to the scorer. The channel and the word
# counts rank them first; eight is past where the true word stops turning up.
POOL = 8

# Intermediate strings explored when a word needs two repairs. A cap, because
# a long word under a generous channel has tens of thousands.
DEPTH2_CAP = 3000

# A one-letter word is proposed as half of a split only if it is this common.
# `a`, `e` and `o` are Latin; so is nearly any letter, once, in some
# abbreviation, and without a floor every damaged word splits into confetti.
SHORT_FLOOR = 500


def split_edges(word: str) -> Tuple[str, str, str]:
    body = word.lstrip(LEAD_PUNCT)
    lead = word[:len(word) - len(body)]
    inner = body.rstrip(TRAIL_PUNCT)
    return lead, inner, body[len(inner):]


def recase(candidate: str, like: str) -> str:
    if len(like) > 1 and like.isupper():
        return candidate.upper()
    if like[:1].isupper():
        return candidate[:1].upper() + candidate[1:]
    return candidate


class Repairer:
    name = "?"

    def fit(self, pairs: Sequence[Tuple[str, str]], lexicon: Dict[str, int]) -> None:
        pass

    def repair(self, noisy: str) -> str:
        raise NotImplementedError

    def suspicion(self, noisy: str) -> Optional[List[float]]:
        return None


class Identity(Repairer):
    name = "identity"

    def repair(self, noisy: str) -> str:
        return noisy


class Proposing(Repairer):
    """Shared by every system that repairs word by word through a fitted channel."""

    def __init__(self, margin: float = 0.0):
        # Nats by which a change must beat leaving the word alone, ON TOP of
        # the channel's own prior against change. Tune on dev, never on test.
        self.margin = margin

    def fit(self, pairs, lexicon) -> None:
        self.lexicon = lexicon
        self.total = sum(lexicon.values()) or 1
        self.channel = _channel.fit(pairs)
        self.ops = [(t, n, lp) for t, n, lp in self.channel.ops()
                    if " " not in t and " " not in n]
        # The two segmentation repairs are not substitutions inside a word, so
        # they are looked up by name. A channel that never saw one cannot
        # propose it.
        self.lp_split = self.channel.logp.get(("", " "))    # a space appeared
        self.lp_merge = self.channel.logp.get((" ", ""))    # a space vanished

    # -- proposals ---------------------------------------------------------

    def _step(self, word: str, known_only: bool) -> Dict[str, float]:
        """Every string one inverse channel operation away from `word`."""
        out: Dict[str, float] = {}

        def offer(w: str, lp: float) -> None:
            if w and w != word and (not known_only or w in self.lexicon):
                if lp > out.get(w, -math.inf):
                    out[w] = lp

        for t, n, lp in self.ops:
            if n:
                at = word.find(n)
                while at != -1:
                    offer(word[:at] + t + word[at + len(n):], lp)
                    at = word.find(n, at + 1)
            else:
                for at in range(len(word) + 1):
                    offer(word[:at] + t + word[at:], lp)
        return out

    def propose(self, inner: str) -> Dict[str, float]:
        """Candidate true forms of one noisy word -> log P(noisy | candidate)."""
        word = inner.lower()
        found = self._step(word, known_only=True)
        if not found and word not in self.lexicon:
            # Two errors in one word. Only tried when the word is unknown AND
            # nothing is one step away, because every known word is two steps
            # from some other known word and would otherwise be offered it.
            halfway = self._step(word, known_only=False)
            for mid, lp1 in sorted(halfway.items(), key=lambda kv: -kv[1])[:DEPTH2_CAP]:
                for w, lp2 in self._step(mid, known_only=True).items():
                    if w != word and lp1 + lp2 > found.get(w, -math.inf):
                        found[w] = lp1 + lp2
        if self.lp_merge is not None:
            for cut in range(1, len(word)):
                a, b = word[:cut], word[cut:]
                if self._splittable(a) and self._splittable(b):
                    found[f"{a} {b}"] = self.lp_merge
        return found

    def _splittable(self, part: str) -> bool:
        floor = SHORT_FLOOR if len(part) == 1 else 1
        return self.lexicon.get(part, 0) >= floor

    def prior(self, candidate: str) -> float:
        """log P(candidate) from word counts alone, add-one over the lexicon."""
        return sum(math.log((self.lexicon.get(w, 0) + 1) / (self.total + len(self.lexicon)))
                   for w in candidate.split())

    # -- the loop ----------------------------------------------------------

    def repair(self, noisy: str) -> str:
        words = noisy.split()
        out: List[str] = []
        k = 0
        while k < len(words):
            lead, inner, trail = split_edges(words[k])
            if not inner:
                out.append(words[k])
                k += 1
                continue
            # (replacement text, channel log-prob, noisy words consumed)
            pool: List[Tuple[str, float, int]] = [(inner, 0.0, 1)]
            pool += [(recase(c, inner), lp, 1) for c, lp in self.propose(inner).items()]
            if self.lp_split is not None and k + 1 < len(words) and not trail:
                lead2, inner2, trail2 = split_edges(words[k + 1])
                joined = (inner + inner2).lower()
                if inner2 and not lead2 and joined in self.lexicon:
                    pool.append((recase(joined, inner), self.lp_split, 2))

            if len(pool) > 1:
                pool = pool[:1] + sorted(
                    pool[1:], key=lambda p: -(p[1] + self.prior(p[0].lower())))[:POOL]
                scores = self.score(out, pool, words, k)
                best = max(range(len(pool)), key=lambda c: scores[c])
                if best and scores[best] - scores[0] <= self.margin:
                    best = 0
            else:
                best = 0

            text, _, used = pool[best]
            if used == 2:
                trail = split_edges(words[k + 1])[2]
            out.append(lead + text + trail)
            k += used
        return " ".join(out)

    def score(self, done: List[str], pool: List[Tuple[str, float, int]],
              words: List[str], k: int) -> List[float]:
        raise NotImplementedError


class Unigram(Proposing):
    name = "unigram"

    def score(self, done, pool, words, k) -> List[float]:
        # A candidate that joins two noisy words is priced for both, so one
        # that does not must pay for the second as it stands. Without this
        # `suu m` went to `suum m`: mending the first fragment alone was
        # cheaper than the join, only because the join was charged for an `m`
        # its rival never had to account for.
        widest = max(p[2] for p in pool)
        return [lp + self.prior(text.lower()) + sum(
                    self.prior(split_edges(w)[1].lower() or w)
                    for w in words[k + used:k + widest])
                for text, lp, used in pool]

    def suspicion(self, noisy: str) -> List[float]:
        return [-self.prior(split_edges(w)[1].lower() or w) for w in noisy.split()]


class Lii(Proposing):
    """The same proposals, judged by the model across the text either side.

    left    what has ALREADY been repaired, so an early fix helps a later one.
    right   the next few noisy words, as they are. They may be damaged too.
            Every candidate is scored over the same ones, so the damage costs
            all of them alike and the ranking survives it -- the mechanism
            demos/lacuna.py measures, at 40% against 8% without it.
    """

    def __init__(self, model: str = "", left_words: int = 40,
                 right_words: int = 5, margin: float = 0.0):
        super().__init__(margin)
        sys.path.insert(0, str(ROOT / "demos"))
        import _common
        self._common = _common
        self.m = _common.load(model or _common.DEFAULT_MODEL)
        self.left_words, self.right_words = left_words, right_words
        self.name = f"lii:{self.m.name}" + ("" if right_words else ":left-only")

    def score(self, done, pool, words, k) -> List[float]:
        left = " ".join(done[-self.left_words:])
        scores: List[float] = []
        # Every candidate must be scored over the SAME stretch of noisy text or
        # the totals do not compare. One that swallows two noisy words has the
        # second inside its fill, so the others get that word as right context
        # -- even in left-only mode, where it is the only right context there
        # is, and the alternative is comparing one word's cost against two's.
        end = k + max(p[2] for p in pool) + self.right_words
        for used in sorted({p[2] for p in pool}):
            group = [c for c, p in enumerate(pool) if p[2] == used]
            right = " ".join(words[k + used:end])
            rows = self._common.score_readings(
                self.m, left, [pool[c][0] for c in group], right)
            scores += [(c, row["logprob"] + pool[c][1]) for c, row in zip(group, rows)]
        return [s for _, s in sorted(scores)]

    def suspicion(self, noisy: str) -> Optional[List[float]]:
        from score import score_text
        from surprisal import to_words
        result = score_text(self.m.model, self.m.tok, noisy, self.m.context)
        scored = to_words(self.m.tok, result["ids"], result["logprobs"],
                          result["prefix_len"])
        # The tokenizer's idea of a word and str.split()'s must agree, or the
        # scores would be laid against the wrong labels without any error.
        if [w.text for w in scored] != noisy.split():
            return None
        return [w.bits for w in scored]


def make(spec: str) -> Repairer:
    """identity | unigram | lii | lii:<checkpoint> | lii-left | lii-left:<checkpoint>"""
    kind, _, arg = spec.partition(":")
    if kind == "identity":
        return Identity()
    if kind == "unigram":
        return Unigram()
    if kind == "lii":
        return Lii(arg)
    if kind == "lii-left":
        return Lii(arg, right_words=0)
    sys.exit(f"unknown system {spec!r} -- see ocr/repairers.py:make")
