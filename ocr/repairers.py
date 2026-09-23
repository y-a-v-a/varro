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

FOUR SYSTEMS, AND THE GAPS BETWEEN THE LAST THREE ARE THE EXPERIMENT
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

    fim        the same again, judged by a checkpoint from src/fim.py that was
               trained to price a fill with the text after it in front of it,
               rather than recovering that from the cost of the text after.

    `lii` and `fim` are reference points, not the experiment's answer: greedy,
    one word at a time, right context still noisy, and every candidate from
    the same channel-and-lexicon proposer -- which tune.py's `ceiling` shows
    is the wall on real scans.
"""

import json
import math
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, NamedTuple, Optional, Sequence, Tuple

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(Path(__file__).resolve().parent))

import channel as _channel  # noqa: E402
from metrics import LEAD_PUNCT, TRAIL_PUNCT  # noqa: E402

# How many proposals per word go to the scorer. The channel and the word
# counts rank them first. Sixteen, because with the wider proposer the cap
# was binding on nine words in ten and the true word was among what it cut.
POOL = 16

# Intermediate strings explored when a word needs two repairs. A cap,
# because a long word under a generous channel has tens of thousands; and
# the likeliest few hundred, because the channel's own rates order them.
DEPTH2_CAP = 600

# Beyond two repairs the channel is not enumerated -- three steps through
# 300 operations is a hundred million strings a word -- and the lexicon is
# searched instead: known words within this many edits, from a bucket of
# words that share the first two letters and roughly the length.
FAR_EDITS = 3
FAR_FLOOR = 20            # a word this rare is not worth reaching for
FAR_KEEP = 6              # the commonest few, by corpus count

# A token this long that the lexicon does not know may be several words run
# together, and is offered its best segmentation into up to this many.
SEGMENT_MIN_LEN = 6
SEGMENT_MAX_PARTS = 4

# Sampling into the gap: how many draws, and how far a draw may be from what
# the page shows before it stops being a reading of the page and becomes a
# rewriting of it. The distance is in letters, case and spaces ignored.
SAMPLES = 12
SAMPLE_STEPS = 8
SAMPLE_REACH = (2, 3)      # at most max(2, len // 3), capped at 3

# A word is offered with a capital when it carries one at least this often in
# the training corpus, and without one when it does so at most this often.
CAPITAL_HIGH = 0.5
CAPITAL_LOW = 0.1

# A one-letter word is proposed as half of a split only if it is this common.
# `a`, `e` and `o` are Latin; so is nearly any letter, once, in some
# abbreviation, and without a floor every damaged word splits into confetti.
SHORT_FLOOR = 500

# Arabic or Roman, the latter as the corpus spells it: v is folded to u, so
# forty-eight is XLUIII.
NUMERAL = re.compile(r"^(?:[0-9]+|[IUXLCDM]+)$")


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


class Rule(NamedTuple):
    """When a proposed change is allowed to stand. Tuned on dev by ocr/tune.py.

    The scorer says how much likelier a candidate makes the text; the channel
    says how likely the scanner was to have produced what is on the page from
    it. Added raw, those two decide too eagerly, and what they break is not
    random: it is the word the lexicon has not met -- a name, a medieval
    spelling, a line of Folengo's macaronics -- which looks exactly like
    damage one letter away from a commoner word. Each margin below is extra
    evidence demanded, in nats, before a change of that kind is believed.
    """
    weight: float = 1.0     # scorer nats per channel nat
    margin: float = 0.0     # any change at all
    known: float = 0.0      # + when the word on the page is itself a known word
    unknown: float = 0.0    # + when it is not
    capital: float = 0.0    # + when it is capitalised: names live here
    join: float = 0.0       # + for joining two fragments that are each known words
    split: float = 0.0      # + for cutting one token in two
    short: float = 0.0      # + when the word on the page has three letters or fewer
    repeated: float = 0.0   # + when the same form stands elsewhere on the page:
                            #   a name recurs, a misreading seldom recurs the same
    seen: float = 0.0       # - when the candidate itself stands elsewhere on the
                            #   page: the page has already vouched for it


def needed(rule: Rule, flags: Dict[str, bool], text: str, used: int) -> float:
    """Evidence this candidate must bring, given what kind of change it is."""
    need = rule.margin
    need += rule.known if flags["known"] else rule.unknown
    if flags["capital"]:
        need += rule.capital
    if flags["short"]:
        need += rule.short
    if used == 2 and flags["halves_known"]:
        need += rule.join
    if " " in text:
        need += rule.split
    if flags.get("repeated"):
        need += rule.repeated
    if text in flags.get("seen", ()):
        need -= rule.seen
    return need


def decide(rule: Rule, pool: Sequence[Tuple[str, float, int]],
           lang: Sequence[float], flags: Dict[str, bool]) -> int:
    """Index of the candidate to keep; 0 is the word as found.

    Pure, and separate from the loop, so that tune.py can replay a recorded
    run under a thousand rules without asking the model anything again.
    """
    stay = rule.weight * lang[0]
    best, lead = 0, 0.0
    for c in range(1, len(pool)):
        text, lp, used = pool[c]
        gain = rule.weight * lang[c] + lp - stay - needed(rule, flags, text, used)
        if gain > lead:
            best, lead = c, gain
    return best


def page_flags(words: Sequence[str], k: int,
               pool: Sequence[Tuple[str, float, int]]) -> Dict[str, Any]:
    """What the rest of the page says about this word and its candidates.

    `repeated`: the form on the page, edge punctuation aside, occurs again
    elsewhere on it. `seen`: the candidates whose core occurs elsewhere on
    the page as it stands. Both are cheap, both are computed from the noisy
    text alone, and both are the kind of evidence a human corrector uses
    first -- a strange word that appears three times is a name.
    """
    cores = [split_edges(w)[1].lower() for w in words]
    mine = cores[k]
    others = cores[:k] + cores[k + 1:]
    seen = []
    for text, _, used in pool[1:]:
        core = " ".join(split_edges(w)[1].lower() for w in text.split())
        if core and core != mine and all(part in others for part in core.split()):
            seen.append(text)
    return {"repeated": bool(mine) and mine in others, "seen": seen}


def load_rule(system: str, tier: int = 1) -> Rule:
    """The rule tuned for this system on this tier's dev split, if any.

    Per tier, because a rule is a judgement about how much evidence a change
    needs, and that depends on how often the scanner is wrong and in what way.
    Tier 2 falls back to the Tier 1 rule rather than to none: a rule tuned on
    invented noise is still more cautious than the raw sum.
    """
    path = Path(__file__).resolve().parent / "rules.json"
    if path.exists():
        saved = json.loads(path.read_text(encoding="utf-8")).get("rules", {})
        for key in ([f"{system}#tier2"] if tier == 2 else []) + [system]:
            if key in saved:
                return Rule(**saved[key])
    return Rule()


class Repairer:
    name = "?"

    def fit(self, pairs: Sequence[Tuple[str, str]], lexicon: Dict[str, int],
            capitalised: Optional[Dict[str, int]] = None) -> None:
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

    def __init__(self, rule: Optional[Rule] = None):
        self.rule = rule or Rule()
        # Set to a list to have every decision recorded: the pool, what the
        # scorer said of each candidate, and what kind of word it was.
        self.trace: Optional[List[Dict[str, Any]]] = None

    def fit(self, pairs, lexicon, capitalised=None) -> None:
        self.lexicon = lexicon
        self.capitalised = capitalised or {}
        self.total = sum(lexicon.values()) or 1
        self.channel = _channel.fit(pairs)
        self.ops = [(t, n, lp) for t, n, lp in self.channel.ops()
                    if " " not in t and " " not in n]
        # The two segmentation repairs are not substitutions inside a word, so
        # they are looked up by name. A channel that never saw one cannot
        # propose it.
        self.lp_split = self.channel.logp.get(("", " "))    # a space appeared
        self.lp_merge = self.channel.logp.get((" ", ""))    # a space vanished
        # What an operation the channel never saw is charged: one nat under
        # the rarest it did see. Real scans do things no sample of them
        # showed -- `%uam`, `nia$aa` -- and a proposer that could only undo
        # what it had been shown left a third of real damage unoffered.
        self.lp_unseen = min((lp for _, _, lp in self.ops), default=-8.0) - 1.0
        # Buckets for the far search: (first two letters, length) -> words.
        self.buckets: Dict[Tuple[str, int], List[str]] = {}
        for w, n in lexicon.items():
            if n >= FAR_FLOOR and len(w) >= 4:
                self.buckets.setdefault((w[:2], len(w)), []).append(w)

    def known(self, word: str) -> bool:
        return word.lower() in self.lexicon

    def capital_share(self, word: str) -> float:
        low = word.lower()
        return self.capitalised.get(low, 0) / max(1, self.lexicon.get(low, 0))

    # -- proposals ---------------------------------------------------------

    def _step(self, word: str, known_only: bool) -> Dict[str, float]:
        """Every string one inverse channel operation away from `word`.

        Case is kept as the channel wrote it. The channel was fitted on text
        as printed, so it knows `I -> l` and `R -> B` as operations, and a
        step through one of them restores the capital. Lowercasing first --
        the earlier way -- made `Ioue` unreachable from `loue`: the step
        produced `ioue`, the lexicon knew it, and the word came back with the
        wrong case for the wrong reason.
        """
        out: Dict[str, float] = {}

        def offer(w: str, lp: float) -> None:
            if w and w != word and (not known_only or self.known(w)):
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

    def _far(self, word: str) -> List[Tuple[str, int]]:
        """Known words within FAR_EDITS of `word`, commonest first.

        The bucket is by the first two letters and the length, give or take
        two, which assumes the scanner got the first two letters right. It
        usually did; and a word wrong from its first letter is one the model
        will have to propose from context, not the lexicon.
        """
        from metrics import align
        low = word.lower()
        if len(low) < 4:
            return []
        hits: List[Tuple[int, int, str]] = []
        for length in range(len(low) - 2, len(low) + 3):
            for w in self.buckets.get((low[:2], length), ()):
                d = align(low, w)[0]
                if 0 < d <= FAR_EDITS:
                    hits.append((-self.lexicon[w], d, w))
        hits.sort()
        out = []
        for _, d, w in hits[:FAR_KEEP]:
            out.append((word[:1] + w[1:] if word[:1].isupper() else w, d))
        return out

    @staticmethod
    def _parts(word: str, cut: str) -> List[str]:
        """`word` divided as `cut` is, so the two can be compared part by part."""
        out, at = [], 0
        for part in cut.split():
            out.append(word[at:at + len(part)])
            at += len(part)
        return out

    def _wild(self, word: str) -> Dict[str, float]:
        """Anything that is not a letter, taken for a letter the scanner
        misread, or for dirt: each such character is dropped, or replaced by
        every letter, and whatever the lexicon knows is offered."""
        out: Dict[str, float] = {}
        for at, ch in enumerate(word):
            if ch.isalpha():
                continue
            letters = "abcdefghilmnopqrstuxyz"
            if at == 0 or word[:at].isupper():
                letters += letters.upper()
            for repl in [""] + list(letters):
                w = word[:at] + repl + word[at + 1:]
                if w != word and self.known(w) and w not in out:
                    out[w] = self.lp_unseen
        return out

    def _segment(self, word: str) -> Optional[str]:
        """The likeliest way to read `word` as two to four known words.

        A line of verse the scanner ran together (`Hospitatectalocis.`) is
        not a word with three errors; it is four words with none, and only a
        segmentation can offer it back.
        """
        n = len(word)
        # A part is a known word, or -- at the channel's price -- one step
        # from one: `Almafalutiferam` is `Alma salutiferam` with a long s.
        def readings(part: str) -> List[Tuple[float, str]]:
            out = []
            if self._splittable(part.lower()):
                out.append((self.prior(part.lower()), part))
            if len(part) >= 3:
                for w, lp in self._step(part, known_only=True).items():
                    if " " not in w:
                        out.append((self.prior(w.lower()) + lp, w))
            return out

        # best[i][k]: (score, cut, reading) for word[:i] as k parts.
        best: List[List[Optional[Tuple[float, int, str]]]] = \
            [[None] * (SEGMENT_MAX_PARTS + 1) for _ in range(n + 1)]
        best[0][0] = (0.0, 0, "")
        for i in range(1, n + 1):
            for j in range(max(0, i - 24), i):
                for gain, reading in readings(word[j:i]):
                    for k in range(1, SEGMENT_MAX_PARTS + 1):
                        prev = best[j][k - 1]
                        if prev is not None and (best[i][k] is None
                                                 or prev[0] + gain > best[i][k][0]):
                            best[i][k] = (prev[0] + gain, j, reading)
        choices = [(best[n][k][0], k) for k in range(2, SEGMENT_MAX_PARTS + 1)
                   if best[n][k] is not None]
        if not choices:
            return None
        _, k = max(choices)
        parts: List[str] = []
        i = n
        while k:
            _, j, reading = best[i][k]
            parts.append(reading)
            i, k = j, k - 1
        return " ".join(reversed(parts))

    def propose(self, inner: str) -> Dict[str, float]:
        """Candidate true forms of one noisy word -> log P(noisy | candidate).

        In order of how far the proposer has to reach: one channel operation;
        two, then three, for a word the lexicon does not know; a letter for a
        symbol; a segmentation; and the other case for a word that usually
        carries one. Each stage is here because the missed-proposal breakdown
        on real scans (ocr/README.md §3) named it.
        """
        found = self._step(inner, known_only=True)
        unknown = not self.known(inner)
        if unknown:
            halfway = self._step(inner, known_only=False)
            second = sorted(halfway.items(), key=lambda kv: -kv[1])
            for mid, lp1 in second[:DEPTH2_CAP]:
                for w, lp2 in self._step(mid, known_only=True).items():
                    if w != inner and lp1 + lp2 > found.get(w, -math.inf):
                        found[w] = lp1 + lp2
            if not found:
                for w, distance in self._far(inner):
                    found[w] = self.lp_unseen * distance
            for w, lp in self._wild(inner).items():
                if lp > found.get(w, -math.inf):
                    found[w] = lp
        if self.lp_merge is not None:
            low = inner.lower()
            for cut in range(1, len(inner)):
                a, b = low[:cut], low[cut:]
                if self._splittable(a) and self._splittable(b):
                    found[f"{inner[:cut]} {inner[cut:]}"] = self.lp_merge
            if unknown and len(inner) >= SEGMENT_MIN_LEN:
                cut = self._segment(inner)
                if cut and cut not in found:
                    # A space the scanner lost per cut, and a step for each
                    # part it also misread.
                    steps = sum(1 for a, b in zip(cut.split(), self._parts(inner, cut))
                                if a != b)
                    found[cut] = self.lp_merge * cut.count(" ") + self.lp_unseen * steps
        # The other case, where the corpus says the word usually has it.
        # Offered rather than imposed: which case a heading or a sentence
        # start wants is the scorer's question, and the margin's.
        for w in [inner] + list(found):
            share = self.capital_share(w.split()[0]) if self.known(w.split()[0]) else 0.0
            head = w[:1]
            if head.islower() and share >= CAPITAL_HIGH:
                alt = head.upper() + w[1:]
            elif head.isupper() and share <= CAPITAL_LOW and not w[:2].isupper():
                alt = head.lower() + w[1:]
            else:
                continue
            if alt != inner and alt not in found:
                found[alt] = found.get(w, 0.0) + self.lp_unseen
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
            # Nothing without a letter in it is touched, and no numeral. `5.`
            # is one channel step from `s.` and `XL III` joins into a known
            # `XLIII`; both were being "repaired", and no amount of context
            # tells a model which number a charter meant.
            if not any(ch.isalpha() for ch in inner) or NUMERAL.match(inner):
                out.append(words[k])
                k += 1
                continue
            # (replacement token, channel log-prob, noisy words consumed). The
            # token is whole -- edge punctuation included -- because the
            # punctuation is part of what the scanner may have got wrong.
            pool: List[Tuple[str, float, int]] = [(words[k], 0.0, 1)]
            proposed = self.propose(inner)
            for c, lp in self.extra(inner, out, words, k).items():
                if lp > proposed.get(c, -math.inf):
                    proposed[c] = lp
            pool += [(lead + c + trail, lp, 1) for c, lp in proposed.items()]
            for text, lp in self._edges(lead, inner, trail):
                pool.append((text, lp, 1))
            halves_known = False
            if self.lp_split is not None and k + 1 < len(words) and not trail:
                lead2, inner2, trail2 = split_edges(words[k + 1])
                joined = inner + inner2
                if inner2 and not lead2 and self.known(joined) and not NUMERAL.match(inner2):
                    pool.append((lead + joined + trail2, self.lp_split, 2))
                    halves_known = self.known(inner) and self.known(inner2)

            best = 0
            if len(pool) > 1:
                pool = pool[:1] + sorted(
                    pool[1:], key=lambda p: -(p[1] + self.prior(split_edges(p[0])[1].lower())))[:POOL]
                lang = self.score(out, pool, words, k)
                flags = {"known": self.known(inner),
                         "capital": inner[:1].isupper(),
                         "short": len(inner) <= 3,
                         "halves_known": halves_known,
                         **page_flags(words, k, pool)}
                best = decide(self.rule, pool, lang, flags)
                if self.trace is not None:
                    self.trace.append({"k": k, "pool": pool, "lang": lang,
                                       "flags": flags, "chose": best})

            out.append(pool[best][0])
            k += pool[best][2]
        return " ".join(out)

    def extra(self, inner: str, done: List[str], words: List[str],
              k: int) -> Dict[str, float]:
        """Proposals that need the context. None, unless a subclass has a way."""
        return {}

    def _edges(self, lead: str, inner: str, trail: str) -> List[Tuple[str, float]]:
        """The same word with its edge punctuation dropped, or one mark added.

        Only where the channel has seen that mark come or go: a scanner that
        loses full stops is a scanner that may have lost this one.
        """
        out: List[Tuple[str, float]] = []
        if trail:
            lp = self.channel.logp.get(("", trail))
            if lp is not None:
                out.append((lead + inner, lp))
        else:
            for mark in ".,":
                lp = self.channel.logp.get((mark, ""))
                if lp is not None:
                    out.append((lead + inner + mark, lp))
        return out

    def score(self, done: List[str], pool: List[Tuple[str, float, int]],
              words: List[str], k: int) -> List[float]:
        """The scorer's opinion of each candidate, in nats, WITHOUT the channel.

        Kept apart from the channel term so that Rule.weight can set the
        exchange rate between them after the fact.
        """
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
        return [self.prior(split_edges(text)[1].lower() or text) + sum(
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
                 right_words: int = 5, rule: Optional[Rule] = None):
        super().__init__(rule)
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
            scores += [(c, row["logprob"]) for c, row in zip(group, rows)]
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


class Fim(Lii):
    """The same proposals, judged by a model trained to fill a gap directly.

    `lii` recovers P(candidate | both sides) by Bayes: it prices the candidate
    AND the right context, and the right context's share is common to every
    candidate. That is exact, but the quantity it ranks on is a sum over a
    dozen tokens of noisy text of which only the first few depend on the
    candidate at all. A checkpoint from src/fim.py has the right context in
    front of it and prices only the candidate's own tokens and the token that
    closes the gap. Same information, a different estimator; the benchmark
    says which is better.
    """

    def __init__(self, model: str = "", left_words: int = 40,
                 right_words: int = 5, rule: Optional[Rule] = None,
                 sample: bool = False):
        super().__init__(model or "latin-lii-fim", left_words, right_words, rule)
        self.name = "fim:" + self.name.split(":", 1)[1]
        self.seam = self.m.tok.token_to_id("<|pad|>")
        self.eos = self.m.tok.token_to_id("<|endoftext|>")
        self.sample = sample
        if sample:
            self.name += "+samples"

    def extra(self, inner, done, words, k) -> Dict[str, float]:
        """Ask the model what stood here, knowing both sides, and keep the
        answers that could be a reading of what the page shows.

        Only for a word the lexicon does not know. Those are where every
        lexicon-bound stage runs out -- a name, a coinage, a word broken
        three ways -- and where a model that has read the whole corpus, and
        the sentence either side, may still know. The draws are then held
        to a small edit distance from the page: the model proposes, the page
        disposes, and a fluent word that is not a reading of the letters is
        not offered however likely it is.
        """
        if not self.sample or self.known(inner):
            return {}
        import mlx.core as mx
        from metrics import align

        enc = self._common.encode
        suffix = enc(self.m, " ".join(words[k + 1:k + 1 + self.right_words]))
        room = self.m.context - 3 - len(suffix) - SAMPLE_STEPS
        prefix = enc(self.m, " ".join(done[-self.left_words:]))[-max(0, room):]
        prompt = [self.eos, self.seam, *suffix, self.seam, *prefix]
        ids = mx.array([prompt] * SAMPLES)
        key = mx.random.key(k)
        for _ in range(SAMPLE_STEPS):
            logits = self.m.model(ids)[:, -1].astype(mx.float32)
            key, use = mx.random.split(key)
            ids = mx.concatenate([ids, mx.random.categorical(logits, key=use)[:, None]], axis=1)
            mx.eval(ids)

        plain = re.sub(r"[^a-z]", "", inner.lower())
        reach = min(SAMPLE_REACH[1], max(SAMPLE_REACH[0], len(plain) // 3))
        out: Dict[str, float] = {}
        for row in ids[:, len(prompt):].tolist():
            if self.eos in row:
                row = row[:row.index(self.eos)]
            words_ = self.m.tok.decode(row).split()
            # The first word of the draw, or the first few: the checkpoint
            # measured here (reports/train-latin-lii-fim.json) almost never
            # closes a middle with <|endoftext|> -- P(eos | true fill) was
            # 0.000 on every case tried -- so a draw is read as a
            # continuation and its opening words are the proposal.
            for n in range(1, min(SEGMENT_MAX_PARTS, len(words_)) + 1):
                cand = " ".join(w.strip(LEAD_PUNCT + TRAIL_PUNCT) for w in words_[:n]).strip()
                if not cand or cand == inner or any(not w for w in cand.split()):
                    continue
                letters = re.sub(r"[^a-z]", "", cand.lower())
                if len(letters) > len(plain) + reach:
                    break
                distance = align(plain, letters)[0]
                if distance <= reach:
                    lp = self.lp_unseen * max(1, distance)
                    if lp > out.get(cand, -math.inf):
                        out[cand] = lp
        return out

    def score(self, done, pool, words, k) -> List[float]:
        import mlx.core as mx
        from fim import frame

        enc = self._common.encode
        widest = max(p[2] for p in pool)
        suffix = enc(self.m, " ".join(words[k + widest:k + widest + self.right_words]))
        middles = []
        for text, _, used in pool:
            # The token is whole, edge punctuation included, as training cut
            # middles: from one word boundary to the next, with the comma
            # after a word on the word's side of that boundary.
            rest = words[k + used:k + widest]
            middles.append(enc(self.m, " ".join([text] + rest)))
        room = self.m.context - 3 - len(suffix) - max(len(b) for b in middles)
        prefix = enc(self.m, " ".join(done[-self.left_words:]))[-max(0, room):] \
            if room > 0 else []

        rows = [[self.eos] + frame(prefix, b, suffix, self.seam, self.eos)
                for b in middles]
        width = max(len(r) for r in rows)
        padded = mx.array([r + [self.seam] * (width - len(r)) for r in rows])
        logits = self.m.model(padded).astype(mx.float32)
        logp = logits - mx.logsumexp(logits, axis=-1, keepdims=True)
        picked = mx.take_along_axis(logp[:, :-1], padded[:, 1:, None], axis=-1)[:, :, 0]
        mx.eval(picked)
        values = picked.tolist()
        out = []
        for row, body, v in zip(rows, middles, values):
            # The middle and the end-of-text that closes it are the last
            # len(body) + 1 tokens of the row; v[t] prices row[t + 1].
            out.append(sum(v[len(row) - len(body) - 2:len(row) - 1]))
        return out


def make(spec: str, tuned: bool = True, tier: int = 1) -> Repairer:
    """identity | unigram | lii | lii-left | fim, then optionally :<checkpoint>,
    @<words of right context> and, for fim, +samples -- e.g. `lii:latin-vii`,
    `fim@12`, `fim+samples`.

    With `tuned`, a system takes the rule ocr/rules.json holds for its name,
    if there is one. Without, it decides on the raw sum, as the first dev
    reports did.
    """
    spec, _, width = spec.partition("@")
    spec, _, extra = spec.partition("+")
    kind, _, arg = spec.partition(":")
    right = int(width) if width else 5
    if extra and extra != "samples":
        sys.exit(f"unknown option +{extra}; only +samples exists, and only for fim")
    if kind == "identity":
        return Identity()
    if kind == "unigram":
        system: Proposing = Unigram()
    elif kind == "lii":
        system = Lii(arg, right_words=right)
    elif kind == "lii-left":
        system = Lii(arg, right_words=0)
    elif kind == "fim":
        system = Fim(arg, right_words=right, sample=extra == "samples")
    else:
        sys.exit(f"unknown system {spec!r} -- see ocr/repairers.py:make")
    if width:
        system.name += f"@{right}"
    if tuned:
        system.rule = load_rule(system.name, tier)
    return system
