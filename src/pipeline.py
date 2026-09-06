#!/usr/bin/env python3
"""Run the whole corpus pipeline, skipping stages whose inputs have not changed.

One command to rebuild everything after adding a source, editing an authority
file, or changing a normalisation rule — and it does only the work that the
change actually invalidated.

    fetch -> extract -> enrich -> normalize -> dedupe -> build
          -> tokenizer -> baselines -> provenance

WHY FINGERPRINTS CHAIN RATHER THAN LOOK AT FILES
    Three stages rewrite data/interim/ IN PLACE. Fingerprinting a stage by the
    modification times of its inputs would therefore mark it stale the instant it
    finished, since its own output is its neighbour's input and its own input's
    mtime moved. Instead each stage's fingerprint is

        sha256(previous stage's fingerprint + its own config files + its own source)

    so the chain is ordered by construction. Editing normalize.py invalidates
    normalize and everything after it, and nothing before it. Editing an
    authority file invalidates enrich onward. Adding a source invalidates
    everything, which is correct.

    A consequence worth knowing: the fingerprint covers *code and configuration*,
    not the fetched bytes. If an upstream repository changes under a floating
    pin, re-run with --force. The lockfile is what makes that detectable.

WHY THE LOCKFILE IS PROJECTED RATHER THAN HASHED
    fetch.py rewrites manifest/fetch.lock.json on every run, stamping a fresh
    `generated` and a fresh per-source `fetched` even when every resolved_pin
    comes back identical. Hashing the file's bytes therefore marked extract
    stale after every fetch, and — because fingerprints chain — every stage
    after it too. "Run only what changed" quietly became "rebuild the whole
    corpus, always". Extract is keyed instead on a projection of the lockfile
    that keeps what determines its output (pins, checksums, licence, tier) and
    drops the timestamps.

WHY A STAGE IS FINGERPRINTED AGAIN, AND WHEN
    fetch mutates the lockfile, which is an input to the stage after it, so a
    plan computed up front describes the inputs as they were *before* the run.
    Recording those would be wrong in both directions: unchanged pins would
    look changed, and — worse — a pin that genuinely moved during the fetch
    would be recorded under its old value, marking a stale build fresh.

    So each stage is fingerprinted again — but at the moment it STARTS, not
    the moment it finishes. By then every earlier stage has run and the
    lockfile holds the pins this stage is about to read, which is what the
    paragraph above requires. Fingerprinting afterwards instead would record
    whatever the inputs became *during* the run: edit an extractor while it is
    working and its output is stamped with code that never produced it, and
    the stage then reads as fresh for good. A fingerprint has to describe the
    inputs the stage actually used.

Usage:
    python src/pipeline.py --status         # what is stale and why
    python src/pipeline.py                  # run what is needed
    python src/pipeline.py --force          # run everything regardless
    python src/pipeline.py --from normalize # this stage onward
"""

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from functools import partial
from pathlib import Path
from typing import Any, Callable, Dict, List, NamedTuple, Optional, Tuple

ROOT = Path(__file__).resolve().parent.parent
STATE = ROOT / "data" / "pipeline.state.json"
PY = sys.executable


class Part(NamedTuple):
    """One independently invalidatable unit inside a stage."""
    name: str
    argv: List[str]
    derived: Optional[Callable[[], str]] = None


class Stage:
    """One pipeline step: what invalidates it, and how to run it."""

    def __init__(self, name: str, script: str, config: List[str],
                 argv: Optional[List[List[str]]] = None, note: str = "",
                 derived: Optional[Callable[[], str]] = None,
                 parts: Optional[List["Part"]] = None,
                 concurrent: bool = False):
        self.name = name
        self.script = script
        self.config = config
        self.argv = argv or [[]]
        self.note = note
        # Whether this stage's invocations may run at the same time. Off by
        # default, and it is not a property of being partitioned: extract's
        # parts write one interim file each and never read another's, while
        # build's four invocations each read every interim file into memory to
        # assemble a corpus, so running them together would multiply the peak
        # by four for no parallelism the machine can actually deliver.
        self.concurrent = concurrent
        # For an input whose bytes churn without its meaning changing. Called
        # at fingerprint time, never cached, so it sees the file as it is now.
        self.derived = derived
        # A partitioned stage does independently invalidatable units of work.
        # Extract has one part per source, so adding the tenth source does not
        # re-extract the nine already there — the cost of adding a source stays
        # constant instead of growing with the corpus.
        self.parts = parts

    def part_fingerprint(self, previous: str, part: "Part") -> str:
        h = hashlib.sha256()
        h.update(self._common(previous))
        h.update(json.dumps(part.argv, sort_keys=True).encode())
        h.update(part.derived().encode() if part.derived else b"")
        return h.hexdigest()[:16]

    def _common(self, previous: str) -> bytes:
        """Everything a stage's parts share: the chain, its script, its config."""
        h = hashlib.sha256()
        h.update(previous.encode())
        for rel in [self.script, *self.config]:
            path = ROOT / rel
            h.update(rel.encode())
            h.update(path.read_bytes() if path.exists() else b"<missing>")
        return h.digest()

    def fingerprint(self, previous: str) -> str:
        """The whole stage, for chaining onward.

        For a partitioned stage this folds in every part, so a change to any
        one source still invalidates everything downstream of extract — which
        is correct, because the corpus changed.
        """
        h = hashlib.sha256()
        h.update(self._common(previous))
        if self.parts is not None:
            for part in self.parts:
                h.update(part.name.encode())
                h.update(self.part_fingerprint(previous, part).encode())
            return h.hexdigest()[:16]
        h.update(json.dumps(self.argv, sort_keys=True).encode())
        if self.derived is not None:
            h.update(self.derived().encode())
        return h.hexdigest()[:16]

    def _invoke(self, extra: List[str], capture: bool) -> Tuple[str, int, str]:
        cmd = [PY, str(ROOT / self.script), *extra]
        printable = " ".join([Path(self.script).name, *extra])
        if not capture:
            print(f"    $ {printable}", flush=True)
            return printable, subprocess.run(cmd, cwd=ROOT).returncode, ""
        done = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True)
        return printable, done.returncode, done.stdout + done.stderr

    def run(self, argv: Optional[List[List[str]]] = None, jobs: int = 1) -> None:
        """Run this stage's invocations, concurrently where that is declared safe.

        WHY THE OUTPUT IS CAPTURED WHEN THEY OVERLAP
            Sequentially each invocation inherits this terminal and streams as
            it goes, which is what you want when watching one long extractor.
            Run several at once and their lines interleave into something no
            one can read, so a concurrent run collects each invocation's output
            and prints it whole when it finishes. The trade is that a slow part
            says nothing until it is done; the alternative says everything at
            once and means nothing.

        Every invocation is run even if an earlier one fails, because they are
        independent by construction — that is the same reason they may overlap
        at all — and there is no sense in abandoning eight sources because the
        ninth has a malformed file. The stage still fails, and pipeline state
        is not advanced.
        """
        commands = self.argv if argv is None else argv
        if jobs <= 1 or not self.concurrent or len(commands) < 2:
            for extra in commands:
                printable, code, _ = self._invoke(extra, capture=False)
                if code != 0:
                    raise RuntimeError(f"{printable} exited {code}")
            return

        print(f"    ({len(commands)} invocations, {min(jobs, len(commands))} "
              f"at a time)", flush=True)
        failures: List[str] = []
        with ThreadPoolExecutor(max_workers=jobs) as pool:
            for printable, code, output in pool.map(
                    lambda extra: self._invoke(extra, capture=True), commands):
                mark = "" if code == 0 else f"   <-- EXIT {code}"
                print(f"    $ {printable}{mark}", flush=True)
                for line in output.rstrip("\n").splitlines():
                    print(f"      {line}")
                if code != 0:
                    failures.append(f"{printable} exited {code}")
        if failures:
            raise RuntimeError("; ".join(failures))


def extractable_sources() -> List[str]:
    """Sources that are fetched, permitted in a corpus, and have a reader.

    The permitted-in-a-corpus half lives in extract.py, which excludes the
    eval tier — the UD treebanks are CoNLL-U exactly like DigilibLT, so
    dispatching on format alone would extract held-out evaluation data into
    the training corpus.
    """
    sys.path.insert(0, str(ROOT / "src"))
    try:
        from extract import extractable_sources as declared
    except Exception:
        return []
    lock_path = ROOT / "manifest" / "fetch.lock.json"
    if not lock_path.exists():
        return []
    with lock_path.open(encoding="utf-8") as fh:
        fetched = json.load(fh).get("sources", {})
    # n_files, not mere presence in the lockfile. A `manual` source is recorded
    # there the moment it is declared, with present:false and no files — and
    # once its format gains a reader, the pipeline would try to extract it and
    # fail the whole run. Gutenberg is exactly that case.
    return [s for s in declared()
            if s in fetched and "error" not in fetched[s]
            and (fetched[s].get("n_files") or 0) > 0]


LOCKFILE_IDENTITY_KEYS = ("resolved_pin", "license", "tier", "method", "url",
                          "n_files", "bytes")


def lockfile_identity(source: Optional[str] = None) -> str:
    """What in the lockfile determines what extraction will produce.

    With `source`, only that one entry — which is what lets extract invalidate
    a single source rather than all of them.

    Everything that fixes the fetched bytes — the resolved commit for a git
    source, the checksums for a downloaded one, and the licence and tier that
    decide whether it may enter the corpus at all. Deliberately excluded:
    `generated`, `fetched`, and the display `name`, all of which fetch.py
    rewrites on every run without any change to what was fetched.
    """
    lock = ROOT / "manifest" / "fetch.lock.json"
    if not lock.exists():
        return "<missing>"
    try:
        with lock.open(encoding="utf-8") as fh:
            sources = json.load(fh).get("sources", {})
    except (json.JSONDecodeError, OSError):
        # A corrupt lockfile must not silently fingerprint as "unchanged".
        return f"<unreadable:{lock.stat().st_size if lock.exists() else 0}>"
    if source is not None:
        sources = {source: sources.get(source, {})}
    identity = {
        name: {**{k: rec.get(k) for k in LOCKFILE_IDENTITY_KEYS},
               "files": sorted((f.get("name"), f.get("sha256"), f.get("bytes"))
                               for f in rec.get("files", []))}
        for name, rec in sources.items()
    }
    return json.dumps(identity, sort_keys=True)


def tokenizer_identity() -> str:
    """The vocabulary the floors are measured through.

    Not redundant with the chain. The tokenizer stage sits immediately before
    baselines, so in practice the chain already covers it — but the floors are
    meaningless if measured through a different vocabulary than the models
    use, and that relationship deserves to be stated rather than inferred from
    stage ordering. Measured 2026-08-18: retraining the 4k vocabulary moved the
    classical floor 158.8 -> 149.8 on a corpus that had changed by three words.
    """
    path = ROOT / "data" / "processed" / "tokenizers" / "latin-bpe-4000.json"
    if not path.exists():
        return "<missing>"
    return hashlib.sha256(path.read_bytes()).hexdigest()[:16]


def build_stages() -> List[Stage]:
    sources = extractable_sources()
    return [
        # Track B, because the build stage assembles Track B corpora and a
        # pipeline that builds a corpus it never fetched the sources for is
        # not reproducible. Track B is a superset of Track A: green + amber.
        Stage("fetch", "src/fetch.py", ["manifest/sources.yaml"],
              argv=[["--track", "B"]],
              note="green + amber; idempotent, existing clones are updated"),
        # Concurrent: one part per source, each reading data/raw/<source>/ and
        # writing data/interim/<source>.jsonl. No part reads another's input or
        # output, which is the same independence that lets them be invalidated
        # separately — extraction is XML parsing, so this is the stage with the
        # most idle cores to reclaim.
        Stage("extract", "src/extract.py", [],
              parts=[Part(s, ["--source", s], partial(lockfile_identity, s))
                     for s in sources],
              argv=[["--source", s] for s in sources] or [["--help"]],
              concurrent=True,
              note=f"{len(sources)} source(s), invalidated independently"),
        Stage("enrich", "src/enrich.py",
              ["manifest/authorities.yaml", "manifest/genres.yaml"],
              note="dates, periods, form and genre"),
        Stage("normalize", "src/normalize.py", [],
              note="orthography; rewrites interim in place"),
        Stage("dedupe", "src/dedupe.py", [], argv=[["--show", "0"]],
              note="content-based; marks rather than deletes"),
        # Four corpora from one marking pass. Track A is what gets published;
        # Track B adds the non-commercial sources and is never redistributed,
        # so both are built every time rather than one being a manual extra —
        # a research corpus nobody rebuilds drifts from the code that made it.
        Stage("build", "src/build.py", [],
              argv=[[],
                    ["--name", "classical", "--period", "CLASSICAL"],
                    ["--name", "corpus-b", "--track", "B"],
                    ["--name", "classical-b", "--track", "B",
                     "--period", "CLASSICAL"]],
              note="Track A and Track B, full and classical"),
        # The tokenizer and the floors were outside the pipeline until
        # 2026-08-19, which meant both had to be remembered after a rebuild.
        # They are not optional: a vocabulary trained on a corpus that has
        # since changed, or a floor measured through a vocabulary that has,
        # produce numbers that look valid and are not. Making them stages is
        # what stops that being a thing anyone has to remember.
        Stage("tokenizer", "src/tokenizer.py", [],
              argv=[["--vocab", "4000"]],
              note="the 4k vocabulary the models are trained through"),
        Stage("baselines", "src/evaluate.py", [],
              argv=[["--baselines", "--corpus", "corpus"],
                    ["--baselines", "--corpus", "classical"]],
              derived=tokenizer_identity,
              note="the bigram floors every model is judged against"),
        Stage("provenance", "src/provenance.py", ["manifest/sources.yaml"],
              note="regenerates ATTRIBUTION.md and audits the build"),
    ]


def load_state() -> Dict[str, Any]:
    if not STATE.exists():
        return {}
    with STATE.open(encoding="utf-8") as fh:
        return json.load(fh)


def save_state(state: Dict[str, Any]) -> None:
    STATE.parent.mkdir(parents=True, exist_ok=True)
    with STATE.open("w", encoding="utf-8") as fh:
        json.dump(state, fh, indent=2, sort_keys=True)
        fh.write("\n")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--status", action="store_true", help="report staleness, run nothing")
    ap.add_argument("--force", action="store_true", help="run every stage")
    ap.add_argument("--from", dest="start", help="run from this stage onward")
    ap.add_argument("--skip-fetch", action="store_true",
                    help="assume data/raw is current (fetch is the slow, networked stage)")
    # Four rather than every core: each extractor holds one source's records in
    # memory before writing them, and the largest are hundreds of megabytes. The
    # bound here is RAM, not CPU.
    ap.add_argument("--jobs", type=int, default=min(4, os.cpu_count() or 1),
                    metavar="N",
                    help="invocations to run at once within a stage that "
                         "declares it safe (default 4; 1 runs everything in "
                         "order, streaming output as it goes)")
    args = ap.parse_args()

    stages = build_stages()
    state = load_state()
    recorded = state.get("stages", {})
    recorded_parts: Dict[str, Dict[str, str]] = state.get("parts", {})

    previous = ""
    plan: List[tuple] = []
    downstream = False
    started_at = None
    for stage in stages:
        incoming = previous
        fingerprint = stage.fingerprint(incoming)
        previous = fingerprint
        if args.start and stage.name == args.start:
            started_at = True
        stale = recorded.get(stage.name) != fingerprint
        upstream_changed = downstream
        run = bool(args.force or downstream or stale)
        if args.start and not started_at:
            run = False
        if args.skip_fetch and stage.name == "fetch":
            run = False

        # A partitioned stage does only the parts whose own inputs moved —
        # unless something upstream changed, in which case every part has to
        # be redone anyway. `pending is None` means "run the whole stage".
        pending: Optional[List[Part]] = None
        if stage.parts is not None and run and not args.force and not upstream_changed:
            known = recorded_parts.get(stage.name, {})
            pending = [part for part in stage.parts
                       if known.get(part.name)
                       != stage.part_fingerprint(incoming, part)]
            if not pending:
                # Every part is current; only the stage-level bookkeeping was
                # stale. Nothing to do, and nothing downstream to invalidate.
                run = False

        if run:
            downstream = True
        reason = ("forced" if args.force else
                  "upstream changed" if upstream_changed and run else
                  # A stale stage that is skipped anyway was skipped by a flag,
                  # not because it is current — saying "inputs changed" next to
                  # "skip" reads as a contradiction.
                  "stale, held back by --from/--skip-fetch" if stale and not run else
                  "inputs changed" if stale else "up to date")
        if run and pending is not None and len(pending) < len(stage.parts):
            reason += f"  [{len(pending)} of {len(stage.parts)}]"
        plan.append((stage, fingerprint, run, reason, pending))

    width = max(len(s.name) for s in stages)
    print(f"{'stage':<{width}}  {'action':<9}  reason")
    print("-" * (width + 40))
    for stage, _, run, reason, _pending in plan:
        print(f"{stage.name:<{width}}  {'RUN' if run else 'skip':<9}  {reason}"
              + (f"  ({stage.note})" if run and stage.note else ""))

    if args.status:
        pending = sum(1 for _, _, run, _, _ in plan if run)
        print(f"\n{pending} of {len(plan)} stage(s) would run")
        return 0

    print()
    started = time.time()
    # The chain is rebuilt as we go: a stage that ran is fingerprinted again
    # just before it starts, so inputs an earlier stage rewrote (fetch
    # rewrites the lockfile) are recorded as they now are rather than as they
    # were at plan time — and so what is recorded is what the stage read.
    chained = ""
    for stage, planned, run, _, pending in plan:
        incoming = chained
        if not run:
            chained = planned
            continue
        print(f"[{stage.name}]", flush=True)
        used = stage.fingerprint(incoming)
        used_parts = ({part.name: stage.part_fingerprint(incoming, part)
                       for part in stage.parts}
                      if stage.parts is not None else None)
        try:
            # None means the whole stage; a list means only these parts.
            stage.run([part.argv for part in pending] if pending else None,
                      jobs=args.jobs)
        except RuntimeError as exc:
            print(f"\nFAILED at {stage.name}: {exc}", file=sys.stderr)
            print("State not advanced; fix and re-run — completed stages are "
                  "remembered.", file=sys.stderr)
            save_state({"stages": recorded, "parts": recorded_parts,
                        "updated": datetime.now(timezone.utc).isoformat(timespec="seconds")})
            return 1
        chained = used
        recorded[stage.name] = used
        if used_parts is not None:
            recorded_parts[stage.name] = used_parts
        save_state({"stages": recorded, "parts": recorded_parts,
                    "updated": datetime.now(timezone.utc).isoformat(timespec="seconds")})
        print()

    print(f"pipeline complete in {(time.time()-started)/60:.1f} minutes")
    return 0


if __name__ == "__main__":
    sys.exit(main())
