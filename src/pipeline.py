#!/usr/bin/env python3
"""Run the whole corpus pipeline, skipping stages whose inputs have not changed.

One command to rebuild everything after adding a source, editing an authority
file, or changing a normalisation rule — and it does only the work that the
change actually invalidated.

    fetch -> extract -> enrich -> normalize -> dedupe -> build -> provenance

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

WHY A STAGE IS FINGERPRINTED AGAIN AFTER IT RUNS
    fetch mutates the lockfile, which is an input to the stage after it, so a
    plan computed up front describes the inputs as they were *before* the run.
    Recording those would be wrong in both directions: unchanged pins would
    look changed, and — worse — a pin that genuinely moved during the fetch
    would be recorded under its old value, marking a stale build fresh. Each
    stage is therefore re-fingerprinted at the moment its result is recorded.

Usage:
    python src/pipeline.py --status         # what is stale and why
    python src/pipeline.py                  # run what is needed
    python src/pipeline.py --force          # run everything regardless
    python src/pipeline.py --from normalize # this stage onward
"""

import argparse
import hashlib
import json
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

ROOT = Path(__file__).resolve().parent.parent
STATE = ROOT / "data" / "pipeline.state.json"
PY = sys.executable


class Stage:
    """One pipeline step: what invalidates it, and how to run it."""

    def __init__(self, name: str, script: str, config: List[str],
                 argv: Optional[List[List[str]]] = None, note: str = "",
                 derived: Optional[Callable[[], str]] = None):
        self.name = name
        self.script = script
        self.config = config
        self.argv = argv or [[]]
        self.note = note
        # For an input whose bytes churn without its meaning changing. Called
        # at fingerprint time, never cached, so it sees the file as it is now.
        self.derived = derived

    def fingerprint(self, previous: str) -> str:
        h = hashlib.sha256()
        h.update(previous.encode())
        for rel in [self.script, *self.config]:
            path = ROOT / rel
            h.update(rel.encode())
            h.update(path.read_bytes() if path.exists() else b"<missing>")
        h.update(json.dumps(self.argv, sort_keys=True).encode())
        if self.derived is not None:
            h.update(self.derived().encode())
        return h.hexdigest()[:16]

    def run(self) -> None:
        for extra in self.argv:
            cmd = [PY, str(ROOT / self.script), *extra]
            printable = " ".join([Path(self.script).name, *extra])
            print(f"    $ {printable}", flush=True)
            result = subprocess.run(cmd, cwd=ROOT)
            if result.returncode != 0:
                raise RuntimeError(f"{printable} exited {result.returncode}")


def extractable_sources() -> List[str]:
    """Sources that are both fetched and have an extractor implemented."""
    sys.path.insert(0, str(ROOT / "src"))
    try:
        from extract import EXTRACTORS
    except Exception:
        return []
    lock_path = ROOT / "manifest" / "fetch.lock.json"
    if not lock_path.exists():
        return []
    with lock_path.open(encoding="utf-8") as fh:
        fetched = json.load(fh).get("sources", {})
    return sorted(s for s in EXTRACTORS
                  if s in fetched and "error" not in fetched[s])


LOCKFILE_IDENTITY_KEYS = ("resolved_pin", "license", "tier", "method", "url",
                          "n_files", "bytes")


def lockfile_identity() -> str:
    """What in the lockfile determines what extraction will produce.

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
    identity = {
        name: {**{k: rec.get(k) for k in LOCKFILE_IDENTITY_KEYS},
               "files": sorted((f.get("name"), f.get("sha256"), f.get("bytes"))
                               for f in rec.get("files", []))}
        for name, rec in sources.items()
    }
    return json.dumps(identity, sort_keys=True)


def build_stages() -> List[Stage]:
    sources = extractable_sources()
    return [
        Stage("fetch", "src/fetch.py", ["manifest/sources.yaml"],
              note="idempotent: existing clones are updated, not re-cloned"),
        Stage("extract", "src/extract.py", [],
              argv=[["--source", s] for s in sources] or [["--help"]],
              derived=lockfile_identity,
              note=f"{len(sources)} source(s) with an extractor"),
        Stage("enrich", "src/enrich.py",
              ["manifest/authorities.yaml", "manifest/genres.yaml"],
              note="dates, periods, form and genre"),
        Stage("normalize", "src/normalize.py", [],
              note="orthography; rewrites interim in place"),
        Stage("dedupe", "src/dedupe.py", [], argv=[["--show", "0"]],
              note="content-based; marks rather than deletes"),
        Stage("build", "src/build.py", [],
              argv=[[], ["--name", "classical", "--period", "CLASSICAL"]],
              note="full corpus and the classical subset"),
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
    args = ap.parse_args()

    stages = build_stages()
    state = load_state()
    recorded = state.get("stages", {})

    previous = ""
    plan: List[tuple] = []
    downstream = False
    started_at = None
    for stage in stages:
        fingerprint = stage.fingerprint(previous)
        previous = fingerprint
        if args.start and stage.name == args.start:
            started_at = True
        stale = recorded.get(stage.name) != fingerprint
        run = bool(args.force or downstream or stale)
        if args.start and not started_at:
            run = False
        if args.skip_fetch and stage.name == "fetch":
            run = False
        if run:
            downstream = True
        reason = ("forced" if args.force else
                  "upstream changed" if downstream and not stale and run else
                  # A stale stage that is skipped anyway was skipped by a flag,
                  # not because it is current — saying "inputs changed" next to
                  # "skip" reads as a contradiction.
                  "stale, held back by --from/--skip-fetch" if stale and not run else
                  "inputs changed" if stale else "up to date")
        plan.append((stage, fingerprint, run, reason))

    width = max(len(s.name) for s in stages)
    print(f"{'stage':<{width}}  {'action':<9}  reason")
    print("-" * (width + 40))
    for stage, _, run, reason in plan:
        print(f"{stage.name:<{width}}  {'RUN' if run else 'skip':<9}  {reason}"
              + (f"  ({stage.note})" if run and stage.note else ""))

    if args.status:
        pending = sum(1 for _, _, run, _ in plan if run)
        print(f"\n{pending} of {len(plan)} stage(s) would run")
        return 0

    print()
    started = time.time()
    # The chain is rebuilt as we go: a stage that ran is fingerprinted again
    # afterwards, so inputs it or an earlier stage rewrote (fetch rewrites the
    # lockfile) are recorded as they ended up, not as they were at plan time.
    chained = ""
    for stage, planned, run, _ in plan:
        if not run:
            chained = planned
            continue
        print(f"[{stage.name}]", flush=True)
        try:
            stage.run()
        except RuntimeError as exc:
            print(f"\nFAILED at {stage.name}: {exc}", file=sys.stderr)
            print("State not advanced; fix and re-run — completed stages are "
                  "remembered.", file=sys.stderr)
            save_state({"stages": recorded,
                        "updated": datetime.now(timezone.utc).isoformat(timespec="seconds")})
            return 1
        chained = stage.fingerprint(chained)
        recorded[stage.name] = chained
        save_state({"stages": recorded,
                    "updated": datetime.now(timezone.utc).isoformat(timespec="seconds")})
        print()

    print(f"pipeline complete in {(time.time()-started)/60:.1f} minutes")
    return 0


if __name__ == "__main__":
    sys.exit(main())
