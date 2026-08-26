#!/usr/bin/env python3
"""Fetch corpus sources declared in manifest/sources.yaml into data/raw/.

The manifest is the authoritative record of what may enter the corpus and under
what licence; this script is the only sanctioned way to populate data/raw/.

Two invariants it exists to protect:

  1. data/raw/ is immutable. Nothing here edits a fetched file, and neither
     should you. Every correction belongs downstream, in code.
  2. Track discipline (see feasibility.md §3). Track A draws only on
     free-licence sources and is what gets published; Track B adds
     non-commercial sources and is never redistributed. Track membership is
     derived from the licence tier, so the manifest cannot assert it directly.

Usage:
    python src/fetch.py --list
    python src/fetch.py --validate
    python src/fetch.py                       # Track A: redistributable
    python src/fetch.py --track B             # research build, NOT redistributable
    python src/fetch.py --only perseus-canonical-latinLit
    python src/fetch.py --tier green --dry-run
"""

import argparse
import hashlib
import json
import re
import shutil
import subprocess
import time
import sys
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

try:
    import yaml
except ImportError:
    sys.exit(
        "PyYAML is required.\n"
        "    python3.14 -m venv .venv && source .venv/bin/activate\n"
        "    pip install -r requirements.txt"
    )

ROOT = Path(__file__).resolve().parent.parent
MANIFEST = ROOT / "manifest" / "sources.yaml"
LOCKFILE = ROOT / "manifest" / "fetch.lock.json"
RAW = ROOT / "data" / "raw"

VALID_TIERS = {"green", "amber", "unknown", "eval"}

# Which licence tiers each track may draw on. Track membership is derived here
# and nowhere else — the manifest never names a track directly.
TRACKS = {
    "A": {"green"},                # redistributable: published corpus and weights
    "B": {"green", "amber"},       # research only: adds non-commercial sources
    "eval": {"eval"},              # held-out evaluation data; never in either corpus
}
VALID_METHODS = {"git", "http", "zenodo", "manual", "corpus-corporum"}

# How a source's text came to be text. Not a licence question and not a quality
# one: it is about whether a later model would be learning from an earlier
# model's output.
#
#   transcribed      keyed or proofread by people
#   ocr              machine-read, possibly machine-corrected by a third party
#   model-corrected  corrected by a model from THIS project — the feedback risk
#   unknown          not established; treated as suspect, like tier: unknown
#
# `unknown` is not a placeholder to be filled in later with a guess. The Latin
# Library carries it because sources.md says its provenance is inconsistent and
# it has known OCR errors, and a source whose provenance nobody has established
# should say so rather than claim to be transcribed.
VALID_TEXT_PROVENANCE = {"transcribed", "ocr", "model-corrected", "unknown"}
VALID_PERIODS = {
    "CLASSICAL", "LATE", "MEDIEVAL", "NEOLATIN", "MIXED", "DOCUMENTARY",
}
USER_AGENT = "open-latin-corpus/0.1 (+https://github.com/)"


# ---------------------------------------------------------------------------
# Manifest loading and validation
# ---------------------------------------------------------------------------

def load_manifest(path: Path = MANIFEST) -> Dict[str, Any]:
    if not path.exists():
        sys.exit(f"Manifest not found: {path}")
    with path.open(encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def validate(manifest: Dict[str, Any]) -> List[str]:
    """Return a list of problems. Empty list means the manifest is sound."""
    errors: List[str] = []
    sources = manifest.get("sources")

    if not isinstance(sources, list) or not sources:
        return ["manifest has no `sources` list"]

    seen_ids = set()
    for i, src in enumerate(sources):
        where = src.get("id") or f"sources[{i}]"

        for field in ("id", "name", "url", "method", "license", "tier", "include"):
            if field not in src:
                errors.append(f"{where}: missing required field `{field}`")

        sid = src.get("id")
        if sid in seen_ids:
            errors.append(f"{where}: duplicate id")
        seen_ids.add(sid)

        if src.get("tier") not in VALID_TIERS:
            errors.append(f"{where}: tier must be one of {sorted(VALID_TIERS)}")
        if src.get("method") not in VALID_METHODS:
            errors.append(f"{where}: method must be one of {sorted(VALID_METHODS)}")
        if "period" in src and src["period"] not in VALID_PERIODS:
            errors.append(f"{where}: period must be one of {sorted(VALID_PERIODS)}")
        # Required, and deliberately with no default. A source that does not
        # say how its text was produced should fail the validator rather than
        # be silently assumed to be human-transcribed — that assumption is
        # exactly the one that would let model-corrected text in unnoticed.
        provenance = src.get("text_provenance")
        if provenance is None:
            errors.append(f"{where}: text_provenance is required "
                          f"(one of {sorted(VALID_TEXT_PROVENANCE)})")
        elif provenance not in VALID_TEXT_PROVENANCE:
            errors.append(f"{where}: text_provenance must be one of "
                          f"{sorted(VALID_TEXT_PROVENANCE)}, not {provenance!r}")

        # An amber source with include:true is now correct — it means "wanted,
        # in Track B". What must never happen is amber reaching a Track A build,
        # and that is enforced at fetch and build time rather than here.
        if src.get("tier") == "unknown" and src.get("include"):
            errors.append(
                f"{where}: tier `unknown` with include:true — a licence nobody has "
                f"resolved cannot enter either track; settle it or leave it out"
            )
        if src.get("tier") in ("amber", "unknown") and not src.get("restriction"):
            errors.append(
                f"{where}: tier `{src.get('tier')}` requires a `restriction` "
                f"explaining what the terms are and why"
            )

    return errors


def selected(manifest: Dict[str, Any], only: Optional[str], tier: Optional[str],
             include_all: bool, track: str) -> List[Dict[str, Any]]:
    """Sources to act on.

    A bare `--track X` selects only what belongs to X. An explicit `--only`
    bypasses that filter so the track guard can refuse it by name — a source
    asked for and silently skipped is worse than one asked for and rejected.
    """
    out = []
    for src in manifest["sources"]:
        if only and src["id"] != only:
            continue
        if tier and src.get("tier") != tier:
            continue
        if not only:
            if not include_all and not src.get("include"):
                continue
            if src.get("tier") not in TRACKS[track]:
                continue
        out.append(src)
    return out


# ---------------------------------------------------------------------------
# Fetch methods
# ---------------------------------------------------------------------------

def _run(cmd: List[str], cwd: Optional[Path] = None) -> str:
    result = subprocess.run(
        cmd, cwd=cwd, capture_output=True, text=True, check=False,
    )
    if result.returncode != 0:
        raise RuntimeError(f"{' '.join(cmd)}\n{result.stderr.strip()}")
    return result.stdout.strip()


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _tree_stats(path: Path) -> Dict[str, int]:
    files = [p for p in path.rglob("*") if p.is_file() and ".git" not in p.parts]
    return {"n_files": len(files), "bytes": sum(p.stat().st_size for p in files)}


def fetch_git(src: Dict[str, Any], dest: Path) -> Dict[str, Any]:
    """Partial clone (blobless) so large repos stay cheap but any SHA is checkoutable."""
    if not shutil.which("git"):
        raise RuntimeError("git not found on PATH")

    # Test for .git, not for the directory: main() pre-creates dest, so an
    # existing-but-empty directory is the normal first-run state.
    if (dest / ".git").exists():
        print(f"    already cloned, fetching updates")
        _run(["git", "fetch", "--quiet", "origin"], cwd=dest)
    else:
        if dest.exists() and any(dest.iterdir()):
            raise RuntimeError(
                f"{dest} is non-empty but is not a git clone — refusing to "
                f"overwrite. Remove it by hand if this is stale."
            )
        dest.parent.mkdir(parents=True, exist_ok=True)
        print(f"    cloning {src['url']}")
        _run(["git", "clone", "--filter=blob:none", "--quiet", src["url"], str(dest)])

    pin = src.get("pin")
    if pin and pin != "latest":
        print(f"    checking out pin {pin}")
        _run(["git", "checkout", "--quiet", pin], cwd=dest)

    sha = _run(["git", "rev-parse", "HEAD"], cwd=dest)
    return {"resolved_pin": sha, **_tree_stats(dest)}


def _download(url: str, target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(req) as resp, target.open("wb") as fh:
        total = int(resp.headers.get("Content-Length") or 0)
        done = 0
        while chunk := resp.read(1 << 20):
            fh.write(chunk)
            done += len(chunk)
            # Carriage-return progress only works on a terminal. Redirected to a
            # log it emits one line per megabyte, which buried a 465MB download
            # in several hundred lines of noise.
            if not sys.stdout.isatty():
                continue
            if total:
                pct = 100 * done / total
                print(f"\r    {done/1e6:.1f}/{total/1e6:.1f} MB ({pct:.0f}%)",
                      end="", flush=True)
            else:
                print(f"\r    {done/1e6:.1f} MB", end="", flush=True)
    if sys.stdout.isatty():
        print()
    else:
        print(f"    {done/1e6:.1f} MB downloaded")


def fetch_http(src: Dict[str, Any], dest: Path) -> Dict[str, Any]:
    url = src["url"]
    filename = url.rstrip("/").split("/")[-1] or f"{src['id']}.bin"
    target = dest / filename
    if target.exists():
        print(f"    already present: {filename}")
    else:
        print(f"    downloading {filename}")
        _download(url, target)
    return {
        "resolved_pin": src.get("pin"),
        "files": [{"name": filename, "sha256": _sha256(target),
                   "bytes": target.stat().st_size}],
        **_tree_stats(dest),
    }


def fetch_zenodo(src: Dict[str, Any], dest: Path) -> Dict[str, Any]:
    """Resolve a Zenodo record id to its files via the API, then download each."""
    record_id = str(src.get("pin") or src["url"].rstrip("/").split("/")[-1])
    api = f"https://zenodo.org/api/records/{record_id}"
    print(f"    querying {api}")
    req = urllib.request.Request(api, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(req) as resp:
        record = json.loads(resp.read())

    files_meta = []
    for entry in record.get("files", []):
        name = entry.get("key") or entry.get("filename")
        link = (entry.get("links", {}).get("self")
                or entry.get("links", {}).get("download")
                or f"{api}/files/{name}/content")
        target = dest / name
        if target.exists():
            print(f"    already present: {name}")
        else:
            print(f"    downloading {name}")
            _download(link, target)
        files_meta.append({"name": name, "sha256": _sha256(target),
                           "bytes": target.stat().st_size})

    return {
        "resolved_pin": record_id,
        "doi": record.get("doi"),
        "files": files_meta,
        **_tree_stats(dest),
    }


# ---------------------------------------------------------------------------
# Corpus Corporum — one work per request, slowly
# ---------------------------------------------------------------------------
#
# There is no bulk export. The site serves one zipped TEI file per work from
#   download.php?type=file-pos-xml&idno=<id>
# and the ids come from a metadata table published by the Centre for
# Computational Studies in Plzeň (CC BY-SA 4.0), which also supplies dates and
# word counts.
#
# This is a small university server, so the request rate is a courtesy, not a
# throttle to tune down: 20 seconds between works, single-threaded, and no
# retry storm. The whole selection takes about fourteen hours and is meant to.
#
# It is RESUMABLE by construction — a work whose file already exists is
# skipped without a request — because a fourteen-hour job will be interrupted.

CC_DELAY_SECONDS = 20.0
CC_MAX_RETRIES = 3
CC_BACKOFF_SECONDS = 60.0

# The endpoint answers a missing work with `File not found. Sorry` and HTTP
# 200. Nothing in the status line says anything is wrong, so without this
# check a long run ends with thousands of 21-byte files that every reader
# downstream reports as empty, and the fetch reports success.
ZIP_MAGIC = b"PK\x03\x04"


def _cc_selection(src: Dict[str, Any], meta_path: Path) -> List[Dict[str, str]]:
    """The works to fetch: everything except what the corpus already holds.

    Excluding by series rather than trusting dedupe to clean up afterwards.
    Patrologia Latina alone is 5,248 works and 118M words against the ~66M we
    already carry through Vicifons, so fetching it would triple the run time
    to re-acquire text we have.
    """
    import csv as _csv
    exclude = [re.compile(p, re.I) for p in (src.get("exclude_series") or [])]
    rows = []
    with meta_path.open(encoding="utf-8") as fh:
        for row in _csv.DictReader(fh):
            series = row.get("series_title") or ""
            if any(p.search(series) for p in exclude):
                continue
            # `type=file-pos-xml` serves only works that have been lemmatised,
            # and the metadata says which those are. 129 of 7,819 have not
            # been; asking for them earns a soft 404 apiece, so they are
            # skipped rather than discovered one 20-second wait at a time.
            if (row.get("prelemmatized") or "").strip().lower() != "true":
                continue
            if row.get("work_id"):
                rows.append(row)
    return rows


def fetch_corpus_corporum(src: Dict[str, Any], dest: Path) -> Dict[str, Any]:
    dest.mkdir(parents=True, exist_ok=True)
    meta_dir = dest / "_metadata"
    meta_dir.mkdir(exist_ok=True)
    meta_path = meta_dir / "cc_metadata.csv"
    if not meta_path.exists():
        print(f"    fetching the work list from {src['id_source']}")
        _download(src["id_source"], meta_path)

    works = _cc_selection(src, meta_path)
    delay = float(src.get("delay_seconds") or CC_DELAY_SECONDS)
    template = src["url_template"]
    print(f"    {len(works):,} works selected; {delay:.0f}s between requests "
          f"— about {len(works) * delay / 3600:.1f} h if none are cached")

    got = skipped = failed = unavailable = 0
    for i, row in enumerate(works, 1):
        idno = row["work_id"]
        target = dest / "works" / f"{idno}.zip"
        if target.exists() and target.stat().st_size:
            skipped += 1
            continue
        if got:                       # only wait between actual requests
            time.sleep(delay)
        url = template.format(idno=idno)
        for attempt in range(1, CC_MAX_RETRIES + 1):
            try:
                _download(url, target)
                if target.read_bytes()[:4] != ZIP_MAGIC:
                    # A soft 404, not a transient fault: retrying it three
                    # times with a backoff would waste three minutes to be
                    # told the same thing.
                    body = target.read_text(errors="replace")[:60].strip()
                    target.unlink(missing_ok=True)
                    unavailable += 1
                    print(f"    [{idno}] not served: {body!r}", flush=True)
                    break
                got += 1
                break
            except (urllib.error.URLError, urllib.error.HTTPError, OSError) as exc:
                target.unlink(missing_ok=True)
                if attempt == CC_MAX_RETRIES:
                    failed += 1
                    print(f"    [{idno}] giving up after {attempt}: {exc}")
                    break
                # Back off hard rather than hammering a server that is
                # already unhappy.
                wait = CC_BACKOFF_SECONDS * attempt
                print(f"    [{idno}] attempt {attempt} failed ({exc}); "
                      f"waiting {wait:.0f}s", flush=True)
                time.sleep(wait)
        if i % 25 == 0 or i == len(works):
            print(f"    {i:,}/{len(works):,}  fetched {got:,}  cached {skipped:,}"
                  f"  unavailable {unavailable:,}  failed {failed:,}", flush=True)

    if failed:
        print(f"    {failed:,} work(s) could not be fetched — re-run to retry "
              f"only those")
    return {
        "resolved_pin": None,
        "works_selected": len(works),
        "works_fetched": got,
        "works_cached": skipped,
        "works_failed": failed,
        "works_unavailable": unavailable,
        **_tree_stats(dest),
    }


def fetch_manual(src: Dict[str, Any], dest: Path) -> Dict[str, Any]:
    present = dest.exists() and any(dest.iterdir())
    status = "present" if present else "MISSING"
    print(f"    manual source — {status}")
    print(f"    obtain from: {src['url']}")
    print(f"    place files in: {dest.relative_to(ROOT)}")
    if src.get("notes"):
        print(f"    note: {' '.join(src['notes'].split())}")
    return {"resolved_pin": None, "manual": True, "present": present,
            **(_tree_stats(dest) if present else {"n_files": 0, "bytes": 0})}


DISPATCH = {
    "git": fetch_git,
    "http": fetch_http,
    "zenodo": fetch_zenodo,
    "manual": fetch_manual,
    "corpus-corporum": fetch_corpus_corporum,
}


# ---------------------------------------------------------------------------
# Lockfile
# ---------------------------------------------------------------------------

def write_lock(records: Dict[str, Any]) -> None:
    existing: Dict[str, Any] = {}
    if LOCKFILE.exists():
        with LOCKFILE.open(encoding="utf-8") as fh:
            existing = json.load(fh).get("sources", {})
    existing.update(records)
    payload = {
        "generated": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "note": "Resolved versions of fetched sources. Commit this file — it is "
                "what makes the corpus reproducible and the licence trail auditable.",
        "sources": existing,
    }
    with LOCKFILE.open("w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2, sort_keys=True)
        fh.write("\n")
    print(f"\nLockfile updated: {LOCKFILE.relative_to(ROOT)}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def cmd_list(manifest: Dict[str, Any]) -> None:
    rows = []
    for src in manifest["sources"]:
        rows.append((
            src.get("tier", "?"),
            "yes" if src.get("include") else "no",
            src.get("id", "?"),
            src.get("license", "?"),
            src.get("method", "?"),
            src.get("period", "-"),
            f"{src.get('est_words', 0):,}" if src.get("est_words") else "-",
        ))
    headers = ("TIER", "SHIP", "ID", "LICENCE", "METHOD", "PERIOD", "EST.WORDS")
    widths = [max(len(str(r[i])) for r in (*rows, headers)) for i in range(len(headers))]
    line = "  ".join(h.ljust(w) for h, w in zip(headers, widths))
    print(line)
    print("-" * len(line))
    for tier in ("green", "eval", "amber"):
        for row in rows:
            if row[0] == tier:
                print("  ".join(str(c).ljust(w) for c, w in zip(row, widths)))

    ship = [s for s in manifest["sources"] if s.get("include")]
    total = sum(s.get("est_words", 0) for s in ship)
    print(f"\n{len(ship)} source(s) marked for inclusion; "
          f"~{total:,} est. words (planning figure, not a measurement)")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--list", action="store_true", help="show manifest contents and exit")
    ap.add_argument("--validate", action="store_true", help="validate manifest and exit")
    ap.add_argument("--only", metavar="ID", help="fetch a single source by id")
    ap.add_argument("--tier", choices=sorted(VALID_TIERS), help="restrict to one tier")
    ap.add_argument("--dry-run", action="store_true", help="show what would be fetched")
    ap.add_argument("--track", choices=sorted(TRACKS), default="A",
                    help="A = redistributable sources only (default); "
                         "B = research build, adds non-commercial sources")
    ap.add_argument("--include-all", action="store_true",
                    help="ignore include:false (does NOT override the track guard)")
    args = ap.parse_args()

    manifest = load_manifest()
    errors = validate(manifest)
    if errors:
        print("Manifest validation FAILED:", file=sys.stderr)
        for err in errors:
            print(f"  - {err}", file=sys.stderr)
        return 1

    if args.validate:
        print(f"Manifest OK — {len(manifest['sources'])} sources declared.")
        return 0

    if args.list:
        cmd_list(manifest)
        return 0

    targets = selected(manifest, args.only, args.tier, args.include_all, args.track)
    if not targets:
        print("Nothing selected.")
        return 0

    # Track guard. Track A may draw only on green sources; anything else named
    # explicitly is refused rather than silently skipped, because a quiet skip
    # is how a published corpus ends up missing what someone assumed was in it.
    allowed = TRACKS[args.track]
    blocked = [s for s in targets if s.get("tier") not in allowed]
    if blocked:
        print(f"Track {args.track} may not draw on these sources:", file=sys.stderr)
        for src in blocked:
            why = " ".join((src.get("restriction") or src.get("notes") or "").split())
            print(f"  - {src['id']} [{src['tier']}]: {why[:160]}", file=sys.stderr)
        tiers = {s.get("tier") for s in blocked}
        if "eval" in tiers:
            print("\nEvaluation data lives in its own track so it can never reach a\n"
                  "training corpus by accident: re-run with --track eval.", file=sys.stderr)
        if "unknown" in tiers:
            print("\nTier `unknown` means the licence is unresolved, not merely\n"
                  "restrictive. Settle the terms before fetching.", file=sys.stderr)
        if "amber" in tiers and args.track == "A":
            print("\nTrack A is the redistributable build. Non-commercial sources belong\n"
                  "to Track B: re-run with --track B for a research build, which is NOT\n"
                  "redistributable. See feasibility.md §3.", file=sys.stderr)
        return 1

    if args.track == "eval":
        print("=" * 72)
        print("EVAL TRACK — held-out data. Never mix any of this into pretraining.")
        print("=" * 72 + "\n")

    if args.track == "B":
        research = [s for s in targets if s.get("tier") == "amber"]
        if research:
            print("=" * 72)
            print(f"TRACK B — RESEARCH BUILD. NOT REDISTRIBUTABLE.")
            print(f"{len(research)} non-commercial source(s) included:")
            for src in research:
                print(f"  - {src['id']} ({src['license']})")
            print("Nothing built from this may be published as a corpus or as weights.")
            print("=" * 72 + "\n")

    print(f"Fetching {len(targets)} source(s) into {RAW.relative_to(ROOT)}/ "
          f"[track {args.track}]\n")
    records: Dict[str, Any] = {}

    for src in targets:
        sid = src["id"]
        dest = RAW / sid
        print(f"[{sid}]  tier={src['tier']}  licence={src['license']}")
        if args.dry_run:
            print(f"    would fetch via {src['method']} -> {dest.relative_to(ROOT)}")
            continue
        try:
            dest.mkdir(parents=True, exist_ok=True)
            result = DISPATCH[src["method"]](src, dest)
            records[sid] = {
                "url": src["url"],
                "method": src["method"],
                "license": src["license"],
                "tier": src["tier"],
                "name": src.get("name"),
                "fetched": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                **result,
            }
            size = result.get("bytes", 0)
            print(f"    ok — {result.get('n_files', 0):,} files, {size/1e6:.1f} MB")
        except (RuntimeError, urllib.error.URLError, OSError) as exc:
            print(f"    FAILED: {exc}", file=sys.stderr)
            records[sid] = {"error": str(exc),
                            "attempted": datetime.now(timezone.utc).isoformat(timespec="seconds")}
        print()

    if records and not args.dry_run:
        write_lock(records)
    return 0


if __name__ == "__main__":
    sys.exit(main())
