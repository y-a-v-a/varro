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
import shutil
import subprocess
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
}
VALID_METHODS = {"git", "http", "zenodo", "manual"}
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

        # An amber source with include:true is now correct — it means "wanted,
        # in Track B". What must never happen is amber reaching a Track A build,
        # and that is enforced at fetch and build time rather than here.
        if src.get("tier") == "eval" and src.get("include"):
            errors.append(
                f"{where}: tier `eval` with include:true — evaluation data must "
                f"never be mixed into pretraining"
            )
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
             include_all: bool) -> List[Dict[str, Any]]:
    out = []
    for src in manifest["sources"]:
        if only and src["id"] != only:
            continue
        if tier and src.get("tier") != tier:
            continue
        if not only and not include_all and not src.get("include"):
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
            if total:
                pct = 100 * done / total
                print(f"\r    {done/1e6:.1f}/{total/1e6:.1f} MB ({pct:.0f}%)",
                      end="", flush=True)
            else:
                print(f"\r    {done/1e6:.1f} MB", end="", flush=True)
    print()


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

    targets = selected(manifest, args.only, args.tier, args.include_all)
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
        if args.track == "A":
            print("\nTrack A is the redistributable build. Non-commercial sources belong\n"
                  "to Track B: re-run with --track B for a research build, which is NOT\n"
                  "redistributable. See feasibility.md §3.", file=sys.stderr)
        else:
            print("\nTier `unknown` means the licence is unresolved, not merely\n"
                  "restrictive. Settle the terms before fetching.", file=sys.stderr)
        return 1

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
