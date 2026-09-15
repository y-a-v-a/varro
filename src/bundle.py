#!/usr/bin/env python3
"""Assemble a Hugging Face bundle for a trained model — dist/<name>/.

The first bundle was assembled by hand: files copied, numbers transcribed
from reports into a model card, a config.json typed out. Hand-transcribing
numbers is how this project's docs drifted from its builds before, and a
hand-assembled bundle also sits outside the Track A guard that fetch.py,
build.py and provenance.py all enforce. This script closes both holes:

  1. Every figure in the published card comes from a generated report —
     the training report, the corpus-matched baseline, the memorisation
     report — or the assembly fails. Prose lives in a template under
     docs/modelcards/<name>.md; numbers are substituted, never retyped.

  2. Only Track A models can be bundled for publication. LATIN-LII trains on
     NonCommercial material and its weights inherit that restriction; asking
     for it here is a hard error, not a warning.

     --research-only lifts that for local inspection alone. It cannot be
     reached by accident, it writes to dist/research/ rather than the dist/
     directory the upload command points at, and the bundle it produces
     declares its own restriction: NC licence in the card frontmatter, a
     research-only header, and a DO-NOT-REDISTRIBUTE.md beside the weights.
     A bundle that escapes should say what it is without needing this script.

The same fingerprint discipline as train.py applies: the baseline floor is
quoted only if its report describes the same corpus and the same tokenizer
the model trained with. A floor measured through a different tokenizer is
refused, not approximated.

Usage:
    python src/bundle.py --name latin-vii
    hf upload <user>/latin-vii dist/latin-vii . --repo-type model

    python src/bundle.py --name latin-lii-latinitas --research-only
"""

import argparse
import hashlib
import json
import shutil
import string
import sys
from pathlib import Path
from typing import Any, Dict, List, NoReturn

ROOT = Path(__file__).resolve().parent.parent
REPORTS = ROOT / "reports"
CHECKPOINTS = ROOT / "checkpoints"
TOKENIZERS = ROOT / "data" / "processed" / "tokenizers"
MODELCARDS = ROOT / "docs" / "modelcards"
DIST = ROOT / "dist"
RESEARCH_DIST = DIST / "research"
LOADER = ROOT / "src" / "hf_loader.py"

# The HF frontmatter wants a licence identifier from its fixed list. Only
# licences this project can actually produce are mapped; anything else is a
# decision for a person, not a default.
HF_LICENSE_IDS = {
    "CC-BY-SA-4.0": "cc-by-sa-4.0",
    "CC-BY-4.0": "cc-by-4.0",
    "CC-BY-SA-4.0 + UBPL-1.0": "cc-by-sa-4.0",
}

# Reachable only under --research-only. Kept separate so a NonCommercial
# licence can never be resolved on the publication path: a corpus that
# resolves to NC while claiming Track A is a contradiction, and it should
# fail rather than quietly stamp the bundle with an NC identifier.
RESEARCH_LICENSE_IDS = {
    "CC-BY-NC-SA-4.0": "cc-by-nc-sa-4.0",
    "NonCommercial — research only": "other",
}

REDISTRIBUTION_NOTICE = """# Do not redistribute

These weights were trained on the Varro **Track B** corpus, which includes
NonCommercial sources. They are a research artifact: publishable as results,
not as a download.

- Effective licence of the training corpus: **{licence}**
- NonCommercial sources: {nc_sources}

The project's position is that NonCommercial propagates into weights, which
is why `src/bundle.py` refuses Track B models on the publication path and why
this bundle was written to `dist/research/` instead. It exists so the result
can be inspected and run locally.

If you want a Latin model to publish or build on, use LATIN-VII, whose corpus
carries no NonCommercial term anywhere: https://huggingface.co/vincent-b/latin-vii

Attribution for every source, including the NonCommercial ones, is in
ATTRIBUTION.md — the BY in CC BY-NC-SA is the same BY as anywhere else.
"""


def fail(msg: str) -> NoReturn:
    sys.exit(f"bundle: {msg}")


def load_report(path: Path, hint: str) -> Dict[str, Any]:
    if not path.exists():
        fail(f"{path.relative_to(ROOT)} not found — {hint}")
    with path.open(encoding="utf-8") as fh:
        return json.load(fh)


def guard_track(train_report: Dict[str, Any], name: str,
                research_only: bool = False) -> str:
    """Refuse to bundle a model whose training corpus was not Track A.

    The track is read from the training report — what the model actually
    saw — not from the corpus currently on disk, which may have been
    rebuilt since. Returns the track, so callers need not re-read it.

    `research_only` permits Track B for local inspection. It never permits
    an unknown track: a report that does not say what it trained on is a
    reason to stop, not a reason to guess.
    """
    track = (train_report.get("corpus") or {}).get("track")
    if track == "A":
        return track
    if track == "B" and research_only:
        return track
    if track == "B":
        fail(f"{name} trained on a Track {track!r} corpus. Only Track A "
             f"(redistributable) models may be bundled; NonCommercial terms "
             f"propagate into the weights. Pass --research-only to assemble "
             f"it locally for inspection instead.")
    fail(f"{name} has no usable track in its training report (found "
         f"{track!r}). Retrain or repair the report; the track is derived, "
         f"never assumed.")


def matched_floor(train_report: Dict[str, Any], corpus: str) -> float:
    """The bigram floor, only if measured on this model's corpus + tokenizer."""
    report = load_report(REPORTS / f"baselines-{corpus}.json",
                         "run `python src/evaluate.py --baselines` first")
    for key in ("corpus", "tokenizer"):
        recorded, trained = report.get(key), train_report.get(key)
        if recorded != trained:
            fail(f"baselines-{corpus}.json describes a different {key} than "
                 f"the model trained on ({recorded} != {trained}). "
                 f"Re-run the baseline; a mismatched floor is not quotable.")
    return report["baselines"]["overall_perplexity"]


def fmt_m(n: float) -> str:
    """151_552_000 -> '151.5M' — the card's compact figure style."""
    if n >= 1e9:
        return f"{n / 1e9:.2f}B"
    if n >= 1e6:
        return f"{n / 1e6:.1f}M".replace(".0M", "M")
    return f"{n:,.0f}"


def source_table(rows: List[Dict[str, Any]], mark_nc: bool = False) -> str:
    """The per-source table.

    `mark_nc` flags the sources that restrict the whole corpus — on a Track B
    card the reader needs to see which two of eleven sources are the reason
    the weights cannot be redistributed.
    """
    lines = []
    for r in rows:
        licence = "public domain" if r["license"] == "public-domain" else r["license"]
        if mark_nc and r["tier"] == "amber":
            licence += " **(NC)**"
        lines.append(f"| {r['name']} | {r['share']:.1%} | {licence} |")
    return "\n".join(lines)


def memo_table(curve: Dict[str, Dict[str, float]]) -> str:
    lines = []
    for window in sorted(curve, key=int):
        row = curve[window]
        lines.append(f"| {window} | {row['text']:.1%} | {row['control']:.1%} |")
    return "\n".join(lines)


def build_values(name: str, train_report: Dict[str, Any],
                 floor: float, memo_report: Dict[str, Any],
                 sources: List[Dict[str, Any]], effective_license: str,
                 default_period: str,
                 research_only: bool = False) -> Dict[str, str]:
    cfg = train_report["config"]
    corpus_info = train_report["corpus"]
    final = train_report["final"]
    ppl = final["val_perplexity"]
    params = train_report["parameters"]
    train_tokens = train_report["train_tokens"]
    tokens_seen = cfg["steps"] * cfg["batch"] * cfg["context"]
    memo = memo_report["memorization"]

    known = dict(HF_LICENSE_IDS)
    if research_only:
        known.update(RESEARCH_LICENSE_IDS)
    if effective_license not in known:
        hint = ("" if research_only else
                " (NonCommercial licences map only under --research-only)")
        fail(f"no HF licence identifier mapped for {effective_license!r}"
             f"{hint} — decide one and add it to HF_LICENSE_IDS")

    return {
        "name": name,
        "recipe": cfg["config"],
        "hf_license": known[effective_license],
        "effective_license": effective_license,
        "track": corpus_info["track"],
        "parameters": f"{params:,}",
        "parameters_m": f"{params / 1e6:.2f}M",
        "parameters_m_round": f"{params / 1e6:.0f}M",
        "corpus_words": f"{corpus_info['words']:,}",
        "corpus_words_m": f"{corpus_info['words'] / 1e6:.2f}M",
        "records": f"{corpus_info['records']:,}",
        "tokens_seen_m": fmt_m(tokens_seen),
        "train_tokens_m": f"{train_tokens / 1e6:.2f}M",
        "train_tokens_m_round": f"{train_tokens / 1e6:.0f}M",
        "tokens_per_param": f"{tokens_seen / params:.0f}",
        "epochs": f"{train_report['epochs']:.1f}",
        "ppl": f"{ppl:.1f}",
        "floor": f"{floor:.1f}",
        "ratio": f"{floor / ppl:.2f}",
        "memorised": f"{memo['memorised']}",
        "longest_run": f"{memo['longest_verbatim_words']}",
        "trained_date": train_report["generated"][:10],
        "hours": f"{final['elapsed_s'] / 3600:.0f}",
        "vocab_fmt": f"{cfg['vocab']:,}",
        "dim": str(cfg["dim"]),
        "layers": str(cfg["layers"]),
        "heads": str(cfg["heads"]),
        "context": str(cfg["context"]),
        "tokenizer_sha": train_report["tokenizer"]["sha256"],
        "period_tag": default_period or "none",
        "period_tag_lower": (default_period or "none").lower(),
        "nc_share": f"{sum(r['share'] for r in sources if r['tier'] == 'amber'):.1%}",
        "source_table": source_table(sources, mark_nc=research_only),
        "memo_table": memo_table(memo_report["curve"]),
    }


def render_card(name: str, values: Dict[str, str]) -> str:
    template_path = MODELCARDS / f"{name}.md"
    if not template_path.exists():
        fail(f"no model card template at {template_path.relative_to(ROOT)} — "
             f"the prose is written by a person; only the numbers are filled in")
    try:
        return string.Template(template_path.read_text(encoding="utf-8")) \
            .substitute(values)
    except KeyError as exc:
        fail(f"template placeholder ${{{exc.args[0]}}} has no value — "
             f"add it in build_values()")


def default_period_tag(train_report: Dict[str, Any]) -> str:
    """The tag every training document carried, if there was exactly one."""
    if not train_report["config"].get("period_tags"):
        return None
    periods = (train_report.get("corpus") or {}).get("period_filter") or []
    return periods[0] if len(periods) == 1 else None


def write_config(out: Path, name: str, train_report: Dict[str, Any],
                 floor: float, memo_report: Dict[str, Any],
                 tokenizer_path: Path, default_period: str) -> None:
    cfg = train_report["config"]
    tok = json.loads(tokenizer_path.read_text(encoding="utf-8"))
    ppl = train_report["final"]["val_perplexity"]
    config = {
        "model_type": "gpt-varro",
        "architecture": ("decoder-only transformer, pre-norm, tied embeddings, "
                         "learned positions"),
        "vocab_size": cfg["vocab"],
        "dim": cfg["dim"],
        "layers": cfg["layers"],
        "heads": cfg["heads"],
        "context": cfg["context"],
        "mlp_ratio": 4,
        "activation": "gelu",
        "bias": False,
        "tie_word_embeddings": True,
        "parameters": train_report["parameters"],
        "default_period_tag": default_period,
        "redistributable": train_report["corpus"]["track"] == "A",
        "special_tokens": [t["content"] for t in tok.get("added_tokens", [])],
        "training": {
            "corpus": train_report["corpus_name"],
            "track": train_report["corpus"]["track"],
            "corpus_words": train_report["corpus"]["words"],
            "train_tokens": train_report["train_tokens"],
            "epochs": round(train_report["epochs"], 1),
            "steps": cfg["steps"],
            "batch": cfg["batch"],
            "lr": cfg["lr"],
            "warmup": cfg["warmup"],
            "weight_decay": cfg["weight_decay"],
            "seed": cfg["seed"],
            "tokenizer_sha256": train_report["tokenizer"]["sha256"],
            "recipe": cfg["config"],
            "framework": "mlx",
        },
        "evaluation": {
            "held_out_perplexity": round(ppl, 1),
            "bigram_floor_same_split": round(floor, 1),
            "times_better_than_floor": round(floor / ppl, 2),
            "memorised_12word_windows": memo_report["memorization"]["memorised"],
            "longest_verbatim_run_words":
                memo_report["memorization"]["longest_verbatim_words"],
        },
    }
    (out / "config.json").write_text(json.dumps(config, indent=2) + "\n",
                                     encoding="utf-8")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--name", required=True, help="run name, e.g. latin-vii")
    ap.add_argument("--research-only", action="store_true",
                    help="assemble a Track B model into dist/research/ for "
                         "local inspection. Never publishable.")
    args = ap.parse_args()
    name = args.name

    train_report = load_report(REPORTS / f"train-{name}.json",
                               "no training report; has this model been trained?")
    track = guard_track(train_report, name, args.research_only)
    research = track == "B"
    corpus = train_report["corpus_name"]
    floor = matched_floor(train_report, corpus)
    memo_report = load_report(
        REPORTS / f"memorization-{name}.json",
        f"run `python src/sample.py --name {name}` and "
        f"`python src/evaluate.py --memorization` first")
    if memo_report.get("tokenizer") != train_report.get("tokenizer"):
        fail(f"memorization-{name}.json was measured through a different "
             f"tokenizer than the model trained with")

    # Weights and the exact tokenizer, verified by content hash rather than
    # trusted from the filename.
    weights = CHECKPOINTS / f"{name}.safetensors"
    if not weights.exists():
        fail(f"no checkpoint at {weights.relative_to(ROOT)}")
    sha = train_report["tokenizer"]["sha256"]
    vocab = train_report["config"]["vocab"]
    tokenizer_path = TOKENIZERS / f"latin-bpe-{vocab}.{sha}.json"
    if not tokenizer_path.exists():
        fail(f"tokenizer {tokenizer_path.relative_to(ROOT)} not found — the "
             f"fingerprinted copy of the tokenizer this model trained with")
    actual = hashlib.sha256(tokenizer_path.read_bytes()).hexdigest()[:16]
    if actual != sha:
        fail(f"tokenizer file content hashes to {actual}, filename says {sha}")

    # Attribution and the source table come from the corpus on disk, so they
    # must describe the same build the model trained on.
    import provenance
    records, _, _ = provenance.load_corpus(corpus)
    if len(records) != train_report["corpus"]["records"]:
        fail(f"corpus {corpus!r} on disk has {len(records)} records; the model "
             f"trained on {train_report['corpus']['records']}. Rebuild the "
             f"corpus the model saw, or retrain, before bundling.")
    sources = provenance.summarise(records, provenance.load_manifest())
    effective_license = provenance.corpus_licence(sources)
    attribution = provenance.attribution_path(corpus)
    if not attribution.exists():
        fail(f"{attribution.name} not found — run "
             f"`python src/provenance.py --corpus {corpus}` first")

    default_period = default_period_tag(train_report)
    values = build_values(name, train_report, floor, memo_report,
                          sources, effective_license, default_period, research)
    card = render_card(name, values)

    out = (RESEARCH_DIST if research else DIST) / name
    out.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(weights, out / "model.safetensors")
    shutil.copyfile(tokenizer_path, out / "tokenizer.json")
    shutil.copyfile(attribution, out / "ATTRIBUTION.md")
    shutil.copyfile(LOADER, out / "generate.py")
    write_config(out, name, train_report, floor, memo_report,
                 tokenizer_path, default_period)
    (out / "README.md").write_text(card, encoding="utf-8")

    if research:
        nc = [r["name"] for r in sources
              if r["tier"] == "amber"] or ["(none listed)"]
        (out / "DO-NOT-REDISTRIBUTE.md").write_text(
            REDISTRIBUTION_NOTICE.format(licence=effective_license,
                                         nc_sources=", ".join(nc)),
            encoding="utf-8")

    rel = out.relative_to(ROOT)
    print(f"bundled {name} -> {rel}/")
    for f in sorted(out.iterdir()):
        print(f"  {f.name:24} {f.stat().st_size:>12,} bytes")
    if research:
        print(f"\nTrack B — research only. Effective licence "
              f"{effective_license}.\nDo not upload this bundle. "
              f"LATIN-VII is the publishable model.")
    else:
        print(f"\nupload with:\n"
              f"  hf upload <user>/{name} {rel} . --repo-type model")
    return 0


if __name__ == "__main__":
    sys.exit(main())
