"""Every script under src/ must run from a clone anywhere.

src/token_growth.py carried `ROOT = Path("/Users/vincentb/Projects/claudius")`,
which worked on exactly one machine. That is the opposite of what this project
promises: the corpus is meant to be rebuildable from the manifest by anyone,
and a stage that only runs in one home directory breaks the chain quietly —
it does not fail on a fresh clone, it fails on someone else's.
"""

import ast
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "src"

# How every script locates the repository: from its own position on disk.
CANONICAL_ROOT = "Path(__file__).resolve().parent.parent"

ABSOLUTE = re.compile(r"""["'](/(?:Users|home|var|opt|tmp|private)/[^"']*)["']""")


def sources():
    return sorted(SRC.glob("*.py"))


def test_no_script_hardcodes_an_absolute_path():
    offenders = {}
    for path in sources():
        found = ABSOLUTE.findall(path.read_text(encoding="utf-8"))
        if found:
            offenders[path.name] = found
    assert not offenders, f"absolute paths baked into src/: {offenders}"


def test_every_script_that_needs_the_repo_finds_it_the_same_way():
    """One idiom, so a reader knows what ROOT means without checking."""
    wrong = {}
    for path in sources():
        text = path.read_text(encoding="utf-8")
        for line in text.splitlines():
            if line.startswith("ROOT ") and CANONICAL_ROOT not in line:
                wrong[path.name] = line.strip()
    assert not wrong, f"ROOT derived some other way: {wrong}"


def test_the_scan_is_not_vacuous():
    assert len(sources()) > 15
    assert any("ROOT = " in p.read_text(encoding="utf-8") for p in sources())


def test_no_script_does_its_work_at_import_time():
    """conftest puts src/ on sys.path, so an import must not start a long job.

    token_growth.py tokenised four corpora at module level. Importing it — which
    any test collecting src/ would do — was a multi-minute job with side effects.
    """
    # Import-time setup that is not work: making src/ importable the way the
    # scripts import each other, which conftest.py does for the same reason.
    allowed = {"sys.path.insert", "sys.path.append"}

    def call_name(node):
        func = node.func
        parts = []
        while isinstance(func, ast.Attribute):
            parts.append(func.attr)
            func = func.value
        if isinstance(func, ast.Name):
            parts.append(func.id)
        return ".".join(reversed(parts))

    naked = []
    for path in sources():
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in tree.body:
            if isinstance(node, (ast.For, ast.While)):
                naked.append(f"{path.name}:{node.lineno} loop")
            elif (isinstance(node, ast.Expr) and isinstance(node.value, ast.Call)
                    and call_name(node.value) not in allowed):
                naked.append(f"{path.name}:{node.lineno} {call_name(node.value)}()")
    assert not naked, f"work done at import time: {naked}"


def test_every_script_answers_help_rather_than_doing_the_work():
    """`--help` must never start the job.

    token_growth.py had no argument parser, so argv was ignored entirely: the
    typo you meant to catch, and `--help` itself, both tokenised four corpora
    and rewrote token-growth.json. A script whose arguments are silently
    discarded is a trap even when it takes none.
    """
    import subprocess
    import sys as _sys
    missing = []
    for path in sources():
        if path.name == "encoding.py":       # a library, not a command
            continue
        done = subprocess.run([_sys.executable, str(path), "--help"],
                              capture_output=True, text=True, timeout=90)
        if done.returncode != 0 or "usage:" not in done.stdout:
            missing.append(path.name)
    assert not missing, f"no argument parser reached before the work: {missing}"


# ---------------------------------------------------------------------------
# One more thing that has to hold across every script, for the same kind of
# reason: a rule kept in one place and broken in another is not a rule.
# ---------------------------------------------------------------------------

# `latin-bpe-{...}.json` assembled from a vocabulary size. Loading a model
# through whatever file currently has that name is how a retrained tokenizer
# turns a trained checkpoint into fluent nonsense, silently — it happened on
# 2026-08-20. encoding.resolve_tokenizer exists to refuse that, and sft.py
# named the file directly anyway because the function lived in sample.py,
# which cannot be imported without MLX.
BY_NAME = re.compile(r'latin-bpe-\{|latin-bpe-\{vocab')

# Who is allowed to build that name. tokenizer.py writes the file; encoding.py
# is the resolver itself; evaluate.py and token_growth.py measure the corpus
# and load no checkpoint, so they have no embedding table to misalign — and
# evaluate.py records the digest it used, which compare.py checks against the
# run's before it will call two numbers comparable.
# train.py names it twice and both are correct: a FRESH run is defined by the
# tokenizer currently on disk and stamps it into its own report, and the other
# site is the stamping itself. The case that is not safe there is --resume,
# which is checked in load_checkpoint against the fingerprint the checkpoint
# carries. bundle.py builds the content-named archive path from the run's
# recorded sha and then re-hashes the file, which is stricter than the
# resolver, not looser.
MAY_NAME_A_TOKENIZER = {"tokenizer.py", "encoding.py", "evaluate.py",
                        "token_growth.py", "train.py", "bundle.py"}


def test_only_the_resolver_names_a_tokenizer_file():
    offenders = {}
    for path in sources():
        if path.name in MAY_NAME_A_TOKENIZER:
            continue
        hits = [line.strip() for line in path.read_text(encoding="utf-8").splitlines()
                if BY_NAME.search(line) and not line.strip().startswith("#")]
        if hits:
            offenders[path.name] = hits
    assert not offenders, (
        "these load a tokenizer by name instead of through "
        f"encoding.resolve_tokenizer: {offenders}")


def test_the_tokenizer_scan_is_not_vacuous():
    """The allowlist must still contain the pattern it is exempting."""
    text = (SRC / "tokenizer.py").read_text(encoding="utf-8")
    assert BY_NAME.search(text)


def test_every_script_that_loads_a_checkpoint_resolves_its_vocabulary():
    """The positive half: naming nothing is not the same as checking."""
    for name in ("sample.py", "sft.py", "score.py"):
        text = (SRC / name).read_text(encoding="utf-8")
        assert "resolve_tokenizer(" in text, f"{name} loads a model unchecked"


def test_resuming_a_run_checks_the_vocabulary_it_started_with():
    """train.py may name the tokenizer, but not for --resume.

    A fresh run defines its own vocabulary. A resumed one inherits embeddings
    keyed to whatever the earlier steps used, so it has to be told what that
    was and refuse if the file has moved underneath it.
    """
    text = (SRC / "train.py").read_text(encoding="utf-8")
    assert "def load_checkpoint(name: str, model, optimizer,\n" in text
    assert "tok_fingerprint)" in text, "--resume passes no fingerprint to check"
