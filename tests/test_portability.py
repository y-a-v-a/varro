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
