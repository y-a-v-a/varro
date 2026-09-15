"""requirements.txt must cover every third-party import under src/.

README.md's quickstart is `pip install -r requirements.txt`, so a dependency
that exists only in the author's venv is a reproducibility hole — and this
project's central claim is that its corpus and its models can be rebuilt from
the manifest. mlx and tokenizers were both undeclared while being hard
requirements of the training half and of the tokenizer stage respectively.

Written as a scan rather than a hand-kept list, so adding an import to src/ is
what fails the test, not remembering to update it.
"""

import ast
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "src"
REQUIREMENTS = ROOT / "requirements.txt"

# Distribution name -> the module name it installs under, where they differ.
INSTALLS_AS = {"pyyaml": {"yaml"}, "mlx": {"mlx"}, "tokenizers": {"tokenizers"},
               "pytest": {"pytest"}}


def declared():
    """Module names the requirements file promises will be importable."""
    names = set()
    for line in REQUIREMENTS.read_text(encoding="utf-8").splitlines():
        line = line.split("#")[0].split(";")[0].strip()
        if not line:
            continue
        dist = line.split(">=")[0].split("==")[0].split("[")[0].strip().lower()
        names |= INSTALLS_AS.get(dist, {dist.replace("-", "_")})
    return names


def imported():
    """Top-level modules imported anywhere under src/, by parsing rather than running."""
    found = {}
    for path in sorted(SRC.glob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    found.setdefault(alias.name.split(".")[0], set()).add(path.name)
            elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
                found.setdefault(node.module.split(".")[0], set()).add(path.name)
    return found


def third_party():
    """Imports that are neither stdlib nor another module in src/."""
    local = {p.stem for p in SRC.glob("*.py")}
    return {name: where for name, where in imported().items()
            if name not in sys.stdlib_module_names and name not in local}


def test_every_third_party_import_is_declared():
    missing = {name: sorted(where) for name, where in third_party().items()
               if name not in declared()}
    assert not missing, (
        f"undeclared in requirements.txt: {missing}. "
        f"`pip install -r requirements.txt` would not make src/ runnable.")


def test_the_scan_finds_the_dependencies_we_know_about():
    """A scan that silently found nothing would pass the test above vacuously."""
    assert {"yaml", "tokenizers", "mlx"} <= set(third_party())


def test_mlx_is_marked_apple_silicon_only():
    """The corpus half must stay installable on a machine MLX has no wheel for."""
    line = next(l for l in REQUIREMENTS.read_text(encoding="utf-8").splitlines()
                if l.strip().startswith("mlx"))
    assert "sys_platform ==" in line and "platform_machine ==" in line


def test_the_tested_interpreter_is_recorded():
    """3.14 is not incidental: annotations are lazy there and eager before it.

    src/train.py carried `Optional` in a signature without importing it and ran
    anyway for exactly that reason. The floor belongs somewhere a reader will
    find it rather than in the shape of a traceback.
    """
    quickstart = (ROOT / "README.md").read_text(encoding="utf-8")
    assert "python3.14" in quickstart
