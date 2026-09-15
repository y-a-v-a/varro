#!/usr/bin/env python3
"""A local HTTP front end for the three demos.

WHAT IT IS
    One process holding one copy of the weights, exposing lacuna, surprisal and
    aetas over JSON, plus a page that drives all three. The point is that the
    model loads once: every CLI invocation pays 0.2s of load and, more
    importantly, cannot keep a warm allocator between calls.

WHAT IT IS NOT
    Not a service. It binds 127.0.0.1 by default, has no authentication, no
    rate limiting and no sandbox, and it runs a model on any text posted to it.
    Do not put it on a public interface. --host is there for reaching it from
    another machine on a trusted network, and it prints a warning when used.

    There is one lock around all inference, so concurrent requests queue rather
    than overlapping. That is deliberate: two forward passes racing on a 16GB
    machine is how the allocator cache blew up in the first place.

Usage:
    python3 demos/serve.py
    python3 demos/serve.py --port 8080 --name latin-vii
"""

import argparse
import json
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Callable, Dict

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from _common import (  # noqa: E402
    DEFAULT_MODEL, PERIODS, Loaded, load, to_corpus_orthography,
)
import aetas  # noqa: E402
import lacuna  # noqa: E402
import surprisal  # noqa: E402
from score import score_text  # noqa: E402

WEB = HERE / "web"

# A demo server should refuse a novel rather than truncate one silently.
MAX_CHARS = 20_000
MAX_CANDIDATES = 64

STATE: Dict[str, Any] = {"model": None, "lock": threading.Lock()}


def model() -> Loaded:
    return STATE["model"]


def clean(payload: Dict[str, Any]) -> str:
    text = (payload.get("text") or "").strip()
    if not text:
        raise ValueError("no text")
    if len(text) > MAX_CHARS:
        raise ValueError(f"text is {len(text):,} characters; "
                         f"this demo accepts {MAX_CHARS:,}")
    return text


def period_of(payload: Dict[str, Any]):
    period = payload.get("period") or None
    if period and period not in PERIODS:
        raise ValueError(f"unknown period {period!r}")
    return period


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

def api_model(_: Dict[str, Any]) -> Dict[str, Any]:
    m = model()
    calibration = aetas.load_calibration(m.name)
    return {
        "name": m.name,
        "parameters": m.run["parameters"],
        "corpus": m.cfg.get("corpus"),
        "perplexity": m.run["final"]["val_perplexity"],
        "periods": list(PERIODS),
        "period_tags": m.tagged,
        "calibrated": calibration is not None,
        "era_accuracy": calibration["accuracy"] if calibration else None,
        "max_chars": MAX_CHARS,
    }


def api_lacuna(payload: Dict[str, Any]) -> Dict[str, Any]:
    m = model()
    text = clean(payload)
    period = period_of(payload)
    raw = bool(payload.get("raw"))
    words = max(1, min(int(payload.get("words") or 1), 8))
    k = max(4, min(int(payload.get("candidates") or 32), MAX_CANDIDATES))

    # Split before folding: normalize() strips square brackets, which would
    # delete the gap this endpoint exists to fill.
    parts, n_gaps = lacuna.split_gaps(text)
    if n_gaps == 0:
        raise ValueError("no gap in the text -- mark one with [...]")

    folding: Dict[str, int] = {}
    if not raw:
        folded = []
        for part in parts:
            piece, changes = to_corpus_orthography(part)
            folded.append(piece)
            for what, n in changes.items():
                folding[what] = folding.get(what, 0) + n
        parts = folded

    given = [to_corpus_orthography(c)[0] if not raw else c
             for c in (payload.get("candidates_given") or []) if c.strip()]

    gaps = []
    filled = [parts[0].strip()]
    for gap in range(n_gaps):
        left = " ".join(p for p in filled if p)
        right = parts[gap + 1].strip()
        proposals = lacuna.propose(m, left, words, k, 1.0, 60, 7 + gap, period)
        pool = list(dict.fromkeys(given + proposals))
        if not pool:
            raise ValueError("the model proposed nothing usable")

        rows = lacuna.rank(m, left, pool, right, period)
        blind = lacuna.rank(m, left, pool, "", period)
        best = rows[0]["logprob"]
        best_blind = blind[0]["logprob"]
        import math
        gaps.append({
            "left": left, "right": right, "n_candidates": len(pool),
            "ranked": [{"fill": r["fill"], "tokens": r["fill_tokens"],
                        "bits_worse": (best - r["logprob"]) / math.log(2)}
                       for r in rows[:15]],
            "blind": [{"fill": r["fill"], "tokens": r["fill_tokens"],
                       "bits_worse": (best_blind - r["logprob"]) / math.log(2)}
                      for r in blind[:15]],
        })
        filled.append(rows[0]["fill"])
        filled.append(right)

    return {
        "folding": folding,
        "gaps": gaps,
        "reconstruction": " ".join(" ".join(p for p in filled if p).split()),
    }


def api_surprisal(payload: Dict[str, Any]) -> Dict[str, Any]:
    m = model()
    text = clean(payload)
    period = period_of(payload)
    folding: Dict[str, int] = {}
    if not payload.get("raw"):
        text, folding = to_corpus_orthography(text)

    result = score_text(m.model, m.tok, text, m.context, period)
    words = surprisal.to_words(m.tok, result["ids"], result["logprobs"],
                               result["prefix_len"])
    if not words:
        raise ValueError("nothing scorable in that input")

    per_word = sorted(w.bits for w in words)
    return {
        "folding": folding,
        "perplexity": result["perplexity"],
        "n_tokens": result["n_tokens"],
        "median_bits": per_word[len(per_word) // 2],
        "p90_bits": per_word[int(len(per_word) * 0.9)],
        "bands": [b[0] for b in surprisal.BANDS[:-1]],
        "words": [{"text": w.text, "bits": w.bits, "tokens": w.tokens}
                  for w in words],
    }


def api_aetas(payload: Dict[str, Any]) -> Dict[str, Any]:
    m = model()
    if not m.tagged:
        raise ValueError(f"{m.name} was trained without period tags")
    text = clean(payload)
    folding: Dict[str, int] = {}
    if not payload.get("raw"):
        text, folding = to_corpus_orthography(text)

    scores = aetas.era_scores(m, text)
    best, second, margin = aetas.verdict(scores)
    calibration = aetas.load_calibration(m.name)
    band = aetas.band_of(margin)
    bucket = (calibration or {}).get("bands", {}).get(band)

    return {
        "folding": folding,
        "best": best, "runner_up": second,
        "margin_bits": margin, "band": band,
        "n_tokens": scores[best]["n_tokens"],
        "scores": {p: s["perplexity"] for p, s in scores.items()},
        "calibration": None if calibration is None else {
            "n": calibration["n"],
            "accuracy": calibration["accuracy"],
            "band_n": (bucket or {}).get("n"),
            "band_accuracy": (bucket or {}).get("accuracy"),
            "chance": 1 / len(PERIODS),
        },
    }


ROUTES: Dict[str, Callable[[Dict[str, Any]], Dict[str, Any]]] = {
    "/api/model": api_model,
    "/api/lacuna": api_lacuna,
    "/api/surprisal": api_surprisal,
    "/api/aetas": api_aetas,
}


# ---------------------------------------------------------------------------
# Plumbing
# ---------------------------------------------------------------------------

class Handler(BaseHTTPRequestHandler):
    server_version = "varro-demos"

    def log_message(self, fmt, *args):
        sys.stderr.write(f"  {self.address_string()} {fmt % args}\n")

    def _send(self, code: int, body: bytes, content_type: str) -> None:
        self.send_response(code)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        # No external anything: the page is self-contained, and saying so
        # keeps a stray CDN reference from working by accident in development
        # and then failing for anyone offline.
        self.send_header("Content-Security-Policy",
                         "default-src 'self' 'unsafe-inline'")
        self.end_headers()
        self.wfile.write(body)

    def _json(self, code: int, payload: Dict[str, Any]) -> None:
        self._send(code, json.dumps(payload).encode("utf-8"),
                   "application/json; charset=utf-8")

    def do_GET(self) -> None:
        path = self.path.split("?")[0]
        if path in ("/", "/index.html"):
            page = WEB / "index.html"
            if not page.exists():
                self._json(500, {"error": f"missing {page}"})
                return
            self._send(200, page.read_bytes(), "text/html; charset=utf-8")
        elif path == "/api/model":
            with STATE["lock"]:
                self._json(200, api_model({}))
        else:
            self._json(404, {"error": f"no route {path}"})

    def do_POST(self) -> None:
        path = self.path.split("?")[0]
        handler = ROUTES.get(path)
        if handler is None:
            self._json(404, {"error": f"no route {path}"})
            return
        length = int(self.headers.get("Content-Length") or 0)
        if length > MAX_CHARS * 4:
            self._json(413, {"error": "request too large"})
            return
        try:
            payload = json.loads(self.rfile.read(length) or b"{}")
        except json.JSONDecodeError as exc:
            self._json(400, {"error": f"bad JSON: {exc}"})
            return

        try:
            with STATE["lock"]:
                self._json(200, handler(payload))
        except ValueError as exc:
            self._json(400, {"error": str(exc)})
        except Exception as exc:  # noqa: BLE001 -- a demo must not 500 silently
            self.log_message("unhandled %s: %s", type(exc).__name__, exc)
            self._json(500, {"error": f"{type(exc).__name__}: {exc}"})


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--name", default=DEFAULT_MODEL)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8000)
    args = ap.parse_args()

    STATE["model"] = load(args.name)
    m = model()
    print(m.describe())
    if not m.tagged:
        print("  [no period tags in this model: /api/aetas will refuse]")
    if aetas.load_calibration(m.name) is None:
        print(f"  [no reports/era-{m.name}.json: the era panel will report an "
              f"uncalibrated ranking]")
    if args.host not in ("127.0.0.1", "localhost", "::1"):
        print(f"\n  WARNING: binding {args.host}. This server has no "
              f"authentication and runs\n  a model on anything posted to it. "
              f"Trusted networks only.")

    server = ThreadingHTTPServer((args.host, args.port), Handler)
    print(f"\nhttp://{args.host}:{args.port}   (ctrl-c to stop)\n")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nstopped")
    return 0


if __name__ == "__main__":
    sys.exit(main())
