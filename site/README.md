# site/ — the public page

A static, dependency-free website for the project: what the corpus is, how it
is built, the model versions, the training curves, and every source with its
licence. Plain HTML, CSS and JavaScript; no build step.

```
index.html        the page — prose lives here
styles.css        parchment by day, lamplight by night
app.js            draws the figures, versions and sources from data.js
theme.js          applies a remembered theme before first paint
data.js           GENERATED — every number on the page
favicon.svg, og.png
vercel.json       security headers, clean URLs
content/
  models.yaml     the model versions: names, status, what each one showed
  sources.yaml    a public one-liner per source (optional)
tools/
  sync.py         regenerates data.js from the repository's own records
  og.html         source for og.png
```

## Nothing numeric is written by hand

`data.js` is generated from files the pipeline already maintains, so the page
cannot drift a build behind:

| on the page | read from |
|---|---|
| sources, licences, tiers, periods | `manifest/sources.yaml` |
| what each source contributed | `ATTRIBUTION.md`, `ATTRIBUTION-classical.md`, `ATTRIBUTION-corpus-b.md` |
| corpus totals, periods, tokens | `data/processed/*.summary.json`, `token-growth.json` |
| parameters, curves, perplexity | `reports/train-*.json` |
| recitation measurements | `reports/memorization-*.json` |
| bigram floors | `reports/baselines-*.json` (only where corpus *and* tokenizer match) |

```bash
source .venv/bin/activate
python site/tools/sync.py           # rewrite site/data.js
python site/tools/sync.py --check   # exit 1 if it is out of date
```

`data/` is gitignored; on a machine without a built corpus, `sync.py` keeps the
corpus totals already in `data.js` and says so.

## Extending the corpus

1. Add the source to `manifest/sources.yaml` and build as usual.
2. Regenerate the notices: `python src/provenance.py --check`, and again with
   `--corpus classical` and `--corpus corpus-b`.
3. Optionally give it a line in `site/content/sources.yaml`.
4. `python site/tools/sync.py`, and commit `site/data.js`.

The source appears under the right heading on its own — Track A, Track B, held
out, or not yet in the corpus — because the grouping is derived from its tier,
`include` flag and whether an attribution notice credits it, exactly as track
membership is in the pipeline. `hidden: true` in `content/sources.yaml` keeps a
source off the page without touching the manifest.

## Adding a model

Train it so `reports/train-<run>.json` exists, add an entry to
`site/content/models.yaml` (the file documents every field), and run `sync.py`.
Set `curve: true` to draw it in the perplexity figure. The figure has seven
validated pigment colours, assigned in training order; an eighth run is drawn in
the neutral control colour rather than an invented hue — switch an old run to
`curve: hidden` or remove its `curve` before that happens.

## Local preview

```bash
cd site && python3 -m http.server 8000     # http://localhost:8000
```

## Deploying to Vercel

1. Import the repository in Vercel.
2. **Root Directory: `site`**. Framework preset: *Other*. Leave the build
   command empty and the output directory as the default.
3. Deploy. `.vercelignore` keeps `content/`, `tools/` and this README off the
   served site.

With the CLI: `cd site && vercel` (then `vercel --prod`).

The production URL is `https://varro.vercel.app/`; `og:url`, `og:image` and the
canonical link in `index.html` are absolute, so change them if the domain moves.

The Content-Security-Policy in `vercel.json` allows scripts and styles from the
site itself and fonts from Google Fonts only. A new external resource needs an
entry there.
