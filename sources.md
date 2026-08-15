# Classical Latin Corpus — Copyright-Clear Source List

> **Status, 2026-08-15.** This document is the original licence survey and remains the reference for
> *what exists and under what terms*. Three of its Tier A sources are now fetched, extracted, dated
> and deduplicated; see `README.md` for measured figures and `manifest/sources.yaml` for the
> machine-readable manifest that actually drives the pipeline.
>
> | source | estimated here | **measured** |
> |---|---:|---:|
> | Perseus canonical-latinLit | 6.0M | **5,596,757** |
> | DigilibLT | 4.0M | **6,522,099** |
> | OGL CSEL | 5.0M | **6,039,916** |
> | *total, post-dedup* | 15.0M | **17,271,153** |
>
> The estimates held to within ~21% in aggregate. Two corrections the survey got wrong are recorded
> in `manifest/sources.yaml`: the CIRCSE DigilibLT mirror is **CoNLL-U + RDF, not TEI**, and its
> README contradicts itself on licence (CC badge says BY-NC-SA 3.0, anchor text says BY-SA 4.0 —
> resolved as BY-SA 4.0 for the texts, so only the text layer is ingested).
>
> The headline finding since: of that 17.3M words, only **4,638,175 are CLASSICAL**. The ~10M
> classical figure below is an upper bound on what survives, not on what is reachable from literary
> repositories — see `feasibility.md` §2.

Goal: assemble as close as possible to the complete surviving corpus of **Classical Latin**
(conventionally: earliest inscriptions ~600 BC through ~AD 200, i.e. the PHI scope), restricted to
material that is **public domain** or under a **permissive / free-culture licence**.

Compiled 2026-08-12. Every licence below was checked against the source or its repository; items
flagged **VERIFY** need a direct look at the current terms before you ingest them.

---

## 0. The copyright situation in one paragraph

The ancient texts themselves are long out of copyright. What can be copyrighted is the **modern
editorial work**: the constituted text of a critical edition, conjectures, apparatus criticus,
punctuation/orthography decisions, and the digital markup. In the US, anything published before
**1931** is public domain (the wall moves forward one year each January). In the EU/UK, the relevant
term is life-of-editor + 70 years, and some jurisdictions grant a separate ~25-year *critical
edition* right (Italy, Germany). Practically: the big free digital collections are built on
**pre-1930 Teubner / Oxford / Leipzig editions**, which is exactly why they can be given away.

Two licence families to keep separate in your pipeline:

- **Green (permissive/PD):** CC0, CC BY, CC BY-SA, public domain. Usable for any purpose.
- **Amber (restricted):** CC BY-**NC**-SA, custom "personal use" terms, no stated licence.
  Fine for private research; a problem if the model is ever released commercially. **ShareAlike**
  is also unsettled for model weights — nobody has litigated whether a model trained on CC BY-SA
  text is a derivative work. If you want zero ambiguity, build a green-only core and keep the
  amber material in a separately-tagged shard you can drop.

Realistic ceiling: all surviving Latin to AD 200 is roughly **7–10 million words**. Extend to AD 600
and you get maybe 30M. This is a *small* corpus by LLM standards — plan for heavy epoch counts,
tokenizer care, and possibly staged pretraining rather than expecting web-scale volume.

---

## 1. Tier A — Green licence, bulk-downloadable, start here

### Perseus Digital Library — canonical Latin literature
- **https://github.com/PerseusDL/canonical-latinLit**
- Core classical authors in TEI-XML (CTS/CapiTainS structure), the single best-curated free set.
- Licence: **CC BY-SA 4.0** ("Unless otherwise indicated, all contents of this repository are
  licensed under a Creative Commons Attribution-ShareAlike 4.0 International License").
- Also: `git clone` the whole org — https://github.com/PerseusDL
- Bulk text tarball (older, plain XML): https://www.perseus.tufts.edu/hopper/opensource/download
  (Greek and Roman collection, ~125 MB, CC BY-SA 3.0). Contains Greek too — filter by language.

### Open Greek and Latin
- **https://github.com/OpenGreekAndLatin**
- Machine-corrected OCR of public-domain editions; the continuation/expansion of the Perseus effort.
- Latin-relevant repos: `csel-dev` (Corpus Scriptorum Ecclesiasticorum Latinorum — *late antique/
  patristic*, out of classical scope but useful if you widen), plus various `*-dev` OCR repos.
- Licence: **CC BY-SA 4.0**.
- Reading interface: https://scaife.perseus.org/

### The Latin Library (via CLTK mirror)
- **https://github.com/cltk/lat_text_latin_library** (~108 MB plain text)
- Original site: **https://thelatinlibrary.com/**
- Broadest single free collection of Latin plain text; spans archaic → modern, so you must filter
  to the classical window by author/date.
- Licence: **public domain declaration** in the repo's `LICENSE.md`; site states texts are taken
  from out-of-copyright sources.
- Caveat: no critical apparatus, inconsistent provenance, known OCR/typing errors. Excellent for
  volume, weak for textual authority. Use as a supplement, not the backbone.

### DigilibLT — Digital Library of Late-Antique Latin Texts
- **https://digiliblt.uniupo.it/** (English: `?lang=en`)
- Bulk download page: **https://digiliblt.uniupo.it/g_scarica.php?lang=en** (txt / XML-TEI / PHI-TLG format)
- Mirror with TEI data: **https://github.com/CIRCSE/digilibLT**
- ~400 secular Latin prose texts, 2nd–7th c. AD. Overlaps the tail of your window (2nd c.) and
  extends past it — filter, or keep as adjacent-register data.
- Licence: **CC BY-SA 4.0** on the texts (site chrome is CC BY-NC-SA 3.0). Free account required.

### Latin Wikisource (Vicifons)
- **https://la.wikisource.org/**
- Dumps: **https://dumps.wikimedia.org/lawikisource/** (grab `*-pages-articles.xml.bz2`)
- Indexed by century from the 7th c. BC onward, which makes date-filtering easy.
- Licence: **CC BY-SA 4.0**; all hosted works must themselves be copyright-free.
- Also useful: https://en.wikisource.org/wiki/Portal:Classical_Latin_literature

### Papyri.info / Duke Databank of Documentary Papyri
- **https://papyri.info/** — data: **https://github.com/papyri/idp.data**
- EpiDoc XML. Mostly Greek, but a real body of **documentary Latin** (military records, letters,
  legal texts) that literary corpora completely lack.
- Licence: **CC BY 3.0**.

### Epigraphic Database Heidelberg (EDH)
- **https://edh.ub.uni-heidelberg.de/** — data page: **https://edh.ub.uni-heidelberg.de/data**
- Dumps: **https://github.com/epigraphic-database-heidelberg/data**
- Latin and Latin-Greek inscriptions of the Roman Empire, EpiDoc XML + RDF + GeoJSON, plus a
  live API and DTS API.
- Licence: **CC BY-SA 4.0** across the Open Data Repository.

### EDCS inscriptions, cleaned dataset (SDAM, Aarhus)
- **https://zenodo.org/records/7072337**
- 537,262 Latin inscriptions extracted and cleaned from the Epigraphik-Datenbank Clauss/Slaby, JSON.
- Licence: **CC BY 4.0**. This is the cleanest legal route to EDCS content.
- ETL scripts: https://github.com/sdam-au/EDH_ETL
- Live EDCS itself: https://db.edcs.eu/epigr/epi.php and https://edcs.hist.uzh.ch/en/ —
  **VERIFY** terms before scraping the live DB; use the Zenodo dump instead.

### LatinISE
- **https://lindat.mff.cuni.cz/repository/xmlui/handle/11234/1-2506**
- ~9.9M words, lemmatised, chronologically tagged (built from LacusCurtius, IntraText, Musisque
  Deoque sources). Useful as a ready-made lemmatised comparison corpus.
- Licence: **CC BY-SA 4.0**.

### Project Gutenberg — Latin
- **https://www.gutenberg.org/browse/languages/la**
- ~120 Latin-language titles on the browse page (a few hundred PG items tagged Latin overall).
  Classical slice is thin — Caesar, Cicero, Virgil, Ovid, Horace, Seneca, Plautus, Catullus,
  Sallust, Persius, Tacitus — and every one of them is already covered by Perseus / Latin Library.
  The bulk of the listing is medieval and neo-Latin (Augustine, Boethius, Descartes, Newton,
  Renaissance theology/medicine/botany).
- Licence: **US public domain** — the single cleanest licence in this document. No attribution,
  no ShareAlike, no NC. PG claims no copyright in markup or formatting.
  Trademark caveat: the *text* is unrestricted commercially, but commercial use of the
  "Project Gutenberg" **name** requires a royalty agreement. Strip the header/footer boilerplate
  regardless — it is modern English and would pollute training data.
- **Why include it despite the thin coverage:** it is the only source of real classical Latin with
  zero downstream obligations, so it belongs in a strictly-permissive core alongside the CC BY-SA
  backbone. It is also *human-proofread* (Distributed Proofreaders), not OCR — so wherever it
  overlaps a text you would otherwise have to OCR from Tier C, it is the better starting point.
  It additionally holds an eclectic tail of modern-period Latin grammars and secondary texts not
  easily found elsewhere.
- Caveats: volunteer-produced, may contain errors, often from superseded editions. Filter hard —
  the language tag also catches English translations, bilingual editions, and grammars written
  in English. Expect ~1–3% of final corpus volume.
- Assessment: **take it, but file under "permissive core", not "coverage".**

### Universal Dependencies Latin treebanks
- **https://universaldependencies.org/** — e.g. https://github.com/UniversalDependencies/UD_Latin-PROIEL
- Small but gold-standard annotated Latin (Caesar, Cicero, Vulgate, Palladius; plus Perseus, ITTB,
  LLCT, UDante treebanks). Value here is evaluation and syntactic probing, not pretraining volume.
- Licence: varies per treebank — **CC BY-SA 4.0** for several, **CC BY-NC-SA** for others.
  **VERIFY per repo.**

---

## 2. Tier B — Free to use, but restricted (NC / SA-heavy / unclear)

Keep these in a separate shard. Fine for a research model; drop them if you ever go commercial.

### Corpus Corporum (University of Zurich)
- **https://www.mlat.uzh.ch/**
- The largest structured Latin collection anywhere: ~226 million words, ~30 corpora, antiquity →
  20th c. TEI XML and txt download.
- Licence: **non-commercial use only**, and rights vary by constituent corpus ("heterogeneous
  origin"). Enormous value, real legal friction. Classical subset is a minority of the total.

### LASLA / Opera Latina
- **https://www.lasla.uliege.be/** — corpus release announcement:
  https://ancientworldonline.blogspot.com/2023/10/the-lasla-latin-corpus-has-been.html
- 1,630,825 words, 154 works, 19 classical authors (Caesar, Cicero, Vergil, Ovid, Tacitus,
  Plautus…), **manually verified** lemma + POS + morphology. The highest-quality annotated
  classical Latin in existence.
- Licence: **CC BY-NC-SA 4.0**.
- Linked data version: https://github.com/CIRCSE/LASLA and the LiLa knowledge base
  (https://lila-erc.eu/).

### IntraText — Bibliotheca Latina
- **https://www.intratext.com/LAT/**
- Licence: **CC BY-NC-SA 3.0**.

### Index Thomisticus Treebank / CIRCSE resources
- **https://itreebank.marginalia.it/**
- Medieval (Aquinas), outside classical scope, but the tooling and models are relevant.
- Licence: **CC BY-NC-SA 3.0**.

### Tesserae corpus
- **https://tesserae.caset.buffalo.edu/** — texts: https://github.com/cltk/lat_text_tesserae
- Aggregated from Latin Library, Perseus, DigilibLT, OGL, Musisque Deoque — so it is mostly
  *derivative* of Tier A. Useful mainly for its normalisation.
- Licence: **UB Public License 1.0** (custom university licence — read it; **VERIFY** for your use).

### Musisque Deoque — Latin poetry with critical apparatus
- **https://www.mqdq.it/**
- Complete corpus of ancient Latin poetry, XML-TEI-ish, with apparatus. Open access.
- Licence: **not clearly stated as a free licence — VERIFY / request permission.**
  The MQDQ-derived content inside LatinISE *is* CC BY-SA 4.0, which is a safer path.

### Corpus Grammaticorum Latinorum
- **https://cgl.hypotheses.org/** (and the older kaali.linguist.jussieu.fr interface)
- Latin grammarians, 2nd–7th c. Free access, **licence not explicitly stated — VERIFY.**
- A CLTK mirror exists: `latin_text_corpus_grammaticorum_latinorum`.

### Bibliotheca Augustana
- **https://www.hs-augsburg.de/~harsch/augustana.html**
- Long-standing free collection, largely from superseded (public-domain) editions.
- **No explicit licence.** Texts are very likely PD in substance; the site makes no grant.
  Treat as amber unless you match each text to a pre-1931 edition yourself.

### LacusCurtius
- **https://penelope.uchicago.edu/Thayer/E/Roman/Texts/home.html**
- Idiosyncratic but careful. Copyright status is encoded in the URL: one asterisk = public domain,
  two asterisks = copyrighted-by-another used with permission, no asterisk = © Bill Thayer.
  Many texts are drawn from **Loeb** editions — check each one. Filter on the asterisk convention;
  do not bulk-scrape.

### Corpus Scriptorum Latinorum
- **http://www.forumromanum.org/literature/**
- Largely a portal/index pointing at other collections. **No licence statement — VERIFY.**

---

## 3. Tier C — Public-domain page images, requires OCR

This is where you go to close the gaps Tier A leaves, and where the *authoritative* pre-1931
critical editions live (Teubner, Oxford Classical Texts, CSEL, CIL, Loeb vols. before 1931).

- **Internet Archive** — https://archive.org/
  - Teubner collection: https://archive.org/details/Teubner-Edition-Collection
  - Full-text search across PD scans; downloadable DjVu/OCR text and page images.
- **HathiTrust** — https://www.hathitrust.org/ (full view = US public domain, pre-1931)
- **Google Books** — https://books.google.com/ (see
  https://wiki.digitalclassicist.org/Classical_texts_on_Google_Book_Search)
- **Corpus Inscriptionum Latinarum, out-of-copyright volumes (pre-1940)**
  - https://cil.bbaw.de/en/ and https://www.aarome.org/corpus-inscriptionum-latinarum-digital-cil
  - Vols. I–XVI + auctaria, scanned and openly available.
- **Lace / Bruce Robertson OCR** — OCR'd Greek and Latin from Archive.org and Google Books;
  useful prior art for your own OCR pass.
(Project Gutenberg was previously listed here; it belongs in Tier A — it is proofread text, not scans.
See §1.)

OCR notes: Tesseract with the `lat` model plus a Latin-specific post-correction pass is the standard
route (`pytesseract`). Expect to spend real effort on hyphenation, running heads, apparatus-criticus
stripping (the apparatus is the copyrightable part *and* it is noise — remove it), long-s, and
ligatures.

---

## 4. Explicitly exclude — commercial / licensed

Do not train on these; they are subscription products with active copyright in the constituted text.

- **Bibliotheca Teubneriana Latina (BTL)** — https://www.brepols.net/ (subscription)
- **Library of Latin Texts (LLT), Brepolis** — https://www.brepolis.net/ (subscription)
- **Thesaurus Formarum Totius Latinitatis** — commercial
- **Loeb Classical Library (post-1930 volumes)** — https://www.loebclassics.com/ (subscription)
- **Thesaurus Linguae Graecae** — https://www.tlg.uci.edu/ (Greek, licensed)
- **PHI Classical Latin Texts** — https://latin.packhum.org/
  Free to *read* online and the best-normalised classical corpus in existence (essentially all Latin
  before AD 200), but it is **not** openly licensed: bulk data requires a signed licence agreement
  sent to phi@packhum.org, and the PHI 5 CD-ROM is distributed under restrictive terms.
  **If you want one thing that would transform this project, write to PHI and ask.** Treat the
  answer as authoritative and do not scrape in the meantime.

---

## 5. Suggested build order

1. **Backbone:** Perseus `canonical-latinLit` + Open Greek and Latin (CC BY-SA, TEI, well-structured).
2. **Volume fill:** The Latin Library (PD) — filter to classical authors, deduplicate against (1).
3. **Documentary/epigraphic:** EDH + EDCS-Zenodo + Papyri.info Latin subset (CC BY / CC BY-SA).
   This adds register diversity no literary corpus has.
4. **Gap analysis:** diff your author/work list against the PHI author index and against
   https://wiki.digitalclassicist.org/Greek_and_Latin_texts_in_digital_form — whatever is missing,
   source from Tier C scans and OCR it.
5. **Optional amber shard:** Corpus Corporum + LASLA + IntraText, tagged and separable.
6. **Held-out evaluation:** UD Latin treebanks and LASLA (never mix into pretraining).

Practical hygiene: keep a per-document manifest recording `source`, `licence`, `edition`,
`edition_year`, `author`, `work`, `date_of_composition`. Deduplication is critical — Tesserae,
LatinISE, CLTK and Corpus Corporum are all substantially *derived from* Perseus and The Latin
Library, so naive concatenation will duplicate Vergil half a dozen times. Near-duplicate detection
must be edition-aware: two editions of the same work are genuinely different texts, but you probably
want only one.

---

## 6. Meta-resources worth bookmarking

- **Digital Classicist Wiki, Greek and Latin texts in digital form** —
  https://wiki.digitalclassicist.org/Greek_and_Latin_texts_in_digital_form
- **Digital Corpora of Latin Texts (J. Ramminger)** — https://jramminger.github.io/corpora/
  (concise table of corpora with licences)
- **AWOL — The Ancient World Online** — https://ancientworldonline.blogspot.com/
  (announces new open-access classical resources; searchable archive)
- **Open Greek & Latin project site** — https://www.opengreekandlatin.org/
- **CLTK data index** — https://docs.cltk.org/en/latest/data.html
- **LiLa: Linking Latin** — https://lila-erc.eu/ (knowledge base tying these corpora together)
