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
> | vicifons | 3.0M | **86,368,214** |
> | edcs-sdam | 3.0M | **4,356,994** |
> | *total, post-dedup* | 21.0M | **78,975,089** |
>
> The estimates held for the three TEI sources and failed badly for Vicifons,
> which was put at 3M words and delivered 86M — Latin Wikisource hosts Migne's
> Patrologia Latina at scale. EDCS also overshot by 45%. Two corrections the survey got wrong are recorded
> in `manifest/sources.yaml`: the CIRCSE DigilibLT mirror is **CoNLL-U + RDF, not TEI**, and its
> README contradicts itself on licence (CC badge says BY-NC-SA 3.0, anchor text says BY-SA 4.0 —
> resolved as BY-SA 4.0 for the texts, so only the text layer is ingested).
>
> The headline finding since: of the 79M words now assembled, only **4,539,515
> are CLASSICAL**. The ~10M classical figure below is an upper bound on what
> survives, not on what is reachable from literary repositories — and the corpus
> growing 4.6× barely moved it, because the new material is late antique and
> medieval. See `feasibility.md` §2.

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
- **Two layouts, not one.** `csel-dev` ships the texts twice: `data/` holds one file per work
  in CapiTainS layout, and `Volumes/` holds the printed volumes those were split out of
  (`Volumes/split_parse.py` does the splitting). `Volumes/` is 292 MB, half the repository,
  and was unread until the volume filter was written.
- What is actually in `Volumes/` beyond the per-work tree, measured:
  - **2.2M words of apparatus** — index locorum, index verborum, manuscript sigla, tables of
    contents, corrigenda. Training on it would teach page-and-line references, not Latin.
  - **~313k words of duplicate text** under identifiers that do not match the CTS tree.
    `@n` is present on only 274 of 302 work divisions and does not always agree, so these are
    caught by containment rather than by identifier.
  - **618k words of genuine prose the CTS tree never received** — overwhelmingly the prefaces
    and prolegomena, which the split produced no per-work file for.
- The prefaces are Latin prose written between **1867 and 1934**, not late antiquity. They are
  ingested as `NEOLATIN`, dated by their editor's earliest volume, and credited to that editor
  (`manifest/authorities.yaml`) — not to the patristic author whose volume they introduce.
- The markup cannot be trusted to tell text from apparatus in either direction: CSEL48 files
  68,471 words of *index verborum* under `subtype="chapter"`, and CSEL65 files a real work,
  *Liber ad Constantium*, under `subtype="MSS"`. CSEL48's header is wrong too — it claims
  Boethius and contains Optatus of Milevis. See `classify_division` in `src/extract.py`.

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

### Corpus Corporum (University of Zurich) — the Stage 2 corpus, pending an ask
- **https://www.mlat.uzh.ch/** — Philipp Roelli, Institute for Greek and Latin Philology.
  Contact: turicense@gmail.com.
- Size, from the site's own changelog: **over 200 million words** (19.12.2024). Note the
  15.3.2024 entry — a reload recounted more strictly, "paratextual elements and punctuation are
  no longer included", and the total *dropped* to 161M. Treat any single figure with care.
- Licence, quoted from `?p=about`: "The data we transcribed ourselves ... is published under
  **Creative Commons Share-Alike** and may thus be reused freely (**but non-commercially**) as
  long as the source is indicated. Most of the texts present here, however, stem from various
  online sources. As far as we could determine they are either in the public domain or their use
  was granted us by their owners." Also: "Texts may be downloaded as **TEI xml for
  non-commercial use**", and loaded/downloaded "unless copyrights or the texts' providers
  restrict this".
  → **amber, Track B.** Consistent with what the manifest already assumed.
- **robots.txt is `User-agent: * / Allow: /`** — the opposite of IntraText. Nothing technical
  or stated forbids automated access.
- **Do not extrapolate Tesserae's 95% duplication here, and the reason is on the page.** The
  site names its sixteen constituent projects and which corpus number each feeds. Three of them
  are already three of our own sources:
  | constituent | corpus no. | we already have it |
  |---|---|---|
  | Perseus | 5 | yes |
  | OpenGreekAndLatin | 2, 19 | yes (CSEL) |
  | digilibLT | 16 | yes |
  The other thirteen are **not** in this corpus and are largely not in Perseus or the Latin
  Library either: dMGH (Monumenta Germaniae Historica), CAMENA (neo-Latin), CroALa (Croatian
  Latin), ALIM (Italian medieval), mirabile/SISMEL, Nova scientia/noscemus (early modern
  science), Biblioteca Italiana, Bullinger Digital (Reformation letters), PAL (astrological),
  SourcEncyMe, CGL, Richard Rufus, Peter King. That is medieval and neo-Latin at scale —
  precisely the material Stage 2 wants and the one kind our 70%-Vicifons corpus is not already
  saturated with. **Overlap is excludable by corpus number**, which is a far better position
  than dedup-and-hope.
- **What blocks it: there is no bulk export.** Texts download individually as TEI XML; there is
  no documented dump, API or archive. Pulling 200M words text-by-text off a small university
  server is an imposition even where robots.txt permits it, and the site explicitly invites
  contact. **The unblocking action is an email to turicense@gmail.com asking for an export of
  the non-Perseus/OGL/digilibLT corpora** — same move as LTA/BBAW and IntraText.
- **The `tei-xml` reader now exists and is tested** — written against real data, not guessed.
  The route in: Corpus Corporum's largest corpus is fed from
  `github.com/OpenGreekAndLatin/patrologia_latina-dev`, whose 6,411 files carry a literal
  `[CC]` prefix because they *are* the Corpus Corporum copies. No scraping needed. Those files
  also declare **CC BY-SA 4.0** in their own teiHeader, so that constituent is greener than the
  platform's blanket NC term.
- Two defects found there, both of which a guessed reader would have sailed past:
  | defect | scale | effect if ignored |
  |---|---|---|
  | unterminated character references (`&#171 ;`, `&gt5`, `&#0160`) | 5 of 21 files in one volume | the file is a `parse_errors` counter and nothing else |
  | words split across a printed line (`Victo-\nrina`) | **4.06% of all words** | ~3.5M broken tokens at PL's scale |
  Repairing both took one volume from **50,980 words to 218,818** — a 4.3× difference between a
  reader that looks like it works and one that does.
- Repair placement matters and is not arbitrary. Entity repair runs *only after a parse fails*,
  so a file broken for some other reason still reports a genuine error instead of being masked.
  De-hyphenation runs *before* `tidy()`, which collapses the newline and destroys the only
  evidence that the hyphen was a line break rather than a real one.
- The TEI header's `<date>` is **not** a composition date — the PL files say `<date>2015</date>`,
  meaning when Leipzig encoded them. The reader therefore leaves `approximate_date` unset and
  recovers the **Patrologia Latina volume from the filename** instead, feeding the `pl_volume`
  interpolation `enrich` already has. Taking the header date would repeat the mistake that filed
  the CSEL prefaces under LATE.

### Two Corpus Corporum derivatives, assessed 2026-08-26

**`CCS-ZCU/corpus-corporum`** (Vojtěch Kaše, Centre for Computational Studies, Plzeň) —
**take the metadata and the method, not the data.** CC BY-SA 4.0, so green in its own right.
- `data/cc_metadata.csv` — **7,819 works** with `work_id`, title, author, `year` /
  `not_before` / `not_after` (**70.4% dated**), `word_count`, series title, edition, editor,
  publisher, VIAF and Wikidata ids. This is simultaneously the id list needed to iterate the
  corpus and a dating layer better than anything we could infer. Directly relevant to the 42%
  `UNASSIGNED` problem.
- `scripts/0_downloading-lemmatized-cc.ipynb` — reveals the per-work endpoint
  `https://mlat.uzh.ch/php_modules/download.php?type=file-pos-xml&idno=<id>`.
  Verified: `idno=10045` → `144_Petrus-Damianus_Epistolae.POS.xml.zip`, 1.18 MB, HTTP 200.
- Its own `data/` is a research sample (a few POS XMLs, sentence JSONs), not a corpus dump.

**`Fece228/latin-literature-dataset-170M` on HuggingFace — do not ingest.**
- 6.15 GB, two CSVs (`latin_raw.csv`, `latin_lemmas.csv`), "collected from all the texts
  available at Corpus Corporum".
- **`license: None`.** No licence field, no licence file. The upstream material is
  NonCommercial with rights varying per constituent corpus; a third party republishing it does
  not launder those terms, and this project's entire premise is that a *properly licensed*
  Latin corpus does not yet exist. Ingesting an unlicensed scrape of an NC source would make us
  the thing we are trying to replace.
- Its own card admits the raw half "contains text in Greek, Hebrew, and other languages, with
  references and contractions" — uncleaned.
- It also flattens away per-work provenance: which edition, which constituent corpus, which
  licence. Going to the source costs more requests and returns strictly more information.
- Useful only as confirmation that a full pull is feasible.

### LASLA / Opera Latina
- **https://www.lasla.uliege.be/** — corpus release announcement:
  https://ancientworldonline.blogspot.com/2023/10/the-lasla-latin-corpus-has-been.html
- 1,630,825 words, 154 works, 19 classical authors (Caesar, Cicero, Vergil, Ovid, Tacitus,
  Plautus…), **manually verified** lemma + POS + morphology. The highest-quality annotated
  classical Latin in existence.
- Licence: **CC BY-NC-SA 4.0**.
- Linked data version: https://github.com/CIRCSE/LASLA and the LiLa knowledge base
  (https://lila-erc.eu/).

### IntraText — Bibliotheca Latina — BLOCKED (access terms, not licence)
- **https://www.intratext.com/LAT/** — 716 IntraTexts / 876 works, EuloTech SRL.
- Licence: **CC BY-NC-SA 3.0**. That is amber, and amber is fine: Track B never redistributes.
  The licence is **not** what stops us.
- **The access terms are.** robots.txt disallows `/Dwnld/` and denies roughly 25 named mirroring
  agents outright — Wget, HTTrack, WebZIP and the rest — and the copyright page asks readers to
  "link to the contents in The IntraText Digital Library itself" rather than republish them.
  A user-agent not on that list might technically slip through; the publisher's intent does not.
  This project honours terms as offered rather than as argued, so: **do not write a scraper.**
- The way in is to **ask EuloTech for a dataset**, the same move as the LTA/BBAW question.
- Worth knowing before spending that ask: substantially derived from other collections, and the
  closest measured comparison — Tesserae — came back 95% already present.

### Index Thomisticus Treebank / CIRCSE resources
- **https://itreebank.marginalia.it/**
- Medieval (Aquinas), outside classical scope, but the tooling and models are relevant.
- Licence: **CC BY-NC-SA 3.0**.

### Tesserae corpus — INGESTED, green
- **https://tesserae.caset.buffalo.edu/** — texts: https://github.com/cltk/lat_text_tesserae
- Licence **RESOLVED, 2026-08-26**: UB Public License 1.0 (https://cse.buffalo.edu/sneps/ubpl.pdf)
  is a **Mozilla Public License 1.1 derivative**. §2.1 grants a "world-wide, royalty-free,
  non-exclusive license ... to use, reproduce, modify, display, perform, sublicense and
  distribute". The strings "non-commercial", "research only" and "academic use only" appear
  **nowhere in it**. §3.1 is file-level copyleft: redistribute the source form under the same
  licence and ship a copy of the licence with it.
  → **green, Track A**, obligation closer to CC BY-SA than to anything amber. The PDF does not
  extract through ordinary fetching; the text was recovered from its FlateDecode streams.
- Format is **`.tess`, not `.txt`** as the manifest first claimed: 748 files, one citation per
  line (`<verg. aen. 1.1>\tArma virumque cano`). Reader hazards, all from the real data —
  byte-order marks on three files; citation tags that lose their opening `<`
  (`cic. phil. 7.1>`); and angle brackets that are *editorial supplements*, not references, so
  `pronos> intelligebat` is Latin and must survive.
- **metadata/metadata.json dates every one of the 748 files** and gives a prose/verse/mixed
  mode. That is per-work evidence, better than an author-level floruit, and it is why
  `Authority.resolve` now has a `source-metadata` rank above interpolation.
- **Measured yield: 6,735,635 words extracted, 95.0% already in the corpus.**
  Only **334,010 words are new** — 128,523 of them classical. The derivation warning in §5 was
  right, and this is the number to calibrate the other derived collections against.

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
