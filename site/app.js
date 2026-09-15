/* Varro — renders the page's figures from window.VARRO (site/data.js).
   No dependencies. Every value that reaches the DOM goes through textContent. */
(function () {
  "use strict";

  var D = window.VARRO;
  var SVGNS = "http://www.w3.org/2000/svg";

  // ------------------------------------------------------------------ helpers

  function slot(name) { return document.querySelector('[data-render="' + name + '"]'); }

  function el(tag, attrs) {
    var node = document.createElement(tag);
    applyAttrs(node, attrs);
    for (var i = 2; i < arguments.length; i++) append(node, arguments[i]);
    return node;
  }
  function sv(tag, attrs) {
    var node = document.createElementNS(SVGNS, tag);
    applyAttrs(node, attrs);
    for (var i = 2; i < arguments.length; i++) append(node, arguments[i]);
    return node;
  }
  function applyAttrs(node, attrs) {
    if (!attrs) return;
    Object.keys(attrs).forEach(function (k) {
      var v = attrs[k];
      if (v === null || v === undefined || v === false) return;
      if (k === "text") node.textContent = v;
      else if (k === "class") node.setAttribute("class", v);
      else if (k.slice(0, 2) === "on") node.addEventListener(k.slice(2), v);
      else node.setAttribute(k, v === true ? "" : v);
    });
  }
  function append(node, child) {
    if (child === null || child === undefined || child === false) return;
    if (Array.isArray(child)) { child.forEach(function (c) { append(node, c); }); return; }
    node.appendChild(typeof child === "string" || typeof child === "number"
      ? document.createTextNode(String(child)) : child);
  }
  function clear(node) { while (node && node.firstChild) node.removeChild(node.firstChild); }

  // ------------------------------------------------------------ Roman numbers

  var NUMERALS = [[1000, "M"], [900, "CM"], [500, "D"], [400, "CD"], [100, "C"], [90, "XC"],
                  [50, "L"], [40, "XL"], [10, "X"], [9, "IX"], [5, "V"], [4, "IV"], [1, "I"]];

  function roman(n) {
    if (!Number.isInteger(n) || n <= 0 || n >= 4000) return String(n);
    var out = "";
    NUMERALS.forEach(function (pair) { while (n >= pair[0]) { out += pair[1]; n -= pair[0]; } });
    return out;
  }

  var MONTHS = ["Ian.", "Feb.", "Mart.", "Apr.", "Mai.", "Iun.", "Iul.", "Aug.", "Sept.", "Oct.", "Nov.", "Dec."];

  /* A date the way a Roman would have written it: counted inclusively down to
     the next Kalends, Nones or Ides. In March, May, July and October the Nones
     fall on the 7th and the Ides on the 15th; elsewhere on the 5th and 13th. */
  function romanDay(iso) {
    var parts = String(iso).split("-").map(Number);
    var y = parts[0], m = parts[1] - 1, d = parts[2];
    if (!y || isNaN(m) || !d) return "";
    var long = m === 2 || m === 4 || m === 6 || m === 9;
    var nones = long ? 7 : 5, ides = long ? 15 : 13;
    var days = new Date(Date.UTC(y, m + 1, 0)).getUTCDate();

    function count(n, mark, month) {
      if (n === 2) return "prid. " + mark + " " + month;
      return "a.d. " + roman(n) + " " + mark + " " + month;
    }
    if (d === 1) return "Kal. " + MONTHS[m];
    if (d === nones) return "Non. " + MONTHS[m];
    if (d < nones) return count(nones - d + 1, "Non.", MONTHS[m]);
    if (d === ides) return "Id. " + MONTHS[m];
    if (d < ides) return count(ides - d + 1, "Id.", MONTHS[m]);

    var next = MONTHS[(m + 1) % 12];
    if (m === 1 && days === 29) {           // the bissextile day
      if (d === 24) return "a.d. bis VI Kal. Mart.";
      if (d > 24) d -= 1;
      days = 28;
    }
    return count(days - d + 2, "Kal.", next);
  }

  // ------------------------------------------------------------ number format

  var nf = new Intl.NumberFormat("en-GB");
  function full(n) { return n === null || n === undefined ? "—" : nf.format(n); }

  function compactParts(n) {
    if (n === null || n === undefined) return ["—", ""];
    var abs = Math.abs(n);
    function fix(v) { return v >= 10 ? v.toFixed(1) : v.toFixed(2); }
    if (abs >= 1e9) return [fix(n / 1e9), "B"];
    if (abs >= 1e6) return [fix(n / 1e6), "M"];
    if (abs >= 1e3) return [fix(n / 1e3), "k"];
    return [String(n), ""];
  }
  function compact(n) { var p = compactParts(n); return p[0] + p[1]; }
  // Axis ticks are round numbers: 1M, 30M, 1B — no trailing zeros.
  function tickCompact(n) { var p = compactParts(n); return String(parseFloat(p[0])) + p[1]; }
  function withUnit(n) { var p = compactParts(n); return [p[0], p[1] ? el("small", { text: p[1] }) : null]; }
  function pct(v, digits) { return (v * 100).toFixed(digits === undefined ? 1 : digits) + "%"; }

  var PERIODS = [
    { key: "CLASSICAL", label: "Classical", cls: "c-p1" },
    { key: "LATE", label: "Late antique", cls: "c-p2" },
    { key: "MEDIEVAL", label: "Medieval", cls: "c-p3" },
    { key: "NEOLATIN", label: "Neo-Latin", cls: "c-p4" },
    { key: "UNASSIGNED", label: "Undated", cls: "c-p0" }
  ];
  var PERIOD_NAMES = { CLASSICAL: "Classical", LATE: "Late antique", MEDIEVAL: "Medieval",
                       NEOLATIN: "Neo-Latin", MIXED: "Many periods", DOCUMENTARY: "Documentary" };
  var CORPUS_NAMES = { "corpus": "Track A · full", "classical": "Track A · classical",
                       "corpus-b": "Track B · full", "classical-b": "Track B · classical" };

  function licenceName(id) {
    if (!id) return "—";
    var special = { "public-domain": "Public domain", "unclear": "No licence stated",
                    "NonCommercial": "NonCommercial", "UB-Public-License-1.0": "UB Public Licence 1.0" };
    if (special[id]) return special[id];
    var m = /^CC-(.+)-(\d\.\d)$/.exec(id);
    return m ? "CC " + m[1] + " " + m[2] : id;
  }

  // ------------------------------------------------------------ tooltip

  var tip = document.querySelector(".tooltip");

  function showTip(title, rows, x, y) {
    if (!tip) return;
    clear(tip);
    tip.appendChild(el("p", { class: "tooltip__title", text: title }));
    rows.forEach(function (r) {
      tip.appendChild(el("div", { class: "tooltip__row" },
        el("span", { class: (r.key === "rect" ? "key-rect " : "key-line ") + (r.cls || "") }),
        el("strong", { text: r.value }),
        el("span", { text: r.label })));
    });
    tip.hidden = false;
    var w = tip.offsetWidth, h = tip.offsetHeight, pad = 14;
    var left = x + pad, top = y + pad;
    if (left + w > window.innerWidth - 8) left = x - w - pad;
    if (top + h > window.innerHeight - 8) top = y - h - pad;
    tip.style.left = Math.max(8, left) + "px";
    tip.style.top = Math.max(8, top) + "px";
  }
  function hideTip() { if (tip) tip.hidden = true; }
  window.addEventListener("scroll", hideTip, { passive: true });

  // ------------------------------------------------------------ theme

  (function theme() {
    var btn = document.querySelector(".theme-toggle");
    if (!btn) return;
    var media = window.matchMedia ? window.matchMedia("(prefers-color-scheme: dark)") : null;
    function current() {
      var set = document.documentElement.getAttribute("data-theme");
      if (set === "light" || set === "dark") return set;
      return media && media.matches ? "dark" : "light";
    }
    function label() {
      btn.setAttribute("aria-label", current() === "dark"
        ? "Lamplight (dark). Switch to daylight."
        : "Daylight (light). Switch to lamplight.");
    }
    btn.addEventListener("click", function () {
      var next = current() === "dark" ? "light" : "dark";
      document.documentElement.setAttribute("data-theme", next);
      try { localStorage.setItem("varro-theme", next); } catch (e) { /* not remembered */ }
      label();
    });
    if (media && media.addEventListener) media.addEventListener("change", label);
    label();
  })();

  // ------------------------------------------------------------ footer

  (function footer() {
    var year = new Date().getFullYear();
    var y = slot("year");
    if (y) { y.textContent = roman(year); y.setAttribute("title", String(year)); }
    var auc = slot("auc");
    if (auc) auc.textContent = "anno urbis conditae " + roman(year + 753);
  })();

  if (!D) return;

  document.querySelectorAll("[data-repo-link]").forEach(function (a) { if (D.repo) a.href = D.repo; });

  (function updated() {
    var node = slot("updated");
    if (!node || !D.attribution_generated) return;
    clear(node);
    var iso = D.attribution_generated;
    append(node, [el("span", { lang: "la", text: romanDay(iso) + " " + roman(Number(iso.slice(0, 4))) }),
                  " (" + iso + ")"]);
  })();

  // ------------------------------------------------------------ derived data

  var allRuns = [];
  D.models.forEach(function (m) {
    m.runs.forEach(function (r) { allRuns.push({ run: r, model: m }); });
  });

  function runLabel(r, m) {
    if (r.role === "ablation") return m.name + " ablation";
    return r.run.replace(/-latinitas$/, "").toUpperCase();
  }

  // Series colours follow the run, in the order runs were trained — never rank.
  var curveRuns = allRuns.filter(function (x) { return x.run.curve; });
  var colourOf = {};
  curveRuns.forEach(function (x, i) { colourOf[x.run.run] = i < 7 ? "c-s" + (i + 1) : "c-control"; });

  function colourFor(runId, model) {
    if (colourOf[runId]) return colourOf[runId];
    if (colourOf[runId + "-latinitas"]) return colourOf[runId + "-latinitas"];
    var sibling = model && model.runs.filter(function (r) { return colourOf[r.run]; })[0];
    return sibling ? colourOf[sibling.run] : "c-control";
  }

  function lineage(model, headline) {
    var byId = {};
    model.runs.forEach(function (r) { byId[r.run] = r; });
    var chain = [headline], cur = headline;
    while (cur.init_from && byId[cur.init_from]) { cur = byId[cur.init_from]; chain.unshift(cur); }
    return chain;
  }

  // ------------------------------------------------------------ hero stats

  (function heroStats() {
    var node = slot("hero-stats");
    if (!node) return;
    var corpus = D.corpora.corpus || {};
    var trackA = D.sources.filter(function (s) { return s.group === "track-a"; }).length;
    var headlines = D.models.map(function (m) { return m.runs.filter(function (r) { return r.headline; })[0] || m.runs[0]; });
    var params = headlines.map(function (r) { return r.parameters; });
    var windows = 0, recited = 0;
    allRuns.forEach(function (x) {
      if (x.run.memorization) { windows += x.run.memorization.windows; recited += x.run.memorization.memorised; }
    });

    function stat(value, label) {
      return el("div", null, el("dt", { text: label }), el("dd", null, value));
    }
    append(node, [
      stat(withUnit(corpus.words), "words of Latin in the redistributable corpus"),
      stat(String(trackA), "free-licence sources, every version pinned"),
      stat(String(D.models.length), "model versions, " + compact(Math.min.apply(null, params)) +
        " to " + compact(Math.max.apply(null, params)) + " parameters"),
      stat(String(recited), "twelve-word windows recited, in " + full(windows) + " generated")
    ]);
  })();

  // ------------------------------------------------------------ corpora

  (function corpora() {
    var node = slot("corpora");
    if (!node) return;
    ["corpus", "classical", "corpus-b", "classical-b"].forEach(function (key) {
      var c = D.corpora[key];
      if (!c) return;
      var licence = c.licence ? licenceName(c.licence) : null;
      node.appendChild(el("article", { class: "corpus-card corpus-card--" + c.track },
        el("p", { class: "corpus-card__label", text: c.label }),
        el("p", { class: "corpus-card__big" }, withUnit(c.words), el("small", { text: " words" })),
        el("dl", null,
          el("dt", { text: "documents" }), el("dd", { text: full(c.records) }),
          el("dt", { text: "training tokens" }), el("dd", { text: c.train_tokens ? compact(c.train_tokens) : "—" }),
          el("dt", { text: "classical" }), el("dd", { text: c.periods && c.periods.CLASSICAL ? pct(c.periods.CLASSICAL / c.words) : "—" })),
        el("p", { class: "corpus-card__licence" },
          licence ? [c.redistributable ? "Offered under " : "Carries ", el("strong", { text: licence })]
                  : el("strong", { text: c.redistributable ? "Redistributable" : "Research only" }),
          c.redistributable ? "" : " — never redistributed")));
    });
  })();

  // ------------------------------------------------------------ periods

  (function periods() {
    var node = slot("periods"), legend = slot("periods-legend"), table = slot("periods-table");
    if (!node) return;
    var rows = ["corpus", "corpus-b"].filter(function (k) { return D.corpora[k]; });

    PERIODS.forEach(function (p) {
      legend.appendChild(el("span", { class: "legend__item" },
        el("span", { class: "key-rect " + p.cls }), p.label));
    });

    rows.forEach(function (key) {
      var c = D.corpora[key];
      var bar = el("div", { class: "period-bar" });
      PERIODS.forEach(function (p) {
        var words = (c.periods || {})[p.key] || 0;
        if (!words) return;
        var share = words / c.words;
        var desc = p.label + ": " + full(words) + " words, " + pct(share);
        var seg = el("span", {
          class: "period-seg " + p.cls, tabindex: "0", role: "img", "aria-label": c.label + ", " + desc
        });
        seg.style.flexGrow = String(share);
        seg.style.flexBasis = "0";
        function show(ev) {
          var box = seg.getBoundingClientRect();
          var x = ev && ev.clientX !== undefined ? ev.clientX : box.left + box.width / 2;
          var y = ev && ev.clientY !== undefined ? ev.clientY : box.bottom;
          showTip(c.label, [{ key: "rect", cls: p.cls, value: pct(share), label: p.label + " · " + compact(words) + " words" }], x, y);
        }
        seg.addEventListener("pointermove", show);
        seg.addEventListener("pointerleave", hideTip);
        seg.addEventListener("focus", function () { show(); });
        seg.addEventListener("blur", hideTip);
        bar.appendChild(seg);
      });
      node.appendChild(el("div", { class: "period-row" },
        el("p", { class: "period-row__label" }, c.label, el("small", { text: compact(c.words) + " words" })),
        bar));
    });

    var thead = el("tr", null, el("th", { text: "Period" }));
    rows.forEach(function (k) { thead.appendChild(el("th", { class: "num", text: D.corpora[k].label })); });
    var tbody = el("tbody");
    PERIODS.forEach(function (p) {
      var tr = el("tr", null, el("td", { text: p.label }));
      rows.forEach(function (k) {
        var c = D.corpora[k], w = (c.periods || {})[p.key] || 0;
        tr.appendChild(el("td", { class: "num", text: full(w) + " (" + pct(w / c.words) + ")" }));
      });
      tbody.appendChild(tr);
    });
    table.appendChild(el("div", { class: "table-scroll" }, el("table", null, el("thead", null, thead), tbody)));
  })();

  // ------------------------------------------------------------ fasti

  var STATUS = {
    published:  [["Publicatum", "pill pill--solid"], ["released", "pill pill--a"]],
    research:   [["Ad studium", "pill pill--b"], ["research only", "pill pill--b"]],
    superseded: [["Superseded", "pill pill--unknown"]],
    first:      [["Prima lux", "pill pill--unknown"], ["first light", "pill pill--unknown"]]
  };
  var ROLE = { latinitas: "Latinitas — broad corpus", romanitas: "Romanitas — classical narrowing",
               single: "single phase", ablation: "ablation — classical only" };

  (function fasti() {
    var node = slot("fasti");
    if (!node) return;

    D.models.forEach(function (m) {
      var head = m.runs.filter(function (r) { return r.headline; })[0] || m.runs[m.runs.length - 1];
      var chain = lineage(m, head);
      var tokens = chain.reduce(function (s, r) { return s + r.tokens_seen; }, 0);
      var hours = chain.reduce(function (s, r) { return s + r.hours; }, 0);
      var epochs = chain.map(function (r) { return String(r.epochs); }).join(" + ");

      function stat(value, label, cls) {
        return el("div", null, el("dt", { text: label }), el("dd", { class: cls || null, text: value }));
      }

      var pills = el("p", { class: "fastus__status" });
      (STATUS[m.status] || []).forEach(function (p) { pills.appendChild(el("span", { class: p[1], text: p[0] })); });
      pills.appendChild(el("span", { class: "pill " + (m.track === "B" ? "pill--b" : "pill--a"), text: "Track " + m.track }));

      var notes = el("ul", { class: "fastus__notes" });
      (m.notes || []).forEach(function (n) { notes.appendChild(el("li", { text: n })); });

      var tbody = el("tbody");
      m.runs.forEach(function (r) {
        var memo = r.memorization ? r.memorization.memorised + " / " + full(r.memorization.windows) : "—";
        tbody.appendChild(el("tr", null,
          el("td", null, el("code", { text: r.run })),
          el("td", { class: "role", text: ROLE[r.role] || r.role }),
          el("td", { text: CORPUS_NAMES[r.corpus] || r.corpus }),
          el("td", { class: "num", text: compact(r.parameters) }),
          el("td", { class: "num", text: compact(r.tokens_seen) }),
          el("td", { class: "num", text: String(r.epochs) }),
          el("td", { class: "num", text: r.perplexity.toFixed(1) }),
          el("td", { class: "num", text: r.x_floor ? r.x_floor.toFixed(2) + "×" : "—" }),
          el("td", { class: "num", text: r.hours.toFixed(1) }),
          el("td", { class: "num", text: memo }),
          el("td", { text: r.date })));
      });
      var acta = el("details", { class: "fastus__acta" },
        el("summary", null, "Acta · " + m.runs.length + (m.runs.length === 1 ? " run" : " runs")),
        el("div", { class: "table-scroll" }, el("table", null,
          el("thead", null, el("tr", null,
            el("th", { text: "Run" }), el("th", { text: "Phase" }), el("th", { text: "Corpus" }),
            el("th", { class: "num", text: "Params" }), el("th", { class: "num", text: "Tokens seen" }),
            el("th", { class: "num", text: "Epochs" }), el("th", { class: "num", text: "Perplexity" }),
            el("th", { class: "num", text: "× floor" }), el("th", { class: "num", text: "Hours" }),
            el("th", { class: "num", text: "Recited" }), el("th", { text: "Finished" }))),
          tbody)));

      var year = Number(String(m.date).slice(0, 4));
      node.appendChild(el("li", { class: "fastus fastus--" + m.status, id: m.id },
        el("p", { class: "fastus__when" },
          el("span", { class: "fastus__roman", lang: "la", text: romanDay(m.date) }),
          el("span", { class: "fastus__roman", lang: "la", text: roman(year) }),
          el("time", { class: "fastus__iso", datetime: m.date, text: m.date })),
        el("article", { class: "fastus__card", "aria-labelledby": m.id + "-name" },
          el("div", { class: "fastus__top" },
            el("h3", { class: "fastus__name", id: m.id + "-name", text: m.name }),
            pills),
          el("p", { class: "fastus__numeral", text: m.numeral }),
          el("p", { class: "fastus__summary", text: m.summary }),
          el("dl", { class: "fastus__stats" },
            stat(compact(head.parameters), "parameters"),
            stat(compact(tokens), "tokens seen"),
            stat(epochs, chain.length > 1 ? "epochs, by phase" : "epochs"),
            stat(hours.toFixed(1), "hours on a laptop"),
            stat(head.perplexity.toFixed(1), "perplexity · " + (CORPUS_NAMES[head.corpus] || head.corpus)),
            head.x_floor
              ? stat(head.x_floor.toFixed(2) + "×", "better than its floor", "is-emph")
              : stat("—", "no matching floor survives", "is-none")),
          notes,
          m.link ? el("p", { class: "fastus__link" }, el("a", { href: m.link.url, text: m.link.label + " →" })) : null,
          acta)));
    });
  })();

  // ------------------------------------------------------------ licence note

  (function licenceA() {
    var node = slot("licence-a");
    if (node && D.corpora.corpus && D.corpora.corpus.licence) node.textContent = licenceName(D.corpora.corpus.licence);
  })();

  // ------------------------------------------------------------ chart scaffolding

  function linear(d0, d1, r0, r1) {
    var f = function (v) { return r0 + (v - d0) / (d1 - d0 || 1) * (r1 - r0); };
    f.invert = function (px) { return d0 + (px - r0) / (r1 - r0 || 1) * (d1 - d0); };
    return f;
  }
  function logScale(d0, d1, r0, r1) {
    var l0 = Math.log10(d0), l1 = Math.log10(d1);
    var f = function (v) { return r0 + (Math.log10(v) - l0) / (l1 - l0 || 1) * (r1 - r0); };
    f.invert = function (px) { return Math.pow(10, l0 + (px - r0) / (r1 - r0 || 1) * (l1 - l0)); };
    return f;
  }
  function niceStep(span, maxTicks) {
    var steps = [0.5, 1, 2, 2.5, 5, 10, 20, 25, 50, 100];
    for (var i = 0; i < steps.length; i++) if (span / steps[i] <= maxTicks) return steps[i];
    return steps[steps.length - 1];
  }
  function watchWidth(node, draw) {
    var last = 0, frame = 0;
    function run() {
      frame = 0;
      var w = Math.round(node.clientWidth);
      if (w && w !== last) { last = w; draw(w); }
    }
    if (window.ResizeObserver) new ResizeObserver(function () { if (!frame) frame = requestAnimationFrame(run); }).observe(node);
    else window.addEventListener("resize", run);
    run();
    return function redraw() { last = 0; run(); };
  }

  /* Stack end-labels that would overlap, without leaving them far from their
     line: pushes are small, and a label that would travel further is dropped
     (the legend chips and the tooltip still carry it). */
  function placeLabels(labels, minGap, bottom) {
    labels.sort(function (a, b) { return a.y - b.y; });
    for (var i = 0; i < labels.length; i++) {
      for (var j = 0; j < i; j++) {
        var a = labels[j], b = labels[i];
        if (a.hidden || Math.abs(a.x - b.x) > 120) continue;
        if (b.y - a.y < minGap) b.y = a.y + minGap;
      }
      labels[i].hidden = labels[i].y - labels[i].anchorY > 22 || labels[i].y > bottom + 6;
    }
    return labels;
  }

  function hover(svgNode, plot, positions, render) {
    // positions: sorted array of pixel x values the crosshair may snap to
    var group = sv("g", { "aria-hidden": "true" });
    var line = sv("line", { class: "crosshair", y1: plot.top, y2: plot.bottom, visibility: "hidden" });
    group.appendChild(line);
    var dots = sv("g");
    group.appendChild(dots);
    var hit = sv("rect", {
      class: "plot-hit", x: plot.left, y: plot.top, width: plot.right - plot.left,
      height: plot.bottom - plot.top, tabindex: "0", role: "img",
      "aria-label": "Chart. Use the left and right arrow keys to step through the values."
    });
    var frame = sv("rect", { class: "focus-frame", x: plot.left - 4, y: plot.top - 4,
      width: plot.right - plot.left + 8, height: plot.bottom - plot.top + 8, rx: 3 });
    svgNode.appendChild(group);
    svgNode.appendChild(hit);
    svgNode.appendChild(frame);

    var index = -1;
    function nearest(px) {
      var lo = 0, hi = positions.length - 1;
      while (hi - lo > 1) { var mid = (lo + hi) >> 1; if (positions[mid] < px) lo = mid; else hi = mid; }
      return Math.abs(positions[lo] - px) <= Math.abs(positions[hi] - px) ? lo : hi;
    }
    function show(i, clientX, clientY) {
      if (!positions.length) return;
      index = Math.max(0, Math.min(positions.length - 1, i));
      var px = positions[index];
      line.setAttribute("x1", px); line.setAttribute("x2", px);
      line.setAttribute("visibility", "visible");
      clear(dots);
      var content = render(px, dots);
      if (!content) return;
      if (clientX === undefined) {
        var box = svgNode.getBoundingClientRect();
        var scale = box.width / (svgNode.viewBox.baseVal.width || box.width);
        clientX = box.left + px * scale;
        clientY = box.top + plot.top * scale + 20;
      }
      showTip(content.title, content.rows, clientX, clientY);
    }
    function hide() { line.setAttribute("visibility", "hidden"); clear(dots); hideTip(); }

    hit.addEventListener("pointermove", function (ev) {
      var box = svgNode.getBoundingClientRect();
      var scale = (svgNode.viewBox.baseVal.width || box.width) / box.width;
      show(nearest((ev.clientX - box.left) * scale), ev.clientX, ev.clientY);
    });
    hit.addEventListener("pointerleave", hide);
    hit.addEventListener("focus", function () { show(index < 0 ? positions.length - 1 : index); });
    hit.addEventListener("blur", hide);
    hit.addEventListener("keydown", function (ev) {
      var step = ev.shiftKey ? 10 : 1;
      if (ev.key === "ArrowRight") { show(index + step); ev.preventDefault(); }
      else if (ev.key === "ArrowLeft") { show(index - step); ev.preventDefault(); }
      else if (ev.key === "Home") { show(0); ev.preventDefault(); }
      else if (ev.key === "End") { show(positions.length - 1); ev.preventDefault(); }
      else if (ev.key === "Escape") { hide(); }
    });
  }

  // ------------------------------------------------------------ perplexity chart

  (function perplexity() {
    var node = slot("ppl-chart"), chips = slot("ppl-chips"), table = slot("ppl-table");
    if (!node || !curveRuns.length) return;

    var state = { x: "epochs", on: {} };
    curveRuns.forEach(function (x) { state.on[x.run.run] = x.run.curve !== "hidden"; });

    var series = curveRuns.map(function (x) {
      return { id: x.run.run, label: runLabel(x.run, x.model), cls: colourOf[x.run.run], run: x.run, model: x.model };
    });

    var redraw;

    series.forEach(function (s) {
      var chip = el("button", { type: "button", class: "chip " + s.cls, "aria-pressed": String(state.on[s.id]) },
        el("span", { class: "key-line" }), s.label);
      chip.addEventListener("click", function () {
        var visible = series.filter(function (t) { return state.on[t.id]; }).length;
        if (state.on[s.id] && visible === 1) return;     // always keep one line
        state.on[s.id] = !state.on[s.id];
        chip.setAttribute("aria-pressed", String(state.on[s.id]));
        redraw();
      });
      chips.appendChild(chip);
    });

    document.querySelectorAll('[data-x]').forEach(function (b) {
      b.addEventListener("click", function () {
        state.x = b.getAttribute("data-x");
        document.querySelectorAll('[data-x]').forEach(function (o) {
          o.setAttribute("aria-checked", String(o === b));
        });
        redraw();
      });
    });

    function xOf(p) { return state.x === "epochs" ? p[0] : p[1]; }

    function draw(width) {
      clear(node);
      hideTip();
      var visible = series.filter(function (s) { return state.on[s.id]; });
      var narrow = width < 620;
      var m = { top: 26, right: narrow ? 14 : 132, bottom: 48, left: 46 };
      var plotH = narrow ? 280 : Math.max(300, Math.min(440, width * 0.42));
      var height = m.top + plotH + m.bottom;
      var plot = { left: m.left, right: width - m.right, top: m.top, bottom: m.top + plotH };

      var ys = [], xs = [];
      visible.forEach(function (s) { s.run.history.forEach(function (p) { ys.push(p[2]); xs.push(xOf(p)); }); });
      var yMin = Math.min.apply(null, ys) * 0.88, yMax = Math.max.apply(null, ys) * 1.12;
      var y = logScale(yMin, yMax, plot.bottom, plot.top);

      var x, xTicks = [], xFmt;
      if (state.x === "epochs") {
        var xMax = Math.max.apply(null, xs);
        var step = niceStep(xMax, narrow ? 5 : 9);
        xMax = Math.ceil(xMax / step) * step;
        x = linear(0, xMax, plot.left, plot.right);
        for (var t = 0; t <= xMax + 1e-9; t += step) xTicks.push(t);
        xFmt = function (v) { return String(Math.round(v * 10) / 10); };
      } else {
        var lo = Math.pow(10, Math.floor(Math.log10(Math.min.apply(null, xs))));
        var hi = Math.max.apply(null, xs) * 1.15;
        x = logScale(lo, hi, plot.left, plot.right);
        for (var e = Math.log10(lo); Math.pow(10, e) <= hi; e++) {
          [1, 3].forEach(function (k) { var v = k * Math.pow(10, e); if (v >= lo && v <= hi) xTicks.push(v); });
        }
        if (narrow) xTicks = xTicks.filter(function (v) { return String(v)[0] === "1"; });
        xFmt = tickCompact;
      }

      var yTicks = [10, 15, 20, 30, 50, 70, 100, 150, 200, 300, 400, 700, 1000]
        .filter(function (v) { return v >= yMin && v <= yMax; });
      if (yTicks.length > 7) yTicks = yTicks.filter(function (v) { return [15, 70, 150, 700].indexOf(v) < 0; });
      var kept = [];                                   // keep labels a line apart on short plots
      yTicks.forEach(function (v) {
        if (!kept.length || Math.abs(y(kept[kept.length - 1]) - y(v)) >= 18) kept.push(v);
      });
      yTicks = kept;

      var svg = sv("svg", { viewBox: "0 0 " + width + " " + height, width: width, height: height,
        role: "group", "aria-label": "Held-out perplexity against " + (state.x === "epochs" ? "epochs" : "tokens seen") });

      var grid = sv("g", { class: "grid" });
      yTicks.forEach(function (v) {
        grid.appendChild(sv("line", { x1: plot.left, x2: plot.right, y1: y(v), y2: y(v) }));
        svg.appendChild(sv("text", { x: plot.left - 8, y: y(v) + 4, "text-anchor": "end", text: String(v) }));
      });
      svg.insertBefore(grid, svg.firstChild);
      svg.appendChild(sv("line", { class: "baseline", x1: plot.left, x2: plot.right, y1: plot.bottom, y2: plot.bottom }));
      xTicks.forEach(function (v) {
        svg.appendChild(sv("text", { x: x(v), y: plot.bottom + 20, "text-anchor": "middle", text: xFmt(v) }));
      });
      svg.appendChild(sv("text", { class: "axis-title", x: plot.left - 8, y: plot.top - 12, "text-anchor": "start",
        text: "perplexity (log)" }));
      svg.appendChild(sv("text", { class: "axis-title", x: plot.right, y: plot.bottom + 40, "text-anchor": "end",
        text: state.x === "epochs" ? "epochs over each run’s own corpus" : "tokens seen (log)" }));

      var labels = [];
      visible.forEach(function (s) {
        var g = sv("g", { class: "series " + s.cls });
        var d = s.run.history.map(function (p, i) {
          return (i ? "L" : "M") + x(xOf(p)).toFixed(1) + " " + y(p[2]).toFixed(1);
        }).join("");
        g.appendChild(sv("path", { d: d }));
        var last = s.run.history[s.run.history.length - 1];
        var lx = x(xOf(last)), ly = y(last[2]);
        g.appendChild(sv("circle", { cx: lx, cy: ly, r: 4 }));
        svg.appendChild(g);
        labels.push({ s: s, g: g, x: lx + 9, y: ly + 4, anchorY: ly + 4, value: last[2] });
      });

      if (!narrow) {
        // A label that would sit on another run's line moves above or below it.
        labels.forEach(function (l) {
          var width = (l.s.label.length + 5) * 6.6;
          function crowded(yy) {
            return visible.some(function (o) {
              if (o === l.s) return false;
              return o.run.history.some(function (p) {
                var px = x(xOf(p)), py = y(p[2]);
                return px >= l.x - 4 && px <= l.x + width && Math.abs(py - (yy - 4)) < 9;
              });
            });
          }
          if (!crowded(l.y)) return;
          [l.y - 14, l.y + 16, l.y - 26, l.y + 28].some(function (yy) {
            if (yy < plot.top + 8 || yy > plot.bottom - 2 || crowded(yy)) return false;
            l.y = yy; l.anchorY = yy; return true;
          });
        });
        placeLabels(labels, 15, plot.bottom).forEach(function (l) {
          if (l.hidden) return;
          l.g.appendChild(sv("text", { class: "end-label", x: l.x, y: l.y },
            sv("tspan", { text: l.s.label + " " }),
            sv("tspan", { class: "end-value", text: l.value.toFixed(1) })));
        });
      }

      // snap positions: every data x of every visible series
      var positions = [];
      visible.forEach(function (s) { s.run.history.forEach(function (p) { positions.push(x(xOf(p))); }); });
      positions.sort(function (a, b) { return a - b; });
      positions = positions.filter(function (v, i) { return !i || v - positions[i - 1] > 0.5; });

      hover(svg, plot, positions, function (px, dots) {
        var rows = [], xVal = x.invert(px);
        visible.forEach(function (s) {
          var h = s.run.history, first = x(xOf(h[0])), lastX = x(xOf(h[h.length - 1]));
          if (px < first - 6 || px > lastX + 6) return;
          var best = h[0], dist = Infinity;
          h.forEach(function (p) { var dd = Math.abs(x(xOf(p)) - px); if (dd < dist) { dist = dd; best = p; } });
          dots.appendChild(sv("circle", { class: "hover-dot " + s.cls, cx: x(xOf(best)), cy: y(best[2]), r: 4.5 }));
          rows.push({ cls: s.cls, value: best[2].toFixed(1), label: s.label, v: best[2] });
        });
        if (!rows.length) return null;
        rows.sort(function (a, b) { return a.v - b.v; });
        return {
          title: state.x === "epochs" ? "Epoch " + (Math.round(xVal * 10) / 10) : compact(xVal) + " tokens seen",
          rows: rows
        };
      });

      node.appendChild(svg);
    }

    redraw = watchWidth(node, draw);

    // table view: where every run ended
    var tbody = el("tbody");
    series.forEach(function (s) {
      var r = s.run;
      tbody.appendChild(el("tr", null,
        el("td", null, el("span", { class: "key-line " + s.cls }), " ", s.label),
        el("td", { text: CORPUS_NAMES[r.corpus] || r.corpus }),
        el("td", { class: "num", text: compact(r.parameters) }),
        el("td", { class: "num", text: String(r.epochs) }),
        el("td", { class: "num", text: compact(r.tokens_seen) }),
        el("td", { class: "num", text: r.history.length ? r.history[0][2].toFixed(1) : "—" }),
        el("td", { class: "num", text: r.perplexity.toFixed(1) }),
        el("td", { class: "num", text: r.x_floor ? r.x_floor.toFixed(2) + "×" : "—" })));
    });
    table.appendChild(el("div", { class: "table-scroll" }, el("table", null,
      el("thead", null, el("tr", null,
        el("th", { text: "Run" }), el("th", { text: "Corpus" }), el("th", { class: "num", text: "Params" }),
        el("th", { class: "num", text: "Epochs" }), el("th", { class: "num", text: "Tokens seen" }),
        el("th", { class: "num", text: "First eval" }), el("th", { class: "num", text: "Final" }),
        el("th", { class: "num", text: "× floor" }))),
      tbody)));
  })();

  // ------------------------------------------------------------ memorisation chart

  (function memoria() {
    var node = slot("memo-chart"), picker = slot("memo-picker"), legend = slot("memo-legend"), table = slot("memo-table");
    if (!node) return;

    // One entry per model version: its headline run if measured, else any measured run.
    var options = [];
    D.models.forEach(function (m) {
      var measured = m.runs.filter(function (r) { return r.memorization && r.memorization.curve && r.role !== "ablation"; });
      if (!measured.length) return;
      var pick = measured.filter(function (r) { return r.headline; })[0] || measured[0];
      options.push({ id: pick.run, label: m.name, run: pick, model: m, cls: colourFor(pick.run, m) });
    });
    if (!options.length) return;

    var state = { id: (options.filter(function (o) { return o.model.status === "research"; })[0] || options[options.length - 1]).id };
    var redraw;

    options.forEach(function (o) {
      var b = el("button", { type: "button", role: "radio", "aria-checked": String(o.id === state.id), text: o.label });
      b.addEventListener("click", function () {
        state.id = o.id;
        picker.querySelectorAll("button").forEach(function (x) { x.setAttribute("aria-checked", String(x === b)); });
        redraw();
        renderTable();
      });
      picker.appendChild(b);
    });

    function current() { return options.filter(function (o) { return o.id === state.id; })[0]; }

    function renderLegend(o) {
      clear(legend);
      legend.appendChild(el("span", { class: "legend__item" }, el("span", { class: "key-line " + o.cls }), o.label + "’s own output"));
      legend.appendChild(el("span", { class: "legend__item" }, el("span", { class: "key-line c-control" }), "Genuine Latin it never saw"));
    }

    function renderTable() {
      var o = current();
      clear(table);
      var tbody = el("tbody");
      o.run.memorization.curve.forEach(function (c) {
        tbody.appendChild(el("tr", null,
          el("td", { class: "num", text: c[0] + " words" }),
          el("td", { class: "num", text: pct(c[1]) }),
          el("td", { class: "num", text: pct(c[2]) })));
      });
      var memo = o.run.memorization;
      table.appendChild(el("div", { class: "table-scroll" }, el("table", null,
        el("thead", null, el("tr", null,
          el("th", { class: "num", text: "Window" }),
          el("th", { class: "num", text: o.label + " output" }),
          el("th", { class: "num", text: "Unseen Latin" }))),
        tbody)));
      table.appendChild(el("p", { class: "table-view__note",
        text: o.run.run + ": " + memo.memorised + " of " + full(memo.windows) + " " + memo.window +
          "-word windows found in training text; longest verbatim run " + memo.longest + " words." }));
    }

    function draw(width) {
      clear(node);
      hideTip();
      var o = current();
      renderLegend(o);
      var curve = o.run.memorization.curve;
      var narrow = width < 620;
      var m = { top: 26, right: narrow ? 14 : 150, bottom: 48, left: 46 };
      var plotH = narrow ? 240 : Math.max(240, Math.min(340, width * 0.32));
      var height = m.top + plotH + m.bottom;
      var plot = { left: m.left, right: width - m.right, top: m.top, bottom: m.top + plotH };

      var peak = Math.max.apply(null, curve.map(function (c) { return Math.max(c[1], c[2]); }));
      var yStep = peak > 0.3 ? 0.1 : peak > 0.12 ? 0.05 : 0.02;
      var yMax = Math.max(yStep, Math.ceil(peak / yStep) * yStep);
      var y = linear(0, yMax, plot.bottom, plot.top);
      var band = (plot.right - plot.left) / curve.length;
      var xAt = function (i) { return plot.left + band * (i + 0.5); };

      var svg = sv("svg", { viewBox: "0 0 " + width + " " + height, width: width, height: height, role: "group",
        "aria-label": "Share of word windows found in training text, by window length" });
      var grid = sv("g", { class: "grid" });
      for (var t = yStep; t <= yMax + 1e-9; t += yStep) {
        grid.appendChild(sv("line", { x1: plot.left, x2: plot.right, y1: y(t), y2: y(t) }));
      }
      svg.appendChild(grid);
      for (var t2 = 0; t2 <= yMax + 1e-9; t2 += yStep) {
        svg.appendChild(sv("text", { x: plot.left - 8, y: y(t2) + 4, "text-anchor": "end", text: Math.round(t2 * 100) + "%" }));
      }
      svg.appendChild(sv("line", { class: "baseline", x1: plot.left, x2: plot.right, y1: plot.bottom, y2: plot.bottom }));
      curve.forEach(function (c, i) {
        svg.appendChild(sv("text", { x: xAt(i), y: plot.bottom + 20, "text-anchor": "middle", text: String(c[0]) }));
      });
      svg.appendChild(sv("text", { class: "axis-title", x: plot.left - 8, y: plot.top - 12, text: "found verbatim in training text" }));
      svg.appendChild(sv("text", { class: "axis-title", x: plot.right, y: plot.bottom + 40, "text-anchor": "end", text: "window length, in words" }));

      var lines = [
        { cls: o.cls, idx: 1, label: o.label },
        { cls: "c-control", idx: 2, label: "unseen Latin" }
      ];
      var labels = [];
      lines.forEach(function (l) {
        var g = sv("g", { class: "series " + l.cls });
        g.appendChild(sv("path", { d: curve.map(function (c, i) { return (i ? "L" : "M") + xAt(i) + " " + y(c[l.idx]); }).join("") }));
        curve.forEach(function (c, i) { g.appendChild(sv("circle", { cx: xAt(i), cy: y(c[l.idx]), r: 4 })); });
        svg.appendChild(g);
        var last = curve[curve.length - 1];
        labels.push({ g: g, x: xAt(curve.length - 1) + 10, y: y(last[l.idx]) + 4, anchorY: y(last[l.idx]) + 4, text: l.label, value: last[l.idx] });
      });
      if (!narrow) {
        // both series can end on 0%: keep them apart, never on top of each other
        labels.sort(function (a, b) { return a.y - b.y; });
        if (labels[1].y - labels[0].y < 16) labels[0].y = labels[1].y - 16;
        labels.forEach(function (l) {
          l.g.appendChild(sv("text", { class: "end-label", x: l.x, y: l.y },
            sv("tspan", { text: l.text + " " }), sv("tspan", { class: "end-value", text: pct(l.value) })));
        });
      }

      hover(svg, plot, curve.map(function (c, i) { return xAt(i); }), function (px, dots) {
        var i = Math.max(0, Math.min(curve.length - 1, Math.round((px - plot.left) / band - 0.5)));
        var c = curve[i];
        lines.forEach(function (l) {
          dots.appendChild(sv("circle", { class: "hover-dot " + l.cls, cx: xAt(i), cy: y(c[l.idx]), r: 5 }));
        });
        return {
          title: c[0] + "-word windows",
          rows: [
            { cls: o.cls, value: pct(c[1]), label: o.label + " output" },
            { cls: "c-control", value: pct(c[2]), label: "unseen Latin" }
          ]
        };
      });

      node.appendChild(svg);
    }

    redraw = watchWidth(node, draw);
    renderTable();
  })();

  // ------------------------------------------------------------ fontes

  var GROUPS = [
    { key: "track-a", la: "In corpore publico", en: "Track A — redistributable",
      desc: "Free licences and the public domain. Together they make the corpus that is published." },
    { key: "track-b", la: "Ad studium tantum", en: "Track B — research only",
      desc: "NonCommercial terms. Used in research builds, never redistributed — and neither are weights trained on them." },
    { key: "eval", la: "Ad probandum seposita", en: "Held out",
      desc: "Gold-standard annotation kept out of every training corpus, so that a probe measures something." },
    { key: "pending", la: "Nondum recepta", en: "Not in the corpus",
      desc: "Declared in the manifest but not ingested — waiting on a download, a licence, or an answer." }
  ];
  var CONTRIB = [["corpus", "Track A", "A"], ["classical", "Classical", "C"], ["corpus-b", "Track B", "B"]];

  (function fontes() {
    var node = slot("sources");
    if (!node) return;

    GROUPS.forEach(function (g) {
      var members = D.sources.filter(function (s) { return s.group === g.key; });
      if (!members.length) return;
      var section = el("section", { class: "fontes__group", "aria-labelledby": "fg-" + g.key },
        el("header", { class: "fontes__group-head" },
          el("h3", { id: "fg-" + g.key }, el("span", { lang: "la", text: g.la }), " · " + g.en),
          el("p", { text: members.length + (members.length === 1 ? " source" : " sources") })),
        el("p", { class: "figure__desc", style: null, text: g.desc }));

      members.forEach(function (s) {
        var version = null;
        CONTRIB.forEach(function (c) { var x = s.contributions[c[0]]; if (!version && x && x.version) version = x.version; });

        var meta = el("p", { class: "fons__meta" },
          el("span", { class: "licence" }, s.licence_url
            ? el("a", { href: s.licence_url, text: licenceName(s.licence) })
            : licenceName(s.licence)),
          s.period ? el("span", { text: PERIOD_NAMES[s.period] || s.period }) : null,
          s.format ? el("code", { text: s.format }) : null,
          version ? el("span", { title: version, text: "pinned " + (version.length > 12 ? version.slice(0, 7) : version) }) : null);

        var contrib = el("div", { class: "fons__contrib" });
        var any = false;
        CONTRIB.forEach(function (c) {
          var x = s.contributions[c[0]];
          if (!x) return;
          any = true;
          var fill = el("div", { class: "contrib__fill" });
          fill.style.width = Math.max(0.4, Math.min(100, x.share)) + "%";
          contrib.appendChild(el("div", { class: "contrib contrib--" + c[2],
            title: full(x.words) + " words across " + full(x.documents) + " documents" },
            el("span", { class: "contrib__label", text: c[1] }),
            el("div", { class: "contrib__track", role: "img",
              "aria-label": c[1] + ": " + x.share + "% of the corpus" }, fill),
            el("span", { class: "contrib__value", text: compact(x.words) + " words · " + (x.share ? x.share.toFixed(1) + "%" : "<0.1%") })));
        });
        if (!any) {
          var why = s.tier === "eval" ? "Held out of every training corpus."
            : s.tier === "unknown" ? "Licence unresolved — enters neither track."
            : !s.include ? "Declared, and deliberately not included."
            : "Declared; not yet fetched.";
          contrib.appendChild(el("p", { class: "fons__none", text: why }));
        }

        section.appendChild(el("article", { class: "fons", id: "fons-" + s.id },
          el("div", null,
            el("h4", { class: "fons__name" }, s.url ? el("a", { href: s.url, text: s.name }) : s.name),
            s.blurb ? el("p", { class: "fons__blurb", text: s.blurb }) : null,
            meta),
          contrib));
      });
      node.appendChild(section);
    });
  })();
})();
