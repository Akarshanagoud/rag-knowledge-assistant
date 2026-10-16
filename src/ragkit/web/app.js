/* Knowledge Assistant: views over the ragkit HTTP API. */
(function () {
  "use strict";
  const { h, api, fmt } = Kit;

  const MODE_LABEL = { hybrid: "Hybrid (RRF)", dense: "Dense", lexical: "Lexical (BM25)" };
  const MODE_NOTE = {
    hybrid: "Fuses both rankings by reciprocal rank",
    dense: "Embedding similarity: catches paraphrases",
    lexical: "BM25 term matching: catches exact identifiers",
  };

  let overviewCache = null;
  let docColors = {};
  const conversation = [];
  let selectedTurn = -1;
  const askSettings = { mode: "hybrid", top_k: 5 };

  async function overview() {
    if (!overviewCache) overviewCache = await api.get("/api/overview");
    return overviewCache;
  }

  async function loadDocColors() {
    const docs = await api.get("/api/documents");
    docColors = {};
    docs.forEach((d, i) => (docColors[d.doc_id] = i < 8 ? Kit.seriesColor(i) : "var(--series-other)"));
    return docs;
  }

  function docChip(docId, label) {
    return h("span", { class: "doc-chip" }, h("i", { style: { background: docColors[docId] || "var(--series-other)" } }), label || docId);
  }

  function termSpans(text, terms) {
    if (!terms || !terms.length) return [];
    const escaped = terms.filter(Boolean).map((t) => t.replace(/[.*+?^${}()|[\]\\]/g, "\\$&"));
    if (!escaped.length) return [];
    const re = new RegExp("\\b(" + escaped.join("|") + ")\\w*", "gi");
    const spans = [];
    let m;
    while ((m = re.exec(text)) !== null) {
      spans.push({ start: m.index, end: m.index + m[0].length, cls: "m-info" });
      if (m[0].length === 0) re.lastIndex++;
    }
    return spans;
  }

  // ------------------------------------------------------------------ ask
  function answerText(turn, onMarker) {
    const text = turn.answer.answer;
    const parts = text.split(/(\[\d+\])/g);
    return h(
      "div",
      { class: "answer" },
      parts.map((part) => {
        const m = /^\[(\d+)\]$/.exec(part);
        if (!m) return part.replace(/\s*\n\s*/g, " ");
        const n = Number(m[1]);
        const b = h("button", { class: "marker", type: "button", title: "Show evidence " + n, "data-n": String(n) }, String(n));
        b.addEventListener("click", (e) => {
          e.stopPropagation();
          onMarker(n);
        });
        return b;
      }),
    );
  }

  function statusRow(a) {
    const row = [];
    if (a.abstained) row.push(Kit.badge("Abstained", "warning"));
    else row.push(a.grounded ? Kit.badge("Grounded", "good") : Kit.badge("Not grounded", "critical"));
    row.push(Kit.badge(MODE_LABEL[a.mode] || a.mode, "", { icon: false }));
    row.push(h("span", { class: "small muted" }, fmt.ms(a.elapsed_ms) + " · top " + a.top_k));
    return h("div", { class: "row" }, row);
  }

  function abstainDetail(a) {
    const md = a.metadata || {};
    const rows = [];
    if (md.coverage !== undefined && md.min_coverage !== undefined) rows.push(["Question coverage", fmt.num(md.coverage, 2) + " (needs " + fmt.num(md.min_coverage, 2) + ")"]);
    if (md.semantic !== undefined && md.semantic_floor !== undefined) rows.push(["Semantic similarity", fmt.num(md.semantic, 3) + " (floor " + fmt.num(md.semantic_floor, 2) + ")"]);
    if (md.best_score !== undefined) rows.push(["Best fused score", fmt.num(md.best_score, 4)]);
    return h(
      "div",
      { class: "stack", style: { gap: "8px", marginTop: "10px" } },
      h("div", { class: "small ink-2" }, "Why: " + (md.reason || "evidence below the answer threshold")),
      rows.length ? Kit.kv(rows) : null,
    );
  }

  function evidencePanel(host, turn, activeN) {
    Kit.clear(host);
    if (!turn) {
      host.append(Kit.card({ title: "Evidence", body: Kit.empty("Ask a question", "The chunks retrieved for the selected answer appear here, ranked, with each retriever's score.") }));
      return;
    }
    const a = turn.answer;
    const items = a.retrieved.map((c) => {
      const comps = c.components || {};
      const el = h(
        "div",
        { class: "evidence" + (activeN === c.rank ? " is-active" : ""), id: "ev-" + c.rank },
        h("div", { class: "row-between" }, h("div", { class: "row" }, h("span", { class: "marker", style: { cursor: "default" } }, String(c.rank)), docChip(c.doc_id, c.title || c.doc_id)), h("span", { class: "small muted num" }, "fused " + fmt.num(c.score, 4))),
        h("div", { class: "small muted", style: { marginTop: "4px" } }, (c.section ? c.section + " · " : "") + "chars " + c.start + "–" + c.end),
        h(
          "div",
          { class: "stack", style: { gap: "4px", marginTop: "8px" } },
          comps.dense !== undefined
            ? h("div", { class: "score-row" }, h("span", null, "Dense"), Kit.meter(Math.max(0, comps.dense), "", "Dense similarity"), h("span", { class: "num" }, fmt.num(comps.dense, 3) + (c.ranks && c.ranks.dense ? " #" + c.ranks.dense : "")))
            : null,
          comps.lexical !== undefined
            ? h("div", { class: "score-row" }, h("span", null, "BM25"), Kit.meter(Math.min(1, comps.lexical / 8), "", "BM25 score"), h("span", { class: "num" }, fmt.num(comps.lexical, 2) + (c.ranks && c.ranks.lexical ? " #" + c.ranks.lexical : "")))
            : null,
        ),
        c.matched_terms && c.matched_terms.length ? h("div", { class: "row", style: { marginTop: "8px", gap: "4px" } }, h("span", { class: "small muted" }, "matched"), c.matched_terms.map((t) => h("span", { class: "tag" }, t))) : null,
        h("div", { class: "chunk-text" }, Kit.highlight(c.text, termSpans(c.text, c.matched_terms))),
      );
      return el;
    });
    host.append(
      Kit.card({
        title: "Evidence",
        subtitle: a.retrieved.length + " chunks retrieved · numbers match the markers in the answer",
        body: items.length ? h("div", { class: "stack", style: { gap: "10px" } }, items) : Kit.empty("Nothing retrieved", "The index returned no chunks for this question."),
      }),
    );
    if (activeN) {
      const target = host.querySelector("#ev-" + activeN);
      if (target) target.scrollIntoView({ block: "nearest", behavior: "smooth" });
    }
  }

  async function askView(view) {
    const ov = await overview();
    await loadDocColors();
    const input = h("input", { class: "input", type: "text", placeholder: "Ask about refunds, rate limits, data retention, incidents…", "aria-label": "Question" });
    const submit = h("button", { class: "btn btn-primary", type: "submit" }, "Ask");
    const thread = h("div", { class: "list" });
    const evidence = h("div", null);
    const threadCard = Kit.card({ title: "Conversation", subtitle: "Every answer is copied from retrieved text and cited. Click a marker to see its evidence.", flush: true, body: thread });

    const renderThread = () => {
      Kit.clear(thread);
      if (!conversation.length) {
        thread.append(Kit.empty("No questions yet", "Try one of the suggestions above, or ask your own."));
        evidencePanel(evidence, null);
        return;
      }
      conversation.forEach((turn, i) => {
        const el = h(
          "div",
          { class: "turn" + (i === selectedTurn ? " selected" : ""), tabindex: "0" },
          h("div", { class: "turn-q" }, turn.question),
          answerText(turn, (n) => {
            selectedTurn = i;
            renderThread();
            evidencePanel(evidence, turn, n);
          }),
          h("div", { style: { marginTop: "10px" } }, statusRow(turn.answer)),
          turn.answer.abstained ? abstainDetail(turn.answer) : null,
          !turn.answer.abstained && turn.answer.citations.length
            ? h(
                "div",
                { class: "stack", style: { gap: "6px", marginTop: "10px" } },
                turn.answer.citations.map((c) =>
                  h("div", { class: "score-row", style: { gridTemplateColumns: "minmax(0,1fr) 120px 48px" } }, h("span", { class: "truncate", title: c.citation }, docChip(c.doc_id, c.citation)), Kit.meter(c.support, c.support >= 0.3 ? "good" : "warning", "Support"), h("span", { class: "num" }, fmt.num(c.support, 2))),
                ),
              )
            : null,
        );
        el.addEventListener("click", () => {
          selectedTurn = i;
          renderThread();
          evidencePanel(evidence, turn);
        });
        thread.append(el);
      });
      evidencePanel(evidence, conversation[selectedTurn]);
    };

    const ask = async (question) => {
      if (!question.trim()) return;
      await Kit.withButton(submit, async () => {
        submit.textContent = "Searching…";
        const answer = await api.post("/api/ask", { question, mode: askSettings.mode, top_k: askSettings.top_k });
        conversation.unshift({ question, answer });
        selectedTurn = 0;
        input.value = "";
        renderThread();
      });
    };

    const form = h("form", { class: "ask-bar" }, input, submit);
    form.addEventListener("submit", (e) => {
      e.preventDefault();
      ask(input.value);
    });
    const suggestions = h(
      "div",
      { class: "chips" },
      ov.suggested.map((q, i) =>
        h(
          "button",
          { class: "chip", type: "button", onclick: () => ask(q), title: i >= ov.suggested.length - 2 ? "Not covered by the corpus: the assistant should decline" : "" },
          q,
          i >= ov.suggested.length - 2 ? " — not in corpus" : "",
        ),
      ),
    );
    const settings = h(
      "div",
      { class: "row" },
      h("span", { class: "small muted" }, "Retrieval"),
      Kit.seg(["hybrid", "dense", "lexical"].map((m) => ({ value: m, label: MODE_LABEL[m] })), askSettings.mode, (v) => (askSettings.mode = v)),
      h("span", { class: "small muted", style: { marginLeft: "8px" } }, "Chunks"),
      Kit.select([3, 5, 8], askSettings.top_k, (v) => (askSettings.top_k = Number(v)), { style: { width: "72px" }, "aria-label": "Chunks to retrieve" }),
    );

    view.append(
      Kit.pageHead("Ask", ov.documents + " documents · " + ov.chunks + " chunks · " + ov.embedder.name + " embeddings · " + ov.store.name + " store. Answers are extractive, so they cannot invent facts."),
      Kit.card({ body: h("div", { class: "stack" }, form, h("div", { class: "row-between" }, suggestions), settings) }),
      h("div", { class: "split-r mt" }, threadCard, evidence),
    );
    renderThread();
    input.focus();
  }

  // ------------------------------------------------------------------ retrieval
  async function retrievalView(view, params) {
    await loadDocColors();
    const gold = await api.get("/api/gold");
    const input = h("input", { class: "input", type: "text", value: params.q || "Can I ask for a higher throughput allowance?", "aria-label": "Query" });
    const topk = Kit.select([3, 5, 8], 5, null, { style: { width: "72px" }, "aria-label": "Results per mode" });
    const run = h("button", { class: "btn btn-primary", type: "submit" }, "Compare");
    const pick = Kit.select([{ value: "", label: "Pick a gold question…" }].concat(gold.map((g) => ({ value: g.question, label: g.question + (g.unanswerable ? " (unanswerable)" : "") }))), "", (v) => {
      if (v) {
        input.value = v;
        go();
      }
    }, { "aria-label": "Gold questions" });
    const out = h("div", null);

    const go = () =>
      Kit.withButton(run, async () => {
        const data = await Kit.busy(out, () => api.post("/api/retrieve", { query: input.value, top_k: Number(topk.value) }));
        render(data);
      });

    const render = (data) => {
      Kit.clear(out);
      const relevant = new Set(data.gold ? data.gold.relevant_doc_ids : []);
      if (data.gold) {
        out.append(
          Kit.banner({
            kind: "info",
            title: data.gold.unanswerable ? "Gold question: deliberately unanswerable" : "Gold question: relevant document" + (relevant.size > 1 ? "s " : " ") + [...relevant].join(", "),
            text: "Results from a relevant document are marked with a green rank." + (data.gold.tags.length ? " Tags: " + data.gold.tags.join(", ") : ""),
          }),
        );
      }
      const cols = ["dense", "lexical", "hybrid"].map((mode) => {
        const rows = data.runs[mode];
        const hits = rows.filter((r) => relevant.has(r.doc_id)).length;
        const firstHit = rows.findIndex((r) => relevant.has(r.doc_id));
        return Kit.card({
          cls: "mode-col",
          title: MODE_LABEL[mode],
          subtitle: MODE_NOTE[mode],
          actions: relevant.size ? Kit.badge(firstHit === -1 ? "Missed" : firstHit === 0 ? "Hit at #1" : "Hit at #" + (firstHit + 1), firstHit === -1 ? "critical" : firstHit === 0 ? "good" : "warning") : null,
          body: rows.length
            ? h(
                "div",
                null,
                rows.map((r) =>
                  h(
                    "div",
                    { class: "rank-item" + (relevant.has(r.doc_id) ? " relevant" : "") },
                    h("span", { class: "rank-n" }, String(r.rank)),
                    h(
                      "div",
                      { style: { minWidth: "0" } },
                      h("div", { class: "row-between" }, docChip(r.doc_id), h("span", { class: "small muted num" }, fmt.num(r.score, 4))),
                      h("div", { class: "small muted truncate" }, r.section || r.title),
                      h("div", { class: "small ink-2", style: { marginTop: "4px", display: "-webkit-box", WebkitLineClamp: "2", WebkitBoxOrient: "vertical", overflow: "hidden" } }, r.text.replace(/\s+/g, " ")),
                    ),
                  ),
                ),
                h("div", { class: "small muted", style: { marginTop: "6px" } }, relevant.size ? hits + " of " + rows.length + " results from a relevant document" : ""),
              )
            : Kit.empty("No results"),
        });
      });
      out.append(h("div", { class: "grid cols-3 mt" }, cols));
    };

    const form = h("form", { class: "stack" }, h("div", { class: "ask-bar" }, input, topk, run), h("div", { class: "row" }, pick));
    form.addEventListener("submit", (e) => {
      e.preventDefault();
      go();
    });
    view.append(
      Kit.pageHead("Retrieval inspector", "The same query through each retriever. Dense and lexical retrieval fail in opposite directions; hybrid search fuses their ranks so neither failure reaches the answer."),
      Kit.card({ body: form }),
      out,
    );
    go();
  }

  // ------------------------------------------------------------------ documents
  async function documentsView(view, params) {
    const docs = await loadDocColors();
    let current = params.doc || (docs[0] && docs[0].doc_id);
    const list = h("div", { class: "list" });
    const viewer = h("div", null);

    const showDoc = async (docId) => {
      current = docId;
      list.querySelectorAll(".list-item").forEach((el) => el.classList.toggle("selected", el.dataset.doc === docId));
      const d = await Kit.busy(viewer, () => api.get("/api/documents/" + encodeURIComponent(docId)));
      Kit.clear(viewer);
      const text = h("div", { class: "doc" });
      let pos = 0;
      d.chunks.forEach((c, i) => {
        if (c.start > pos) text.append(d.text.slice(pos, c.start));
        const seg = h("span", { class: "chunk-seg" + (i % 2 ? " odd" : ""), id: "chunk-" + c.index, title: c.citation }, d.text.slice(Math.max(pos, c.start), c.end));
        text.append(seg);
        pos = Math.max(pos, c.end);
      });
      if (pos < d.text.length) text.append(d.text.slice(pos));
      const removeBtn = d.added
        ? h("button", {
            class: "btn btn-sm",
            type: "button",
            onclick: () =>
              Kit.withButton(removeBtn, async () => {
                await api.del("/api/documents/" + encodeURIComponent(d.doc_id));
                Kit.toast("Removed " + d.title + " from the index");
                overviewCache = null;
                app.go("documents");
              }),
          }, "Remove")
        : null;
      viewer.append(
        Kit.card({
          title: d.title,
          subtitle: d.doc_id + " · " + d.chunks.length + " chunks · shading marks chunk boundaries",
          actions: removeBtn,
          body: h(
            "div",
            { class: "stack" },
            text,
            Kit.table({
              columns: [
                { key: "index", label: "#", align: "right", width: "40px" },
                { key: "section", label: "Section", render: (c) => c.section || h("span", { class: "muted" }, "—") },
                { key: "chars", label: "Chars", align: "right" },
                { key: "citation", label: "Citation", cls: "hide-sm", render: (c) => h("code", null, c.citation) },
              ],
              rows: d.chunks,
              sort: { key: "index", dir: "asc" },
              onRowClick: (c) => {
                viewer.querySelectorAll(".chunk-seg.is-active").forEach((x) => x.classList.remove("is-active"));
                const seg = viewer.querySelector("#chunk-" + c.index);
                if (seg) {
                  seg.classList.add("is-active");
                  seg.scrollIntoView({ block: "center", behavior: "smooth" });
                }
              },
            }),
          ),
        }),
      );
    };

    docs.forEach((d) => {
      const item = h(
        "div",
        { class: "list-item clickable", tabindex: "0", "data-doc": d.doc_id },
        h(
          "div",
          { style: { flex: "1", minWidth: "0" } },
          h("div", { class: "row-between" }, docChip(d.doc_id, d.title), d.added ? Kit.badge("Added", "info", { icon: false }) : null),
          h("div", { class: "small muted" }, d.doc_id + " · " + fmt.int(d.words) + " words · " + d.chunks + " chunks"),
        ),
      );
      item.addEventListener("click", () => showDoc(d.doc_id));
      item.addEventListener("keydown", (e) => e.key === "Enter" && showDoc(d.doc_id));
      list.append(item);
    });

    const title = h("input", { class: "input", placeholder: "e.g. Travel and Expenses Policy", "aria-label": "Title" });
    const body = h("textarea", { class: "textarea", rows: "7", placeholder: "Paste the document text. Markdown headings become sections.", "aria-label": "Document text" });
    const add = h("button", { class: "btn btn-primary", type: "submit" }, "Add to index");
    const addForm = h("form", { class: "stack" }, Kit.field("Title", title), Kit.field("Text", body), h("div", { class: "row-between" }, h("span", { class: "small muted" }, "Indexed immediately: chunked, embedded and searchable."), add));
    addForm.addEventListener("submit", (e) => {
      e.preventDefault();
      Kit.withButton(add, async () => {
        const doc = await api.post("/api/documents", { title: title.value, text: body.value });
        Kit.toast("Indexed " + doc.title + " as " + doc.chunks + " chunks");
        overviewCache = null;
        app.go("documents", { doc: doc.doc_id });
      });
    });

    view.append(
      Kit.pageHead("Documents", "The indexed corpus. Each document is split by section so a citation always knows its heading and character offsets."),
      h(
        "div",
        { class: "split" },
        h("div", { class: "stack" }, Kit.card({ title: "Library", subtitle: docs.length + " documents", flush: true, body: list }), Kit.card({ title: "Add a document", body: addForm })),
        viewer,
      ),
    );
    if (current) showDoc(current);
  }

  // ------------------------------------------------------------------ evaluation
  async function evaluationView(view) {
    const state = { k: 5, mode: "hybrid" };
    const results = h("div", null);
    const compareHost = h("div", { class: "mt" });
    const runBtn = h("button", { class: "btn btn-primary", type: "button" }, "Run evaluation");
    const cmpBtn = h("button", { class: "btn", type: "button" }, "Compare configurations");

    const run = () =>
      Kit.withButton(runBtn, async () => {
        runBtn.textContent = "Scoring…";
        const r = await Kit.busy(results, () => api.post("/api/evaluate", state));
        renderReport(r);
      });

    const renderReport = (r) => {
      Kit.clear(results);
      const tiles = [
        ["Recall@" + r.k, fmt.num(r["recall@k"], 3), "relevant docs found"],
        ["MRR", fmt.num(r.mrr, 3), "how high the first hit ranks"],
        ["nDCG@" + r.k, fmt.num(r["ndcg@k"], 3), "ranking quality"],
        ["Answer accuracy", fmt.num(r.answer_accuracy, 3), "expected facts present"],
        ["Grounded rate", fmt.num(r.grounded_rate, 3), "claims supported by citations"],
        ["Abstention accuracy", fmt.num(r.abstention_accuracy, 3), "declined what it should"],
        ["False abstentions", fmt.num(r.false_abstention_rate, 3), "answerable but declined"],
        ["Latency p95", fmt.ms(r.latency_ms_p95), "p50 " + fmt.ms(r.latency_ms_p50)],
      ];
      results.append(h("div", { class: "grid cols-4" }, tiles.map(([label, value, note]) => Kit.stat({ label, value, note }))));
      let filter = "all";
      const rows = r.per_question.map((q, i) => Object.assign({ n: i + 1 }, q));
      const tbl = Kit.table({
        columns: [
          { key: "n", label: "#", align: "right", width: "36px" },
          { key: "question", label: "Question", render: (q) => h("div", null, q.question, h("div", { style: { marginTop: "3px" } }, q.tags.map((t) => h("span", { class: "tag" }, t)))) },
          {
            key: "hit",
            label: "Retrieval",
            render: (q) => (q.unanswerable ? Kit.badge("n/a", "", { icon: false }) : q.hit ? Kit.badge(q.reciprocal_rank === 1 ? "Hit #1" : "Hit #" + Math.round(1 / q.reciprocal_rank), q.reciprocal_rank === 1 ? "good" : "warning") : Kit.badge("Miss", "critical")),
            sortValue: (q) => (q.unanswerable ? -1 : q.reciprocal_rank),
          },
          {
            key: "answer_correct",
            label: "Answer",
            render: (q) =>
              q.unanswerable
                ? q.abstained
                  ? Kit.badge("Declined", "good")
                  : Kit.badge("Answered", "critical")
                : q.abstained
                  ? Kit.badge("Declined", "warning")
                  : q.answer_correct === null
                    ? Kit.badge("Answered", "", { icon: false })
                    : q.answer_correct
                      ? Kit.badge("Correct", "good")
                      : Kit.badge("Wrong", "critical"),
          },
          { key: "retrieved_docs", label: "Top documents", sortable: false, cls: "hide-sm", render: (q) => h("span", { class: "small ink-2" }, q.retrieved_docs.slice(0, 3).join(", ")) },
          { key: "latency_ms", label: "Time", align: "right", cls: "hide-sm", format: (v) => fmt.ms(v) },
        ],
        rows,
        maxHeight: 520,
      });
      const apply = () =>
        tbl.update(
          rows.filter((q) => filter === "all" || (filter === "misses" && !q.unanswerable && (!q.hit || q.abstained)) || (filter === "unanswerable" && q.unanswerable)),
        );
      const filters = Kit.seg(
        [
          { value: "all", label: "All " + rows.length },
          { value: "misses", label: "Misses and declines" },
          { value: "unanswerable", label: "Unanswerable" },
        ],
        "all",
        (v) => {
          filter = v;
          apply();
        },
      );
      results.append(Kit.card({ cls: "mt", title: "Per question", subtitle: r.questions + " labelled questions, scored at k=" + r.k + " with " + (MODE_LABEL[r.label] || r.label) + " retrieval", actions: filters, flush: true, body: tbl }));
    };

    const compareRun = () =>
      Kit.withButton(cmpBtn, async () => {
        cmpBtn.textContent = "Rebuilding indexes…";
        const reports = await Kit.busy(compareHost, () => api.post("/api/compare", state));
        Kit.clear(compareHost);
        const best = Math.max(...reports.map((x) => x["ndcg@k"]));
        const bars = reports
          .slice()
          .sort((a, b) => b["ndcg@k"] - a["ndcg@k"])
          .map((x) => ({ label: x.label, value: x["ndcg@k"], color: x.label === "hybrid (default)" ? "var(--series-1)" : "var(--series-other)", extra: [{ label: "MRR", value: fmt.num(x.mrr, 3) }, { label: "Recall@k", value: fmt.num(x["recall@k"], 3) }] }));
        compareHost.append(
          h(
            "div",
            { class: "split" },
            Kit.chartCard({
              title: "nDCG@" + state.k + " by configuration",
              subtitle: "The shipped default is highlighted; best score " + fmt.num(best, 3),
              chart: () => Kit.hbars(bars, { format: (v) => fmt.num(v, 3), valueLabel: "nDCG", max: 1 }),
              table: () => ({ columns: [{ key: "label", label: "Configuration" }, { key: "value", label: "nDCG", align: "right", format: (v) => fmt.num(v, 3) }], rows: bars }),
            }),
            Kit.card({
              title: "Side by side",
              subtitle: "Same corpus, same questions, one variable changed at a time",
              flush: true,
              body: Kit.table({
                columns: [
                  { key: "label", label: "Configuration", render: (x) => h("span", { class: x.label === "hybrid (default)" ? "" : "" }, x.label, x.label === "hybrid (default)" ? h("span", { class: "muted" }, " · shipped") : null) },
                  { key: "recall@k", label: "Recall", align: "right", format: (v) => fmt.num(v, 3) },
                  { key: "mrr", label: "MRR", align: "right", format: (v) => fmt.num(v, 3) },
                  { key: "ndcg@k", label: "nDCG", align: "right", format: (v) => fmt.num(v, 3) },
                  { key: "answer_accuracy", label: "Answer", align: "right", cls: "hide-sm", format: (v) => fmt.num(v, 3) },
                  { key: "abstention_accuracy", label: "Abstain", align: "right", cls: "hide-sm", format: (v) => fmt.num(v, 3) },
                  { key: "latency_ms_p50", label: "p50", align: "right", cls: "hide-sm", format: (v) => fmt.ms(v) },
                ],
                rows: reports,
                sort: { key: "ndcg@k", dir: "desc" },
              }),
            }),
          ),
        );
      });

    runBtn.addEventListener("click", run);
    cmpBtn.addEventListener("click", compareRun);
    view.append(
      Kit.pageHead("Evaluation", "The gold set is labelled with relevant documents, expected facts and deliberately unanswerable questions. The same harness gates CI.", [
        Kit.seg([{ value: "hybrid", label: "Hybrid" }, { value: "dense", label: "Dense" }, { value: "lexical", label: "Lexical" }], state.mode, (v) => (state.mode = v)),
        Kit.select([{ value: 3, label: "k = 3" }, { value: 5, label: "k = 5" }, { value: 8, label: "k = 8" }], 5, (v) => (state.k = Number(v)), { style: { width: "86px" }, "aria-label": "k" }),
        cmpBtn,
        runBtn,
      ]),
      results,
      compareHost,
    );
    run();
  }

  const app = Kit.createApp({
    title: "Knowledge Assistant",
    health: "/health",
    healthLabel: (info) => info.documents + " documents indexed",
    routes: [
      { path: "ask", label: "Ask", render: askView },
      { path: "retrieval", label: "Retrieval", render: retrievalView },
      { path: "documents", label: "Documents", render: documentsView },
      { path: "evaluation", label: "Evaluation", render: evaluationView },
    ],
  });
})();
