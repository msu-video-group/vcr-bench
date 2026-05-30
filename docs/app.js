(function () {
  "use strict";
  const APP_VERSION = "2026-05-25-vcr-full-white-box";

  const DEFAULT_DATA_ROOT = "data";

  const METRIC_DIRECTIONS = {
    asr: "desc",
    target_sr: "desc",
    adv_acc: "desc",
    clear_acc: "desc",
    psnr: "desc",
    ssim: "desc",
    vmaf: "desc",
    lpips: "asc",
    dists: "asc",
    mse: "asc",
    time: "asc",
    iter_count: "asc",
    mean_time: "asc",
    mean_time_ms: "asc",
    mean_iterations: "asc",
  };

  const PREFERRED_METRICS = [
    "eps",
    "max_iters",
    "asr",
    "target_sr",
    "adv_acc",
    "clear_acc",
    "psnr",
    "ssim",
    "lpips",
    "dists",
    "vmaf",
    "mse",
    "iter_count",
    "time",
    "mean_time",
    "mean_time_ms",
    "mean_iterations",
  ];

  const state = {
    view: "attacks",
    runs: [],
    filteredRuns: [],
    tableRows: [],
    tableSort: { key: "asr", dir: "desc" },
    rawByKey: {},
    rawRowsCache: {},
    lastCsvToken: 0,
    chart: null,
    matrixScatterChart: null,
  };

  const CHART_PRESETS = {
    asr: ["asr", "target_sr", "clear_acc", "adv_acc"],
    quality: ["vmaf", "psnr", "ssim", "mse", "lpips"],
    speed: ["mean_time_ms", "mean_iterations"],
    all: ["asr", "target_sr", "adv_acc", "clear_acc", "psnr", "ssim", "lpips", "dists", "vmaf", "mse", "mean_time_ms", "mean_iterations"],
  };

  const AXIS_OPTIONS = [
    { id: "model", label: "Model", fields: ["model"] },
    { id: "attack", label: "Attack", fields: ["attack"] },
    { id: "defence", label: "Defence", fields: ["defence"] },
    { id: "model_attack", label: "Model | Attack", fields: ["model", "attack"] },
    { id: "model_defence", label: "Model | Defence", fields: ["model", "defence"] },
    { id: "attack_defence", label: "Attack | Defence", fields: ["attack", "defence"] },
  ];

  const MATRIX_COLORS = [
    "rgba(20, 103, 169, 0.85)",
    "rgba(180, 53, 72, 0.85)",
    "rgba(33, 150, 83, 0.85)",
    "rgba(160, 99, 28, 0.85)",
    "rgba(121, 87, 213, 0.85)",
    "rgba(0, 121, 107, 0.85)",
    "rgba(233, 30, 99, 0.85)",
    "rgba(96, 125, 139, 0.85)",
    "rgba(3, 169, 244, 0.85)",
    "rgba(255, 87, 34, 0.85)",
  ];

  const MATRIX_POINT_STYLES = [
    "circle",
    "triangle",
    "rect",
    "rectRot",
    "cross",
    "crossRot",
    "star",
    "line",
    "dash",
  ];

  function byId(id) {
    return document.getElementById(id);
  }

  function setStatus(msg, cls) {
    const el = byId("status");
    el.className = "status" + (cls ? " " + cls : "");
    el.textContent = msg;
  }

  function parseCsv(text) {
    return Papa.parse(text, { header: true, dynamicTyping: true, skipEmptyLines: true }).data || [];
  }

  async function fetchText(url) {
    const res = await fetch(url, { cache: "no-store" });
    if (!res.ok) throw new Error("HTTP " + res.status + " for " + url);
    return res.text();
  }

  async function fetchJson(url) {
    const res = await fetch(url, { cache: "no-store" });
    if (!res.ok) throw new Error("HTTP " + res.status + " for " + url);
    return res.json();
  }

  async function tryLoadCache(base) {
    const url = `${base.replace(/\/+$/, "")}/website_cache.json`;
    const data = await fetchJson(url);
    return { data, url };
  }

  async function loadCacheWithFallback() {
    const roots = [];
    const seen = new Set();
    const add = (v) => {
      const x = (v || "").trim().replace(/\/+$/, "");
      if (!x || seen.has(x)) return;
      seen.add(x);
      roots.push(x);
    };
    add(DEFAULT_DATA_ROOT);
    add("./data");
    add("website/data");
    add("/website/data");

    const errors = [];
    for (const r of roots) {
      try {
        const out = await tryLoadCache(r);
        return { ...out, usedRoot: r, errors };
      } catch (e) {
        errors.push(String(e && e.message ? e.message : e));
      }
    }
    throw new Error(errors.join(" | "));
  }

  function asNumber(v) {
    if (v === null || v === undefined || v === "") return null;
    const n = Number(v);
    return Number.isFinite(n) ? n : null;
  }

  function fmt(v, digits) {
    const n = asNumber(v);
    if (n == null) return "N/A";
    const d = digits != null ? digits : 3;
    const factor = Math.pow(10, d);
    return (Math.trunc(n * factor) / factor).toFixed(d);
  }

  function metricLabel(k) {
    const map = {
      key: "Entity",
      runCount: "Runs",
      nVideos: "Videos",
      num_total: "Total Num",
      eps: "Epsilon",
      max_iters: "Max Iters",
      asr: "ASR (%)",
      target_sr: "TargetSR (%)",
      clear_acc: "Clear Acc (%)",
      adv_acc: "Robust Acc (%)",
      psnr: "PSNR",
      ssim: "SSIM",
      lpips: "LPIPS",
      dists: "DISTS",
      vmaf: "VMAF",
      mse: "MSE",
      iter_count: "Iter",
      time: "Time (s)",
      mean_time: "Mean Time (s)",
      mean_time_ms: "Mean Time (ms)",
      mean_iterations: "Mean Iters",
    };
    return map[k] || k;
  }

  function axisOptionById(id) {
    return AXIS_OPTIONS.find((o) => o.id === id) || AXIS_OPTIONS[0];
  }

  function axisValue(run, axisId) {
    if (axisId === "model") return run.model || "unknown";
    if (axisId === "attack") return run.attack || "unknown";
    if (axisId === "defence") return run.defence || "unknown";
    if (axisId === "model_attack") return `${run.model || "unknown"} | ${run.attack || "unknown"}`;
    if (axisId === "model_defence") return `${run.model || "unknown"} | ${run.defence || "unknown"}`;
    if (axisId === "attack_defence") return `${run.attack || "unknown"} | ${run.defence || "unknown"}`;
    return "unknown";
  }

  function isDisjoint(a, b) {
    const s = new Set(a);
    return !b.some((x) => s.has(x));
  }

  function syncMatrixAxisSelectors(preferredRow, preferredCol) {
    const rowSel = byId("matrixRowAxis");
    const colSel = byId("matrixColAxis");
    if (!rowSel || !colSel) return;

    const currentRow = preferredRow || rowSel.value || "model";
    rowSel.innerHTML = AXIS_OPTIONS.map((o) => `<option value="${o.id}">${o.label}</option>`).join("");
    rowSel.value = AXIS_OPTIONS.some((o) => o.id === currentRow) ? currentRow : "model";

    const rowOpt = axisOptionById(rowSel.value);
    const colCandidates = AXIS_OPTIONS.filter((o) => isDisjoint(o.fields, rowOpt.fields));
    const currentCol = preferredCol || colSel.value || "attack";
    colSel.innerHTML = colCandidates.map((o) => `<option value="${o.id}">${o.label}</option>`).join("");
    if (colCandidates.some((o) => o.id === currentCol)) colSel.value = currentCol;
    else if (colCandidates.length) colSel.value = colCandidates[0].id;
  }

  function syncScatterAxisSelectors(preferredRow, preferredCol) {
    const rowSel = byId("scatterRowAxis");
    const colSel = byId("scatterColAxis");
    if (!rowSel || !colSel) return;

    const currentRow = preferredRow || rowSel.value || "model";
    rowSel.innerHTML = AXIS_OPTIONS.map((o) => `<option value="${o.id}">${o.label}</option>`).join("");
    rowSel.value = AXIS_OPTIONS.some((o) => o.id === currentRow) ? currentRow : "model";

    const rowOpt = axisOptionById(rowSel.value);
    const colCandidates = AXIS_OPTIONS.filter((o) => isDisjoint(o.fields, rowOpt.fields));
    const currentCol = preferredCol || colSel.value || "attack";
    colSel.innerHTML = colCandidates.map((o) => `<option value="${o.id}">${o.label}</option>`).join("");
    if (colCandidates.some((o) => o.id === currentCol)) colSel.value = currentCol;
    else if (colCandidates.length) colSel.value = colCandidates[0].id;
  }

  function shouldKeepMetric(name) {
    const lower = String(name).toLowerCase();
    if (
      lower.endsWith("_class") ||
      lower === "gt_class" ||
      lower === "clear_class" ||
      lower === "target_class" ||
      lower === "attacked_class"
    ) {
      return false;
    }
    if (lower === "num_total" || lower === "num_videos" || lower === "iter") {
      return false;
    }
    return true;
  }

  function parseAttackMeta(sourceId, attackFolder) {
    const raw = String(attackFolder || "").trim();
    const m = raw.match(/^(.+?)_(target|untarget)_(.+)_(adaptive|non-adaptive)(?:_(full_video))?$/);
    if (m) {
      return {
        attack: m[1],
        targetMode: m[2],
        defence: m[3],
        adaptiveMode: m[4],
        fullVideo: Boolean(m[5]),
      };
    }
    const legacyNew = raw.match(/^(.+?)_(target|untarget)_(.+)$/);
    if (legacyNew) {
      return {
        attack: legacyNew[1],
        targetMode: legacyNew[2],
        defence: legacyNew[3],
        adaptiveMode: "non-adaptive",
        fullVideo: false,
      };
    }
    const old = raw.match(/^(.+?)_(target|untarget)$/);
    if (old) {
      return {
        attack: old[1],
        targetMode: old[2],
        defence: "no_defence",
        adaptiveMode: "non-adaptive",
        fullVideo: false,
      };
    }
    return {
      attack: raw,
      targetMode: "untarget",
      defence: "no_defence",
      adaptiveMode: "non-adaptive",
      fullVideo: false,
    };
  }

  function runKey(sourceId, attackFolder, model) {
    return sourceId + "|" + attackFolder + "|" + model;
  }

  function normalizeCacheRun(r) {
    const sourceId = r.sourceId || "unknown";
    const attackFolder = r.attackFolder || r.attack || "unknown";
    const model = r.model || "unknown";
    const meta = parseAttackMeta(sourceId, attackFolder);
    const key = r.key || runKey(sourceId, attackFolder, model);
    const metrics = r.metrics || {};
    const sourceFolder = r.sourceFolder || (sourceId === "new" ? "test-defence" : sourceId === "old" ? "benchmark-kinetics400" : "");
    const csvPath =
      r.csvPath ||
      (sourceFolder ? `${sourceFolder}/${attackFolder}/${model}.csv` : "");
    return {
      key,
      sourceId,
      sourceFolder,
      attackFolder,
      attack: meta.attack,
      model,
      defence: r.defence || meta.defence || "no_defence",
      targetMode: r.targetMode || meta.targetMode || "untarget",
      adaptiveMode: r.adaptiveMode || meta.adaptiveMode || "non-adaptive",
      fullVideo: Boolean(r.fullVideo ?? meta.fullVideo),
      runCount: 1,
      nVideos: asNumber(r.nVideos) || 0,
      numTotal: asNumber(r.numTotal) || 0,
      clearCorrectCount: asNumber(r.clearCorrectCount) || 0,
      attackedSuccessCount: asNumber(r.attackedSuccessCount) || 0,
      targetSuccessCount: asNumber(r.targetSuccessCount) || 0,
      metrics,
      asr: asNumber(r.asr),
      target_sr: asNumber(r.target_sr),
      clear_acc: asNumber(r.clear_acc),
      adv_acc: asNumber(r.adv_acc),
      csvPath,
    };
  }

  function normalizeRun(source, logRow, rows, attackFolder, model) {
    const meta = parseAttackMeta(source.id, attackFolder);
    const numericStats = {};
    const numericCount = {};

    let targetSuccessFromRows = 0;
    let untargetSuccessFromRows = 0;
    let clearCorrectFromRows = 0;

    for (const row of rows) {
      for (const [k, v] of Object.entries(row)) {
        if (!shouldKeepMetric(k)) continue;
        const n = asNumber(v);
        if (n === null) continue;
        numericStats[k] = (numericStats[k] || 0) + n;
        numericCount[k] = (numericCount[k] || 0) + 1;
      }

      const targetClass = asNumber(row.target_class);
      const attackedClass = asNumber(row.attacked_class);
      const clearClass = asNumber(row.clear_class);
      const gtClass = asNumber(row.gt_class);

      if (targetClass !== null && targetClass !== -1) {
        if (attackedClass !== null && attackedClass === targetClass) targetSuccessFromRows += 1;
      } else {
        if (attackedClass !== null && clearClass !== null && attackedClass !== clearClass) {
          untargetSuccessFromRows += 1;
        }
      }
      if (clearClass !== null && gtClass !== null && clearClass === gtClass) clearCorrectFromRows += 1;
    }

    const metrics = {};
    Object.keys(numericStats).forEach((k) => {
      metrics[k] = numericStats[k] / numericCount[k];
    });

    const meanPsnrLog = asNumber(logRow.mean_psnr);
    const meanVmafLog = asNumber(logRow.mean_vmaf);
    const meanTimeLog = asNumber(logRow.mean_time);
    const meanTimeMsLog = asNumber(logRow.mean_time_ms);
    const meanIterLog = asNumber(logRow.mean_iterations);
    if (metrics.psnr == null && meanPsnrLog != null) metrics.psnr = meanPsnrLog;
    if (metrics.vmaf == null && meanVmafLog != null) metrics.vmaf = meanVmafLog;
    if (metrics.mean_time == null && meanTimeLog != null) metrics.mean_time = meanTimeLog;
    if (metrics.mean_time_ms == null && meanTimeMsLog != null) metrics.mean_time_ms = meanTimeMsLog;
    if (metrics.mean_iterations == null && meanIterLog != null) metrics.mean_iterations = meanIterLog;

    const attackedSuccessFromRows = targetSuccessFromRows + untargetSuccessFromRows;
    const clearCorrect = asNumber(logRow.clear_correct) ?? clearCorrectFromRows;
    const attackedSuccess = asNumber(logRow.attacked_success) ?? attackedSuccessFromRows;
    const targetSuccess = asNumber(logRow.target_success) ?? targetSuccessFromRows;
    const numTotal = asNumber(logRow.num_total) && asNumber(logRow.num_total) > 0 ? asNumber(logRow.num_total) : rows.length;

    const clearDen = clearCorrect > 0 ? clearCorrect : null;
    return {
      key: runKey(source.id, attackFolder, model),
      sourceId: source.id,
      sourceFolder: source.folder,
      attackFolder,
      attack: meta.attack,
      model,
      defence: meta.defence,
      targetMode: meta.targetMode,
      adaptiveMode: meta.adaptiveMode,
      fullVideo: meta.fullVideo,
      runCount: 1,
      nVideos: rows.length,
      numTotal,
      clearCorrectCount: clearCorrect,
      attackedSuccessCount: attackedSuccess,
      targetSuccessCount: targetSuccess,
      metrics,
      asr: clearDen ? (100 * attackedSuccess) / clearDen : null,
      target_sr: clearDen ? (100 * targetSuccess) / clearDen : null,
      clear_acc: numTotal > 0 ? (100 * clearCorrect) / numTotal : null,
      adv_acc: numTotal > 0 ? (100 * (clearCorrect - attackedSuccess)) / numTotal : null,
    };
  }

  async function loadSource(resultsRoot, source) {
    const logPath = `${resultsRoot}/${source.folder}/${source.log}`;
    const logRows = parseCsv(await fetchText(logPath));
    const keyToLog = new Map();
    const uniqKeys = new Set();

    for (const row of logRows) {
      const attackFolder = String(row.attack || "").trim();
      const model = String(row.model || "").trim();
      if (!attackFolder || !model) continue;
      const key = runKey(source.id, attackFolder, model);
      keyToLog.set(key, row);
      uniqKeys.add(key);
    }

    const loadTasks = [...uniqKeys].map((key) => {
      const parts = key.split("|");
      const attackFolder = parts[1];
      const model = parts[2];
      const csvPath = `${resultsRoot}/${source.folder}/${attackFolder}/${model}.csv`;
      return fetchText(csvPath)
        .then((text) => ({ ok: true, key, rows: parseCsv(text), csvPath }))
        .catch((err) => ({ ok: false, key, err, csvPath }));
    });

    const loaded = await Promise.all(loadTasks);
    const runs = [];
    const rawByKey = {};
    const misses = [];

    for (const item of loaded) {
      if (!item.ok) {
        misses.push(item.key);
        continue;
      }
      const parts = item.key.split("|");
      const attackFolder = parts[1];
      const model = parts[2];
      const logRow = keyToLog.get(item.key) || {};
      if (!item.rows.length) continue;
      const run = normalizeRun(source, logRow, item.rows, attackFolder, model);
      runs.push(run);
      rawByKey[item.key] = {
        sourceId: source.id,
        sourceFolder: source.folder,
        attackFolder,
        model,
        defence: run.defence,
        targetMode: run.targetMode,
        adaptiveMode: run.adaptiveMode,
        csvPath: item.csvPath,
        rows: item.rows,
      };
    }

    return {
      source: source.id,
      runs,
      rawByKey,
      misses,
      logRows: logRows.length,
      files: uniqKeys.size,
    };
  }

  function getFilters() {
    syncMatrixAxisSelectors();
    syncScatterAxisSelectors();
    const chartMetrics = [...byId("chartMetrics").selectedOptions].map((o) => o.value);
    return {
      resultsRoot: DEFAULT_DATA_ROOT,
      targetMode: byId("targetMode").value,
      adaptiveMode: byId("adaptiveMode").value,
      defenceFilter: byId("defenceFilter").value,
      primaryMetric: byId("primaryMetric").value,
      sortOrder: byId("sortOrder").value,
      topN: Math.max(3, Math.min(100, asNumber(byId("topN").value) || 15)),
      chartMetrics: chartMetrics.length ? chartMetrics : ["asr", "psnr", "ssim"],
      matrixRowAxis: byId("matrixRowAxis").value,
      matrixColAxis: byId("matrixColAxis").value,
      matrixMetric1: byId("matrixMetric1").value,
      matrixMetric2: byId("matrixMetric2").value,
      scatterRowAxis: byId("scatterRowAxis").value,
      scatterColAxis: byId("scatterColAxis").value,
      scatterMetricX: byId("scatterMetricX").value,
      scatterMetricY: byId("scatterMetricY").value,
      csvDetailsMode: byId("csvDetailsMode").checked,
      csvComboSelect: byId("csvComboSelect").value,
      csvRunKey: byId("csvRunSelect").value,
      csvRowLimit: Math.max(20, Math.min(5000, asNumber(byId("csvRowLimit").value) || 200)),
    };
  }

  function setChartMetrics(values) {
    const select = byId("chartMetrics");
    const wanted = new Set(values);
    [...select.options].forEach((o) => {
      o.selected = wanted.has(o.value);
    });
  }

  function baseFilterBySourceTargetAdaptive(runs, filters) {
    return runs.filter((r) => {
      if (filters.targetMode !== "all" && r.targetMode !== filters.targetMode) return false;
      if (filters.adaptiveMode !== "all" && r.adaptiveMode !== filters.adaptiveMode) return false;
      return true;
    });
  }

  function applyRunFilters(runs, filters) {
    return runs.filter((r) => {
      if (filters.defenceFilter !== "all" && r.defence !== filters.defenceFilter) return false;
      return true;
    });
  }

  function aggregateRuns(runList) {
    if (!runList.length) return null;
    const agg = {
      runCount: 0,
      nVideos: 0,
      sumNumTotal: 0,
      sumClearCorrect: 0,
      sumAttackedSuccess: 0,
      sumTargetSuccess: 0,
      weighted: {},
      weights: {},
    };
    for (const run of runList) {
      agg.runCount += 1;
      agg.nVideos += run.nVideos || 0;
      agg.sumNumTotal += run.numTotal || 0;
      agg.sumClearCorrect += run.clearCorrectCount || 0;
      agg.sumAttackedSuccess += run.attackedSuccessCount || 0;
      agg.sumTargetSuccess += run.targetSuccessCount || 0;
      const w = run.numTotal > 0 ? run.numTotal : run.nVideos > 0 ? run.nVideos : 1;
      for (const [mk, mv] of Object.entries(run.metrics)) {
        if (mv == null || !Number.isFinite(mv)) continue;
        agg.weighted[mk] = (agg.weighted[mk] || 0) + mv * w;
        agg.weights[mk] = (agg.weights[mk] || 0) + w;
      }
    }
    const row = {
      runCount: agg.runCount,
      nVideos: agg.nVideos,
      num_total: agg.sumNumTotal,
    };
    for (const [mk, sum] of Object.entries(agg.weighted)) {
      row[mk] = sum / agg.weights[mk];
    }
    const clearDen = agg.sumClearCorrect > 0 ? agg.sumClearCorrect : null;
    const totalDen = agg.sumNumTotal > 0 ? agg.sumNumTotal : null;
    row.asr = clearDen ? (100 * agg.sumAttackedSuccess) / clearDen : null;
    row.target_sr = clearDen ? (100 * agg.sumTargetSuccess) / clearDen : null;
    row.clear_acc = totalDen ? (100 * agg.sumClearCorrect) / totalDen : null;
    row.adv_acc = totalDen ? (100 * (agg.sumClearCorrect - agg.sumAttackedSuccess)) / totalDen : null;
    return row;
  }

  function aggregateBy(runs, field) {
    const groups = new Map();
    for (const run of runs) {
      const key = run[field] || "unknown";
      if (!groups.has(key)) groups.set(key, []);
      groups.get(key).push(run);
    }
    const rows = [];
    for (const [key, items] of groups.entries()) {
      const agg = aggregateRuns(items);
      if (!agg) continue;
      rows.push({ key, ...agg });
    }
    return rows;
  }

  function aggregateByKeyFn(runs, keyFn) {
    const groups = new Map();
    for (const run of runs) {
      const key = keyFn(run) || "unknown";
      if (!groups.has(key)) groups.set(key, []);
      groups.get(key).push(run);
    }
    const rows = [];
    for (const [key, items] of groups.entries()) {
      const agg = aggregateRuns(items);
      if (!agg) continue;
      rows.push({ key, ...agg });
    }
    return rows;
  }

  function getMetricColumns(rows) {
    const keys = new Set();
    rows.forEach((r) => Object.keys(r).forEach((k) => keys.add(k)));
    const fixed = ["key", "runCount", "nVideos"];
    let dynamic = [...keys].filter((k) => !fixed.includes(k));
    if (dynamic.includes("max_iters")) {
      dynamic = dynamic.filter((k) => k !== "iter_count");
    }
    const ordered = [];
    for (const m of PREFERRED_METRICS) if (dynamic.includes(m)) ordered.push(m);
    for (const m of dynamic.sort()) if (!ordered.includes(m)) ordered.push(m);
    return ordered;
  }

  function sortRows(rows, metric, sortOrder) {
    const direction = sortOrder === "best" ? (METRIC_DIRECTIONS[metric] || "desc") : sortOrder;
    return [...rows].sort((a, b) => {
      const av = asNumber(a[metric]);
      const bv = asNumber(b[metric]);
      if (av == null && bv == null) return 0;
      if (av == null) return 1;
      if (bv == null) return -1;
      return direction === "asc" ? av - bv : bv - av;
    });
  }

  function populateDefenceFilter(baseRuns) {
    const select = byId("defenceFilter");
    const prev = select.value || "all";
    const defs = [...new Set(baseRuns.map((r) => r.defence).filter(Boolean))].sort();
    select.innerHTML = `<option value="all">all</option>` + defs.map((d) => `<option value="${d}">${d}</option>`).join("");
    if (defs.includes(prev)) select.value = prev;
    else select.value = "all";
  }

  function renderSummary(filters, rows) {
    const el = byId("summary");
    const pills = [
      `View: ${state.view}`,
      `Primary: ${metricLabel(filters.primaryMetric)}`,
      `Entities: ${rows.length}`,
      `Target mode: ${filters.targetMode}`,
      `Adaptive mode: ${filters.adaptiveMode}`,
      `Defence: ${filters.defenceFilter}`,
    ];
    el.innerHTML = pills.map((p) => `<span class="pill">${p}</span>`).join("");
  }

  function renderChart(rows, sortMetric, topN, selectedMetrics) {
    const labels = rows.slice(0, topN).map((r) => r.key);
    const palette = [
      "rgba(20, 103, 169, 0.72)",
      "rgba(180, 53, 72, 0.72)",
      "rgba(33, 150, 83, 0.72)",
      "rgba(160, 99, 28, 0.72)",
      "rgba(121, 87, 213, 0.72)",
      "rgba(0, 121, 107, 0.72)",
      "rgba(233, 30, 99, 0.72)",
    ];
    const percentMetrics = new Set(["asr", "target_sr", "adv_acc", "clear_acc"]);
    const datasets = selectedMetrics.map((m, i) => ({
      label: metricLabel(m),
      data: rows.slice(0, topN).map((r) => asNumber(r[m])),
      yAxisID: percentMetrics.has(m) ? "yPercent" : "yOther",
      type: "bar",
      backgroundColor: palette[i % palette.length],
      borderColor: palette[i % palette.length].replace("0.72", "1"),
      borderWidth: 1,
    }));
    const ctx = byId("mainChart");
    if (state.chart) state.chart.destroy();
    state.chart = new Chart(ctx, {
      type: "bar",
      data: {
        labels,
        datasets,
      },
      options: {
        responsive: true,
        plugins: {
          title: {
            display: true,
            text: `Grouped metrics (sorted by ${metricLabel(sortMetric)})`,
          },
        },
        scales: {
          yPercent: {
            type: "linear",
            position: "left",
            beginAtZero: true,
            title: { display: true, text: "Rate metrics (%)" },
          },
          yOther: {
            type: "linear",
            position: "right",
            title: { display: true, text: "Quality / other metrics" },
            grid: { drawOnChartArea: false },
          },
        },
      },
    });
  }

  function renderTable(rows) {
    const table = byId("mainTable");
    const metricCols = getMetricColumns(rows);
    let html = "<thead><tr>";
    html += `<th data-k="key">Entity</th><th class="num" data-k="runCount">Runs</th><th class="num" data-k="nVideos">Videos</th>`;
    metricCols.forEach((k) => {
      html += `<th class="num" data-k="${k}">${metricLabel(k)}</th>`;
    });
    html += "</tr></thead><tbody>";
    rows.forEach((r) => {
      html += "<tr>";
      html += `<td>${r.key}</td>`;
      html += `<td class="num">${r.runCount}</td>`;
      html += `<td class="num">${asNumber(r.num_total) ?? r.nVideos ?? 0}</td>`;
      metricCols.forEach((k) => {
        const cls = k === "asr" || k === "target_sr" || k === "clear_acc" || k === "adv_acc" ? "good" : "";
        html += `<td class="num ${cls}">${fmt(r[k])}</td>`;
      });
      html += "</tr>";
    });
    html += "</tbody>";
    table.innerHTML = html;
    table.querySelectorAll("th[data-k]").forEach((th) => {
      th.style.cursor = "pointer";
      th.addEventListener("click", () => {
        const k = th.getAttribute("data-k");
        if (!k) return;
        if (state.tableSort.key === k) state.tableSort.dir = state.tableSort.dir === "asc" ? "desc" : "asc";
        else {
          state.tableSort.key = k;
          state.tableSort.dir = "desc";
        }
        state.tableRows = sortRows(state.tableRows, state.tableSort.key, state.tableSort.dir);
        renderTable(state.tableRows);
      });
    });
  }

  function heatColor(value, min, max, metric) {
    const v = asNumber(value);
    if (v == null || min == null || max == null || max === min) return "transparent";
    let t = (v - min) / (max - min);
    if ((METRIC_DIRECTIONS[metric] || "desc") === "asc") t = 1 - t;
    const alpha = 0.12 + 0.32 * Math.max(0, Math.min(1, t));
    return `rgba(20, 130, 74, ${alpha.toFixed(3)})`;
  }

  function shapeSymbol(pointStyle) {
    if (pointStyle === "circle") return "●";
    if (pointStyle === "triangle") return "▲";
    if (pointStyle === "rect") return "■";
    if (pointStyle === "rectRot") return "◆";
    if (pointStyle === "cross") return "✚";
    if (pointStyle === "crossRot") return "✖";
    if (pointStyle === "star") return "★";
    if (pointStyle === "line") return "━";
    if (pointStyle === "dash") return "▭";
    return "●";
  }

  function matrixGrid(runs, rowAxis, colAxis) {
    const rowOpt = axisOptionById(rowAxis);
    const colOpt = axisOptionById(colAxis);
    const rowLabels = [...new Set(runs.map((r) => axisValue(r, rowOpt.id)))].sort();
    const colLabels = [...new Set(runs.map((r) => axisValue(r, colOpt.id)))].sort();
    const pairMap = new Map();

    for (const run of runs) {
      const rk = axisValue(run, rowOpt.id);
      const ck = axisValue(run, colOpt.id);
      const k = `${rk}||${ck}`;
      if (!pairMap.has(k)) pairMap.set(k, []);
      pairMap.get(k).push(run);
    }

    const cells = {};
    for (const r of rowLabels) {
      for (const c of colLabels) {
        cells[`${r}||${c}`] = aggregateRuns(pairMap.get(`${r}||${c}`) || []);
      }
    }
    return { rowOpt, colOpt, rowLabels, colLabels, cells };
  }

  function renderMatrix(runs, rowAxis, colAxis, metric1, metric2) {
    const tableEl = byId("matrixTable");
    if (!tableEl) return;

    const { rowOpt, colOpt, rowLabels, colLabels, cells } = matrixGrid(runs, rowAxis, colAxis);
    const m1Vals = [];
    for (const r of rowLabels) {
      for (const c of colLabels) {
        const agg = cells[`${r}||${c}`];
        const m1 = agg ? asNumber(agg[metric1]) : null;
        if (m1 != null) m1Vals.push(m1);
      }
    }
    const m1Min = m1Vals.length ? Math.min(...m1Vals) : null;
    const m1Max = m1Vals.length ? Math.max(...m1Vals) : null;

    let html = `<thead><tr><th>${rowOpt.label} \\ ${colOpt.label}</th>`;
    colLabels.forEach((c) => {
      html += `<th>${c}</th>`;
    });
    html += "</tr></thead><tbody>";
    rowLabels.forEach((r) => {
      html += `<tr><th>${r}</th>`;
      colLabels.forEach((c) => {
        const cell = cells[`${r}||${c}`];
        if (!cell) {
          html += `<td class="matrix-cell">-</td>`;
          return;
        }
        const v1 = asNumber(cell[metric1]);
        const v2 = asNumber(cell[metric2]);
        const bg = heatColor(v1, m1Min, m1Max, metric1);
        html += `<td class="matrix-cell" style="background:${bg}"><div class="matrix-m1">${fmt(v1, 2)}</div><div class="matrix-m2">${fmt(v2, 2)}</div></td>`;
      });
      html += "</tr>";
    });
    html += "</tbody>";
    tableEl.innerHTML = html;
  }

  function renderMatrixScatter(runs, rowAxis, colAxis, metricX, metricY) {
    const canvas = byId("matrixScatterChart");
    const legend = byId("matrixScatterLegend");
    if (!canvas) return;
    if (state.matrixScatterChart) {
      state.matrixScatterChart.destroy();
      state.matrixScatterChart = null;
    }

    const { rowOpt, colOpt, rowLabels, colLabels, cells } = matrixGrid(runs, rowAxis, colAxis);
    const colorByRow = new Map(rowLabels.map((r, i) => [r, MATRIX_COLORS[i % MATRIX_COLORS.length]]));
    const shapeByCol = new Map(colLabels.map((c, i) => [c, MATRIX_POINT_STYLES[i % MATRIX_POINT_STYLES.length]]));

    const datasets = [];
    for (const c of colLabels) {
      const points = [];
      const colors = [];
      const styles = [];
      for (const r of rowLabels) {
        const cell = cells[`${r}||${c}`];
        if (!cell) continue;
        const x = asNumber(cell[metricX]);
        const y = asNumber(cell[metricY]);
        if (x == null || y == null) continue;
        points.push({ x, y, row: r, col: c });
        colors.push(colorByRow.get(r));
        styles.push(shapeByCol.get(c));
      }
      if (!points.length) continue;
      datasets.push({
        label: c,
        data: points,
        pointBackgroundColor: colors,
        pointBorderColor: colors,
        pointStyle: styles,
        pointRadius: 5,
        pointHoverRadius: 7,
        showLine: false,
      });
    }

    const rowLegend = rowLabels
      .map((r) => `<span class="pill"><span style="color:${colorByRow.get(r)};font-weight:700;">●</span> ${r}</span>`)
      .join("");
    const colLegend = colLabels
      .map((c) => `<span class="pill"><span style="font-weight:700;">${shapeSymbol(shapeByCol.get(c))}</span> ${c}</span>`)
      .join("");
    legend.innerHTML = `
      <div style="border:1px solid #d9dee6;border-radius:10px;padding:10px;background:#fbfcfe;">
        <div style="display:flex;flex-wrap:wrap;gap:8px;margin-bottom:8px;">
          <span class="pill">X: ${metricLabel(metricX)}</span>
          <span class="pill">Y: ${metricLabel(metricY)}</span>
        </div>
        <div style="margin-bottom:8px;">
          <span class="pill">Color (${rowOpt.label})</span>
          ${rowLegend}
        </div>
        <div>
          <span class="pill">Shape (${colOpt.label})</span>
          ${colLegend}
        </div>
      </div>
    `;

    state.matrixScatterChart = new Chart(canvas, {
      type: "scatter",
      data: { datasets },
      options: {
        responsive: true,
        plugins: {
          legend: { display: true, position: "bottom" },
          title: {
            display: true,
            text: `${rowOpt.label} vs ${colOpt.label}: ${metricLabel(metricY)} vs ${metricLabel(metricX)}`,
          },
          tooltip: {
            callbacks: {
              label: (ctx) => {
                const raw = ctx.raw || {};
                return `${raw.row} | ${raw.col}: ${metricLabel(metricX)}=${fmt(raw.x, 3)}, ${metricLabel(metricY)}=${fmt(raw.y, 3)}`;
              },
            },
          },
        },
        scales: {
          x: { type: "linear", title: { display: true, text: metricLabel(metricX) } },
          y: { type: "linear", title: { display: true, text: metricLabel(metricY) } },
        },
      },
    });
  }

  function refreshCsvRunSelect(runs) {
    const select = byId("csvRunSelect");
    const prev = select.value;
    const opts = runs
      .map((r) => ({
        key: r.key,
        label: `${r.sourceId} | ${r.attackFolder} | ${r.model} | ${r.defence}`,
      }))
      .sort((a, b) => a.label.localeCompare(b.label));
    select.innerHTML = opts.map((o) => `<option value="${o.key}">${o.label}</option>`).join("");
    if (opts.some((o) => o.key === prev)) select.value = prev;
    if (!select.value && opts.length) select.value = opts[0].key;
  }

  function refreshCsvComboSelect(runs) {
    const select = byId("csvComboSelect");
    if (!select) return;
    const prev = select.value;
    const combos = [...new Set(runs.map((r) => `${r.attack} | ${r.defence}`))].sort();
    select.innerHTML = combos.map((c) => `<option value="${c}">${c}</option>`).join("");
    if (combos.some((c) => c === prev)) select.value = prev;
    if (!select.value && combos.length) select.value = combos[0];
  }

  function renderClassifierStatsTable(runs, combo) {
    const table = byId("csvTable");
    const metaEl = byId("csvMeta");
    const select = byId("csvRunSelect");
    const rowLimit = byId("csvRowLimit");
    if (select) select.disabled = true;
    if (rowLimit) rowLimit.disabled = true;

    if (!combo) {
      metaEl.innerHTML = `<span class="pill">No combo selected</span>`;
      table.innerHTML = "";
      return;
    }
    const [attack, defence] = combo.split(" | ");
    const scoped = runs.filter((r) => r.attack === attack && r.defence === defence);
    const rows = aggregateBy(scoped, "model");
    const totalsByModel = new Map();
    for (const r of scoped) {
      totalsByModel.set(r.model, (totalsByModel.get(r.model) || 0) + (asNumber(r.numTotal) || 0));
    }
    let metricCols = getMetricColumns(rows);
    const forced = ["eps", "max_iters"];
    metricCols = [...forced, ...metricCols.filter((k) => !forced.includes(k))];

    metaEl.innerHTML = [
      `<span class="pill">mode=summary-by-classifier</span>`,
      `<span class="pill">combo=${combo}</span>`,
      `<span class="pill">classifiers=${rows.length}</span>`,
      `<span class="pill">runs=${scoped.length}</span>`,
    ].join("");

    let html = "<thead><tr>";
    html += `<th>Classifier</th><th class="num">Runs</th><th class="num">Videos</th>`;
    metricCols.forEach((k) => {
      html += `<th class="num">${metricLabel(k)}</th>`;
    });
    html += "</tr></thead><tbody>";
    rows.forEach((r) => {
      html += "<tr>";
      html += `<td>${r.key}</td>`;
      html += `<td class="num">${r.runCount}</td>`;
      html += `<td class="num">${totalsByModel.get(r.key) || 0}</td>`;
      metricCols.forEach((k) => {
        html += `<td class="num">${fmt(r[k])}</td>`;
      });
      html += "</tr>";
    });
    html += "</tbody>";
    table.innerHTML = html;
  }

  async function renderCsvViewer(runKeyValue, rowLimit) {
    const meta = state.rawByKey[runKeyValue];
    const table = byId("csvTable");
    const metaEl = byId("csvMeta");
    const select = byId("csvRunSelect");
    const rowLimitEl = byId("csvRowLimit");
    if (select) select.disabled = false;
    if (rowLimitEl) rowLimitEl.disabled = false;
    if (!meta) {
      metaEl.innerHTML = `<span class="pill">No run selected</span>`;
      table.innerHTML = "";
      return;
    }
    const token = Date.now();
    state.lastCsvToken = token;
    const resultsRoot = DEFAULT_DATA_ROOT;
    if (!state.rawRowsCache[runKeyValue]) {
      metaEl.innerHTML = `<span class="pill">loading csv...</span>`;
      try {
        const csvText = await fetchText(`${resultsRoot}/${meta.csvPath}`);
        state.rawRowsCache[runKeyValue] = parseCsv(csvText);
      } catch (err) {
        if (token !== state.lastCsvToken) return;
        metaEl.innerHTML = `<span class="pill bad">Failed to load CSV: ${meta.csvPath}</span>`;
        table.innerHTML = "";
        return;
      }
    }
    if (token !== state.lastCsvToken) return;

    const rows = state.rawRowsCache[runKeyValue] || [];
    const limited = rows.slice(0, rowLimit);
    const cols = [...new Set(limited.flatMap((r) => Object.keys(r)))];

    metaEl.innerHTML = [
      `source=${meta.sourceId}`,
      `attack_folder=${meta.attackFolder}`,
      `model=${meta.model}`,
      `defence=${meta.defence}`,
      `rows=${rows.length}`,
      `path=${meta.csvPath}`,
    ]
      .map((s) => `<span class="pill">${s}</span>`)
      .join("");

    let html = "<thead><tr>";
    cols.forEach((c) => {
      html += `<th>${c}</th>`;
    });
    html += "</tr></thead><tbody>";
    limited.forEach((r) => {
      html += "<tr>";
      cols.forEach((c) => {
        const val = r[c];
        const n = asNumber(val);
        html += n == null ? `<td>${val ?? ""}</td>` : `<td class="num">${fmt(n)}</td>`;
      });
      html += "</tr>";
    });
    html += "</tbody>";
    table.innerHTML = html;
  }

  function buildViewRows(filteredRuns, filters) {
    if (state.view === "models") return aggregateBy(filteredRuns, "model");
    if (state.view === "defences") return aggregateBy(filteredRuns, "defence");
    return aggregateBy(filteredRuns, "attack");
  }

  function renderAll() {
    const filters = getFilters();
    const baseRuns = baseFilterBySourceTargetAdaptive(state.runs, filters);
    populateDefenceFilter(baseRuns);
    filters.defenceFilter = byId("defenceFilter").value;
    state.filteredRuns = applyRunFilters(baseRuns, filters);

    const viewRows = buildViewRows(state.filteredRuns, filters);
    state.tableRows = sortRows(viewRows, filters.primaryMetric, filters.sortOrder);
    renderSummary(filters, state.tableRows);
    renderChart(state.tableRows, filters.primaryMetric, filters.topN, filters.chartMetrics);
    renderTable(state.tableRows);
    renderMatrix(state.filteredRuns, filters.matrixRowAxis, filters.matrixColAxis, filters.matrixMetric1, filters.matrixMetric2);
    renderMatrixScatter(state.filteredRuns, filters.scatterRowAxis, filters.scatterColAxis, filters.scatterMetricX, filters.scatterMetricY);
    refreshCsvRunSelect(state.filteredRuns);
    refreshCsvComboSelect(state.filteredRuns);
    const selectedCombo = byId("csvComboSelect").value;
    if (filters.csvDetailsMode) {
      renderCsvViewer(byId("csvRunSelect").value, filters.csvRowLimit);
    } else {
      renderClassifierStatsTable(state.filteredRuns, selectedCombo);
    }
  }

  async function reloadData() {
    setStatus("Loading precomputed cache...", "");

    try {
      const loaded = await loadCacheWithFallback();
      const cache = loaded.data;
      const rawRuns = Array.isArray(cache.runs) ? cache.runs : [];
      state.runs = rawRuns.map(normalizeCacheRun);
      state.rawByKey = {};
      state.rawRowsCache = {};
      state.runs.forEach((r) => {
        state.rawByKey[r.key] = {
          sourceId: r.sourceId,
          sourceFolder: r.sourceFolder,
          attackFolder: r.attackFolder,
          model: r.model,
          defence: r.defence,
          targetMode: r.targetMode,
          adaptiveMode: r.adaptiveMode,
          csvPath: r.csvPath,
        };
      });
      renderAll();
      const generatedAt = cache.generated_at || "unknown";
      setStatus(`Loaded cache: runs=${state.runs.length}, generated_at=${generatedAt}, root=${loaded.usedRoot}`, "good");
    } catch (err) {
      setStatus(
        "Failed to load website/data/website_cache.json via HTTP. " +
          String(err && err.message ? err.message : err),
        "bad"
      );
    }
  }

  function initEvents() {
    byId("reloadBtn").addEventListener("click", reloadData);
    [
      "targetMode",
      "adaptiveMode",
      "defenceFilter",
      "primaryMetric",
      "chartMetrics",
      "sortOrder",
      "topN",
      "matrixRowAxis",
      "matrixColAxis",
      "matrixMetric1",
      "matrixMetric2",
      "scatterRowAxis",
      "scatterColAxis",
      "scatterMetricX",
      "scatterMetricY",
      "csvDetailsMode",
      "csvComboSelect",
      "csvRowLimit",
    ].forEach((id) => byId(id).addEventListener("change", renderAll));
    byId("csvRunSelect").addEventListener("change", renderAll);
    byId("presetAsr").addEventListener("click", () => {
      setChartMetrics(CHART_PRESETS.asr);
      renderAll();
    });
    byId("presetQuality").addEventListener("click", () => {
      setChartMetrics(CHART_PRESETS.quality);
      renderAll();
    });
    byId("presetSpeed").addEventListener("click", () => {
      setChartMetrics(CHART_PRESETS.speed);
      renderAll();
    });
    byId("presetAll").addEventListener("click", () => {
      setChartMetrics(CHART_PRESETS.all);
      renderAll();
    });

    document.querySelectorAll(".tablink").forEach((btn) => {
      btn.addEventListener("click", () => {
        document.querySelectorAll(".tablink").forEach((b) => b.classList.remove("active"));
        btn.classList.add("active");
        state.view = btn.dataset.view;
        renderAll();
      });
    });
  }

  function initFromQuery() {
    return;
  }

  function init() {
    console.log("[website] app version:", APP_VERSION);
    const title = document.title || "Adversarial Benchmark Explorer";
    document.title = `${title} (${APP_VERSION})`;
    setStatus(`app start ${APP_VERSION}`, "");
    initFromQuery();
    setChartMetrics(["asr", "psnr", "ssim"]);
    initEvents();
    reloadData();
  }

  init();
})();
