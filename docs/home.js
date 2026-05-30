(() => {
  const tableBody = document.querySelector("#benchmarkTable tbody");
  const tableHead = document.querySelector("#benchmarkTable thead tr");
  const summaryEl = document.querySelector("#benchmarkSummary");
  const tabsEl   = document.querySelector("#leaderboardTabs");

  const format = (value, digits = 3) => {
    if (value === null || value === undefined || Number.isNaN(value)) return "-";
    const n = Number(value);
    const factor = Math.pow(10, digits);
    return (Math.trunc(n * factor) / factor).toFixed(digits);
  };

  const weightedMean = (sum, weight) => (weight > 0 ? sum / weight : null);

  function aggregateRuns(runs, keyFn) {
    const groups = new Map();
    for (const run of runs) {
      const key = keyFn(run);
      if (!groups.has(key)) {
        groups.set(key, {
          key,
          clearCorrect: 0, attackedSuccess: 0, targetSuccess: 0, numTotal: 0,
          psnrSum: 0, ssimSum: 0, lpipsSum: 0,
          timeSum: 0, metricWeight: 0, timeWeight: 0,
        });
      }
      const g = groups.get(key);
      const videos  = Number(run.nVideos || run.numTotal || 0);
      const metrics = run.metrics || {};
      const meanTime = Number(metrics.mean_time || run.mean_time || 0);

      g.clearCorrect     += Number(run.clearCorrectCount  || 0);
      g.attackedSuccess  += Number(run.attackedSuccessCount || 0);
      g.targetSuccess    += Number(run.targetSuccessCount || 0);
      g.numTotal         += Number(run.numTotal || videos || 0);

      if (videos > 0) {
        if (Number.isFinite(Number(metrics.psnr)))  { g.psnrSum  += Number(metrics.psnr)  * videos; }
        if (Number.isFinite(Number(metrics.ssim)))  { g.ssimSum  += Number(metrics.ssim)  * videos; }
        if (Number.isFinite(Number(metrics.lpips))) { g.lpipsSum += Number(metrics.lpips) * videos; }
        g.metricWeight += videos;
      }
      if (Number.isFinite(meanTime) && meanTime > 0) {
        g.timeSum    += meanTime * (videos || 1);
        g.timeWeight += videos || 1;
      }
    }

    return Array.from(groups.values()).map((g) => {
      const clearDen = g.clearCorrect > 0 ? g.clearCorrect : g.numTotal;
      return {
        key:        g.key,
        asr:        clearDen > 0  ? (100 * g.attackedSuccess) / clearDen : null,
        targetSr:   clearDen > 0  ? (100 * g.targetSuccess)   / clearDen : null,
        robustAcc:  g.numTotal > 0 ? (100 * (g.clearCorrect - g.attackedSuccess)) / g.numTotal : null,
        psnr:       weightedMean(g.psnrSum,  g.metricWeight),
        ssim:       weightedMean(g.ssimSum,  g.metricWeight),
        lpips:      weightedMean(g.lpipsSum, g.metricWeight),
        meanTime:   weightedMean(g.timeSum,  g.timeWeight),
      };
    });
  }

  const COLS = [
    { key: "asr",       label: "ASR (%)"         },
    { key: "targetSr",  label: "TargetSR (%)"     },
    { key: "robustAcc", label: "Robust Acc (%)"   },
    { key: "psnr",      label: "PSNR"             },
    { key: "ssim",      label: "SSIM"             },
    { key: "lpips",     label: "LPIPS"            },
    { key: "meanTime",  label: "Mean Time (s)"    },
  ];

  const VIEWS = {
    attacks:  { rowLabel: "Attack",  keyFn: (r) => r.attack  || "unknown",    sortKey: "asr",       sortDir: "desc" },
    models:   { rowLabel: "Model",   keyFn: (r) => r.model   || "unknown",    sortKey: "robustAcc", sortDir: "desc" },
    defences: { rowLabel: "Defence", keyFn: (r) => r.defence || "no_defence", sortKey: "robustAcc", sortDir: "desc" },
  };

  let cache = null;

  function renderTable(viewKey) {
    const view = VIEWS[viewKey];

    const rows = aggregateRuns(cache.runs, view.keyFn).sort((a, b) => {
      const av = a[view.sortKey] ?? -Infinity;
      const bv = b[view.sortKey] ?? -Infinity;
      return bv - av;
    });

    tableHead.innerHTML =
      `<th>${view.rowLabel}</th>` +
      COLS.map((c) => `<th class="has-text-right">${c.label}</th>`).join("");

    tableBody.innerHTML = rows.map((row) =>
      `<tr><td><code>${row.key}</code></td>` +
      COLS.map((c) => `<td class="has-text-right">${format(row[c.key])}</td>`).join("") +
      `</tr>`
    ).join("");
  }

  function renderSummary() {
    const stats   = cache.stats || {};
    const attacks  = new Set(cache.runs.map((r) => r.attack)).size;
    const defences = new Set(cache.runs.map((r) => r.defence)).size;
    summaryEl.innerHTML = [
      `<span class="pill">Runs: ${cache.runs.length}</span>`,
      `<span class="pill">Attacks: ${attacks}</span>`,
      `<span class="pill">Defences: ${defences}</span>`,
      `<span class="pill">Folders: ${stats.ordinary_folders ?? "-"}</span>`,
    ].join("");
  }

  function setView(viewKey) {
    tabsEl.querySelectorAll("button").forEach((btn) => {
      btn.classList.toggle("is-active", btn.dataset.view === viewKey);
    });
    renderTable(viewKey);
  }

  async function init() {
    try {
      const res = await fetch("./data/website_cache.json", { cache: "no-store" });
      if (!res.ok) throw new Error(`HTTP ${res.status}`);
      cache = await res.json();

      tabsEl.querySelectorAll("button").forEach((btn) => {
        btn.addEventListener("click", () => setView(btn.dataset.view));
      });

      renderSummary();
      setView("attacks");
    } catch (err) {
      tableBody.innerHTML = `<tr><td colspan="8">Failed to load benchmark data: ${err.message}</td></tr>`;
      summaryEl.innerHTML = '<span class="pill">Data unavailable</span>';
    }
  }

  init();
})();
