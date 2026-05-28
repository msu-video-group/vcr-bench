(() => {
  const tableBody = document.querySelector("#benchmarkTable tbody");
  const summaryEl = document.querySelector("#benchmarkSummary");

  const format = (value, digits = 2) => {
    if (value === null || value === undefined || Number.isNaN(value)) return "-";
    return Number(value).toFixed(digits);
  };

  const weightedMean = (sum, weight) => (weight > 0 ? sum / weight : null);

  function addRun(groups, run) {
    const attack = run.attack || "unknown";
    if (!groups.has(attack)) {
      groups.set(attack, {
        attack,
        runs: 0,
        videos: 0,
        clearCorrect: 0,
        attackedSuccess: 0,
        targetSuccess: 0,
        numTotal: 0,
        psnrSum: 0,
        ssimSum: 0,
        lpipsSum: 0,
        timeSum: 0,
        metricWeight: 0,
        timeWeight: 0,
      });
    }

    const group = groups.get(attack);
    const videos = Number(run.nVideos || run.numTotal || 0);
    const metrics = run.metrics || {};
    const meanTimeSource = metrics.mean_time !== undefined && metrics.mean_time !== null
      ? metrics.mean_time
      : run.mean_time;
    const meanTime = Number(meanTimeSource || 0);

    group.runs += 1;
    group.videos += videos;
    group.clearCorrect += Number(run.clearCorrectCount || 0);
    group.attackedSuccess += Number(run.attackedSuccessCount || 0);
    group.targetSuccess += Number(run.targetSuccessCount || 0);
    group.numTotal += Number(run.numTotal || videos || 0);

    if (videos > 0) {
      if (Number.isFinite(Number(metrics.psnr))) group.psnrSum += Number(metrics.psnr) * videos;
      if (Number.isFinite(Number(metrics.ssim))) group.ssimSum += Number(metrics.ssim) * videos;
      if (Number.isFinite(Number(metrics.lpips))) group.lpipsSum += Number(metrics.lpips) * videos;
      group.metricWeight += videos;
    }

    if (Number.isFinite(meanTime) && meanTime > 0) {
      group.timeSum += meanTime * (videos || 1);
      group.timeWeight += videos || 1;
    }
  }

  function finalize(group) {
    const clearDen = group.clearCorrect > 0 ? group.clearCorrect : group.numTotal;
    return {
      ...group,
      asr: clearDen > 0 ? (100 * group.attackedSuccess) / clearDen : null,
      targetSr: clearDen > 0 ? (100 * group.targetSuccess) / clearDen : null,
      advAcc: group.numTotal > 0 ? (100 * (group.clearCorrect - group.attackedSuccess)) / group.numTotal : null,
      psnr: weightedMean(group.psnrSum, group.metricWeight),
      ssim: weightedMean(group.ssimSum, group.metricWeight),
      lpips: weightedMean(group.lpipsSum, group.metricWeight),
      meanTime: weightedMean(group.timeSum, group.timeWeight),
    };
  }

  function renderSummary(cache, rows) {
    const stats = cache.stats || {};
    const attacks = new Set(cache.runs.map((run) => run.attack)).size;
    const defences = new Set(cache.runs.map((run) => run.defence)).size;
    summaryEl.innerHTML = [
      `<span class="pill">Runs: ${cache.runs.length}</span>`,
      `<span class="pill">Attacks: ${attacks}</span>`,
      `<span class="pill">Defences: ${defences}</span>`,
      `<span class="pill">Folders: ${stats.ordinary_folders !== undefined ? stats.ordinary_folders : "-"}</span>`,
      `<span class="pill">Completed rows: ${stats.processed_runs !== undefined ? stats.processed_runs : rows.reduce((sum, row) => sum + row.runs, 0)}</span>`,
    ].join("");
  }

  function renderTable(rows) {
    tableBody.innerHTML = rows.map((row) => `
      <tr>
        <td><code>${row.attack}</code></td>
        <td class="num">${row.runs}</td>
        <td class="num">${row.videos}</td>
        <td class="num">${format(row.asr)}</td>
        <td class="num">${format(row.targetSr)}</td>
        <td class="num">${format(row.advAcc)}</td>
        <td class="num">${format(row.psnr)}</td>
        <td class="num">${format(row.ssim, 4)}</td>
        <td class="num">${format(row.lpips, 4)}</td>
        <td class="num">${format(row.meanTime)}</td>
      </tr>
    `).join("");
  }

  async function init() {
    try {
      const response = await fetch("./data/website_cache.json", { cache: "no-store" });
      if (!response.ok) throw new Error(`HTTP ${response.status}`);
      const cache = await response.json();
      const groups = new Map();
      cache.runs.forEach((run) => addRun(groups, run));
      const rows = Array.from(groups.values())
        .map(finalize)
        .sort((a, b) => {
          const bValue = b.targetSr !== null && b.targetSr !== undefined ? b.targetSr : -Infinity;
          const aValue = a.targetSr !== null && a.targetSr !== undefined ? a.targetSr : -Infinity;
          return bValue - aValue;
        });

      renderSummary(cache, rows);
      renderTable(rows);
    } catch (error) {
      tableBody.innerHTML = `<tr><td colspan="10">Failed to load benchmark data: ${error.message}</td></tr>`;
      summaryEl.innerHTML = '<span class="pill bad">Data unavailable</span>';
    }
  }

  init();
})();
