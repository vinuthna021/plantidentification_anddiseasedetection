(function setupHistory() {
  const STORAGE_KEY = "scan_history_v1";
  const MAX_ITEMS = 50;

  function loadHistory() {
    const raw = localStorage.getItem(STORAGE_KEY);
    return raw ? JSON.parse(raw) : [];
  }

  function saveHistory(items) {
    localStorage.setItem(STORAGE_KEY, JSON.stringify(items.slice(0, MAX_ITEMS)));
  }

  function saveScan(entry) {
    const items = loadHistory();
    items.unshift(entry);
    saveHistory(items);
  }

  function clearHistory() {
    localStorage.removeItem(STORAGE_KEY);
    renderHistoryPage();
  }

  function exportCsv() {
    const rows = loadHistory();
    const header = ["timestamp", "plant", "disease", "confidence", "severity"];
    const lines = [
      header.join(","),
      ...rows.map((r) => [r.timestamp, r.plant, r.disease, r.confidence, r.severity].map((v) => `"${String(v ?? "").replaceAll("\"", "\"\"")}"`).join(",")),
    ];
    const blob = new Blob([lines.join("\n")], { type: "text/csv;charset=utf-8;" });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = "scan_history.csv";
    a.click();
    URL.revokeObjectURL(url);
  }

  function renderHistoryPage() {
    const grid = document.getElementById("historyGrid");
    if (!grid) return;
    const filterValue = (document.getElementById("historyFilter")?.value || "").toLowerCase().trim();
    const items = loadHistory().filter((item) => {
      const hay = `${item.plant} ${item.disease}`.toLowerCase();
      return hay.includes(filterValue);
    });
    grid.innerHTML = "";
    if (!items.length) {
      grid.innerHTML = `<div class="card p-4">No history found.</div>`;
      return;
    }

    items.forEach((item) => {
      const card = document.createElement("div");
      card.className = "card p-4";
      card.innerHTML = `
        <img src="${item.image || ""}" class="h-40 w-full object-contain bg-white dark:bg-slate-800 rounded mb-3" alt="scan image" />
        <p><b>Plant:</b> ${item.plant}</p>
        <p><b>Disease:</b> ${item.disease}</p>
        <p><b>Confidence:</b> ${(item.confidence * 100).toFixed(2)}%</p>
        <p><b>Severity:</b> ${item.severity}</p>
        <p class="text-xs text-slate-500 mt-2">${new Date(item.timestamp).toLocaleString()}</p>
      `;
      grid.appendChild(card);
    });
  }

  document.addEventListener("DOMContentLoaded", () => {
    const filter = document.getElementById("historyFilter");
    filter?.addEventListener("input", renderHistoryPage);
    document.getElementById("exportCsvBtn")?.addEventListener("click", exportCsv);
    document.getElementById("clearHistoryBtn")?.addEventListener("click", clearHistory);
    renderHistoryPage();
  });

  window.HistoryManager = {
    saveScan,
    loadHistory,
    exportCsv,
    clearHistory,
    renderHistoryPage,
  };
}());
