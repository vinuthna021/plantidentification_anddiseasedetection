/* global TranslationManager, HistoryManager, EnhancementManager, OfflineManager */

const state = {
  currentFile: null,
  previewDataUrl: "",
  enhancedDataUrl: "",
  lastResult: null,
  loadingMessages: [
    "Analyzing leaf texture...",
    "Running two-stage model pipeline...",
    "Computing disease confidence...",
    "Generating Grad-CAM heatmap...",
    "Building cure plan timeline...",
  ],
};

const els = {};

function getEl(id) {
  return document.getElementById(id);
}

function initElements() {
  [
    "dropZone", "fileInput", "previewImage", "enhancedPreview", "analyzeBtn",
    "loadingOverlay", "loadingMessage", "resultSection", "plantName", "diseaseName",
    "severityBadge", "confidenceText", "confidenceBar", "top5List", "warningBanner",
    "originalResultImage", "enhancedResultImage", "gradcamImage", "toggleGradcamBtn",
    "cureTabs", "cureContent", "preventionList", "organicList", "chemicalList",
    "plantConfidenceText", "diseaseConfidenceText", "feedbackForm", "rating", "comments", "themeToggle",
    "cacheModelBtn", "offlinePredictBtn", "diagnosisSummary", "timelineList",
  ].forEach((id) => { els[id] = getEl(id); });
}

function setupTheme() {
  const dark = localStorage.getItem("theme") === "dark";
  document.documentElement.classList.toggle("dark", dark);
  els.themeToggle?.addEventListener("click", () => {
    document.documentElement.classList.toggle("dark");
    localStorage.setItem("theme", document.documentElement.classList.contains("dark") ? "dark" : "light");
  });
}

function showLoading(show) {
  if (!els.loadingOverlay) return;
  if (show) {
    els.loadingOverlay.classList.remove("hidden");
    let idx = 0;
    els.loadingMessage.textContent = state.loadingMessages[idx];
    state.loadingInterval = setInterval(() => {
      idx = (idx + 1) % state.loadingMessages.length;
      els.loadingMessage.textContent = state.loadingMessages[idx];
    }, 1200);
  } else {
    clearInterval(state.loadingInterval);
    els.loadingOverlay.classList.add("hidden");
  }
}

function readFileAsDataUrl(file) {
  return new Promise((resolve, reject) => {
    const reader = new FileReader();
    reader.onload = () => resolve(reader.result);
    reader.onerror = reject;
    reader.readAsDataURL(file);
  });
}

function bindUpload() {
  if (!els.dropZone || !els.fileInput) return;

  els.dropZone.addEventListener("click", () => els.fileInput.click());
  els.fileInput.addEventListener("change", async (e) => {
    const file = e.target.files?.[0];
    if (file) await setCurrentFile(file);
  });

  ["dragover", "drop"].forEach((eventName) => {
    els.dropZone.addEventListener(eventName, (event) => event.preventDefault());
  });

  els.dropZone.addEventListener("dragenter", () => els.dropZone.classList.add("dragging"));
  els.dropZone.addEventListener("dragleave", () => els.dropZone.classList.remove("dragging"));

  els.dropZone.addEventListener("drop", async (e) => {
    els.dropZone.classList.remove("dragging");
    const file = e.dataTransfer?.files?.[0];
    if (file) await setCurrentFile(file);
  });
}

async function setCurrentFile(file) {
  if (!["image/jpeg", "image/png"].includes(file.type)) {
    alert("Only JPG and PNG are allowed.");
    return;
  }
  if (file.size > 8 * 1024 * 1024) {
    alert("Maximum file size is 8MB.");
    return;
  }
  state.currentFile = file;
  state.previewDataUrl = await readFileAsDataUrl(file);
  state.enhancedDataUrl = state.previewDataUrl;
  els.previewImage.src = state.previewDataUrl;
  els.enhancedPreview.src = state.enhancedDataUrl;
  if (window.EnhancementManager?.setSource) {
    window.EnhancementManager.setSource(state.previewDataUrl, (newDataUrl) => {
      state.enhancedDataUrl = newDataUrl;
      els.enhancedPreview.src = newDataUrl;
    });
  }
}

function renderTop5(items) {
  els.top5List.innerHTML = "";
  items.forEach((item) => {
    const li = document.createElement("li");
    li.className = "top5-row flex justify-between rounded px-3 py-2";
    li.innerHTML = `<span>${item.label}</span><span>${(item.confidence * 100).toFixed(2)}%</span>`;
    els.top5List.appendChild(li);
  });
}

function renderCurePlan(curePlan) {
  const timeline = curePlan.timeline || [];
  const prevention = curePlan.prevention || [];
  const organic = curePlan.organic_alternatives || [];
  const chemical = curePlan.chemical_options || [];
  els.cureTabs.innerHTML = "";
  els.cureContent.innerHTML = "";
  els.timelineList.innerHTML = "";

  timeline.forEach((dayPlan, idx) => {
    const btn = document.createElement("button");
    btn.className = `btn-secondary ${idx === 0 ? "active-day" : ""}`;
    btn.textContent = dayPlan.day || `Day ${idx + 1}`;
    btn.addEventListener("click", () => {
      document.querySelectorAll("#cureTabs .btn-secondary").forEach((b) => b.classList.remove("active-day"));
      btn.classList.add("active-day");
      els.cureContent.innerHTML = `<h5 class="font-semibold mb-2">${dayPlan.day}</h5><p>${dayPlan.action}</p>`;
    });
    els.cureTabs.appendChild(btn);

    const row = document.createElement("li");
    row.className = "timeline-row";
    row.innerHTML = `<span class="font-semibold">${dayPlan.day || `Day ${idx + 1}`}</span><span>${dayPlan.action || ""}</span>`;
    els.timelineList.appendChild(row);
  });

  if (timeline[0]) {
    els.cureContent.innerHTML = `<h5 class="font-semibold mb-2">${timeline[0].day}</h5><p>${timeline[0].action}</p>`;
  } else {
    els.cureContent.textContent = "No cure plan available.";
  }

  const fillList = (target, items) => {
    target.innerHTML = "";
    items.forEach((text) => {
      const li = document.createElement("li");
      li.textContent = text;
      target.appendChild(li);
    });
    if (!items.length) {
      const li = document.createElement("li");
      li.textContent = "No details available.";
      target.appendChild(li);
    }
  };
  fillList(els.preventionList, prevention);
  fillList(els.organicList, organic);
  fillList(els.chemicalList, chemical);
}

function renderResult(data) {
  state.lastResult = data;
  els.resultSection.classList.remove("hidden");
  els.diagnosisSummary.textContent = `Detected Plant Disease: ${data.plant} - ${data.disease_label}`;
  els.plantName.textContent = data.plant;
  els.diseaseName.textContent = data.disease_label;
  els.severityBadge.textContent = data.severity;
  els.severityBadge.style.backgroundColor = data.severity_color;
  const overallConfidence = typeof data.confidence_score === "number" ? data.confidence_score : data.disease_confidence;
  els.confidenceText.textContent = `${(overallConfidence * 100).toFixed(2)}%`;
  els.plantConfidenceText.textContent = `${(data.plant_confidence * 100).toFixed(2)}%`;
  els.diseaseConfidenceText.textContent = `${(data.disease_confidence * 100).toFixed(2)}%`;
  els.confidenceBar.style.width = `${Math.max(5, overallConfidence * 100)}%`;
  els.confidenceBar.style.backgroundColor = data.severity_color;
  renderTop5(data.top5_diseases || []);
  renderCurePlan(data.cure_plan || {});

  if (data.warning) {
    els.warningBanner.classList.remove("hidden");
    els.warningBanner.textContent = data.warning;
  } else {
    els.warningBanner.classList.add("hidden");
  }

  els.originalResultImage.src = state.previewDataUrl;
  els.enhancedResultImage.src = state.enhancedDataUrl || state.previewDataUrl;
  els.gradcamImage.src = `data:image/jpeg;base64,${data.gradcam_base64}`;
  els.gradcamImage.classList.add("hidden");

  HistoryManager.saveScan({
    timestamp: new Date().toISOString(),
    plant: data.plant,
    disease: data.disease_label,
    confidence: data.disease_confidence,
    severity: data.severity,
    image: state.previewDataUrl,
  });
}

async function callPredict() {
  if (!state.currentFile) {
    alert("Please upload an image first.");
    return;
  }

  showLoading(true);
  try {
    const form = new FormData();
    form.append("image", state.currentFile);
    const response = await fetch("/predict", { method: "POST", body: form });
    const data = await response.json();
    if (!response.ok) throw new Error(data.error || "Prediction failed");
    renderResult(data);
  } catch (error) {
    alert(error.message || "Something went wrong.");
  } finally {
    showLoading(false);
  }
}

function bindActions() {
  els.analyzeBtn?.addEventListener("click", callPredict);

  els.toggleGradcamBtn?.addEventListener("click", () => {
    els.gradcamImage.classList.toggle("hidden");
  });

  els.feedbackForm?.addEventListener("submit", async (e) => {
    e.preventDefault();
    if (!state.lastResult) return;
    try {
      await fetch("/feedback", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          rating: els.rating.value,
          comments: els.comments.value,
          disease: state.lastResult.disease_label,
          confidence: state.lastResult.disease_confidence,
        }),
      });
      els.comments.value = "";
      alert("Feedback submitted.");
    } catch (error) {
      alert("Failed to submit feedback.");
    }
  });

  els.cacheModelBtn?.addEventListener("click", () => OfflineManager.cacheModels());
  els.offlinePredictBtn?.addEventListener("click", async () => {
    if (!state.currentFile) return alert("Upload an image first.");
    const res = await OfflineManager.predict(state.currentFile);
    if (res?.error) alert(res.error);
    else alert(`Offline prediction: ${res.label} (${(res.confidence * 100).toFixed(2)}%)`);
  });
}

window.addEventListener("DOMContentLoaded", () => {
  initElements();
  setupTheme();
  bindUpload();
  bindActions();
  TranslationManager.init();
  EnhancementManager.init();
});
