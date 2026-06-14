(function setupEnhancement() {
  const state = {
    sourceDataUrl: "",
    onUpdate: null,
  };

  function getEl(id) {
    return document.getElementById(id);
  }

  function applySharpening(ctx, width, height, strength) {
    if (strength <= 0) return;
    const imageData = ctx.getImageData(0, 0, width, height);
    const src = imageData.data;
    const out = new Uint8ClampedArray(src);
    const kernel = [
      0, -1 * strength, 0,
      -1 * strength, 1 + 4 * strength, -1 * strength,
      0, -1 * strength, 0,
    ];

    const idx = (x, y, c) => ((y * width + x) * 4 + c);
    for (let y = 1; y < height - 1; y += 1) {
      for (let x = 1; x < width - 1; x += 1) {
        for (let c = 0; c < 3; c += 1) {
          let sum = 0;
          let k = 0;
          for (let ky = -1; ky <= 1; ky += 1) {
            for (let kx = -1; kx <= 1; kx += 1) {
              sum += src[idx(x + kx, y + ky, c)] * kernel[k++];
            }
          }
          out[idx(x, y, c)] = Math.max(0, Math.min(255, sum));
        }
      }
    }
    imageData.data.set(out);
    ctx.putImageData(imageData, 0, 0);
  }

  async function applyManualEnhancement() {
    if (!state.sourceDataUrl || !state.onUpdate) return;
    const canvas = getEl("manualCanvas");
    const brightness = parseFloat(getEl("brightnessSlider").value);
    const contrast = parseFloat(getEl("contrastSlider").value);
    const sharpness = parseFloat(getEl("sharpnessSlider").value);

    const img = new Image();
    img.src = state.sourceDataUrl;
    await img.decode();

    canvas.width = img.width;
    canvas.height = img.height;
    const ctx = canvas.getContext("2d");
    ctx.filter = `brightness(${brightness}) contrast(${contrast})`;
    ctx.drawImage(img, 0, 0);
    applySharpening(ctx, canvas.width, canvas.height, sharpness);
    state.onUpdate(canvas.toDataURL("image/jpeg", 0.92));
  }

  function resetManual() {
    getEl("brightnessSlider").value = "1.0";
    getEl("contrastSlider").value = "1.0";
    getEl("sharpnessSlider").value = "0.5";
    if (state.sourceDataUrl && state.onUpdate) state.onUpdate(state.sourceDataUrl);
  }

  async function callAiEnhance() {
    const fileInput = getEl("fileInput");
    const file = fileInput.files?.[0];
    if (!file) {
      alert("Upload an image first.");
      return;
    }
    const form = new FormData();
    form.append("image", file);
    try {
      const res = await fetch("/enhance", { method: "POST", body: form });
      const data = await res.json();
      if (!res.ok) throw new Error(data.error || "Enhancement failed");
      const enhancedDataUrl = `data:image/jpeg;base64,${data.enhanced_image}`;
      state.onUpdate?.(enhancedDataUrl);
      const list = getEl("enhancementSteps");
      list.innerHTML = "";
      (data.enhancements || []).forEach((step) => {
        const li = document.createElement("li");
        li.textContent = step;
        list.appendChild(li);
      });
    } catch (error) {
      alert(error.message);
    }
  }

  function setupTabs() {
    const aiBtn = getEl("aiEnhanceTab");
    const manualBtn = getEl("manualEnhanceTab");
    const aiPanel = getEl("aiEnhancePanel");
    const manualPanel = getEl("manualEnhancePanel");
    aiBtn?.addEventListener("click", () => {
      aiBtn.classList.add("active");
      manualBtn.classList.remove("active");
      aiPanel.classList.remove("hidden");
      manualPanel.classList.add("hidden");
    });
    manualBtn?.addEventListener("click", () => {
      manualBtn.classList.add("active");
      aiBtn.classList.remove("active");
      manualPanel.classList.remove("hidden");
      aiPanel.classList.add("hidden");
    });
  }

  window.EnhancementManager = {
    init() {
      setupTabs();
      getEl("applyManualBtn")?.addEventListener("click", applyManualEnhancement);
      getEl("resetManualBtn")?.addEventListener("click", resetManual);
      getEl("aiEnhanceBtn")?.addEventListener("click", callAiEnhance);
    },
    setSource(dataUrl, onUpdate) {
      state.sourceDataUrl = dataUrl;
      state.onUpdate = onUpdate;
    },
  };
}());
