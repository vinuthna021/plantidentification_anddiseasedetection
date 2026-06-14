(function setupOffline() {
  const OFFLINE_MODEL_KEY = "indexeddb://disease-offline-model";

  async function cacheModels() {
    if (!window.tf) {
      alert("TensorFlow.js not available.");
      return;
    }
    try {
      const remoteUrl = "/static/model_web/model.json";
      const model = await tf.loadLayersModel(remoteUrl);
      await model.save(OFFLINE_MODEL_KEY);
      alert("Model cached successfully in IndexedDB.");
    } catch (error) {
      alert(`Failed to cache model: ${error.message}`);
    }
  }

  async function loadOfflineModel() {
    const models = await tf.io.listModels();
    if (!models[OFFLINE_MODEL_KEY]) {
      throw new Error("Offline model is not cached yet.");
    }
    return tf.loadLayersModel(OFFLINE_MODEL_KEY);
  }

  async function preprocessFile(file, size = 224) {
    const bitmap = await createImageBitmap(file);
    const canvas = document.createElement("canvas");
    canvas.width = size;
    canvas.height = size;
    const ctx = canvas.getContext("2d");
    ctx.drawImage(bitmap, 0, 0, size, size);
    const tensor = tf.browser.fromPixels(canvas).toFloat().div(255).expandDims(0);
    return tensor;
  }

  async function predict(file) {
    if (!window.tf) return { error: "TensorFlow.js is unavailable." };
    try {
      const model = await loadOfflineModel();
      const tensor = await preprocessFile(file);
      const pred = model.predict(tensor);
      const data = await pred.data();
      tensor.dispose();
      pred.dispose();
      const bestIdx = data.indexOf(Math.max(...data));
      return {
        label: `Class ${bestIdx}`,
        confidence: data[bestIdx],
      };
    } catch (error) {
      return { error: error.message };
    }
  }

  window.OfflineManager = {
    cacheModels,
    predict,
  };
}());
