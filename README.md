# Plant Identification & Disease Detection

An AI-powered agricultural assistant that identifies **Potato**, **Tomato**, and **Pepper** plants from leaf images and detects **15 disease classes** (including healthy states). The project combines deep learning inference with actionable treatment guidance — helping farmers and gardeners diagnose crop issues quickly without expert consultation.

Built with **TensorFlow/Keras** CNN models trained on the PlantVillage dataset, exposed through both a lightweight **Streamlit** demo and a full-featured **Flask** web application.

---

## Live Demo

| Deployment | Link | Description |
|------------|------|-------------|
| **Live App** | [plant-disease-detection-hy63.onrender.com](https://plant-disease-detection-hy63.onrender.com) | Full Flask app — upload images, get diagnoses, cure plans, and Grad-CAM |
| **Landing Page** | [plantidentification-anddiseasedetec.vercel.app](https://plantidentification-anddiseasedetec.vercel.app) | Project overview and quick access to the live app |
| **Source Code** | [github.com/vinuthna021/plantidentification_anddiseasedetection](https://github.com/vinuthna021/plantidentification_anddiseasedetection) | GitHub repository |

> **Note:** The Render free tier sleeps after ~15 minutes of inactivity. The first visit after idle may take 30–60 seconds to load.

---

## Project Description

This system performs **two-stage plant health analysis**:

1. **Plant identification** — automatically detects whether the leaf belongs to Potato, Tomato, or Pepper (via a dedicated plant classifier or confidence-based routing).
2. **Disease classification** — runs a plant-specific CNN to predict the exact disease or healthy status, with calibrated confidence scores.

Beyond raw predictions, the Flask app delivers **Grad-CAM visual explanations**, a **7-day cure timeline**, and **prevention / organic / chemical treatment options** tailored to each diagnosis — turning model output into practical field guidance.

---

## Unique Features

| Feature | Description |
|--------|-------------|
| **Two-stage pipeline** | Plant ID → disease model routing for higher accuracy than a single multi-class model |
| **Temperature-calibrated confidence** | Post-training calibration so reported probabilities better match real accuracy |
| **Test-time augmentation (TTA)** | Horizontal/vertical flips averaged at inference for more stable predictions |
| **Grad-CAM heatmaps** | Visual explanation of which leaf regions drove the diagnosis |
| **7-day cure timeline** | Day-by-day treatment plan per disease from `cure_database.json` |
| **Treatment recommendations** | Prevention, organic alternatives, and chemical options for each class |
| **AI image enhancement** | Auto contrast, brightness, blur, and resize before analysis |
| **Manual enhancement controls** | Brightness, contrast, and sharpness sliders for low-quality photos |
| **Multilingual UI** | English, Telugu, Hindi, and Tamil (Flask app) |
| **Scan history** | Local browser history of past diagnoses |
| **Offline inference support** | TensorFlow.js model caching for use without network (Flask app) |
| **User feedback loop** | Rating and comments logged for continuous improvement |
| **Dual interfaces** | Simple Streamlit prototype + production-ready Flask UI |
| **Retraining pipeline** | Reproducible scripts to retrain, evaluate, and calibrate per-plant models |

---

## Tech Stack

| Layer | Technologies |
|-------|-------------|
| **Deep Learning** | TensorFlow, Keras, MobileNetV2 (transfer learning) |
| **Backend** | Python, Flask |
| **Frontend (Flask app)** | HTML, Tailwind CSS, JavaScript, Font Awesome |
| **Demo UI** | Streamlit |
| **Image Processing** | OpenCV, Pillow (PIL), NumPy |
| **Dataset Loading** | TensorFlow Datasets (PlantVillage), Kaggle API |
| **Client-side ML** | TensorFlow.js (offline mode) |
| **Data Format** | JSON (cure database, class labels, calibration metadata) |

---

## Dataset

Models are trained on the **[PlantVillage](https://www.kaggle.com/datasets/arjuntejaswi/plant-village)** dataset from Kaggle (`arjuntejaswi/plant-village`), a widely used benchmark for plant disease classification.

**Structure:** `PlantVillage/<class_name>/*.jpg`

**Classes used in this project:**

| Plant | Classes | Count |
|-------|---------|-------|
| **Potato** | Early blight, Late blight, Healthy | 3 |
| **Tomato** | Healthy, Spider mites, Target spot, Septoria leaf spot, Mosaic virus, Leaf mold, Bacterial spot, Late blight, Early blight, Yellow leaf curl virus | 10 |
| **Pepper** | Healthy, Bacterial spot | 2 |

**Total disease targets:** 15 classes across 3 crops.

**Data splits:** 70% train / 10% validation / 10% test (deterministic seed).

**Optional download** (requires `~/.kaggle/kaggle.json`):

```bash
python -m train.download_kaggle --out_dir data
```

---

## Model Targets

### Architecture

- **Backbone:** MobileNetV2 (ImageNet weights, frozen then fine-tuned)
- **Head:** GlobalAveragePooling → Dropout(0.2) → Softmax
- **Input size:** 224 × 224 RGB
- **Training:** Two-phase — frozen backbone + fine-tune last 40 layers
- **Augmentation:** Random flip, rotation, zoom
- **Calibration:** Temperature scaling on validation set

### Performance (test set)

| Model | Classes | Test Top-1 Accuracy |
|-------|---------|---------------------|
| Potato | 3 | **96.8%** |
| Pepper | 2 | **92.7%** |
| Tomato | 10 | **91.8%** |
| Plant ID | 3 | Trained via `train/train_plant_id.py` |

Training enforces a **≥ 90% test accuracy** target per plant model. Calibrated confidence stats are stored in each model's `calibration.json`.

### Pretrained model locations

```
potato_trained_models/1/
tomato_trained_models/1/
pepper_trained_models/1/
models/plant_model.keras          # Plant identification model
models/plant_classes.json
```

---

## Competitive Advantage

Compared to typical plant disease apps that only return a label and confidence score, this project offers:

1. **Actionable output** — not just *what* the disease is, but *what to do* with a structured 7-day plan and treatment options.
2. **Explainable AI** — Grad-CAM heatmaps show affected leaf regions, building user trust in low-connectivity farm settings.
3. **Regional accessibility** — multilingual UI (English, Telugu, Hindi, Tamil) for Indian agricultural users.
4. **Robust inference** — TTA + temperature scaling + per-plant specialized models outperform generic single-model approaches.
5. **Field-ready preprocessing** — built-in image enhancement handles poor lighting and phone-camera quality common in the field.
6. **Offline capability** — TensorFlow.js caching enables diagnosis without reliable internet.
7. **End-to-end pipeline** — from Kaggle download → train → calibrate → deploy, all scripts included and reproducible.
8. **Dual deployment** — Streamlit for quick demos; Flask for a full product experience with history, feedback, and rich UI.

---

## Setup

### Prerequisites

- Python 3.10+
- pip

### 1. Clone the repository

```bash
git clone https://github.com/vinuthna021/plantidentification_anddiseasedetection.git
cd plantidentification_anddiseasedetection
```

### 2. Create and activate a virtual environment

**macOS / Linux:**

```bash
python -m venv .venv
source .venv/bin/activate
```

**Windows:**

```cmd
python -m venv .venv
.venv\Scripts\activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

Pretrained models are included in the repo — no training is required to run inference.

### 4. Run the application

**Streamlit (simple demo):**

```bash
streamlit run app.py
```

Open the URL shown in the terminal (default: `http://localhost:8501`). Select a plant, upload a leaf image, and view the prediction.

**Flask (full-featured web app):**

```bash
python flask_app/app.py
```

Open `http://localhost:5000` in your browser.

---

## Retraining Models (Optional)

See [`train/README.md`](train/README.md) for full details.

```bash
# Train per-plant disease models
python -m train.train_plant --plant potato --out_dir potato_trained_models/1
python -m train.train_plant --plant pepper  --out_dir pepper_trained_models/1
python -m train.train_plant --plant tomato  --out_dir tomato_trained_models/1

# Train plant identification model
python -m train.train_plant_id --out_dir models
```

---

## Project Structure

```
├── app.py                      # Streamlit demo
├── predict_pipeline.py         # Shared preprocessing & TTA helpers
├── cure_database.json          # Disease-specific cure timelines & treatments
├── requirements.txt
├── models/                     # Plant identification model
├── potato_trained_models/      # Potato disease model
├── tomato_trained_models/      # Tomato disease model
├── pepper_trained_models/      # Pepper disease model
├── flask_app/                  # Full web application
│   ├── app.py
│   ├── templates/
│   └── static/
└── train/                      # Training & calibration scripts
```

---

## Author

**vinuthna021** — [GitHub](https://github.com/vinuthna021)
