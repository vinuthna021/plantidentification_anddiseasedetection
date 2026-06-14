import base64
import io
import json
import logging
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import tensorflow as tf

# Keep CPU/memory usage predictable on small hosts (e.g. Render free tier).
tf.config.threading.set_intra_op_parallelism_threads(1)
tf.config.threading.set_inter_op_parallelism_threads(1)

from PIL import Image, ImageEnhance, ImageFilter, ImageOps
from flask import Flask, jsonify, render_template, request
from werkzeug.utils import secure_filename

BASE_DIR = Path(__file__).resolve().parent.parent
FLASK_DIR = Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR / "models"
CURE_DB_PATH = BASE_DIR / "cure_database.json"
FEEDBACK_LOG = FLASK_DIR / "feedback_log.jsonl"
ALLOWED_EXTENSIONS = {".jpg", ".jpeg", ".png"}
MAX_FILE_BYTES = 8 * 1024 * 1024

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
logger = logging.getLogger("plant_disease_app")

app = Flask(
    __name__,
    template_folder=str(FLASK_DIR / "templates"),
    static_folder=str(FLASK_DIR / "static"),
)
app.config["MAX_CONTENT_LENGTH"] = MAX_FILE_BYTES

FALLBACK_CLASS_MAP: dict[str, list[str]] = {
    "Potato": ["Potato___Early_blight", "Potato___Late_blight", "Potato___healthy"],
    "Tomato": [
        "Tomato___healthy",
        "Tomato___Spider_mites Two-spotted_spider_mite",
        "Tomato___Target_Spot",
        "Tomato___Septoria_leaf_spot",
        "Tomato___Tomato_mosaic_virus",
        "Tomato___Leaf_Mold",
        "Tomato___Bacterial_spot",
        "Tomato___Late_blight",
        "Tomato___Early_blight",
        "Tomato___Tomato_Yellow_Leaf_Curl_Virus",
    ],
    "Pepper": ["Pepper,_bell___healthy", "Pepper,_bell___Bacterial_spot"],
}


def _load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _normalize_classes(raw: Any) -> list[dict[str, Any]]:
    classes: list[dict[str, Any]] = []
    if isinstance(raw, list):
        for idx, item in enumerate(raw):
            if isinstance(item, str):
                classes.append({"index": idx, "label": item})
            elif isinstance(item, dict):
                classes.append(
                    {
                        "index": int(item.get("index", idx)),
                        "label": item.get("label", item.get("name", f"class_{idx}")),
                        "name": item.get("name", item.get("label", f"class_{idx}")),
                        "plant": item.get("plant", ""),
                        "healthy": bool(item.get("healthy", False)),
                    }
                )
    elif isinstance(raw, dict):
        for key, value in raw.items():
            idx = int(key) if str(key).isdigit() else len(classes)
            if isinstance(value, str):
                classes.append({"index": idx, "label": value})
            elif isinstance(value, dict):
                classes.append(
                    {
                        "index": int(value.get("index", idx)),
                        "label": value.get("label", value.get("name", f"class_{idx}")),
                        "name": value.get("name", value.get("label", f"class_{idx}")),
                        "plant": value.get("plant", ""),
                        "healthy": bool(value.get("healthy", False)),
                    }
                )
    classes.sort(key=lambda x: x["index"])
    return classes


def _parse_label(label: str) -> tuple[str, bool]:
    l = label.lower()
    healthy = "healthy" in l
    cleaned = label.replace("__", " ").replace("_", " ").strip()
    plant = cleaned.split(" ")[0] if cleaned else "Unknown"
    return plant.title(), healthy


def _find_last_conv_layer_name(model: tf.keras.Model) -> str:
    for layer in reversed(model.layers):
        if isinstance(layer, tf.keras.layers.Conv2D):
            return layer.name
    raise RuntimeError("No Conv2D layer found for Grad-CAM generation")


def _model_input_size(model: tf.keras.Model) -> tuple[int, int]:
    shape = model.input_shape
    if isinstance(shape, list):
        shape = shape[0]
    h = int(shape[1] or 224)
    w = int(shape[2] or 224)
    return h, w


def _load_saved_model_predictor(path: Path) -> dict[str, Any]:
    model_obj = tf.saved_model.load(str(path))
    signature = model_obj.signatures.get("serving_default")
    if signature is None:
        raise RuntimeError(f"serving_default signature not found in {path}")
    input_spec = next(iter(signature.structured_input_signature[1].values()))
    shape = input_spec.shape
    h = int(shape[1]) if len(shape) > 2 and shape[1] else 256
    w = int(shape[2]) if len(shape) > 2 and shape[2] else 256
    return {"signature": signature, "input_size": (h, w)}

def _load_temperature(path: Path) -> float:
    """
    Loads temperature scaling value written by `train/train_plant.py`.
    Defaults to 1.0 (no calibration).
    """
    p = path / "calibration.json"
    if not p.exists():
        return 1.0
    try:
        data = _load_json(p)
        t = float(data.get("temperature", 1.0))
        return max(t, 1e-6)
    except Exception:
        return 1.0


def _apply_temperature(probs: np.ndarray, t: float) -> np.ndarray:
    t = max(float(t), 1e-6)
    logits = np.log(np.clip(probs, 1e-8, 1.0))
    scaled = logits / t
    exp = np.exp(scaled - np.max(scaled, axis=-1, keepdims=True))
    return exp / np.sum(exp, axis=-1, keepdims=True)


def _load_fallback_keras_model(model_dir: Path) -> tf.keras.Model:
    """
    Keras 3+ requires `.keras` (or `.h5`) for `load_model()`.
    Our training pipeline writes `model.keras` under each plant dir.
    """
    keras_path = model_dir / "model.keras"
    if not keras_path.exists():
        raise FileNotFoundError(f"Missing {keras_path}")
    return tf.keras.models.load_model(str(keras_path))


def _predict_saved_model(predictor: dict[str, Any], image_path: Path) -> np.ndarray:
    h, w = predictor["input_size"]
    image = Image.open(image_path).convert("RGB").resize((w, h))
    arr = np.asarray(image).astype(np.float32)
    if np.max(arr) <= 1.0:
        arr = arr * 255.0
    input_tensor = tf.convert_to_tensor(np.expand_dims(arr, axis=0), dtype=tf.float32)
    outputs = predictor["signature"](input_tensor)
    output_tensor = next(iter(outputs.values()))
    probs = tf.nn.softmax(output_tensor, axis=-1).numpy()[0]
    return probs


def _load_app_assets() -> dict[str, Any]:
    plant_keras = MODEL_DIR / "plant_model.keras"
    disease_keras = MODEL_DIR / "disease_model.keras"
    plant_json = MODEL_DIR / "plant_classes.json"
    disease_json = MODEL_DIR / "disease_classes.json"

    if all(p.exists() for p in [plant_keras, disease_keras, plant_json, disease_json]):
        plant_model = tf.keras.models.load_model(plant_keras)
        disease_model = tf.keras.models.load_model(disease_keras)
        plant_classes = _normalize_classes(_load_json(plant_json))
        disease_classes = _normalize_classes(_load_json(disease_json))
        last_conv_layer = _find_last_conv_layer_name(disease_model)
        logger.info("Loaded primary models from /models directory")
        return {
            "mode": "two_model",
            "plant_model": plant_model,
            "disease_model": disease_model,
            "plant_classes": plant_classes,
            "disease_classes": disease_classes,
            "cure_db": _load_json(CURE_DB_PATH),
            "last_conv_layer": last_conv_layer,
        }

    fallback_specs = {
        "Potato": BASE_DIR / "potato_trained_models" / "1",
        "Tomato": BASE_DIR / "tomato_trained_models" / "1",
        "Pepper": BASE_DIR / "pepper_trained_models" / "1",
    }
    missing = [name for name, path in fallback_specs.items() if not path.exists()]
    if missing:
        raise FileNotFoundError(
            "No valid model source found. Missing model paths: "
            + ", ".join(str(fallback_specs[name]) for name in missing)
        )

    # Fallback mode: use our per-plant Keras models + temperature scaling.
    plant_models = {
        name: {
            "model": _load_fallback_keras_model(path),
            "temperature": _load_temperature(path),
        }
        for name, path in fallback_specs.items()
    }

    plant_selector_model = None
    plant_selector_classes = ["Potato", "Tomato", "Pepper"]
    if plant_keras.exists():
        try:
            plant_selector_model = tf.keras.models.load_model(plant_keras)
            if plant_json.exists():
                parsed = _normalize_classes(_load_json(plant_json))
                labels = [str(c.get("label", "")).strip() for c in parsed]
                labels = [l for l in labels if l]
                if labels:
                    plant_selector_classes = labels
            logger.info("Loaded plant selector model from /models/plant_model.keras")
        except Exception:
            logger.exception("Failed to load plant selector model from /models")

    disease_classes: list[dict[str, Any]] = []
    disease_index_map: dict[str, list[int]] = {}
    for plant_name, labels in FALLBACK_CLASS_MAP.items():
        disease_index_map[plant_name] = []
        for label in labels:
            idx = len(disease_classes)
            disease_classes.append(
                {
                    "index": idx,
                    "label": label,
                    "name": label.replace("_", " "),
                    "plant": plant_name,
                    "healthy": "healthy" in label.lower(),
                }
            )
            disease_index_map[plant_name].append(idx)

    plant_classes = [
        {"index": 0, "label": "Potato", "name": "Potato", "healthy": False},
        {"index": 1, "label": "Tomato", "name": "Tomato", "healthy": False},
        {"index": 2, "label": "Pepper", "name": "Pepper", "healthy": False},
    ]
    cure_db = _load_json(CURE_DB_PATH)
    logger.info("Loaded fallback per-plant models from trained_models directories")
    return {
        "mode": "per_plant",
        "plant_models": plant_models,
        "plant_classes": plant_classes,
        "disease_classes": disease_classes,
        "disease_index_map": disease_index_map,
        "cure_db": cure_db,
        "plant_selector_model": plant_selector_model,
        "plant_selector_classes": plant_selector_classes,
    }


ASSETS = _load_app_assets()


def _preprocess_image(image_path: Path, model: Any = None) -> np.ndarray:
    # Unified high-quality preprocessing for stable predictions
    image = Image.open(image_path).convert("RGB")
    image = image.resize((224, 224))
    image = image.filter(ImageFilter.SHARPEN)
    arr = np.asarray(image).astype(np.float32) / 255.0
    return np.expand_dims(arr, axis=0)


def _predict_with_tta_keras(model: tf.keras.Model, img: np.ndarray) -> np.ndarray:
    preds: list[np.ndarray] = []
    preds.append(model.predict(img, verbose=0))
    preds.append(model.predict(np.flip(img, axis=1), verbose=0))
    preds.append(model.predict(np.flip(img, axis=2), verbose=0))
    return np.mean(preds, axis=0)


def _predict_with_tta_savedmodel(predictor: dict[str, Any], image_path: Path) -> np.ndarray:
    # Backward-compat shim: some older deployments used TF SavedModel.
    # New deployments should use `_predict_with_tta_keras` via the Keras model path.
    img = _preprocess_image(image_path)
    preds: list[np.ndarray] = []
    for aug_img in (img, np.flip(img, axis=1), np.flip(img, axis=2)):
        input_tensor = tf.convert_to_tensor(aug_img, dtype=tf.float32)
        outputs = predictor["signature"](input_tensor)
        output_tensor = next(iter(outputs.values()))
        probs = tf.nn.softmax(output_tensor, axis=-1).numpy()
        preds.append(probs)
    return np.mean(preds, axis=0)[0]


def _predict_two_stage(image_path: Path) -> dict[str, Any]:
    if ASSETS["mode"] == "two_model":
        plant_arr = _preprocess_image(image_path, ASSETS["plant_model"])
        disease_arr = _preprocess_image(image_path, ASSETS["disease_model"])
        plant_probs = _predict_with_tta_keras(ASSETS["plant_model"], plant_arr)[0]
        disease_probs = _predict_with_tta_keras(ASSETS["disease_model"], disease_arr)[0]
        plant_idx = int(np.argmax(plant_probs))

        # Plant-disease consistency filtering
        plant_name = str(ASSETS["plant_classes"][plant_idx].get("label", "")).lower()
        allowed_idx: list[int] = []
        for idx, meta in enumerate(ASSETS["disease_classes"]):
            label = str(meta.get("label", "")).lower()
            if "potato" in plant_name and "potato" in label:
                allowed_idx.append(idx)
            elif "tomato" in plant_name and "tomato" in label:
                allowed_idx.append(idx)
            elif ("pepper" in plant_name or "bell" in plant_name) and ("pepper" in label or "bell" in label):
                allowed_idx.append(idx)
        if allowed_idx:
            mask = np.zeros_like(disease_probs)
            mask[allowed_idx] = disease_probs[allowed_idx]
            disease_probs = mask
        disease_idx = int(np.argmax(disease_probs))
        return {
            "plant_probs": plant_probs,
            "disease_probs": disease_probs,
            "plant_idx": plant_idx,
            "disease_idx": disease_idx,
            "disease_local_idx": disease_idx,
            "plant_model_for_gradcam": ASSETS["disease_model"],
            "last_conv_layer": ASSETS["last_conv_layer"],
        }

    # Fallback mode: use three trained disease models.
    # Stage 1 chooses plant using the dedicated plant model when available.
    selector_model = ASSETS.get("plant_selector_model")
    selector_classes = ASSETS.get("plant_selector_classes", ["Potato", "Tomato", "Pepper"])
    if selector_model is not None:
        selector_probs = _predict_with_tta_keras(selector_model, _preprocess_image(image_path))[0]
        selector_idx = int(np.argmax(selector_probs))
        raw_label = str(selector_classes[selector_idx]) if selector_idx < len(selector_classes) else ""
        label = raw_label.lower()
        if "potato" in label:
            best_plant = "Potato"
        elif "tomato" in label:
            best_plant = "Tomato"
        elif "pepper" in label or "bell" in label:
            best_plant = "Pepper"
        else:
            best_plant = "Tomato"
        selected_info = ASSETS["plant_models"][best_plant]
        selected_probs_raw = _predict_with_tta_keras(selected_info["model"], _preprocess_image(image_path))[0]
        selected_probs = _apply_temperature(selected_probs_raw, t=float(selected_info.get("temperature", 1.0)))
        local_idx = int(np.argmax(selected_probs))
        global_idx = ASSETS["disease_index_map"][best_plant][local_idx]
        plant_idx = {"Potato": 0, "Tomato": 1, "Pepper": 2}[best_plant]

        all_probs = np.zeros(len(ASSETS["disease_classes"]), dtype=np.float32)
        for local_i, global_i in enumerate(ASSETS["disease_index_map"][best_plant]):
            all_probs[global_i] = float(selected_probs[local_i])

        plant_probs = np.zeros(3, dtype=np.float32)
        if selector_probs.shape[0] >= 3:
            mapped = {"Potato": 0.0, "Tomato": 0.0, "Pepper": 0.0}
            for i, p in enumerate(selector_probs):
                lbl = str(selector_classes[i]).lower() if i < len(selector_classes) else ""
                if "potato" in lbl:
                    mapped["Potato"] = max(mapped["Potato"], float(p))
                elif "tomato" in lbl:
                    mapped["Tomato"] = max(mapped["Tomato"], float(p))
                elif "pepper" in lbl or "bell" in lbl:
                    mapped["Pepper"] = max(mapped["Pepper"], float(p))
            plant_probs = np.array([mapped["Potato"], mapped["Tomato"], mapped["Pepper"]], dtype=np.float32)
        else:
            plant_probs[plant_idx] = 1.0

        return {
            "plant_probs": plant_probs,
            "disease_probs": all_probs,
            "plant_idx": plant_idx,
            "disease_idx": global_idx,
            "disease_local_idx": local_idx,
            "plant_model_for_gradcam": None,
            "last_conv_layer": None,
        }

    # If dedicated plant model isn't available, fall back to confidence heuristic.
    # Stage 1 chooses plant by strongest model confidence.
    plant_scores: dict[str, float] = {}
    plant_outputs: dict[str, np.ndarray] = {}
    for plant_name, info in ASSETS["plant_models"].items():
        model = info["model"]
        t = float(info.get("temperature", 1.0))
        probs_raw = _predict_with_tta_keras(model, _preprocess_image(image_path))[0]
        probs = _apply_temperature(probs_raw, t=t)
        plant_outputs[plant_name] = probs
        plant_scores[plant_name] = float(np.max(probs))

    best_plant = max(plant_scores, key=plant_scores.get)
    selected_probs = plant_outputs[best_plant]
    local_idx = int(np.argmax(selected_probs))
    global_idx = ASSETS["disease_index_map"][best_plant][local_idx]
    plant_idx = {"Potato": 0, "Tomato": 1, "Pepper": 2}[best_plant]

    # Expand per-plant disease probabilities into a global vector for top-5 rendering.
    all_probs = np.zeros(len(ASSETS["disease_classes"]), dtype=np.float32)
    for local_i, global_i in enumerate(ASSETS["disease_index_map"][best_plant]):
        all_probs[global_i] = float(selected_probs[local_i])

    return {
        "plant_probs": np.array([plant_scores["Potato"], plant_scores["Tomato"], plant_scores["Pepper"]], dtype=np.float32),
        "disease_probs": all_probs,
        "plant_idx": plant_idx,
        "disease_idx": global_idx,
        "disease_local_idx": local_idx,
        "plant_model_for_gradcam": None,
        "last_conv_layer": None,
    }


def _gradcam_base64(image_path: Path, model: tf.keras.Model, class_idx: int, conv_name: str) -> str:
    h, w = _model_input_size(model)
    image = Image.open(image_path).convert("RGB").resize((w, h))
    input_arr = np.asarray(image).astype(np.float32) / 255.0
    input_tensor = tf.convert_to_tensor(np.expand_dims(input_arr, axis=0))

    grad_model = tf.keras.models.Model(
        inputs=model.inputs,
        outputs=[model.get_layer(conv_name).output, model.output],
    )

    with tf.GradientTape() as tape:
        conv_out, predictions = grad_model(input_tensor)
        class_channel = predictions[:, class_idx]

    grads = tape.gradient(class_channel, conv_out)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    conv_out = conv_out[0]
    heatmap = conv_out @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)
    heatmap = tf.maximum(heatmap, 0) / (tf.reduce_max(heatmap) + 1e-8)
    heatmap_np = heatmap.numpy()

    heat_img = Image.fromarray(np.uint8(255 * heatmap_np)).resize((w, h))
    heat_arr = np.asarray(heat_img).astype(np.float32)
    rgb_arr = np.asarray(image).astype(np.float32)

    overlay = np.zeros_like(rgb_arr)
    overlay[..., 0] = np.clip(heat_arr * 1.2, 0, 255)
    overlay[..., 1] = np.clip(heat_arr * 0.3, 0, 255)
    overlay[..., 2] = np.clip(heat_arr * 0.1, 0, 255)
    blended = np.uint8((0.65 * rgb_arr) + (0.35 * overlay))

    out = Image.fromarray(blended)
    buff = io.BytesIO()
    out.save(buff, format="JPEG", quality=90)
    return base64.b64encode(buff.getvalue()).decode("utf-8")


def _fallback_heatmap_base64(image_path: Path) -> str:
    image = Image.open(image_path).convert("RGB").resize((256, 256))
    arr = np.asarray(image).astype(np.float32)
    gray = np.mean(arr, axis=-1)
    gy, gx = np.gradient(gray)
    mag = np.sqrt(gx * gx + gy * gy)
    mag = mag / (np.max(mag) + 1e-8)
    heat = np.zeros_like(arr)
    heat[..., 0] = mag * 255
    blended = np.uint8(0.65 * arr + 0.35 * heat)
    buff = io.BytesIO()
    Image.fromarray(blended).save(buff, format="JPEG", quality=90)
    return base64.b64encode(buff.getvalue()).decode("utf-8")


def _severity_and_color(is_healthy: bool, confidence: float) -> tuple[str, str]:
    if is_healthy:
        return "Healthy", "#16a34a"
    if confidence >= 0.85:
        return "Severe", "#dc2626"
    if confidence >= 0.65:
        return "Moderate", "#d97706"
    return "Mild", "#eab308"


def _allowed_file(filename: str) -> bool:
    return Path(filename).suffix.lower() in ALLOWED_EXTENSIONS


def _auto_enhance(image: Image.Image) -> tuple[Image.Image, list[str]]:
    enhancements: list[str] = []
    img = image.convert("RGB")

    img = ImageOps.autocontrast(img, cutoff=2)
    enhancements.append("Autocontrast applied")

    grayscale = np.asarray(img.convert("L"), dtype=np.float32)
    if float(np.std(grayscale)) > 35:
        img = img.filter(ImageFilter.GaussianBlur(radius=0.8))
        enhancements.append("Gaussian blur for noise smoothing")

    mean_lum = float(np.mean(grayscale))
    if mean_lum < 95:
        img = ImageEnhance.Brightness(img).enhance(1.25)
        enhancements.append("Brightness increased")
    elif mean_lum > 185:
        img = ImageEnhance.Brightness(img).enhance(0.85)
        enhancements.append("Brightness reduced")

    img = ImageEnhance.Contrast(img).enhance(1.1)
    enhancements.append("Contrast optimized")

    max_side = max(img.size)
    if max_side > 512:
        ratio = 512 / max_side
        new_size = (int(img.size[0] * ratio), int(img.size[1] * ratio))
        img = img.resize(new_size)
        enhancements.append("Resized to max 512px")

    return img, enhancements


def _img_to_base64(img: Image.Image) -> str:
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=92)
    return base64.b64encode(buf.getvalue()).decode("utf-8")


def _plant_specific_guidance(plant_name: str, disease_label: str, is_healthy: bool) -> dict[str, list[str]]:
    plant_key = plant_name.lower()
    disease_key = disease_label.lower()

    if is_healthy:
        return {
            "prevention": [
                f"Keep {plant_name} canopy dry and maintain weekly scouting.",
                "Use clean tools and remove old/dead foliage promptly.",
                "Maintain balanced nutrients and avoid overwatering.",
            ],
            "organic_alternatives": [
                "Apply neem-based preventive spray at low dose every 7-10 days.",
                "Use Trichoderma/Bacillus bio-inputs as preventive foliar support.",
                "Use compost tea as a mild preventive boost.",
            ],
            "chemical_options": [
                "No curative spray needed; use only preventive labeled products if risk is high.",
                "Use low-dose protectant spray before high humidity/rain periods.",
                "Follow pre-harvest interval and rotation principles.",
            ],
        }

    if "potato" in plant_key:
        return {
            "prevention": [
                "Avoid prolonged leaf wetness; irrigate early morning only.",
                "Remove infected lower leaves and rogue severe plants.",
                "Maintain wide row airflow and field sanitation.",
            ],
            "organic_alternatives": [
                "Neem oil plus potassium bicarbonate rotation for mild pressure.",
                "Use Bacillus subtilis based bio-fungicide as preventive support.",
                "Apply seaweed extract to reduce stress recovery time.",
            ],
            "chemical_options": [
                "Use potato-labeled protectant fungicide and rotate active ingredients.",
                "For blight-prone weather, follow systemic + contact sequence.",
                "Repeat as per label interval and rainfall conditions.",
            ],
        }

    if "tomato" in plant_key:
        vector_hint = "Control whiteflies and aphids aggressively." if "virus" in disease_key else "Monitor lower canopy and splash spread."
        return {
            "prevention": [
                "Prune lower foliage and improve airflow around tomato canopy.",
                "Avoid overhead irrigation and sanitize support tools.",
                vector_hint,
            ],
            "organic_alternatives": [
                "Neem/karanja oil spray in evening with good coverage.",
                "Apply copper soap or bio-fungicide for early symptoms.",
                "Use compost extract and microbial foliar boosters weekly.",
            ],
            "chemical_options": [
                "Use tomato-labeled fungicide/bactericide based on diagnosis class.",
                "Rotate FRAC groups to avoid resistance build-up.",
                "Follow PHI/REI and do not exceed label dosage.",
            ],
        }

    # Pepper guidance fallback
    return {
        "prevention": [
            "Keep pepper canopy ventilated and avoid wet foliage.",
            "Remove diseased leaves and sanitize pruning tools.",
            "Use drip irrigation with consistent moisture management.",
        ],
        "organic_alternatives": [
            "Neem oil spray with soap-based spreader at label dose.",
            "Apply Bacillus-based bio-protectants every 7 days.",
            "Use humic/seaweed foliar tonic after stress events.",
        ],
        "chemical_options": [
            "Apply pepper-approved copper/fungicide chemistry as required.",
            "Rotate active ingredients and avoid repetitive single-mode sprays.",
            "Observe label safety and harvest interval strictly.",
        ],
    }


def _normalize_cure_plan(cure_plan: dict[str, Any], plant_name: str, disease_label: str, is_healthy: bool) -> dict[str, Any]:
    overview = str(cure_plan.get("overview", "Follow hygiene, balanced nutrition, and close monitoring."))
    timeline = cure_plan.get("timeline", [])
    if not isinstance(timeline, list) or not timeline:
        timeline = [
            {"day": "Day 1", "action": "Isolate affected leaves and clean nearby area."},
            {"day": "Day 2", "action": "Apply appropriate preventive treatment."},
            {"day": "Day 3", "action": "Improve airflow and avoid leaf wetness."},
            {"day": "Day 4", "action": "Inspect canopy and remove newly infected tissue."},
            {"day": "Day 5", "action": "Continue monitoring and optimize irrigation schedule."},
            {"day": "Day 6", "action": "Repeat treatment if required by label guidance."},
            {"day": "Day 7", "action": "Review recovery and continue weekly scouting."},
        ]

    prevention = cure_plan.get("prevention", [])
    if not prevention:
        prevention = _plant_specific_guidance(plant_name, disease_label, is_healthy)["prevention"]

    organic_alternatives = cure_plan.get("organic_alternatives", [])
    if not organic_alternatives:
        organic_alternatives = _plant_specific_guidance(plant_name, disease_label, is_healthy)["organic_alternatives"]

    chemical_options = cure_plan.get("chemical_options", [])
    if not chemical_options:
        chemical_options = _plant_specific_guidance(plant_name, disease_label, is_healthy)["chemical_options"]

    return {
        "overview": overview,
        "timeline": timeline[:7],
        "prevention": prevention,
        "organic_alternatives": organic_alternatives,
        "chemical_options": chemical_options,
    }


@app.route("/", methods=["GET"])
def index() -> str:
    return render_template("index.html")


@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok", "mode": ASSETS.get("mode", "unknown")})


@app.route("/history", methods=["GET"])
def history_page() -> str:
    return render_template("history.html")


@app.route("/predict", methods=["POST"])
def predict():
    if "image" not in request.files:
        return jsonify({"error": "No image uploaded"}), 400

    file = request.files["image"]
    if not file or file.filename == "":
        return jsonify({"error": "Empty file submitted"}), 400

    filename = secure_filename(file.filename)
    if not _allowed_file(filename):
        return jsonify({"error": "Only jpg, jpeg, png are supported"}), 400

    file_bytes = file.read()
    if len(file_bytes) > MAX_FILE_BYTES:
        return jsonify({"error": "File exceeds 8MB limit"}), 400
    file.stream.seek(0)

    tmp_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=Path(filename).suffix) as tmp:
            file.save(tmp.name)
            tmp_path = Path(tmp.name)

        prediction = _predict_two_stage(tmp_path)
        plant_probs = prediction["plant_probs"]
        disease_probs = prediction["disease_probs"]
        plant_idx = prediction["plant_idx"]
        disease_idx = prediction["disease_idx"]

        plant_conf = float(np.max(plant_probs))
        disease_conf = float(np.max(disease_probs))

        # Note: per-plant fallback models are already temperature-calibrated.

        plant_meta = ASSETS["plant_classes"][plant_idx]
        disease_meta = ASSETS["disease_classes"][disease_idx]
        disease_label = disease_meta.get("label", f"class_{disease_idx}")

        parsed_plant, inferred_healthy = _parse_label(disease_label)
        plant_name = str(plant_meta.get("label", plant_meta.get("name", parsed_plant)))
        is_healthy = bool(disease_meta.get("healthy", inferred_healthy))

        severity, severity_color = _severity_and_color(is_healthy, disease_conf)
        warning = (
            "Low confidence prediction. Please capture a clearer image for better results."
            if disease_conf < 0.60
            else ""
        )

        top5_idx = np.argsort(disease_probs)[::-1][:5]
        top5 = [
            {
                "index": int(i),
                "label": ASSETS["disease_classes"][int(i)].get("label", f"class_{int(i)}"),
                "confidence": round(float(disease_probs[int(i)]), 4),
            }
            for i in top5_idx
        ]

        if prediction["plant_model_for_gradcam"] is not None:
            heatmap_b64 = _gradcam_base64(
                image_path=tmp_path,
                model=prediction["plant_model_for_gradcam"],
                class_idx=prediction["disease_local_idx"],
                conv_name=prediction["last_conv_layer"],
            )
        else:
            heatmap_b64 = _fallback_heatmap_base64(tmp_path)

        cure_plan_raw = ASSETS["cure_db"].get(disease_label, ASSETS["cure_db"].get("default", {}))
        cure_plan = _normalize_cure_plan(cure_plan_raw, plant_name=plant_name, disease_label=disease_label, is_healthy=is_healthy)
        confidence_score = min(1.0, float((0.45 * plant_conf) + (0.55 * disease_conf)))

        response = {
            "plant": plant_name,
            "plant_confidence": round(plant_conf, 4),
            "disease": disease_label,
            "disease_label": disease_meta.get("name", disease_label),
            "disease_confidence": round(disease_conf, 4),
            "confidence_score": round(confidence_score, 4),
            "is_healthy": is_healthy,
            "severity": severity,
            "severity_color": severity_color,
            "top5_diseases": top5,
            "gradcam_base64": heatmap_b64,
            "cure_plan": cure_plan,
            "warning": warning,
        }
        logger.info("Prediction success: %s | %s", plant_name, disease_label)
        return jsonify(response)
    except Exception as exc:
        logger.exception("Prediction failed: %s", exc)
        return jsonify({"error": "Prediction failed. Please try again."}), 500
    finally:
        if tmp_path and tmp_path.exists():
            tmp_path.unlink(missing_ok=True)


@app.route("/enhance", methods=["POST"])
def enhance():
    if "image" not in request.files:
        return jsonify({"error": "No image uploaded"}), 400

    file = request.files["image"]
    if not file or file.filename == "":
        return jsonify({"error": "Empty file submitted"}), 400

    filename = secure_filename(file.filename)
    if not _allowed_file(filename):
        return jsonify({"error": "Only jpg, jpeg, png are supported"}), 400

    try:
        image = Image.open(file.stream)
        enhanced, steps = _auto_enhance(image)
        return jsonify({"enhanced_image": _img_to_base64(enhanced), "enhancements": steps})
    except Exception as exc:
        logger.exception("Enhancement failed: %s", exc)
        return jsonify({"error": "Enhancement failed"}), 500


@app.route("/feedback", methods=["POST"])
def feedback():
    data = request.get_json(silent=True) or {}
    record = {
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "rating": data.get("rating"),
        "comments": data.get("comments", "").strip(),
        "disease": data.get("disease", ""),
        "confidence": data.get("confidence", ""),
    }
    try:
        FEEDBACK_LOG.parent.mkdir(parents=True, exist_ok=True)
        with FEEDBACK_LOG.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
        logger.info("Feedback logged")
        return jsonify({"status": "ok"})
    except Exception as exc:
        logger.exception("Feedback save failed: %s", exc)
        return jsonify({"error": "Could not save feedback"}), 500


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
