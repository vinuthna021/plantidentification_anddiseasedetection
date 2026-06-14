from pathlib import Path

import numpy as np
from PIL import Image, ImageFilter


def preprocess(img_path: str | Path) -> np.ndarray:
    img = Image.open(img_path).convert("RGB")
    img = img.resize((224, 224))
    img = img.filter(ImageFilter.SHARPEN)
    img = np.array(img) / 255.0
    img = np.expand_dims(img, axis=0)
    return img


def predict_with_tta(model, img: np.ndarray) -> np.ndarray:
    preds = []
    preds.append(model.predict(img))
    preds.append(model.predict(np.flip(img, axis=1)))
    preds.append(model.predict(np.flip(img, axis=2)))
    return np.mean(preds, axis=0)
