import streamlit as st
import numpy as np
from io import BytesIO
from PIL import Image
import tensorflow as tf
import keras
import base64
import cv2
import json
from pathlib import Path

def _load_model(model_dir: str) -> tf.keras.Model:
    """
    Prefer loading the Keras v3 `.keras` artifact. If it's not present yet
    (e.g. training still running), fall back to loading a TF SavedModel as an
    inference-only layer (Keras 3 requirement).
    """
    p = Path(model_dir)
    keras_path = p / "model.keras"
    if keras_path.exists():
        return tf.keras.models.load_model(str(keras_path))

    # Keras 3 does NOT support loading SavedModel via `load_model()`.
    # We can still run inference by wrapping it in a `TFSMLayer`.
    saved_model_dir = p / "saved_model"
    sm_path = saved_model_dir if saved_model_dir.exists() else p

    layer = keras.layers.TFSMLayer(str(sm_path), call_endpoint="serving_default")
    inputs = tf.keras.Input(shape=(224, 224, 3), name="image")
    outputs = layer(inputs)
    return tf.keras.Model(inputs=inputs, outputs=outputs, name=f"{p.name}_tfsm")


POTATO_MODEL = _load_model("./potato_trained_models/1")
TOMATO_MODEL = _load_model("./tomato_trained_models/1")
PEPPER_MODEL = _load_model("./pepper_trained_models/1")

POTATO_CLASSES = ["Potato___Early_blight", "Potato___Late_blight", "Potato___healthy"]
TOMATO_CLASSES = [
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
]
PEPPER_CLASSES = ["Pepper,_bell___healthy", "Pepper,_bell___Bacterial_spot"]


def _load_temperature(model_dir: str) -> float:
    p = Path(model_dir) / "calibration.json"
    if not p.exists():
        return 1.0
    data = json.loads(p.read_text())
    try:
        return float(data.get("temperature", 1.0))
    except Exception:
        return 1.0


POTATO_T = _load_temperature("./potato_trained_models/1")
TOMATO_T = _load_temperature("./tomato_trained_models/1")
PEPPER_T = _load_temperature("./pepper_trained_models/1")
st.set_page_config(
    layout="wide",
    page_title='plant disease detection',
)
st.title("Plant Disease Detection")
st.write("This application is detecting disease in three plants photato, tomato and pepper")
options = ["Select One Plant","Tomato", "Potato", "Pepper"]

# Create a selectbox for the user to choose one option
selected_option = st.selectbox("Select Plant:", options)

# st.write("You selected:", selected_option)

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])

def read_file_as_image(data)->np.array:
    image = np.array(data)
    image = cv2.resize(image, (224, 224))
    return image

def _apply_temperature(probs: np.ndarray, t: float) -> np.ndarray:
    t = max(float(t), 1e-6)
    logits = np.log(np.clip(probs, 1e-8, 1.0))
    scaled = logits / t
    exp = np.exp(scaled - np.max(scaled, axis=-1, keepdims=True))
    return exp / np.sum(exp, axis=-1, keepdims=True)


def _predict(model, classes: list[str], t: float) -> None:
    if uploaded_file is None:
        return
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", width=250)
    image_arr = read_file_as_image(image)
    image_batch = np.expand_dims(image_arr, axis=0)
    raw = model.predict(image_batch, verbose=0)
    probs = _apply_temperature(raw, t=t)[0]
    idx = int(np.argmax(probs))
    confidence = float(np.max(probs))
    st.write("Predicted Class : ", classes[idx], " Confidence Level : ", confidence)


if selected_option == "Potato":
    _predict(POTATO_MODEL, POTATO_CLASSES, POTATO_T)
elif selected_option == "Tomato":
    _predict(TOMATO_MODEL, TOMATO_CLASSES, TOMATO_T)
elif selected_option == "Pepper":
    _predict(PEPPER_MODEL, PEPPER_CLASSES, PEPPER_T)
else:
    st.write("Select a plant and upload an image.")
