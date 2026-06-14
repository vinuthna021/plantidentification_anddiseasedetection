from __future__ import annotations


POTATO_CLASSES = [
    "Potato___Early_blight",
    "Potato___Late_blight",
    "Potato___healthy",
]

PEPPER_CLASSES = [
    "Pepper,_bell___healthy",
    "Pepper,_bell___Bacterial_spot",
]

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


def classes_for_plant(plant: str) -> list[str]:
    key = plant.strip().lower()
    if key == "potato":
        return POTATO_CLASSES
    if key == "pepper":
        return PEPPER_CLASSES
    if key == "tomato":
        return TOMATO_CLASSES
    raise ValueError(f"Unknown plant: {plant!r}. Expected potato/pepper/tomato.")

