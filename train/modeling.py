from __future__ import annotations

import tensorflow as tf


def build_classifier(
    *,
    num_classes: int,
    image_size: tuple[int, int] = (224, 224),
    base_trainable: bool = False,
) -> tf.keras.Model:
    """
    Transfer learning baseline that tends to reach high accuracy quickly on PlantVillage.
    """
    inputs = tf.keras.Input(shape=(image_size[0], image_size[1], 3), name="image")

    # Inputs from TFDS are already float32 in [0, 1]. (For directory-loader
    # paths, image_dataset_from_directory yields uint8; but we rescale there.)
    x = tf.keras.layers.RandomFlip("horizontal")(inputs)
    x = tf.keras.layers.RandomRotation(0.05)(x)
    x = tf.keras.layers.RandomZoom(0.1)(x)

    base = tf.keras.applications.MobileNetV2(
        include_top=False,
        weights="imagenet",
        input_shape=(image_size[0], image_size[1], 3),
    )
    base.trainable = base_trainable

    # MobileNetV2 preprocess expects pixels in [0, 255]
    x = tf.keras.applications.mobilenet_v2.preprocess_input(x * 255.0)
    x = base(x, training=False)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dropout(0.2)(x)
    outputs = tf.keras.layers.Dense(num_classes, activation="softmax", name="probs")(x)
    model = tf.keras.Model(inputs=inputs, outputs=outputs)

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(),
        metrics=[tf.keras.metrics.SparseCategoricalAccuracy(name="acc")],
    )
    return model


def set_backbone_trainable(model: tf.keras.Model, *, trainable: bool, unfreeze_last_n: int = 40) -> None:
    """Unfreeze last N backbone layers for fine-tuning."""
    backbone = next((l for l in model.layers if isinstance(l, tf.keras.Model)), None)
    if backbone is None:
        # Fallback: try by name
        backbone = model.get_layer(index=4)  # best-effort

    backbone.trainable = bool(trainable)
    if trainable:
        layers = backbone.layers
        for l in layers[:-unfreeze_last_n]:
            l.trainable = False

