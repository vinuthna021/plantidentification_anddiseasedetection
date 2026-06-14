from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds

from train.labels import PEPPER_CLASSES, POTATO_CLASSES, TOMATO_CLASSES
from train.modeling import build_classifier, set_backbone_trainable


PLANT_CLASSES = ["Potato", "Tomato", "Pepper"]


def _load_plant_splits_tfds(
    *,
    tfds_data_dir: str | Path,
    image_size: tuple[int, int] = (224, 224),
    batch_size: int = 32,
    seed: int = 1337,
) -> tuple[tf.data.Dataset, tf.data.Dataset, tf.data.Dataset]:
    """Build (train/val/test) datasets labeled by plant only (3-way)."""
    tfds_data_dir = str(tfds_data_dir)
    builder = tfds.builder("plant_village", data_dir=tfds_data_dir)
    label_names = builder.info.features["label"].names
    name_to_id = {n: i for i, n in enumerate(label_names)}

    def ids_for(names: list[str]) -> list[int]:
        missing = [c for c in names if c not in name_to_id]
        if missing:
            raise ValueError(f"Unknown TFDS label(s): {missing}")
        return [name_to_id[c] for c in names]

    potato_ids = ids_for(POTATO_CLASSES)
    tomato_ids = ids_for(TOMATO_CLASSES)
    pepper_ids = ids_for(PEPPER_CLASSES)
    allowed_ids = tf.constant(potato_ids + tomato_ids + pepper_ids, dtype=tf.int64)

    plant_table = tf.lookup.StaticHashTable(
        tf.lookup.KeyValueTensorInitializer(
            keys=tf.constant(potato_ids + tomato_ids + pepper_ids, dtype=tf.int64),
            values=tf.constant(
                [0] * len(potato_ids) + [1] * len(tomato_ids) + [2] * len(pepper_ids),
                dtype=tf.int64,
            ),
        ),
        default_value=-1,
    )

    def is_allowed(_, y):
        y = tf.cast(y, tf.int64)
        return tf.reduce_any(tf.equal(allowed_ids, y))

    def preprocess(x, y):
        x = tf.image.convert_image_dtype(x, tf.float32)  # [0, 1]
        x = tf.image.resize(x, image_size, method="bilinear")
        y = tf.cast(y, tf.int64)
        plant_y = plant_table.lookup(y)
        return x, tf.cast(plant_y, tf.int32)

    ds = tfds.load(
        "plant_village",
        split="train",
        as_supervised=True,
        data_dir=tfds_data_dir,
        shuffle_files=True,
    )
    ds = ds.filter(is_allowed).map(preprocess, num_parallel_calls=tf.data.AUTOTUNE)
    ds = ds.shuffle(10_000, seed=seed, reshuffle_each_iteration=False)

    n_total = int(ds.reduce(tf.constant(0, dtype=tf.int64), lambda x, _: x + 1).numpy())
    n_train = int(0.8 * n_total)
    n_val = int(0.1 * n_total)
    n_test = n_total - n_train - n_val

    def finalize(split: tf.data.Dataset, *, n_examples: int) -> tf.data.Dataset:
        split = split.batch(batch_size, drop_remainder=False)
        n_batches = int((n_examples + batch_size - 1) // batch_size)
        split = split.apply(tf.data.experimental.assert_cardinality(n_batches))
        return split.cache().prefetch(tf.data.AUTOTUNE)

    train = finalize(ds.take(n_train), n_examples=n_train)
    val = finalize(ds.skip(n_train).take(n_val), n_examples=n_val)
    test = finalize(ds.skip(n_train + n_val).take(n_test), n_examples=n_test)
    return train, val, test


def _eval_acc(model: tf.keras.Model, ds: tf.data.Dataset) -> float:
    y_true = []
    y_pred = []
    for x, y in ds:
        p = model.predict(x, verbose=0)
        y_true.append(y.numpy())
        y_pred.append(np.argmax(p, axis=-1))
    yt = np.concatenate(y_true)
    yp = np.concatenate(y_pred)
    return float(np.mean(yt == yp))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tfds_data_dir", default=".tfds")
    ap.add_argument("--out_dir", default="models")
    ap.add_argument("--epochs", type=int, default=6)
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--image_size", type=int, default=224)
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    train, val, test = _load_plant_splits_tfds(
        tfds_data_dir=args.tfds_data_dir,
        image_size=(args.image_size, args.image_size),
        batch_size=args.batch_size,
    )

    model = build_classifier(
        num_classes=3,
        image_size=(args.image_size, args.image_size),
        base_trainable=False,
    )

    callbacks: list[tf.keras.callbacks.Callback] = [
        tf.keras.callbacks.EarlyStopping(monitor="val_acc", patience=2, restore_best_weights=True),
        tf.keras.callbacks.ReduceLROnPlateau(monitor="val_acc", factor=0.5, patience=1),
    ]

    # Phase 1
    model.fit(train, validation_data=val, epochs=max(1, args.epochs // 2), callbacks=callbacks, verbose=1)

    # Phase 2 fine-tune
    set_backbone_trainable(model, trainable=True, unfreeze_last_n=40)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(),
        metrics=[tf.keras.metrics.SparseCategoricalAccuracy(name="acc")],
    )
    model.fit(
        train,
        validation_data=val,
        epochs=args.epochs,
        initial_epoch=max(1, args.epochs // 2),
        callbacks=callbacks,
        verbose=1,
    )

    test_acc = _eval_acc(model, test)

    # Save
    (out_dir / "plant_classes.json").write_text(json.dumps(PLANT_CLASSES, indent=2))
    model.save(out_dir / "plant_model.keras")

    print("=== Plant ID Results ===")
    print(json.dumps({"test_top1_acc": test_acc, "classes": PLANT_CLASSES}, indent=2))


if __name__ == "__main__":
    main()

