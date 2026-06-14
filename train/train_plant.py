from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import tensorflow as tf

from train.calibration import TemperatureScalingResult, apply_temperature, temperature_scale_fit
from train.data import load_splits, load_splits_tfds
from train.labels import classes_for_plant
from train.modeling import build_classifier, set_backbone_trainable


def _evaluate_accuracy(model: tf.keras.Model, ds: tf.data.Dataset) -> float:
    metric = tf.keras.metrics.SparseCategoricalAccuracy()
    for x, y in ds:
        probs = model(x, training=False)
        metric.update_state(y, probs)
    return float(metric.result().numpy())


def _confidence_stats(model: tf.keras.Model, ds: tf.data.Dataset, temperature: float) -> dict:
    confs: list[float] = []
    correct_confs: list[float] = []
    total = 0
    correct = 0

    for x, y in ds:
        probs = model.predict(x, verbose=0)
        probs = apply_temperature(probs, temperature=temperature)
        pred = np.argmax(probs, axis=1)
        conf = np.max(probs, axis=1)
        y_np = y.numpy()

        total += len(y_np)
        correct_mask = pred == y_np
        correct += int(np.sum(correct_mask))

        confs.extend(conf.tolist())
        correct_confs.extend(conf[correct_mask].tolist())

    return {
        "n": int(total),
        "acc": float(correct / max(total, 1)),
        "mean_conf": float(np.mean(confs)) if confs else 0.0,
        "mean_conf_correct": float(np.mean(correct_confs)) if correct_confs else 0.0,
        "p10_conf": float(np.quantile(confs, 0.10)) if confs else 0.0,
        "p90_conf": float(np.quantile(confs, 0.90)) if confs else 0.0,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--plant", required=True, choices=["potato", "tomato", "pepper"])
    parser.add_argument(
        "--source",
        choices=["dir", "tfds"],
        default="tfds",
        help="Dataset source: local directory or TensorFlow Datasets",
    )
    parser.add_argument(
        "--data_dir",
        default="data/PlantVillage",
        help="Path to PlantVillage directory (used when --source=dir)",
    )
    parser.add_argument(
        "--tfds_data_dir",
        default=".tfds",
        help="TFDS data directory (used when --source=tfds)",
    )
    parser.add_argument("--out_dir", required=True, help="SavedModel output directory")
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--image_size", type=int, default=224)
    args = parser.parse_args()

    class_names = classes_for_plant(args.plant)
    if args.source == "dir":
        splits = load_splits(
            data_dir=args.data_dir,
            class_names=class_names,
            image_size=(args.image_size, args.image_size),
            batch_size=args.batch_size,
        )
    else:
        splits = load_splits_tfds(
            tfds_data_dir=args.tfds_data_dir,
            class_names=class_names,
            image_size=(args.image_size, args.image_size),
            batch_size=args.batch_size,
        )

    model = build_classifier(
        num_classes=len(class_names),
        image_size=(args.image_size, args.image_size),
        base_trainable=False,
    )

    callbacks: list[tf.keras.callbacks.Callback] = [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_acc", patience=3, restore_best_weights=True
        ),
        tf.keras.callbacks.ReduceLROnPlateau(monitor="val_acc", factor=0.5, patience=2),
    ]

    # Phase 1: train head with frozen backbone
    model.fit(
        splits.train,
        validation_data=splits.val,
        epochs=max(1, args.epochs // 2),
        callbacks=callbacks,
        verbose=1,
    )

    # Phase 2: fine-tune last backbone layers
    set_backbone_trainable(model, trainable=True, unfreeze_last_n=40)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(),
        metrics=[tf.keras.metrics.SparseCategoricalAccuracy(name="acc")],
    )

    model.fit(
        splits.train,
        validation_data=splits.val,
        epochs=args.epochs,
        initial_epoch=max(1, args.epochs // 2),
        callbacks=callbacks,
        verbose=1,
    )

    test_acc = _evaluate_accuracy(model, splits.test)

    calib: TemperatureScalingResult = temperature_scale_fit(
        model=model, val_ds=splits.val, max_steps=200, lr=0.05
    )

    test_stats = _confidence_stats(model, splits.test, temperature=calib.temperature)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    # Keras 3: `model.save()` requires a file extension. We save a portable
    # `.keras` artifact for this app, and also export a TF SavedModel for
    # TF-Serving/TFLite workflows.
    keras_path = out_dir / "model.keras"
    model.save(keras_path)

    saved_model_dir = out_dir / "saved_model"
    model.export(saved_model_dir)

    meta = {
        "plant": args.plant,
        "class_names": class_names,
        "image_size": [args.image_size, args.image_size],
        "test_top1_acc": test_acc,
        "temperature": calib.temperature,
        "val_nll_before": calib.val_nll_before,
        "val_nll_after": calib.val_nll_after,
        "test_confidence_stats": test_stats,
    }
    (out_dir / "calibration.json").write_text(json.dumps(meta, indent=2))

    print("=== Results ===")
    print(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()

