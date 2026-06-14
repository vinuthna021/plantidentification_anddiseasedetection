from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import tensorflow as tf
import tensorflow_datasets as tfds


@dataclass(frozen=True)
class DatasetSplits:
    train: tf.data.Dataset
    val: tf.data.Dataset
    test: tf.data.Dataset
    class_names: list[str]


def _assert_class_folders(data_dir: Path, class_names: Iterable[str]) -> None:
    missing = [c for c in class_names if not (data_dir / c).exists()]
    if missing:
        raise FileNotFoundError(
            "Dataset is missing expected class folders under "
            f"{str(data_dir)!r}: {missing}"
        )


def load_splits(
    *,
    data_dir: str | Path,
    class_names: list[str],
    image_size: tuple[int, int] = (224, 224),
    batch_size: int = 32,
    seed: int = 1337,
) -> DatasetSplits:
    """
    Loads a labeled dataset from a PlantVillage-like directory:
      data_dir/<class_name>/*.jpg

    We use deterministic TF splitting (subset="training"/"validation") and then
    split that validation portion into val/test.
    """
    data_dir = Path(data_dir)
    _assert_class_folders(data_dir, class_names)

    train = tf.keras.utils.image_dataset_from_directory(
        data_dir,
        labels="inferred",
        label_mode="int",
        class_names=class_names,
        validation_split=0.2,
        subset="training",
        seed=seed,
        image_size=image_size,
        batch_size=batch_size,
        shuffle=True,
    )
    val_plus_test = tf.keras.utils.image_dataset_from_directory(
        data_dir,
        labels="inferred",
        label_mode="int",
        class_names=class_names,
        validation_split=0.2,
        subset="validation",
        seed=seed,
        image_size=image_size,
        batch_size=batch_size,
        shuffle=True,
    )

    # Split the 20% holdout into 10% val and 10% test.
    cardinality = tf.data.experimental.cardinality(val_plus_test).numpy()
    if cardinality <= 2:
        raise RuntimeError(
            "Not enough batches in validation subset to split into val/test. "
            "Add more data or reduce batch_size."
        )
    test_batches = max(1, cardinality // 2)
    test = val_plus_test.take(test_batches)
    val = val_plus_test.skip(test_batches)

    autotune = tf.data.AUTOTUNE
    train = train.cache().prefetch(autotune)
    val = val.cache().prefetch(autotune)
    test = test.cache().prefetch(autotune)

    return DatasetSplits(train=train, val=val, test=test, class_names=class_names)


def load_splits_tfds(
    *,
    tfds_data_dir: str | Path,
    class_names: list[str],
    image_size: tuple[int, int] = (224, 224),
    batch_size: int = 32,
    seed: int = 1337,
) -> DatasetSplits:
    """
    Loads PlantVillage from TensorFlow Datasets (`plant_village`) and filters it down
    to the provided `class_names` (must match TFDS label names exactly).

    Split strategy:
      - Filter to the requested labels
      - Shuffle once deterministically
      - Count examples (single pass)
      - Slice into 80% train, 10% val, 10% test
    """
    tfds_data_dir = str(tfds_data_dir)

    builder = tfds.builder("plant_village", data_dir=tfds_data_dir)
    label_names = builder.info.features["label"].names
    name_to_id = {n: i for i, n in enumerate(label_names)}

    missing = [c for c in class_names if c not in name_to_id]
    if missing:
        raise ValueError(f"Unknown TFDS label(s): {missing}")

    allowed_ids = tf.constant([name_to_id[c] for c in class_names], dtype=tf.int64)

    def is_allowed(_, y):
        y = tf.cast(y, tf.int64)
        return tf.reduce_any(tf.equal(allowed_ids, y))

    # Remap TFDS label ids -> 0..N-1 in the same order as class_names.
    remap_table = tf.lookup.StaticHashTable(
        tf.lookup.KeyValueTensorInitializer(
            keys=allowed_ids,
            values=tf.range(tf.shape(allowed_ids)[0], dtype=tf.int64),
        ),
        default_value=-1,
    )

    def preprocess(x, y):
        x = tf.image.convert_image_dtype(x, tf.float32)  # -> [0, 1]
        x = tf.image.resize(x, image_size, method="bilinear")
        y = tf.cast(y, tf.int64)
        y = remap_table.lookup(y)
        return x, tf.cast(y, tf.int32)

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
        """Batch/cache/prefetch and set an explicit (batch) cardinality."""
        split = split.batch(batch_size, drop_remainder=False)
        n_batches = int((n_examples + batch_size - 1) // batch_size)
        split = split.apply(tf.data.experimental.assert_cardinality(n_batches))
        split = split.cache().prefetch(tf.data.AUTOTUNE)
        return split

    train = finalize(ds.take(n_train), n_examples=n_train)
    val = finalize(ds.skip(n_train).take(n_val), n_examples=n_val)
    test = finalize(ds.skip(n_train + n_val).take(n_test), n_examples=n_test)

    return DatasetSplits(train=train, val=val, test=test, class_names=class_names)

