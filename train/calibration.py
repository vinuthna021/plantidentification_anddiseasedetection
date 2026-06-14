from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import tensorflow as tf


@dataclass(frozen=True)
class TemperatureScalingResult:
    temperature: float
    val_nll_before: float
    val_nll_after: float


def _probs_to_logits(probs: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    p = np.clip(probs, eps, 1.0)
    return np.log(p)


def temperature_scale_fit(
    *,
    model: tf.keras.Model,
    val_ds: tf.data.Dataset,
    max_steps: int = 200,
    lr: float = 0.05,
) -> TemperatureScalingResult:
    """
    Fits a single temperature T>0 on validation data to calibrate probabilities.

    We treat the model output as probabilities and convert to logits via log(p).
    Then we optimize T to minimize NLL of softmax(logits / T).
    """
    y_true_list: list[np.ndarray] = []
    probs_list: list[np.ndarray] = []

    for x, y in val_ds:
        p = model.predict(x, verbose=0)
        probs_list.append(p)
        y_true_list.append(y.numpy())

    probs = np.concatenate(probs_list, axis=0)
    y_true = np.concatenate(y_true_list, axis=0).astype(np.int32)
    logits = _probs_to_logits(probs)

    logits_tf = tf.constant(logits, dtype=tf.float32)
    y_true_tf = tf.constant(y_true, dtype=tf.int32)

    t = tf.Variable(1.0, dtype=tf.float32, constraint=lambda z: tf.maximum(z, 1e-3))
    opt = tf.keras.optimizers.Adam(learning_rate=lr)

    def nll(temp: tf.Tensor) -> tf.Tensor:
        scaled = logits_tf / temp
        return tf.reduce_mean(
            tf.keras.losses.sparse_categorical_crossentropy(
                y_true_tf, tf.nn.softmax(scaled), from_logits=False
            )
        )

    before = float(nll(t).numpy())

    for _ in range(max_steps):
        with tf.GradientTape() as tape:
            loss = nll(t)
        grad = tape.gradient(loss, [t])
        opt.apply_gradients(zip(grad, [t]))

    after = float(nll(t).numpy())
    return TemperatureScalingResult(
        temperature=float(t.numpy()),
        val_nll_before=before,
        val_nll_after=after,
    )


def apply_temperature(
    probs: np.ndarray, *, temperature: float, eps: float = 1e-8
) -> np.ndarray:
    logits = _probs_to_logits(probs, eps=eps)
    scaled = logits / float(max(temperature, 1e-6))
    exp = np.exp(scaled - np.max(scaled, axis=-1, keepdims=True))
    return exp / np.sum(exp, axis=-1, keepdims=True)

