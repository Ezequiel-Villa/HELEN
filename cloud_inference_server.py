"""Flask application that exposes the gesture classifier as a cloud service.

The server is intended to run on the EC2 instance.  It loads the neural network
checkpoint once on startup and keeps it in memory so that inference requests can
reuse the same model instance.  Run in development mode with
``python cloud_inference_server.py`` or in production with Gunicorn:
``gunicorn cloud_inference_server:app --bind 0.0.0.0:5000``.
"""
from __future__ import annotations

import logging
import os
import time
from pathlib import Path
from typing import Any, Dict, Iterable, Optional

import numpy as np
from flask import Flask, jsonify, request

from model.cloud_inference import ModelBundle, load_checkpoint, run_inference

LOGGER = logging.getLogger("helen.cloud_inference_server")


def _configure_logging() -> None:
    level = os.getenv("HELEN_LOG_LEVEL", "INFO").upper()
    logging.basicConfig(
        level=getattr(logging, level, logging.INFO),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )


def _parse_model_paths(raw: Optional[str]) -> Iterable[Path]:
    if raw:
        for part in raw.split(os.pathsep):
            if part:
                yield Path(part)
    # Default fallback order
    yield Path("model/models/sequence_bigru.pt")
    yield Path("models/sequence_bigru.pt")
    yield Path("/mnt/data/sequence_bigru.pt")


_configure_logging()

try:
    _MODEL_BUNDLE: ModelBundle = load_checkpoint(_parse_model_paths(os.getenv("HELEN_MODEL_PATHS")))
    LOGGER.info(
        "Loaded model with seq_len=%s, classes=%s",
        _MODEL_BUNDLE.seq_len,
        ",".join(_MODEL_BUNDLE.class_labels),
    )
except Exception as exc:  # pragma: no cover - startup failure is fatal
    LOGGER.exception("Unable to load model checkpoint: %s", exc)
    raise

app = Flask(__name__)


@app.get("/health")
def health() -> Any:
    return jsonify({"status": "ok", "classes": _MODEL_BUNDLE.num_classes})


@app.get("/metadata")
def metadata() -> Any:
    meta = {
        "seq_len": _MODEL_BUNDLE.seq_len,
        "use_z": _MODEL_BUNDLE.use_z,
        "num_classes": _MODEL_BUNDLE.num_classes,
        "class_labels": _MODEL_BUNDLE.class_labels,
        "feature_dim": _MODEL_BUNDLE.feature_dim,
    }
    return jsonify(meta)


@app.post("/predict")
def predict() -> Any:
    started = time.perf_counter()
    payload: Optional[Dict[str, Any]] = request.get_json(silent=True)
    if not payload:
        return jsonify({"error": "Missing JSON body"}), 400

    sequence = payload.get("sequence")
    mask = payload.get("mask")

    if sequence is None:
        return jsonify({"error": "Field 'sequence' is required"}), 400

    try:
        seq_array = np.asarray(sequence, dtype=np.float32)
        if seq_array.ndim != 2:
            raise ValueError("sequence must be 2D")
        if seq_array.size == 0:
            raise ValueError("sequence must not be empty")
        if seq_array.shape[1] != _MODEL_BUNDLE.feature_dim:
            raise ValueError(
                f"expected feature_dim={_MODEL_BUNDLE.feature_dim}, got {seq_array.shape[1]}"
            )
    except Exception as exc:  # pragma: no cover - defensive against bad input
        LOGGER.warning("Invalid sequence payload: %s", exc)
        return jsonify({"error": "Invalid 'sequence' format"}), 400

    mask_array = None
    if mask is not None:
        try:
            mask_array = np.asarray(mask, dtype=np.float32)
            if mask_array.ndim != 2:
                raise ValueError("mask must be 2D")
            if mask_array.shape[0] != seq_array.shape[0]:
                raise ValueError("mask length must match sequence length")
            if mask_array.shape[1] != 2:
                raise ValueError("mask must have shape (T, 2)")
        except Exception as exc:
            LOGGER.warning("Invalid mask payload: %s", exc)
            return jsonify({"error": "Invalid 'mask' format"}), 400

    try:
        probs = run_inference(_MODEL_BUNDLE, seq_array, mask_array)
    except Exception as exc:  # pragma: no cover - defensive
        LOGGER.exception("Inference failed: %s", exc)
        return jsonify({"error": "Inference failed"}), 500

    top_index = int(probs.argmax())
    top_prob = float(probs[top_index])
    prediction = _MODEL_BUNDLE.class_labels[top_index]
    elapsed_ms = (time.perf_counter() - started) * 1000.0

    LOGGER.info(
        "Inference completed in %.2f ms (label=%s prob=%.3f)",
        elapsed_ms,
        prediction,
        top_prob,
    )

    response = {
        "prediction": prediction,
        "probabilities": probs.tolist(),
        "top_index": top_index,
        "top_probability": top_prob,
        "elapsed_ms": elapsed_ms,
    }
    return jsonify(response)


if __name__ == "__main__":  # pragma: no cover - manual run helper
    app.run(host="0.0.0.0", port=int(os.getenv("PORT", "5000")), debug=os.getenv("FLASK_DEBUG") == "1")
