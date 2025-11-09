"""Utilities for loading and running the gesture classifier in a reusable way.

This module is intended to be imported by the cloud inference server so that
model initialization happens only once per process.  It wraps the PyTorch
classifier together with metadata extracted from the training checkpoint.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Sequence

import numpy as np
import torch
import torch.nn as nn

from .helpers import labels_dict as helpers_labels


class BiGRUClassifier(nn.Module):
    """Sequence classifier with a GRU backbone and a linear head."""

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        num_layers: int,
        num_classes: int,
        dropout: float = 0.2,
        bidirectional: bool = True,
    ) -> None:
        super().__init__()
        self.gru = nn.GRU(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
            bidirectional=bidirectional,
        )
        d = 2 if bidirectional else 1
        self.head = nn.Sequential(
            nn.LayerNorm(d * hidden_dim),
            nn.Dropout(dropout),
            nn.Linear(d * hidden_dim, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # pragma: no cover - tiny wrapper
        out, _ = self.gru(x)
        return self.head(out[:, -1, :])


@dataclass
class ModelBundle:
    model: BiGRUClassifier
    meta: dict
    device: torch.device
    class_labels: List[str]
    feature_dim: int

    @property
    def seq_len(self) -> int:
        return int(self.meta.get("seq_len", 30))

    @property
    def use_z(self) -> bool:
        return bool(self.meta.get("use_z", False))

    @property
    def num_classes(self) -> int:
        return int(self.meta.get("num_classes", len(self.class_labels)))


def load_checkpoint(paths: Iterable[Path | str]) -> ModelBundle:
    """Load the trained model and associated metadata from disk.

    The first existing checkpoint path wins.  We keep the model on CPU because
    the target EC2 instance does not provide a GPU.
    """

    for path in paths:
        p = Path(path)
        if not p.exists():
            continue
        state = torch.load(p, map_location="cpu")
        cfg = state["config"]
        meta = state.get("meta", {})
        model = BiGRUClassifier(
            cfg["input_dim"],
            cfg["hidden_dim"],
            cfg["num_layers"],
            cfg["num_classes"],
            cfg["dropout"],
            cfg["bidirectional"],
        )
        model.load_state_dict(state["model_state"])
        device = torch.device("cpu")
        model.to(device).eval()

        classes_numeric: Sequence[int] = meta.get("classes_numeric", [])
        class_labels = [helpers_labels.get(orig_val, str(orig_val)) for orig_val in classes_numeric]
        if not class_labels:
            class_labels = [str(i) for i in range(cfg["num_classes"])]

        return ModelBundle(
            model=model,
            meta=meta,
            device=device,
            class_labels=class_labels,
            feature_dim=int(cfg["input_dim"]),
        )

    raise FileNotFoundError("No checkpoint found in provided paths")


def run_inference(bundle: ModelBundle, sequence: np.ndarray, mask: np.ndarray | None = None) -> np.ndarray:
    """Run inference on a (T, F) feature sequence and return probabilities."""

    if sequence.ndim != 2:
        raise ValueError("Expected 2D feature array (T, F)")
    if sequence.shape[1] != bundle.feature_dim:
        raise ValueError(f"Expected feature dimension {bundle.feature_dim}, got {sequence.shape[1]}")

    seq_len = min(len(sequence), bundle.seq_len)
    feats = sequence[-seq_len:]

    if mask is not None and mask.ndim == 2:
        mask_seq = mask[-seq_len:]
        F_per_hand = feats.shape[1] // 2
        mask_full = np.concatenate(
            [
                np.repeat(mask_seq[:, 0:1], F_per_hand, axis=1),
                np.repeat(mask_seq[:, 1:2], F_per_hand, axis=1),
            ],
            axis=1,
        )
        feats = feats * mask_full

    with torch.no_grad():
        tensor = torch.from_numpy(feats[None, ...].astype(np.float32)).to(bundle.device)
        logits = bundle.model(tensor)
        probs = torch.softmax(logits, dim=1).cpu().numpy()[0]
    return probs
