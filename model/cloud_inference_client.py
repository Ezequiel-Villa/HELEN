"""HTTP client used by the Raspberry Pi frontend to reach the cloud API."""
from __future__ import annotations

import json
import logging
import os
import time
from typing import Any, Dict, Optional, Sequence

import numpy as np
import requests

LOGGER = logging.getLogger("helen.cloud_inference_client")

_DEFAULT_TIMEOUT = float(os.getenv("HELEN_CLOUD_TIMEOUT", "5.0"))
_MAX_RETRIES = int(os.getenv("HELEN_CLOUD_RETRIES", "2"))
_METADATA_TTL = float(os.getenv("HELEN_CLOUD_METADATA_TTL", "300"))

_API_BASE = os.getenv("HELEN_CLOUD_API_URL", "http://127.0.0.1:8000").rstrip("/")
_SESSION = requests.Session()

_METADATA_CACHE: Optional[Dict[str, Any]] = None
_METADATA_TS = 0.0


def _url(path: str) -> str:
    if not path.startswith("/"):
        path = "/" + path
    return f"{_API_BASE}{path}"


def _request_with_retries(method: str, url: str, *, json_payload: Optional[Dict[str, Any]] = None, timeout: Optional[float] = None) -> Optional[requests.Response]:
    timeout = timeout or _DEFAULT_TIMEOUT
    for attempt in range(_MAX_RETRIES + 1):
        try:
            resp = _SESSION.request(method, url, json=json_payload, timeout=timeout)
            resp.raise_for_status()
            return resp
        except requests.RequestException as exc:
            LOGGER.warning("Cloud request failed (%s/%s): %s", attempt + 1, _MAX_RETRIES + 1, exc)
            time.sleep(min(2 ** attempt * 0.2, 2.0))
    return None


def get_model_metadata(force_refresh: bool = False, *, timeout: Optional[float] = None) -> Optional[Dict[str, Any]]:
    global _METADATA_CACHE, _METADATA_TS
    now = time.time()
    if not force_refresh and _METADATA_CACHE and (now - _METADATA_TS) < _METADATA_TTL:
        return _METADATA_CACHE

    resp = _request_with_retries("GET", _url("/metadata"), timeout=timeout)
    if not resp:
        return None
    try:
        data = resp.json()
    except json.JSONDecodeError:
        LOGGER.error("Invalid JSON in metadata response")
        return None

    _METADATA_CACHE = data
    _METADATA_TS = now
    return data


def get_prediction_from_cloud(
    sequence: Sequence[Sequence[float]] | np.ndarray,
    mask: Optional[Sequence[Sequence[float]] | np.ndarray] = None,
    *,
    timeout: Optional[float] = None,
) -> Optional[Dict[str, Any]]:
    """Send the feature sequence to the cloud API and return the JSON reply."""

    if isinstance(sequence, np.ndarray):
        seq_payload = sequence.tolist()
    else:
        seq_payload = [list(row) for row in sequence]

    payload: Dict[str, Any] = {"sequence": seq_payload}

    if mask is not None:
        if isinstance(mask, np.ndarray):
            payload["mask"] = mask.tolist()
        else:
            payload["mask"] = [list(row) for row in mask]

    resp = _request_with_retries("POST", _url("/predict"), json_payload=payload, timeout=timeout)
    if not resp:
        return None

    try:
        return resp.json()
    except json.JSONDecodeError:
        LOGGER.error("Invalid JSON in prediction response")
        return None
