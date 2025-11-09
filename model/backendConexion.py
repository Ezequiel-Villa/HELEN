"""Client helper for publishing gesture events to the local/web backend."""
from __future__ import annotations

import os
import logging

import requests

LOGGER = logging.getLogger("helen.backendConexion")

SERVER = os.getenv("HELEN_BACKEND_GESTURE_URL", "http://127.0.0.1:5000/gestures/gesture-key")

def post_gesturekey(payload: dict):
    try:
        r = requests.post(SERVER, json=payload, timeout=1.5)
        return r.status_code
    except Exception as e:
        LOGGER.warning("post_gesturekey error: %s", e)
        return 0
