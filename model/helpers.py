from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, Optional


labels_dict = {
    0: '0',
    1: '1',
    2: '2',
    3: '3',
    4: '4',
    5: '5',
    6: '6',
    7: '7',
    8: '8',
    9: '9',
    10: 'Start',        # H
    11: 'Clima',        # C
    12: 'Foco',         # L
    13: 'Ajustes',      # A
    14: 'Inicio',       # I
    15: 'Dispositivos', # D
    16: 'Reloj',        # R
    17: 'Wifi',
    18: 'Alarma',
    19: 'Tutorial',
    20: 'Anterior',
    21: 'Agregar',
}


def _iso_now() -> str:
    return datetime.now(tz=timezone.utc).isoformat()


@dataclass
class GestureData:
    gesture: Optional[str] = None
    character: Optional[str] = None
    score: float = 0.0
    timestamp: str = field(default_factory=_iso_now)
    session_id: Optional[str] = None
    sequence: Optional[int] = None
    source: str = "python-client"

    def to_payload(self) -> Dict[str, Any]:
        payload = asdict(self)
        # Conservar 'gesture' y 'character' aunque sean None para mantener el contrato antiguo.
        preserved = {'gesture', 'character'}
        return {
            key: value
            for key, value in payload.items()
            if value is not None or key in preserved
        }
