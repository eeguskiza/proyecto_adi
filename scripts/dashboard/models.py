from typing import Dict

import numpy as np
import requests


def call_bentoml_scrap(endpoint: str, payload: Dict[str, object]) -> Dict[str, object]:
    try:
        resp = requests.post(endpoint, json=payload, timeout=5)
        if resp.ok:
            return resp.json()
        return {"error": f"Status {resp.status_code}"}
    except Exception as exc:  # pragma: no cover - network call
        return {"error": str(exc)}
